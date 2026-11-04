"""Inpaint-only RGB-anchored depth correction. No fitted plane or GT geometry.

RGB tracks are triangulated without a depth prior. A smooth *correction* to each
camera's z-depth is fitted, not a smooth scene surface. Unsupported regions are
left out; internally consistent hallucinations are not ground truth confidence.
"""
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_erosion
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from source.paintmesh_edgs_init import camera, project, sample, select_pairs, unproject


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def close_camera_pair_graph(pairs, cameras):
    """Add a third matched edge for each reference's widest neighboring baseline.

    Neighbor graphs may be chains with no triangles. This is a matching request,
    not a claim of correspondence validity; pixel closure must still pass later.
    The legacy mixed initializer never calls this function.
    """
    result = {tuple(sorted(pair)) for pair in pairs}
    neighbors = {i: set() for i in range(len(cameras))}
    for a, b in result:
        neighbors[a].add(b); neighbors[b].add(a)
    centers = np.array([cam["c2w"][:3, 3] for cam in cameras])
    for frame in range(len(cameras)):
        candidates = list(combinations(sorted(neighbors[frame]), 2))
        if candidates:
            a, b = max(candidates, key=lambda pair: np.linalg.norm(centers[pair[0]]-centers[pair[1]]))
            result.add((a, b))
    return sorted(result)


def triangulate_tracks(cameras, observations):
    """Three-view RGB-only DLT + robust reprojection refinement, pixel sigma=1.

    Covariance is conditional on correspondences/calibration being correct; a
    singular information matrix must NEVER look certain through a pseudo-inverse.
    """
    n = len(observations[0])
    if not n:
        return np.empty((0, 3)), np.empty((0, len(cameras))), np.empty((0, 3, 3))
    equations = []
    for cam, uv in zip(cameras, observations):
        p = cam["k"] @ cam["w2c"][:3]
        equations.extend((uv[:, 0, None] * p[2] - p[0], uv[:, 1, None] * p[2] - p[1]))
    _, _, vh = np.linalg.svd(np.stack(equations, axis=1), full_matrices=False)
    h = vh[:, -1]
    valid = np.abs(h[:, 3]) > 1e-12
    xyz = h[:, :3] / np.where(valid, h[:, 3], 1)[:, None]
    information = np.zeros((n, 3, 3))
    for iteration in range(5):
        information.fill(0); rhs = np.zeros_like(xyz)
        for cam, uv in zip(cameras, observations):
            r = cam["w2c"][:3, :3]
            pc = xyz @ r.T + cam["w2c"][:3, 3]
            z = np.maximum(pc[:, 2], 1e-8)
            for axis in range(2):
                residual = cam["k"][axis, axis] * pc[:, axis] / z + cam["k"][axis, 2] - uv[:, axis]
                jac = cam["k"][axis, axis] * (r[axis] / z[:, None] - pc[:, axis, None] * r[2] / z[:, None]**2)
                weight = 1 / np.maximum(1, np.abs(residual))
                information += np.einsum("ni,nj,n->nij", jac, jac, weight)
                rhs += jac * (residual * weight)[:, None]
        if iteration < 4:
            # A fixed epsilon disappears numerically for zero-baseline/near-camera
            # tracks. Scale damping by the information matrix; reject rank below.
            damping = np.maximum(1e-8, np.trace(information, axis1=1, axis2=2) * 1e-8)
            xyz -= np.linalg.solve(information + np.eye(3)[None] * damping[:, None, None], rhs)
    eigen = np.linalg.eigvalsh(information)
    valid &= np.isfinite(xyz).all(1) & (eigen[:, 0] > np.maximum(1e-12, eigen[:, -1] * 1e-10))
    covariance = np.zeros_like(information)
    covariance[valid] = np.linalg.inv(information[valid])
    covariance[~valid] = np.eye(3) * 1e30
    errors = np.stack([np.linalg.norm(project(xyz, cam)[0] - uv, axis=1)
                       for cam, uv in zip(cameras, observations)], axis=1)
    errors[~valid] = np.inf
    return xyz, errors, covariance


def rgb_anchors(support, cameras, cfg, init, rgb_pairs=None):
    """Close three existing bidirectional pair matches in IMAGE space.

    Raw pair observations are retained before depth refinement for new runs.
    Historical proposals only have surviving pair observations; this is reported
    and cannot invent missing tracks. No 3-D nearest-neighbor depth snapping.
    """
    if rgb_pairs is None:
        rgb_pairs = []
        pairs = np.column_stack((support["frame"], support["other_frame"]))
        for a, b in np.unique(pairs[support["source_kind"] == 0], axis=0):
            rows = (pairs[:, 0] == a) & (pairs[:, 1] == b) & (support["source_kind"] == 0)
            rgb_pairs.append((int(a), int(b), support["pixel_uv"][rows],
                              support["other_uv"][rows], support["confidence"][rows]))
    pool = {}
    for a, b, uv, other, scores in rgb_pairs:
        if a == b or not 0 <= min(a, b) <= max(a, b) < len(cameras):
            continue
        uv, other, scores = np.asarray(uv), np.asarray(other), np.asarray(scores)
        keep = np.isfinite(uv).all(1) & np.isfinite(other).all(1) & np.isfinite(scores)
        keep &= (scores >= init["confidence_min"]) & (scores <= 1)
        for pixels, cam in ((uv, cameras[a]), (other, cameras[b])):
            keep &= (pixels >= 0).all(1) & (pixels <= [cam["w"]-1, cam["h"]-1]).all(1)
        if a > b:
            a, b, uv, other = b, a, other, uv
        if keep.any():
            pool.setdefault((a, b), []).append(np.column_stack((uv[keep], other[keep], scores[keep])))
    pool = {key: np.concatenate(value) for key, value in pool.items()}
    trees = {key: cKDTree(value[:, :2]) for key, value in pool.items()}
    adjacency = {}
    for a, b in pool:
        adjacency.setdefault(a, set()).add(b)
    output = []; tested = 0
    gap = float(support["spacing"])
    for a, bs in sorted(adjacency.items()):
        for b, c in combinations(sorted(bs), 2):
            if (b, c) not in pool:
                continue
            ab, ac, bc = pool[a, b], pool[a, c], pool[b, c]
            da, ia = trees[a, c].query(ab[:, :2])
            db, ib = trees[b, c].query(ab[:, 2:4])
            valid = (da <= cfg["track_join_pixels"]) & (db <= cfg["track_join_pixels"])
            valid &= np.linalg.norm(ac[ia, 2:4] - bc[ib, 2:4], axis=1) <= cfg["track_join_pixels"]
            rows = np.flatnonzero(valid)
            # Count one measurement per reference pixel, not many duplicated pair samples.
            if len(rows):
                _, first = np.unique(np.floor(ab[rows, :2]).astype(int), axis=0, return_index=True)
                rows = rows[np.sort(first)]
            for start in range(0, len(rows), 4096):
                r = rows[start:start+4096]; tested += len(r)
                observations = [(ab[r, :2] + ac[ia[r], :2]) * .5,
                    (ab[r, 2:4] + bc[ib[r], :2]) * .5, (ac[ia[r], 2:4] + bc[ib[r], 2:4]) * .5]
                cams = [cameras[k] for k in (a, b, c)]
                xyz, errors, covariance = triangulate_tracks(cams, observations)
                good = np.isfinite(errors).all(1) & (errors.max(1) <= init["reprojection_pixels"])
                rays = []
                for cam in cams:
                    good &= project(xyz, cam)[2]
                    ray = xyz - cam["c2w"][:3, 3]
                    rays.append(ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12))
                angle = np.max([np.degrees(np.arccos(np.clip((rays[i]*rays[j]).sum(1), -1, 1)))
                                for i, j in combinations(range(3), 2)], axis=0)
                sigma = np.sqrt(np.linalg.eigvalsh(covariance)[:, -1]) / gap
                good &= (angle >= cfg["anchor_min_angle_deg"]) & (sigma <= cfg["anchor_max_sigma_spacing"])
                if good.any():
                    output.append((xyz[good], covariance[good], np.stack(observations, 1)[good],
                                   np.tile([a, b, c], (int(good.sum()), 1)), errors[good].max(1)))
            if init["max_points"] is not None and sum(len(v[0]) for v in output) > init["max_points"] * 8:
                raise ValueError("RGB anchor pool exceeds 8 * max_points")
    names = ("xyz", "covariance", "uv", "frames", "reprojection")
    if output:
        anchors = {name: np.concatenate([row[i] for row in output]) for i, name in enumerate(names)}
    else:
        anchors = dict(xyz=np.empty((0, 3)), covariance=np.empty((0, 3, 3)), uv=np.empty((0, 3, 2)),
                       frames=np.empty((0, 3), int), reprojection=np.empty(0))
    return anchors, dict(pair_count=len(pool), closed_tracks=tested, accepted=len(anchors["xyz"]))


def grid_basis(uv, origin, shape, step):
    q = np.clip((np.asarray(uv) - origin) / step, 0, np.array(shape[::-1]) - 1 - 1e-9)
    xy = np.floor(q).astype(int); t = q - xy
    columns = np.stack([xy[:, 1]*shape[1]+xy[:, 0], xy[:, 1]*shape[1]+xy[:, 0]+1,
                        (xy[:, 1]+1)*shape[1]+xy[:, 0], (xy[:, 1]+1)*shape[1]+xy[:, 0]+1], axis=1)
    values = np.stack([(1-t[:, 0])*(1-t[:, 1]), t[:, 0]*(1-t[:, 1]),
                        (1-t[:, 0])*t[:, 1], t[:, 0]*t[:, 1]], axis=1)
    return sparse.csr_matrix((values.ravel(), (np.repeat(np.arange(len(uv)), 4), columns.ravel())),
                             shape=(len(uv), int(np.prod(shape))))


def editable_rows(mask, uv):
    """All four bilinear pixels must be editable; no clamping from outside."""
    finite = np.isfinite(uv).all(1)
    xy = np.floor(np.where(finite[:, None], uv, 0)).astype(int); h, w = mask.shape
    valid = finite & (xy >= 0).all(1) & (xy[:, 0] < w-1) & (xy[:, 1] < h-1)
    safe = np.where(valid[:, None], xy, 0)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        valid &= mask[safe[:, 1]+dy, safe[:, 0]+dx]
    return valid


def fit_correction(uv, residual, uncertainty, origin, shape, step, boundary_uv,
                   smooth_weight=.3, ridge_weight=.001, edge_valid=None):
    """Huber IRLS; uncertainty and residual are in world-spacing depth units."""
    a = grid_basis(uv, origin, shape, step)
    boundary = grid_basis(boundary_uv, origin, shape, step)
    ids = np.arange(np.prod(shape)).reshape(shape)
    left = np.r_[ids[:, :-1].ravel(), ids[:-1].ravel()]
    right = np.r_[ids[:, 1:].ravel(), ids[1:].ravel()]
    if edge_valid is not None:
        left, right = left[edge_valid], right[edge_valid]
    reg = sparse.csr_matrix((np.tile([1., -1.], len(left)),
        (np.repeat(np.arange(len(left)), 2), np.column_stack((left, right)).ravel())), shape=(len(left), a.shape[1]))
    fixed = sparse.vstack((boundary * 4, reg * np.sqrt(smooth_weight),
                           sparse.eye(a.shape[1]) * np.sqrt(ridge_weight)), format="csr")
    sigma = np.maximum(uncertainty, .25); solution = np.zeros(a.shape[1])
    weights = np.ones(len(uv))
    for _ in range(8):
        standard = (a @ solution - residual) / sigma
        weights = np.minimum(1., 1.5 / np.maximum(np.abs(standard), 1e-12))
        w = np.sqrt(weights) / sigma
        system = sparse.vstack((a.multiply(w[:, None]), fixed), format="csr")
        fit = lsqr(system, np.r_[residual*w, np.zeros(fixed.shape[0])], atol=1e-8, btol=1e-8, iter_lim=2000)
        if fit[1] not in (0, 1, 2, 4, 5):
            raise ValueError("depth correction linear solver did not converge")
        solution = fit[0]
    return solution, np.abs((a @ solution-residual) / sigma) <= 2.5


def depth_field(cam, target, anchors, frame, gap, cfg):
    """Fit bias only where anchors support the same continuous depth component."""
    depth, mask = array(target["depth"]), array(target["mask"]).astype(bool)
    corrected = depth.copy(); supported = np.zeros(mask.shape, bool)
    stats = dict(frame=frame, observations=0, inliers=0, supported_pixels=0)
    rows, columns = np.where(anchors["frames"] == frame)
    uv = anchors["uv"][rows, columns]
    valid = editable_rows(mask & (depth > 0), uv)
    rows, uv = rows[valid], uv[valid]
    if len(rows) < cfg["min_anchor_observations"]:
        return corrected, supported, {}, stats
    # A track repeated in multiple triangles is still only one observation.
    _, first = np.unique(np.floor(uv).astype(int), axis=0, return_index=True)
    rows, uv = rows[first], uv[first]
    step = cfg["grid_pixels"]
    yy, xx = np.where(mask & (depth > 0))
    origin = np.array([xx.min(), yy.min()], float) - step
    shape = tuple((np.ceil(([yy.max(), xx.max()] - origin[::-1]) / step).astype(int) + 2).tolist())
    gy, gx = np.indices(shape); nodes = np.column_stack((gx.ravel(), gy.ravel())) * step + origin
    node_ok = editable_rows(mask & (depth > 0), nodes)
    nd = sample(depth, nodes)
    ids = np.arange(len(nodes)).reshape(shape)
    left = np.r_[ids[:, :-1].ravel(), ids[:-1].ravel()]
    right = np.r_[ids[:, 1:].ravel(), ids[1:].ravel()]
    mid = (nodes[left] + nodes[right]) * .5
    edges = node_ok[left] & node_ok[right] & editable_rows(mask & (depth > 0), mid)
    edges &= np.abs(nd[left]-nd[right]) <= cfg["depth_edge_relative"] * np.maximum(np.minimum(nd[left], nd[right]), 1e-8)
    # Check edge interiors as well; do not smooth across a narrow depth step/hole.
    for fraction in (.25, .5, .75):
        position = nodes[left] * (1-fraction) + nodes[right] * fraction
        middle_depth = sample(depth, position)
        edges &= editable_rows(mask & (depth > 0), position)
        edges &= np.abs(middle_depth - (nd[left]*(1-fraction)+nd[right]*fraction)) <= cfg["depth_edge_relative"] * np.maximum(middle_depth, 1e-8)
    graph = sparse.csr_matrix((np.ones(int(edges.sum())), (left[edges], right[edges])), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph, directed=False)

    def components(pixels):
        basis = grid_basis(pixels, origin, shape, step)
        corners = basis.indices.reshape(-1, 4)
        label = labels[corners[:, 0]]
        ok = (labels[corners] == label[:, None]).all(1) & node_ok[corners].all(1)
        # A graph can reconnect around a discontinuity: reject such cells explicitly.
        values = nd[corners]
        ok &= np.ptp(values, axis=1) <= cfg["depth_edge_relative"] * np.maximum(values.min(1), 1e-8)
        return np.where(ok, label, -1)

    component = components(uv); good = component >= 0
    rows, uv, component = rows[good], uv[good], component[good]
    stats["observations"] = len(rows)
    if len(rows) < cfg["min_anchor_observations"]:
        return corrected, supported, {}, stats
    unit = gap * np.linalg.norm(cam["w2c"][2, :3])
    z = project(anchors["xyz"][rows], cam)[1]
    residual = (z - sample(depth, uv)) / unit
    direction = cam["w2c"][2, :3]
    uncertainty = np.sqrt(np.maximum(0, np.einsum("i,nij,j->n", direction, anchors["covariance"][rows], direction))) / unit
    boundary = mask & ~binary_erosion(mask)
    by, bx = np.where(boundary)
    boundary_uv = np.column_stack((bx, by))[::max(1, len(bx)//2048)]
    control, inlier = fit_correction(uv, residual, uncertainty, origin, shape, step, boundary_uv,
        cfg["smooth_weight"], cfg["ridge_weight"], edges)
    stats["inliers"] = int(inlier.sum())
    candidates = np.column_stack((xx, yy)).astype(float)
    components_at_pixels = components(candidates)
    keep = np.zeros(len(candidates), bool)
    for label in np.unique(component[inlier]):
        source = uv[inlier & (component == label)]
        if len(source) < cfg["min_anchor_observations"]:
            continue
        selected = components_at_pixels == label
        # At least this many nearby inliers, not one anchor extrapolated across
        # a large connected hole merely because other anchors exist far away.
        distance = cKDTree(source).query(candidates[selected], k=cfg["min_anchor_observations"])[0][:, -1]
        keep[np.flatnonzero(selected)[distance <= cfg["anchor_radius_pixels"]]] = True
    delta = grid_basis(candidates, origin, shape, step) @ control * unit
    values = depth[yy, xx] + delta
    keep &= np.isfinite(values) & (values > 0)
    corrected[yy[keep], xx[keep]] = values[keep].astype(depth.dtype)
    supported[yy[keep], xx[keep]] = True
    stats.update(supported_pixels=int(keep.sum()), correction_median_spacing=float(np.median(delta[keep]/unit)) if keep.any() else 0.)
    field = dict(control=control, origin=origin, shape=np.array(shape), step=np.array(step),
                 unit=np.array(unit), support=supported, anchor_uv=uv[inlier])
    return corrected, supported, field, stats


def correct_surface_support(support, background, views, targets, config, seed=0,
                            max_points=None, *, init_config=None, rgb_pairs=None,
                            artifact_callback=None):
    """Replace mixed XYZ by supported, cross-view-consistent depth samples only.

    callback(model, arrays) runs even on insufficient support, so failures retain
    diagnostics. Empty results fail explicitly; there is no plane/raw-depth fill.
    """
    if config["mode"] != "depth_anchor":
        raise ValueError("surface_init.mode must be depth_anchor; plane mode has been removed")
    if any("depth" not in target for target in targets):
        raise ValueError("depth_anchor requires init.use_depth=true")
    if init_config is None:
        raise ValueError("depth_anchor requires explicit matching thresholds")
    cameras = [camera(view) for view in views]
    gap = float(support["spacing"])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("invalid mixed proposal spacing")
    anchors, track_stats = rgb_anchors(support, cameras, config, init_config, rgb_pairs)
    print(f"Depth-anchor RGB tracks: {track_stats}", flush=True)
    depths, domains, field_stats = {}, {}, []
    artifacts = {"anchor_" + k: v for k, v in anchors.items()}
    original_anchor = np.zeros(len(anchors["xyz"]), bool)
    for frame, (cam, target) in enumerate(zip(cameras, targets)):
        if "original_mask" in target and len(anchors["xyz"]):
            uv, _, valid = project(anchors["xyz"], cam)
            original_anchor |= valid & (sample(array(target["original_mask"]).astype(float), uv, 0) > 0)
        corrected, domain, field, stats = depth_field(cam, target, anchors, frame, gap, config)
        field_stats.append(stats)
        if domain.any():
            depths[frame], domains[frame] = corrected, domain
            artifacts.update({f"frame_{frame:05d}_{key}": value for key, value in field.items()})
    model = dict(mode="depth_anchor", version=1, spacing=gap, tracks=track_stats,
        track_source="raw_bidirectional_pairs" if rgb_pairs is not None else "historical_surviving_pair_observations",
        anchor_in_original_mask=int(original_anchor.sum()), fields=field_stats,
        confidence_kind="anchor_supported_cross_view_depth_consistency; not calibrated accuracy or held-out validation",
        camera_models={str(i): {k: c[k].tolist() if isinstance(c[k], np.ndarray) else c[k]
                      for k in ("w2c", "c2w", "k", "h", "w")} for i, c in enumerate(cameras)},
        color_policy="original source RGB bilinear sampling", shape_policy="original spacing/isotropic/identity/opacity",
        corrected_depth_usage="initialization only; training targets unchanged", seed=seed)
    neighbors = {i: set() for i in depths}
    for a, b in select_pairs(cameras, init_config["neighbors"]):
        if a in depths and b in depths:
            neighbors[a].add(b); neighbors[b].add(a)
    for triple in np.unique(anchors["frames"], axis=0):
        for a, b in combinations(triple, 2):
            if a in depths and b in depths:
                neighbors[a].add(int(b)); neighbors[b].add(int(a))
    batches = []; total = 0; stats_candidates = []
    for frame in sorted(depths):
        cam, target = cameras[frame], targets[frame]
        yy, xx = np.where(domains[frame])
        stride = config["candidate_pixel_step"]
        chosen = (xx % stride == 0) & (yy % stride == 0)
        uv = np.column_stack((xx[chosen], yy[chosen])).astype(float)
        d = sample(depths[frame], uv); xyz = unproject(uv, d, cam)
        votes = np.zeros(len(uv), int); checked = votes.copy(); conflicts = votes.copy()
        for other in sorted(neighbors[frame]):
            projected, z, inside = project(xyz, cameras[other])
            safe_uv = np.where(inside[:, None], projected, 0.)
            target_depth = sample(depths[other], safe_uv)
            tolerance = config["cross_depth_spacing"] * gap * np.linalg.norm(cameras[other]["w2c"][2, :3])
            supported = inside & editable_rows(domains[other], safe_uv) & (target_depth > 0)
            # Camera count alone is not independent geometry: require usable parallax.
            ray1 = xyz - cam["c2w"][:3, 3]; ray2 = xyz - cameras[other]["c2w"][:3, 3]
            cosine = np.sum(ray1*ray2, axis=1) / np.maximum(np.linalg.norm(ray1, axis=1)*np.linalg.norm(ray2, axis=1), 1e-12)
            supported &= np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= config["anchor_min_angle_deg"]
            checked += supported
            votes += supported & (np.abs(z-target_depth) <= tolerance)
            known = inside & (sample(array(targets[other]["mask"]).astype(float), safe_uv, 0) == 0) & (target_depth > 0)
            conflicts += known & (z < target_depth-tolerance)
        keep = (votes >= config["min_cross_depth_views"]) & (votes >= config["min_cross_depth_fraction"] * checked)
        keep &= conflicts == 0
        stats_candidates.append(dict(frame=frame, sampled=len(uv), accepted=int(keep.sum())))
        if keep.any():
            uv, xyz, d = uv[keep], xyz[keep], d[keep]
            rgb = np.stack([sample(array(channel), uv) for channel in target["rgb"]], axis=1)
            original = sample(array(target.get("original_mask", target["mask"])).astype(float), uv, 0) > 0
            batches.append(dict(xyz=xyz, pixel_uv=uv, rgb=rgb, frame=np.full(len(uv), frame),
                raw_depth=sample(array(target["depth"]), uv), corrected_depth=d,
                depth_votes=votes[keep], depth_checks=checked[keep], original_mask=original))
            total += len(uv)
            if max_points is not None and total > max_points * 8:
                raise ValueError("supported depth pool exceeds 8 * max_points; increase candidate_pixel_step")
    model.update(candidate_statistics=stats_candidates, accepted_before_fusion=total,
                 supported_frame_count=len(depths))
    if not batches:
        model.update(status="insufficient_support", output_points=0)
        if artifact_callback:
            artifact_callback(model, artifacts)
        raise ValueError("depth_anchor: no sufficiently anchored, cross-view-consistent points; inspect surface_model.json; no plane/raw-depth fallback")
    result = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    confidence = result["depth_votes"] / np.maximum(result["depth_checks"], 1)
    order = np.argsort(-confidence, kind="stable")
    cells = np.floor(result["xyz"][order] / (gap*config["sampling_distance_factor"])).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    chosen = order[np.sort(first)]
    chosen = chosen[cKDTree(background).query(result["xyz"][chosen])[0] >= gap * .5]
    if max_points is not None and len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen)-1, max_points).astype(int)]
    result = {key: value[chosen] for key, value in result.items()}
    n = len(chosen)
    has_original_masks = all("original_mask" in target for target in targets)
    original_count = int(result["original_mask"].sum())
    usable = n > 0 and (not has_original_masks or original_count > 0)
    model.update(status="complete" if usable else "insufficient_support", output_points=n,
                 output_in_original_mask=original_count)
    if artifact_callback:
        artifact_callback(model, artifacts)
    if not n:
        raise ValueError("depth_anchor: all supported samples overlap retained background; no fallback")
    if not usable:
        raise ValueError("depth_anchor: supported samples only cover the dilated boundary, not the original hole; no fallback")
    _, parents = cKDTree(support["xyz"]).query(result["xyz"])
    result.update(normal=np.zeros((n, 3), np.float32), normal_valid=np.zeros(n, bool),
        other_frame=np.full(n, -1), other_uv=result["pixel_uv"].copy(), confidence=confidence[chosen],
        source_kind=np.full(n, 2, np.int8), parent_index=parents,
        parent_source_kind=support["source_kind"][parents], spacing=np.array(gap))
    result["xyz"], result["rgb"] = result["xyz"].astype(np.float32), result["rgb"].astype(np.float32)
    report = dict(mode="depth_anchor", output_points=n, anchor_count=len(anchors["xyz"]),
        parent_rgb_points=int((result["parent_source_kind"] == 0).sum()),
        parent_depth_points=int((result["parent_source_kind"] == 1).sum()),
        supported_frames=len(depths), original_mask_points=model["output_in_original_mask"],
        confidence_kind=model["confidence_kind"], background_unchanged=True)
    return result, report, model
