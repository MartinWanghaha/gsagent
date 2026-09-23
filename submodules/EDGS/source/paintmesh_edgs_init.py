"""Local RoMa correspondence / optional RGB-D-N initialization, no global Trainer."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

from source.pgsr_geometry import camera_intrinsics


def camera(view):
    w2c = view.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
    c2w = np.linalg.inv(w2c)
    k = camera_intrinsics(view, device="cpu", dtype=torch.float64).numpy()
    return dict(w2c=w2c, c2w=c2w, k=k, h=view.image_height, w=view.image_width)


def project(xyz, cam):
    p = xyz @ cam["w2c"][:3, :3].T + cam["w2c"][:3, 3]
    q = p @ cam["k"].T
    uv = q[:, :2] / np.maximum(p[:, 2:3], 1e-12)
    valid = np.isfinite(uv).all(1) & (p[:, 2] > 0)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= cam["w"] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= cam["h"] - 1)
    return uv, p[:, 2], valid


def unproject(uv, depth, cam):
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(cam["k"]).T
    return rays * depth[:, None] @ cam["c2w"][:3, :3].T + cam["c2w"][:3, 3]


def sample(array, uv, order=1):
    return map_coordinates(array, [uv[:, 1], uv[:, 0]], order=order, mode="nearest")


def select_pairs(cameras, neighbors):
    """Spatial + viewing-direction neighbors, independent of spiral frame order."""
    centers = np.array([c["c2w"][:3, 3] for c in cameras])
    axes = np.array([c["c2w"][:3, 2] for c in cameras])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    typical = np.median(distances[distances > 0]) if (distances > 0).any() else 0
    if typical <= 0:
        raise ValueError("EDGS initialization needs distinct camera centers")
    score = distances / typical + 1 - axes @ axes.T
    score[(distances <= 1e-8) | (axes @ axes.T <= 0)] = np.inf
    pairs = set()
    for i in range(len(cameras)):
        for j in np.argsort(score[i])[:min(neighbors, len(cameras) - 1)]:
            if np.isfinite(score[i, j]):
                pairs.add(tuple(sorted((i, int(j)))))
    if not pairs:
        raise ValueError("no overlapping camera pairs with usable baselines")
    return sorted(pairs)


def matcher_weights(cfg, download=True):
    from source.vendor import bootstrap_roma
    bootstrap_roma()
    from romatch.models.model_zoo import weight_urls
    from edgs_inpaint_io import record
    urls = {"weights": weight_urls["romatch"][cfg["model"]], "dinov2_weights": weight_urls["dinov2"]}
    paths = {}
    for key, url in urls.items():
        path = Path(cfg[key]).expanduser() if cfg[key] else Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[1]
        if not path.is_file() and not cfg[key] and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path))
        paths[key] = record(path)
    return paths


def filter_roma_matches(forward, forward_confidence, reverse, reverse_confidence,
                        mask_a, mask_b, cfg, rng):
    """Gate raw bidirectional certainty BEFORE sampling; never fill a rejected quota.

    Reverse certainty is evaluated at the forward match's target coordinate,
    not the same array index. Certainty is a network score, not calibrated accuracy.
    """
    f, r = np.asarray(forward), np.asarray(reverse)
    fc, rc = np.asarray(forward_confidence), np.asarray(reverse_confidence)
    if (f.ndim != 3 or r.ndim != 3 or f.shape[-1] != 4 or r.shape[-1] != 4
            or fc.shape != f.shape[:2] or rc.shape != r.shape[:2]):
        raise ValueError("invalid RoMa warp/certainty shapes")
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    # RoMa align_corners=False coordinates refer to pixel centers.
    xy = (f[..., :2].reshape(-1, 2) + 1) * [wa / 2, ha / 2] - .5
    uv = (f[..., 2:].reshape(-1, 2) + 1) * [wb / 2, hb / 2] - .5
    valid = np.isfinite(xy).all(1) & np.isfinite(uv).all(1)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] <= wa - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= ha - 1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= wb - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= hb - 1)
    stats = dict(dense=len(xy), finite_in_bounds=int(valid.sum()), confidence_min=cfg["confidence_min"])
    # Invalid coordinates never enter interpolation or become valid through clamping.
    safe_xy = np.where(valid[:, None], xy, 0.)
    safe_uv = np.where(valid[:, None], uv, 0.)
    valid &= sample(np.asarray(mask_a, dtype=float), safe_xy, 0) > 0
    valid &= sample(np.asarray(mask_b, dtype=float), safe_uv, 0) > 0
    stats["hole_candidates"] = int(valid.sum())
    reverse_uv = (safe_uv + .5) * [r.shape[1] / wb, r.shape[0] / hb] - .5
    backward_score = sample(rc, reverse_uv)
    forward_score = fc.reshape(-1)
    scores = np.minimum(forward_score, backward_score)
    valid &= np.isfinite(scores) & (scores > 0) & (forward_score <= 1) & (backward_score <= 1)
    pool = scores[valid]
    stats["bidirectional_confidence_quantiles"] = (
        dict(zip(("min", "p10", "p50", "p90", "max"), np.quantile(pool, [0, .1, .5, .9, 1]).tolist()))
        if len(pool) else {})
    valid &= scores >= cfg["confidence_min"]
    stats["confidence_passed"] = int(valid.sum())
    cycle = np.stack([sample(r[..., 2 + i], reverse_uv) for i in range(2)], axis=1)
    cycle = (cycle + 1) * [wa / 2, ha / 2] - .5
    valid &= np.isfinite(cycle).all(1) & (np.linalg.norm(cycle - xy, axis=1) <= cfg["cycle_pixels"])
    indices = np.flatnonzero(valid)
    stats["cycle_passed"] = len(indices)
    if len(indices) > cfg["samples_per_pair"]:
        probabilities = scores[indices].astype(float)
        indices = rng.choice(indices, cfg["samples_per_pair"], replace=False, p=probabilities / probabilities.sum())
    stats["sampled"] = len(indices)
    return xy[indices], uv[indices], scores[indices], stats


class RomaMatcher:
    def __init__(self, cfg, weights, device="cuda"):
        from source.vendor import bootstrap_roma
        bootstrap_roma()
        from romatch import roma_indoor, roma_outdoor
        create = roma_indoor if cfg["model"] == "indoor" else roma_outdoor
        self.device = device
        self.model = create(device=device, coarse_res=cfg["resolution"],
            weights=torch.load(weights["weights"]["path"], map_location="cpu", weights_only=True),
            dinov2_weights=torch.load(weights["dinov2_weights"]["path"], map_location="cpu", weights_only=True))
        self.model.upsample_preds = False
        self.model.symmetric = False
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, a, b, cfg, rng):
        images = [Image.fromarray(np.rint(t["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for t in (a, b)]
        forward, confidence = self.model.match(*images, device=self.device)
        reverse, reverse_confidence = self.model.match(*images[::-1], device=self.device)
        xy, uv, scores, self.last_diagnostics = filter_roma_matches(
            forward.cpu().numpy(), confidence.cpu().numpy(), reverse.cpu().numpy(),
            reverse_confidence.cpu().numpy(), a["mask"].numpy(), b["mask"].numpy(), cfg, rng)
        return xy, uv, scores


def triangulate(c1, c2, uv1, uv2, scores, cfg):
    from source.correspondence.contracts import MultiViewTracks
    from source.correspondence.geometry import weighted_multiview_dlt
    if not len(uv1):
        return np.empty((0, 3)), np.zeros(0, dtype=bool)
    projections = []
    for c in (c1, c2):
        p = c["k"] @ c["w2c"][:3]
        matrix = np.zeros((4, 4))
        matrix[:, :2], matrix[:, 3] = p[:2].T, p[2]
        projections.append(matrix)
    xy = torch.from_numpy(np.stack((uv1, uv2), axis=1)).double()
    tracks = MultiViewTracks(coordinates=xy, confidence=torch.from_numpy(np.repeat(scores[:, None], 2, axis=1)).double(),
                             valid=torch.ones(xy.shape[:2], dtype=torch.bool), sampling_score=torch.from_numpy(scores).double())
    result = weighted_multiview_dlt(torch.from_numpy(np.array(projections)), tracks,
        camera_centers=torch.from_numpy(np.array([c["c2w"][:3, 3] for c in (c1, c2)])),
        min_views=2, max_reprojection_error=cfg["reprojection_pixels"], min_triangulation_angle_deg=cfg["min_angle_deg"])
    return result.points.numpy(), result.accepted.numpy()


def refine_depth(xyz, cameras, targets, observations, strength):
    """IRLS point updates: reprojection in pixel units plus soft log-z prior."""
    x = xyz.copy()
    for _ in range(3):
        lhs = np.repeat(np.eye(3)[None] * 1e-8, len(x), axis=0)
        rhs = np.zeros_like(x)
        for c, t, uv in zip(cameras, targets, observations):
            pc = x @ c["w2c"][:3, :3].T + c["w2c"][:3, 3]
            z = np.maximum(pc[:, 2], 1e-8)
            for axis in range(2):
                residual = c["k"][axis, axis] * pc[:, axis] / z + c["k"][axis, 2] - uv[:, axis]
                jac = c["k"][axis, axis] * (c["w2c"][axis, :3][None] / z[:, None] - pc[:, axis, None] * c["w2c"][2, :3][None] / z[:, None] ** 2)
                weight = 1 / np.maximum(1, np.abs(residual))
                lhs += np.einsum("ni,nj,n->nij", jac, jac, weight)
                rhs += jac * (residual * weight)[:, None]
            d = sample(t["depth"].numpy(), uv)
            valid = (d > 0) & (pc[:, 2] > 0)
            residual = np.log(z) - np.log(np.maximum(d, 1e-8))
            # Convert relative depth residual to a modest pixel-equivalent prior.
            weight = strength * valid / np.maximum(.05, np.abs(residual))
            jac = c["w2c"][2, :3][None] / z[:, None]
            lhs += np.einsum("ni,nj,n->nij", jac, jac, weight)
            rhs += jac * (residual * weight)[:, None]
        update = np.linalg.solve(lhs, rhs)
        x -= update
    return x


def spacing(points, background):
    """Boundary-near Gaussian spacing, robust to a different far-field density."""
    if len(background) < 2:
        raise ValueError("need at least two retained background Gaussians")
    tree = cKDTree(background)
    queries = points[::max(1, len(points) // 4096)]
    _, rows = tree.query(queries)
    neighbors = tree.query(background[np.unique(rows)], k=min(5, len(background)))[0][:, 1:]
    values = neighbors[np.isfinite(neighbors) & (neighbors > 1e-8)]
    if not len(values):
        raise ValueError("degenerate retained boundary spacing")
    return float(np.median(values))


def initialize_candidates(views, targets, background, cfg, matcher):
    """Returns world candidates and auditable source observations (no hidden targets)."""
    cameras = [camera(v) for v in views]
    pairs = select_pairs(cameras, cfg["init"]["neighbors"])
    rng = np.random.default_rng(cfg["seed"])
    init = cfg["init"]
    batches, pair_stats = [], []
    for i, j in pairs:
        uv, other, confidence = matcher(targets[i], targets[j], init, rng)
        matcher_returned = len(confidence)
        confident = np.isfinite(confidence) & (confidence > 0) & (confidence <= 1) & (confidence >= init["confidence_min"])
        uv, other, confidence = uv[confident], other[confident], confidence[confident]
        xyz, valid = triangulate(cameras[i], cameras[j], uv, other, confidence, init)
        xyz, uv, other, confidence = xyz[valid], uv[valid], other[valid], confidence[valid]
        if init["use_depth"] and len(xyz):
            xyz = refine_depth(xyz, [cameras[i], cameras[j]], [targets[i], targets[j]], [uv, other], init["depth_weight"])
        # Depth refinement must not destroy a previously valid RGB track.
        accepted = np.isfinite(xyz).all(1)
        for cam, observation, t in ((cameras[i], uv, targets[i]), (cameras[j], other, targets[j])):
            projected, _, inside = project(xyz, cam)
            accepted &= inside & (np.linalg.norm(projected - observation, axis=1) <= init["reprojection_pixels"])
            accepted &= sample(t["mask"].numpy().astype(float), observation, 0) > 0
        xyz, uv, other, confidence = xyz[accepted], uv[accepted], other[accepted], confidence[accepted]
        if len(xyz):
            batches.append((xyz, uv, other, np.full(len(xyz), i), np.full(len(xyz), j), confidence, np.zeros(len(xyz), np.int8)))
        if sum(len(b[0]) for b in batches) > init["max_points"] * 8:
            raise ValueError("RGB candidate pool exceeds 8 * max_points resource limit")
        match_stats = dict(getattr(matcher, "last_diagnostics", {}))
        pair_stats.append(dict(source=i, target=j, matcher_returned=matcher_returned, matched=len(valid),
                              triangulated=int(valid.sum()), accepted=len(xyz), matching=match_stats))
        filtering = (f"; bidirectional confidence>={init['confidence_min']:g}: "
                     f"{match_stats['confidence_passed']}/{match_stats['hole_candidates']} hole matches"
                     if match_stats else "")
        print(f"EDGS pair {i:05d}/{j:05d}: {len(xyz)}/{len(valid)} accepted{filtering}", flush=True)
    matched_count = sum(len(b[0]) for b in batches)
    if not matched_count and not init["use_depth"]:
        raise ValueError("RGB-only: no valid triangulation candidates; no depth fallback")
    if init["use_depth"]:
        # Each view proposes samples from its surface-area quota. World-space
        # deduplication below removes repeated observations, not by averaging layers.
        for i, (cam, target) in enumerate(zip(cameras, targets)):
            d, mask = target["depth"].numpy(), target["mask"].numpy()
            yy, xx = np.where(mask & (d > 0))
            if not len(xx):
                continue
            uv = np.column_stack((xx, yy)).astype(float)
            xyz = unproject(uv, d[yy, xx], cam)
            gap = spacing(xyz, background)
            # Tangent area from finite differences; scalar area does not orient GS.
            gy, gx = np.gradient(d.astype(float))
            area = d[yy, xx] ** 2 / (cam["k"][0, 0] * cam["k"][1, 1])
            area *= np.sqrt(1 + (gx[yy, xx] * cam["k"][0, 0] / d[yy, xx]) ** 2 + (gy[yy, xx] * cam["k"][1, 1] / d[yy, xx]) ** 2)
            # Continuous robust area shrinkage at depth discontinuities; no
            # hand-tuned density ratio or surface-coverage acceptance gate.
            area = area / np.maximum(1, area / np.median(area)) ** .5
            count = max(1, int(np.rint(area.sum() / gap ** 2)))
            if count > init["max_points"]:
                raise ValueError(f"depth quota {count} exceeds max_points resource limit")
            indices = rng.choice(len(uv), count, replace=True, p=area / area.sum())
            sampled = uv[indices] + rng.uniform(-.5, .5, (count, 2))
            sampled[:, 0] = np.clip(sampled[:, 0], 0, cam["w"] - 1)
            sampled[:, 1] = np.clip(sampled[:, 1], 0, cam["h"] - 1)
            xyz = unproject(sampled, d[yy[indices], xx[indices]], cam)
            evidence, observations = np.zeros(count), np.zeros(count)
            for a, b in pairs:
                if i not in (a, b):
                    continue
                j = b if a == i else a
                projected, z, inside = project(xyz, cameras[j])
                target_depth = sample(targets[j]["depth"].numpy(), projected)
                visible_hole = inside & (target_depth > 0) & (sample(targets[j]["mask"].numpy().astype(float), projected, 0) > 0)
                relative = np.abs(z - target_depth) / np.maximum(target_depth, 1e-8)
                # A finite pixel footprint defines the depth agreement scale;
                # this is geometric evidence, not a point-density threshold.
                uncertainty = gap / np.maximum(target_depth, gap)
                evidence += visible_hole / (1 + (relative / uncertainty) ** 2)
                observations += visible_hole
            scores = .25 / (1 + observations) * (1 + evidence)
            batches.append((xyz, sampled, sampled, np.full(count, i), np.full(count, -1),
                            scores, np.ones(count, np.int8)))
            if sum(len(b[0]) for b in batches) > init["max_points"] * 8:
                raise ValueError("candidate pool exceeds 8 * max_points resource limit; use fewer views or raise resource budget")
    if not batches:
        raise ValueError("no valid initialization candidates")
    xyz, uv, uv2, frames, frames2, confidence, kind = [np.concatenate([b[k] for b in batches]) for k in range(7)]
    gap = spacing(xyz, background)
    # Best hypothesis per local cell; no position averaging across conflicting layers.
    order = np.argsort(-confidence, kind="stable")
    cells = np.floor(xyz[order] / gap).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    chosen = order[np.sort(first)]
    # Retained background already owns its neighborhoods.
    distance, _ = cKDTree(background).query(xyz[chosen])
    chosen = chosen[distance >= gap * .5]
    if not len(chosen) or len(chosen) > init["max_points"]:
        raise ValueError(f"initialization has {len(chosen)} points; empty or exceeds max_points")
    xyz, uv, uv2, frames, frames2, confidence, kind = [a[chosen] for a in (xyz, uv, uv2, frames, frames2, confidence, kind)]
    rgb, normals = np.zeros((len(xyz), 3)), np.zeros((len(xyz), 3))
    for i, (cam, target) in enumerate(zip(cameras, targets)):
        rows = np.flatnonzero(frames == i)
        rgb[rows] = np.stack([sample(channel.numpy(), uv[rows]) for channel in target["rgb"]], axis=1)
        if init["use_normal"] and len(rows):
            n = np.stack([sample(channel.numpy(), uv[rows]) for channel in target["normal"]], axis=1)
            valid = sample(target["normal_valid"].numpy().astype(float), uv[rows], 0) > 0
            n = n @ np.linalg.inv(cam["c2w"][:3, :3])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
            n[~valid] = 0
            normals[rows] = n
    # Additional matched observation, after world-space sign alignment.
    if init["use_normal"]:
        for j, (cam, target) in enumerate(zip(cameras, targets)):
            rows = np.flatnonzero(frames2 == j)
            n = np.stack([sample(ch.numpy(), uv2[rows]) for ch in target["normal"]], axis=1)
            valid = sample(target["normal_valid"].numpy().astype(float), uv2[rows], 0) > 0
            n = n @ np.linalg.inv(cam["c2w"][:3, :3])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
            n[(n * normals[rows]).sum(1) < 0] *= -1
            normals[rows] += n * valid[:, None]
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-8)
    return dict(xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32), normal=normals.astype(np.float32),
                normal_valid=lengths > 1e-6, frame=frames, pixel_uv=uv, other_frame=frames2,
                other_uv=uv2, confidence=confidence, source_kind=kind, spacing=np.array(gap)), dict(
        pairs=pair_stats, confidence_kind="min_forward_reverse",
        matching_thresholds={key: init[key] for key in ("confidence_min", "cycle_pixels", "reprojection_pixels", "min_angle_deg")},
        rgb_candidates=matched_count, initialized_points=len(xyz),
        rgb_points=int((kind == 0).sum()), depth_points=int((kind == 1).sum()),
        normal_oriented_points=int((lengths > 1e-6).sum()), boundary_spacing=gap)


def compose(background_path, support, config, output):
    from plyfile import PlyData, PlyElement
    from edgs_inpaint_io import atomic_write
    bg = PlyData.read(str(background_path), mmap=True)["vertex"].data
    new = np.zeros(len(support["xyz"]), dtype=bg.dtype)
    for i, key in enumerate("xyz"):
        new[key] = support["xyz"][:, i]
    xyz_bg = np.column_stack([bg[k] for k in "xyz"])
    _, neighbors = cKDTree(xyz_bg).query(support["xyz"])
    for key in bg.dtype.names:
        if key.startswith("obj_"):
            new[key] = bg[key][neighbors]
    for i in range(3):
        new[f"f_dc_{i}"] = (support["rgb"][:, i] - .5) / .28209479177387814
        new[f"scale_{i}"] = np.log(float(support["spacing"]))
    new["opacity"] = np.log(config["opacity"] / (1 - config["opacity"]))
    new["rot_0"] = 1
    if config["use_normal"]:
        valid = support["normal_valid"]
        n = support["normal"][valid]
        quat = np.column_stack((1 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))))
        opposite = np.linalg.norm(quat, axis=1) < 1e-6
        quat[opposite] = [0, 1, 0, 0]
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        for i in range(4):
            new[f"rot_{i}"][valid] = quat[:, i]
        new["scale_2"][valid] += np.log(config["normal_thickness"])
    vertex = np.concatenate((bg, new))
    atomic_write(output, lambda stream: PlyData([PlyElement.describe(vertex, "vertex")]).write(stream))
    return np.arange(len(vertex)) >= len(bg)
