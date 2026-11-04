"""Deterministic, CPU-only repair-surface camera planning.

Visibility is a sampled point-cloud proxy, not a calibrated confidence or a
measurement of the hidden surface. No test images, GT masks or LaMa outputs are
used here. Keep the policy readable by both projects without importing torch.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ALGORITHM_REVISION = 3
SUPPORTED_REVISIONS = (1, 2, 3)
LEGACY_DEFAULTS = dict(
    train_fraction=2 / 3, candidate_multiplier=5, target_roi_fraction=0.12,
    min_roi_fraction=0.005, max_roi_fraction=0.40, min_visible_fraction=0.65,
    texture_angles_deg=[20.0, 35.0], max_incidence_deg=75.0,
    distance_ratio_bounds=[0.35, 1.25], pair_min_overlap=0.55,
    pair_min_angle_deg=3.0, pair_target_angle_deg=12.0, pair_max_angle_deg=35.0,
    background_samples=60000, target_samples=4096, roi_samples=128,
    plane_min_inlier_fraction=0.35, use_view_weights=True, seed=0,
)
RING_DEFAULTS = dict(LEGACY_DEFAULTS, path_strategy="continuous_ring",
    ring_angles_deg=[20.0, 30.0, 35.0, 40.0, 45.0], max_incidence_deg=55.0,
    min_context_support=0.40, tracking_max_rotation_deg=15.0,
    tracking_max_scale_ratio=1.12, tracking_min_overlap=0.8,
    tracking_max_motion_fraction=0.08)
DEFAULTS = dict({k: v for k, v in LEGACY_DEFAULTS.items()
                 if k not in ("train_fraction", "candidate_multiplier", "texture_angles_deg")},
    target_roi_fraction=0.08, inclination_range_deg=[30., 50.],
    distance_ratio_bounds=[0.20, 2.0], max_incidence_deg=55.,
    min_context_support=0.40, tracking_max_rotation_deg=5.,
    tracking_max_scale_ratio=1.03, tracking_min_overlap=0.85,
    tracking_max_motion_fraction=0.02)


def options(value=None, *, revision=ALGORITHM_REVISION):
    if revision not in SUPPORTED_REVISIONS:
        raise ValueError("unsupported adaptive camera revision")
    defaults = {1: LEGACY_DEFAULTS, 2: RING_DEFAULTS, 3: DEFAULTS}[revision]
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("adaptive camera options must be an object with known keys")
    cfg = dict(defaults, **value)
    if revision == 2 and cfg["path_strategy"] not in ("continuous_ring", "selected"):
        raise ValueError("adaptive path_strategy must be continuous_ring or selected")
    if type(cfg["use_view_weights"]) is not bool:
        raise ValueError("adaptive use_view_weights must be boolean")
    for key in ("background_samples", "target_samples", "roi_samples", "seed", *(("candidate_multiplier",) if revision < 3 else ())):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"adaptive {key} must be an integer in range")
    if revision < 3 and not 2 <= cfg["candidate_multiplier"] <= 12:
        raise ValueError("adaptive candidate_multiplier must be in [2,12]")
    for key, low, high in (("background_samples", 128, 200000), ("target_samples", 16, 20000),
                           ("roi_samples", 16, 512)):
        if not low <= cfg[key] <= high:
            raise ValueError(f"adaptive {key} must be in [{low},{high}]")
    for key in set(cfg) - {"candidate_multiplier", "background_samples", "target_samples", "roi_samples",
                            "seed", "texture_angles_deg", "distance_ratio_bounds", "use_view_weights",
                            "path_strategy", "ring_angles_deg", "inclination_range_deg"}:
        x = cfg[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f"adaptive {key} must be a finite number")
        cfg[key] = float(x)
    if revision < 3 and not 0 < cfg["train_fraction"] < 1:
        raise ValueError("adaptive train_fraction must be in (0,1)")
    if not 0 < cfg["min_roi_fraction"] < cfg["target_roi_fraction"] < cfg["max_roi_fraction"] < 1:
        raise ValueError("adaptive ROI fractions must satisfy 0 < min < target < max < 1")
    for key in ("min_visible_fraction", "pair_min_overlap", "plane_min_inlier_fraction"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"adaptive {key} must be in (0,1]")
    if not 0 < cfg["pair_min_angle_deg"] < cfg["pair_target_angle_deg"] < cfg["pair_max_angle_deg"] < 90:
        raise ValueError("adaptive pair angles must satisfy 0 < min < target < max < 90")
    if not 0 < cfg["max_incidence_deg"] < 90:
        raise ValueError("adaptive max_incidence_deg must be in (0,90)")
    array_keys = (["texture_angles_deg"] if revision < 3 else ["inclination_range_deg"]) + ["distance_ratio_bounds"]
    if revision == 2:
        array_keys.append("ring_angles_deg")
    for key in array_keys:
        a = cfg[key]
        if (not isinstance(a, (list, tuple)) or not a or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in a)):
            raise ValueError(f"adaptive {key} must contain finite numbers")
        cfg[key] = [float(x) for x in a]
    a = cfg["distance_ratio_bounds"]
    if len(a) != 2 or not 0 < a[0] < a[1] <= 3:
        raise ValueError("adaptive distance_ratio_bounds must be increasing in (0,3]")
    if revision < 3:
        b = cfg["texture_angles_deg"]
        if any(not 0 < x < cfg["max_incidence_deg"] for x in b) or sorted(set(b)) != b:
            raise ValueError("adaptive texture angles must increase within (0,max_incidence_deg)")
    else:
        b = cfg["inclination_range_deg"]
        if len(b) != 2 or not 0 < b[0] < b[1] < cfg["max_incidence_deg"]:
            raise ValueError("adaptive inclination_range_deg must satisfy 0 < start < end < max_incidence_deg")
    if revision > 1:
        if revision == 2:
            b = cfg["ring_angles_deg"]
            if any(not 0 < x < cfg["max_incidence_deg"] for x in b) or sorted(set(b)) != b:
                raise ValueError("adaptive ring angles must increase within (0,max_incidence_deg)")
        if not 0 <= cfg["min_context_support"] <= 1 or not 0 < cfg["tracking_min_overlap"] <= 1:
            raise ValueError("adaptive context/overlap thresholds must be in [0,1] / (0,1]")
        if (not 0 < cfg["tracking_max_rotation_deg"] <= 180 or
                not 1 <= cfg["tracking_max_scale_ratio"] <= 3 or
                not 0 < cfg["tracking_max_motion_fraction"] <= 1):
            raise ValueError("invalid adaptive tracking rotation/scale/motion limits")
    return cfg


def load_options(path=None):
    return options(json.loads(Path(path).read_text()) if path else None)


def _sample(points, maximum, rng):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > maximum:
        points = points[np.sort(rng.choice(len(points), maximum, replace=False))]
    return points


def load_geometry(model_path, iteration, target_ids, cfg, *, mesh_path=None):
    """Load removal points and require a bound retained mesh, recording exact inputs."""
    from plyfile import PlyData
    cfg = options(cfg)
    rng = np.random.default_rng(cfg["seed"])
    root = Path(model_path) / "point_cloud_object_removal" / f"iteration_{iteration}"
    if (not isinstance(target_ids, (list, tuple)) or not target_ids or
            any(type(i) is not int or i <= 0 for i in target_ids) or len(set(target_ids)) != len(target_ids)):
        raise ValueError("adaptive planning requires unique positive removal target IDs")
    paths = [root / f"point_cloud_{i}.ply" for i in target_ids]
    inputs = []
    def read(path):
        path = path.resolve(strict=True)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        vertex = PlyData.read(str(path), mmap="r")["vertex"].data
        xyz = np.column_stack([vertex[k] for k in "xyz"]).astype(np.float64)
        valid = np.isfinite(xyz).all(1)
        if "opacity" in vertex.dtype.names:
            valid &= np.isfinite(vertex["opacity"]) & (vertex["opacity"] > -4.59511985)
        inputs.append(dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size))
        return xyz[valid]
    targets = []
    for p in paths:
        xyz = read(p)
        if len(xyz) < 8:
            raise ValueError(f"adaptive target has fewer than 8 finite opaque points: {p}")
        # Balance objects before spatial percentiles: a small target must not
        # disappear merely because a second target has many more Gaussians.
        targets.append(_sample(xyz, max(16, cfg["target_samples"] // len(paths)), rng))
    target = _sample(np.concatenate(targets), cfg["target_samples"], rng)
    if len(target) < 8:
        raise ValueError("adaptive planning needs at least 8 finite opaque target points")
    background = read(root / "point_cloud.ply")
    center = np.median(target, axis=0)
    radius = np.percentile(np.linalg.norm(target - center, axis=1), 90)
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("adaptive target extent is degenerate")
    local = background[np.linalg.norm(background - center, axis=1) <= 4 * radius]
    # Preserve small local surfaces even in a very large, unbounded scene.
    bg = _sample(background, cfg["background_samples"], rng)
    bg = np.unique(np.concatenate([bg, _sample(local, min(12000, cfg["background_samples"]), rng)]), axis=0)
    if len(bg) < 32:
        raise ValueError("adaptive planning needs at least 32 retained background points")
    mesh_path = Path(mesh_path) if mesh_path else Path(model_path).parent / "removed_mesh/geometry.ply"
    mesh_path = mesh_path.resolve(strict=True)
    # The published mesh must be bound to the same retained scene used for
    # virtual rendering. A target-only mesh with restored occluders is unsafe.
    semantic_path = Path(model_path).parent / "removed_mesh/semantic_manifest.json"
    if not semantic_path.is_file():
        raise ValueError("adaptive requires the retained mesh semantic_manifest.json; no point-cloud fallback")
    semantic = json.loads(semantic_path.read_text())
    source = semantic.get("inputs", {}).get("gaussian_ply", {}).get("path", "")
    bound_mesh = semantic.get("inputs", {}).get("mesh", {}).get("path", "")
    if (semantic.get("complete") is not True or not source or not bound_mesh or
            Path(source).resolve() != (root / "point_cloud.ply").resolve() or
            Path(bound_mesh).resolve() != mesh_path):
        raise ValueError("adaptive mesh must be produced from the all-selected retained Gaussian model; "
                         "target-only/restored-surrounding meshes cannot substitute for it")
    for key, path in (("gaussian_ply", root / "point_cloud.ply"), ("mesh", mesh_path)):
        record, stat = semantic["inputs"][key], path.stat()
        if record.get("size_bytes") != stat.st_size or record.get("mtime_ns") != stat.st_mtime_ns:
            raise ValueError(f"adaptive mesh provenance is stale for {key}; regenerate the retained mesh")
    for path in (mesh_path, semantic_path.resolve()):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        inputs.append(dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size,
                           kind="retained_mesh" if path == mesh_path else "mesh_provenance"))
    return dict(target_xyz=target, background_xyz=bg, inputs=inputs, mesh_path=str(mesh_path))


def _unit(x):
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(length, 1e-12)


def _basis(normal):
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = _unit(np.cross(normal, axis))
    return u, np.cross(normal, u)


class MeshGuard:
    """Known-surface ray and local distance queries, built only during planning.

    Crop by triangle AABB, not by vertex inclusion: large triangles crossing
    the planning volume must not disappear. No caps, decimation or sampling.
    """
    def __init__(self, geometry, bounds):
        import open3d as o3d
        if "path" in geometry:
            mesh = o3d.io.read_triangle_mesh(str(geometry["path"]))
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            transform = np.asarray(geometry["world_to_pca"])
        else:
            vertices, faces = np.asarray(geometry["vertices"]), np.asarray(geometry["triangles"])
            transform = np.eye(4)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or
                faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
                not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("adaptive requires a finite triangle mesh; no point-cloud fallback")
        selected = []
        low, high = bounds
        for start in range(0, len(faces), 100000):
            ids = faces[start:start + 100000]
            xyz = vertices[ids] @ transform[:3, :3].T + transform[:3, 3]
            keep = (xyz.max(1) >= low).all(1) & (xyz.min(1) <= high).all(1)
            if keep.any():
                selected.append(ids[keep])
        self.scene = None
        queried = 0
        if selected:
            ids, inverse = np.unique(np.concatenate(selected).ravel(), return_inverse=True)
            xyz = vertices[ids] @ transform[:3, :3].T + transform[:3, 3]
            local = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(xyz.astype(np.float32)),
                o3d.core.Tensor(inverse.reshape(-1, 3).astype(np.uint32)))
            self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
            self.scene.add_triangles(local)
            queried = int(len(inverse) // 3)
        self.o3d = o3d
        self.bounds = np.asarray(bounds)
        self.stats = dict(source_triangles=int(len(faces)), queried_triangles=queried,
                          crop_bounds_pca=np.asarray(bounds).tolist(),
                          crop="triangle_aabb_intersection", decimated=False)

    def _tensor(self, values):
        return self.o3d.core.Tensor(np.asarray(values, dtype=np.float32))

    def distance(self, positions):
        if self.scene is None:
            return np.full(len(np.atleast_2d(positions)), np.inf)
        return self.scene.compute_distance(self._tensor(np.atleast_2d(positions)), nthreads=4).numpy()

    def visible(self, position, points, tolerance):
        if self.scene is None:
            # No source triangle intersects this convex query volume: no ray
            # contained in it can hit known mesh. This is not observed surface.
            if not ((points >= self.bounds[0]).all() and (points <= self.bounds[1]).all() and
                    (position >= self.bounds[0]).all() and (position <= self.bounds[1]).all()):
                raise ValueError("mesh ray query leaves its declared crop")
            return np.ones(len(points), bool)
        delta = points - position
        distances = np.linalg.norm(delta, axis=1)
        rays = np.c_[np.broadcast_to(position, delta.shape), _unit(delta)]
        hit = self.scene.cast_rays(self._tensor(rays), nthreads=4)["t_hit"].numpy()
        return hit >= distances - tolerance

    def clear_segments(self, positions, tolerance):
        if self.scene is None:
            if not ((positions >= self.bounds[0]).all() and (positions <= self.bounds[1]).all()):
                raise ValueError("mesh segment query leaves its declared crop")
            return np.ones(len(positions) - 1, bool)
        delta = np.diff(positions, axis=0)
        length = np.linalg.norm(delta, axis=1)
        rays = np.c_[positions[:-1], _unit(delta)]
        hit = self.scene.cast_rays(self._tensor(rays), nthreads=4)["t_hit"].numpy()
        return hit >= length + tolerance


def repair_surface(target, background, train_centers, up, cfg):
    """Robust local support plane; explicitly label a weak fit as a proxy."""
    rng = np.random.default_rng(cfg["seed"])
    center = np.median(target, axis=0)
    radius = float(np.percentile(np.linalg.norm(target - center, axis=1), 90))
    if radius <= 1e-8:
        raise ValueError("adaptive target extent is degenerate")
    local = background[np.linalg.norm(background - center, axis=1) < 4 * radius]
    local = _sample(local, 8000, rng)
    threshold = max(radius * .035, 1e-7)
    best, normal, origin = np.zeros(len(local), bool), np.asarray(up), center
    if len(local) >= 32:
        # Favor support below the object's middle without excluding vertical planes.
        lower = local[(local - center) @ up < radius * .25]
        fit = lower if len(lower) >= 32 else local
        best_score = -1.
        for _ in range(192):
            tri = fit[rng.choice(len(fit), 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.linalg.norm(n) < 1e-10:
                continue
            n = _unit(n)
            inliers = np.abs((local - tri[0]) @ n) < threshold
            score = inliers.sum() * (.8 + .2 * abs(float(n @ up)))
            if score > best_score:
                best_score, best, normal, origin = score, inliers, n, tri[0]
        if best.sum() >= 16:
            origin = np.mean(local[best], axis=0)
            _, _, vt = np.linalg.svd(local[best] - origin, full_matrices=False)
            normal = vt[-1]
            best = np.abs((local - origin) @ normal) < threshold
    fraction = float(best.mean()) if len(best) else 0.
    reliable = best.sum() >= 32 and fraction >= cfg["plane_min_inlier_fraction"]
    if not reliable:
        normal, origin = np.asarray(up), center
    toward = np.median(train_centers, axis=0) - origin
    sign = normal @ up if abs(normal @ up) > .5 else normal @ toward
    normal = _unit(normal * (1 if sign >= 0 else -1))
    u, v = _basis(normal)
    center = center - ((center - origin) @ normal) * normal
    uv = (target - center) @ np.column_stack([u, v])
    low, high = np.percentile(uv, [3, 97], axis=0)
    center = center + .5 * (low[0] + high[0]) * u + .5 * (low[1] + high[1]) * v
    radii = np.maximum((high - low) * .55, radius * .15)
    # Sunflower disk avoids a rectangular ROI and gives stable equal-area probes.
    k = np.arange(cfg["roi_samples"])
    theta, r = k * (math.pi * (3 - math.sqrt(5))), np.sqrt((k + .5) / len(k))
    disk = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) * radii
    points = center + disk[:, :1] * u + disk[:, 1:] * v
    delta = background - center
    xy = delta @ np.column_stack([u, v]) / radii
    ring = (np.linalg.norm(xy, axis=1) >= 1.05) & (np.linalg.norm(xy, axis=1) <= 2.5)
    ring &= np.abs(delta @ normal) <= threshold * 3 if reliable else np.ones(len(delta), bool)
    context = _sample(background[ring], 2048, rng)
    return dict(center=center, normal=normal, u=u, v=v, radii=radii, points=points,
                context=context, threshold=threshold,
                fit=dict(kind="support_plane" if reliable else "target_proxy", local_points=len(local),
                         inliers=int(best.sum()), inlier_fraction=fraction, threshold_pca=threshold))


def _look_at(position, center, up):
    backward = _unit(position - center)
    right = np.cross(up, backward)
    if np.linalg.norm(right) <= 1e-8:
        return None
    return np.column_stack([_unit(right), np.cross(backward, _unit(right)), backward])


def _project(points, position, rotation, tan_fov, width=96, height=64):
    p = (points - position) @ rotation
    depth = -p[:, 2]  # GL camera: right/up/backward.
    uv = np.column_stack([p[:, 0] / np.maximum(depth, 1e-12) / tan_fov[0],
                          -p[:, 1] / np.maximum(depth, 1e-12) / tan_fov[1]])
    pixels = (uv + 1) * [width / 2, height / 2]
    inside = (depth > 1e-8) & (np.abs(uv) < .98).all(1)
    return pixels, depth, inside


def assess_candidate(position, surface, background, up, fovs, cfg, mesh=None):
    """Cheap point-center z-buffer proxy, never an alpha/render quality claim."""
    center, points = surface["center"], surface["points"]
    direction = _unit(position - center)
    cosine = float(direction @ surface["normal"])
    if cosine < math.cos(math.radians(cfg["max_incidence_deg"])):
        return None, "incidence"
    rotation = _look_at(position, center, up)
    if rotation is None:
        return None, "up_singularity"
    tan_fov = np.tan(np.asarray(fovs) / 2)
    pixels, depths, inside = _project(points, position, rotation, tan_fov)
    if inside.mean() < cfg["min_visible_fraction"]:
        return None, "framing"
    # Projected ellipse area is a planning estimate, not the eventual SAM mask.
    from scipy.spatial import ConvexHull
    area = float(ConvexHull(pixels).volume / (96 * 64))
    if not cfg["min_roi_fraction"] <= area <= cfg["max_roi_fraction"]:
        return None, "roi_fraction"
    zbuffer = np.full(96 * 64, np.inf)
    xy, z, valid = _project(background, position, rotation, tan_fov)
    xy = np.floor(xy[valid]).astype(int)
    z = z[valid]
    # Splat to neighboring coarse pixels to reduce holes from subsampling.
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        x, y = xy[:, 0] + dx, xy[:, 1] + dy
        ok = (x >= 0) & (x < 96) & (y >= 0) & (y < 64)
        np.minimum.at(zbuffer, y[ok] * 96 + x[ok], z[ok])
    tolerance = max(surface["threshold"] * 4, float(np.median(depths)) * .025)
    def visibility(p):
        uv, d, valid = _project(p, position, rotation, tan_fov)
        ids = np.floor(uv).astype(int)
        ids = np.clip(ids, [0, 0], [95, 63])
        front = zbuffer[ids[:, 1] * 96 + ids[:, 0]]
        result = valid & (d <= front + tolerance)
        if mesh is not None:
            result &= mesh.visible(position, p, tolerance)
        return result
    visible = visibility(points)
    if visible.mean() < cfg["min_visible_fraction"]:
        return None, "occlusion"
    context = float(visibility(surface["context"]).mean()) if len(surface["context"]) else 0.
    if context < cfg.get("min_context_support", 0.):
        return None, "context_support"
    framing = math.exp(-abs(math.log(area / cfg["target_roi_fraction"])) * .5)
    # Do not downweight the intentionally oblique half of the new sweep merely
    # for its inclination; the visibility/framing gates still apply.
    incidence_weight = 1. if "inclination_range_deg" in cfg else math.sqrt(cosine)
    quality = float(visible.mean() * (.25 + .75 * context) * framing * incidence_weight)
    return dict(position=position, rotation=rotation, direction=direction, visible=visible,
                roi_fraction=area, visible_fraction=float(visible.mean()), context_support=context,
                incidence_deg=math.degrees(math.acos(np.clip(cosine, -1, 1))), quality=quality), None


def geometry_pairs(selected, points, cfg):
    pairs = []
    for i, a in enumerate(selected):
        for j in range(i + 1, len(selected)):
            b = selected[j]
            shared = a["visible"] & b["visible"]
            overlap = float(shared.sum() / max(1, min(a["visible"].sum(), b["visible"].sum())))
            if overlap < cfg["pair_min_overlap"]:
                continue
            ra, rb = _unit(a["position"] - points[shared]), _unit(b["position"] - points[shared])
            angle = float(np.median(np.degrees(np.arccos(np.clip((ra * rb).sum(1), -1, 1)))))
            if cfg["pair_min_angle_deg"] <= angle <= cfg["pair_max_angle_deg"]:
                score = overlap / (1 + abs(angle - cfg["pair_target_angle_deg"]) / cfg["pair_target_angle_deg"])
                pairs.append(dict(i=i, j=j, overlap=overlap, angle_deg=angle, score=score))
    return pairs


def tracking_edges(selected, points, fovs, *, closed=True):
    """Motion is a surface-projection proxy, not flow. Open paths have N-1 edges."""
    projections = [_project(points, c["position"], c["rotation"], np.tan(np.asarray(fovs) / 2))[0]
                   for c in selected]
    steps = []
    for i, b in enumerate(selected):
        if not closed and i == 0:
            continue
        a = selected[i - 1]
        shared = a["visible"] & b["visible"]
        scale = math.sqrt(b["roi_fraction"] / a["roi_fraction"])
        motion = (float(np.percentile(np.linalg.norm((projections[i] - projections[i - 1])[shared], axis=1), 90))
                  / math.hypot(96, 64)) if shared.any() else 1.
        steps.append(dict(i=(i - 1) % len(selected), j=i,
            angle_deg=math.degrees(math.acos(np.clip(a["direction"] @ b["direction"], -1, 1))),
            rotation_deg=math.degrees(math.acos(np.clip((np.trace(a["rotation"].T @ b["rotation"]) - 1) / 2, -1, 1))),
            overlap=float(shared.sum() / min(a["visible"].sum(), b["visible"].sum())),
            scale_ratio=max(scale, 1 / scale), motion_fraction=motion))
    return steps


def _tracking_violation(edge, cfg):
    for key, bound, minimum in (("rotation_deg", "tracking_max_rotation_deg", False),
                                ("scale_ratio", "tracking_max_scale_ratio", False),
                                ("overlap", "tracking_min_overlap", True),
                                ("motion_fraction", "tracking_max_motion_fraction", False)):
        if (edge[key] < cfg[bound] - 1e-9 if minimum else edge[key] > cfg[bound] + 1e-9):
            return key
    return None


def sweep_layout(count, cfg, phase, reference_distance, reference_inclination):
    """The sole revision-3 trajectory: two open turns and a smooth inclination sweep."""
    parameter = np.linspace(0., 1., count)
    smooth = parameter ** 2 * (3 - 2 * parameter)
    low, high = cfg["inclination_range_deg"]
    theta = np.radians(low + (high - low) * smooth)
    phi = phase + 4 * math.pi * parameter
    distance = reference_distance * np.sqrt(np.cos(theta) / math.cos(math.radians(reference_inclination)))
    return parameter, theta, phi, distance


def plan(train_poses, target, background, up, fovs, count, cfg=None, distance_scale=1., *,
         reference_circle=None, mesh_geometry=None):
    """Generate one open double sweep. Failure never selects another strategy."""
    from scipy.spatial import ConvexHull
    cfg = options(cfg)
    if type(count) is not int or not 6 <= count <= 512:
        raise ValueError("adaptive total camera count must be an integer in [6,512]")
    if reference_circle is None or mesh_geometry is None:
        raise ValueError("adaptive requires its circle reference and retained triangle mesh; no fallback")
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("adaptive distance scale must be finite and positive")
    train_centers = np.asarray(train_poses)[:, :3, 3]
    surface = repair_surface(target, background, train_centers, up, cfg)
    center, normal = surface["center"], surface["normal"]
    native = np.asarray(reference_circle, dtype=float)
    if native.ndim != 2 or native.shape[1] != 3 or len(native) < 3 or not np.isfinite(native).all():
        raise ValueError("invalid adaptive reference circle")
    vectors = native - center
    native_distance = np.linalg.norm(vectors, axis=1)
    native_directions = _unit(vectors)
    inclination = float(np.median(np.degrees(np.arccos(np.clip(native_directions @ normal, -1, 1)))))
    if not 0 < inclination < 89:
        raise ValueError("reference circle does not observe the front of the repair surface")
    phase = math.atan2(float(vectors[0] @ surface["v"]), float(vectors[0] @ surface["u"])) % (2 * math.pi)
    capture_distance = float(np.median(np.linalg.norm(train_centers - center, axis=1)))
    lo, hi = np.asarray(cfg["distance_ratio_bounds"]) * capture_distance
    factors = np.sqrt(np.cos(np.radians(cfg["inclination_range_deg"])) / math.cos(math.radians(inclination)))
    lo, hi = lo / factors.min(), hi / factors.max()
    direction = math.cos(math.radians(inclination)) * normal + math.sin(math.radians(inclination)) * (
        math.cos(phase) * surface["u"] + math.sin(phase) * surface["v"])
    def area_at(radius):
        position = center + direction * radius
        rotation = _look_at(position, center, up)
        if rotation is None:
            raise ValueError("adaptive reference up is parallel to its viewing direction")
        pixels, depths, _ = _project(surface["points"], position, rotation, np.tan(np.asarray(fovs) / 2))
        if np.any(depths <= 0):
            return float("inf")
        return float(ConvexHull(pixels).volume / (96 * 64))
    target_area = cfg["target_roi_fraction"]
    if not area_at(lo) >= target_area >= area_at(hi):
        raise ValueError("adaptive target ROI cannot be framed within distance_ratio_bounds; no fallback")
    for _ in range(36):
        mid = (lo + hi) / 2
        if area_at(mid) > target_area:
            lo = mid
        else:
            hi = mid
    reference_distance = (lo + hi) / 2 * distance_scale
    parameter, theta, phi, distances = sweep_layout(count, cfg, phase, reference_distance, inclination)
    directions = np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * (
        np.cos(phi)[:, None] * surface["u"] + np.sin(phi)[:, None] * surface["v"])
    positions = center + distances[:, None] * directions
    bounds_points = np.concatenate([positions, surface["points"], surface["context"]])
    margin = surface["threshold"] * 4
    bounds = [bounds_points.min(0) - margin, bounds_points.max(0) + margin]
    mesh = MeshGuard(mesh_geometry, bounds)
    from scipy.spatial import cKDTree
    if np.any(cKDTree(background).query(positions)[0] < surface["threshold"] * 3):
        raise ValueError("adaptive sweep approaches retained Gaussians; no fallback")
    if np.any(mesh.distance(positions) < surface["threshold"] * 3):
        raise ValueError("adaptive sweep approaches the retained mesh; no fallback")
    if not mesh.clear_segments(positions, surface["threshold"]).all():
        raise ValueError("adaptive sweep crosses the retained mesh between frames; no fallback")
    selected = []
    source_directions = _unit(train_centers - center)
    for i, position in enumerate(positions):
        candidate, reason = assess_candidate(position, surface, background, up, fovs, cfg, mesh)
        if candidate is None:
            raise ValueError(f"adaptive open sweep rejected frame {i}/{count}: {reason}; "
                             "inspect geometry or explicitly adjust configuration; no fallback or frame deletion")
        candidate.update(role="sweep", candidate_id=i,
            source_camera_index=int(np.argmax(source_directions @ directions[i])),
            path_parameter=float(parameter[i]), azimuth_rad=float(phi[i]), distance_pca=float(distances[i]))
        selected.append(candidate)
    for edge in tracking_edges(selected, surface["points"], fovs, closed=False):
        reason = _tracking_violation(edge, cfg)
        if reason:
            raise ValueError(f"adaptive open sweep edge {edge['i']}->{edge['j']} violates {reason}; "
                             "increase VIRTUAL_CAMERA_COUNT or explicitly adjust limits; no fallback")
    search = dict(candidate_count=count, accepted_candidates=count, rejected={},
                  source_camera_count=len(train_centers), sweep_views=count, mesh=mesh.stats)
    poses, metadata = _plan_metadata(selected, surface, up, fovs, cfg, search)
    metadata.update(sampling="repair_surface_open_sweep", continuous_path=dict(
        closed=False, turns=2, inclination_law="smoothstep", distance_law="sqrt_cosine",
        inclination_range_deg=cfg["inclination_range_deg"], start_phase_rad=phase,
        azimuth_step_rad=4 * math.pi / (count - 1), reference_inclination_deg=inclination,
        reference_distance_pca=reference_distance, distance_scale=float(distance_scale),
        reference_circle_centers_pca=native.tolist(),
        original_circle_distance_pca=float(np.median(native_distance))))
    return poses, metadata


def _plan_metadata(selected, surface, up, fovs, cfg, search):
    count = len(selected)
    center, normal = surface["center"], surface["normal"]
    pairs = geometry_pairs(selected, surface["points"], cfg)
    if not pairs:
        raise ValueError("adaptive views have no usable shared-surface triangulation pairs")
    qualities = np.maximum(.1, [c["quality"] for c in selected])
    weights = qualities / np.mean(qualities) if cfg["use_view_weights"] else np.ones(count)
    frames, poses = [], []
    for i, c in enumerate(selected):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = c["rotation"], c["position"]
        poses.append(pose)
        row = {k: c[k] for k in ("candidate_id", "role", "source_camera_index", "roi_fraction",
                                 "visible_fraction", "context_support", "incidence_deg", "quality")}
        row.update(image_name=f"{i:05d}", weight=float(weights[i]),
                   visible_roi_indices=np.flatnonzero(c["visible"]).tolist())
        row.update({k: c[k] for k in ("path_parameter", "azimuth_rad", "distance_pca")})
        frames.append(row)
    degrees = np.bincount([x[k] for x in pairs for k in ("i", "j")], minlength=count)
    steps = tracking_edges(selected, surface["points"], fovs, closed=False)
    observations = np.sum([c["visible"] for c in selected], axis=0)
    paired = np.zeros(len(surface["points"]), bool)
    for p in pairs:
        paired |= selected[p["i"]]["visible"] & selected[p["j"]]["visible"]
    metadata = dict(options=cfg, sampling="repair_surface_open_sweep", orientation="scene_up",
        planning_fov_rad=list(map(float, fovs)), tracking_edges=steps,
        scene_up_pca=np.asarray(up).tolist(), surface_normal_pca=normal.tolist(), surface_fit=surface["fit"],
        roi_points_pca=surface["points"].tolist(), roi_radii_pca=surface["radii"].tolist(),
        frames=frames, geometry_pairs=pairs,
        planner=dict(search,
                     isolated_geometry_views=np.flatnonzero(degrees == 0).tolist(),
                     roi_coverage=dict(one_view_fraction=float((observations > 0).mean()),
                         two_view_fraction=float((observations > 1).mean()),
                         geometry_pair_fraction=float(paired.mean()), minimum_views=int(observations.min()),
                         uncovered_indices=np.flatnonzero(observations == 0).tolist()),
                     visibility_kind="mesh_raycast_and_sampled_point_center_zbuffer", online_refinement=False,
                     appearance_consistency_checked=False),
        center_pca=center.tolist(), focus_pca=center.tolist(),
        radius_pca=float(np.median([np.linalg.norm(c["position"] - center) for c in selected])))
    return poses, metadata


def _validate_sweep(value, count, cfg, arrays, frames, positions):
    """Historical readers never execute generation, mesh queries or fallbacks."""
    path = value.get("continuous_path", {})
    for key in ("start_phase_rad", "azimuth_step_rad", "reference_inclination_deg",
                "reference_distance_pca", "distance_scale", "original_circle_distance_pca"):
        x = path.get(key)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError("invalid adaptive sweep geometry")
    if (path.get("closed") is not False or type(path.get("turns")) is not int or path["turns"] != 2 or
            path.get("inclination_law") != "smoothstep" or path.get("distance_law") != "sqrt_cosine" or
            path.get("inclination_range_deg") != cfg["inclination_range_deg"] or
            not 0 <= path["start_phase_rad"] < 2 * math.pi or
            not 0 < path["reference_inclination_deg"] < 89 or
            min(path[k] for k in ("reference_distance_pca", "distance_scale", "original_circle_distance_pca")) <= 0 or
            not np.isclose(path["azimuth_step_rad"], 4 * math.pi / (count - 1), rtol=0, atol=1e-12)):
        raise ValueError("adaptive open sweep differs from its declared policy")
    native = np.asarray(path.get("reference_circle_centers_pca"), dtype=float)
    if native.shape != (90, 3) or not np.isfinite(native).all():
        raise ValueError("adaptive sweep circle reference is invalid")
    center, normal = np.asarray(value["center_pca"]), arrays["surface_normal_pca"]
    vectors = native - center
    theta0 = float(np.median(np.degrees(np.arccos(np.clip(_unit(vectors) @ normal, -1, 1)))))
    u, v = _basis(normal)
    phase = math.atan2(float(vectors[0] @ v), float(vectors[0] @ u)) % (2 * math.pi)
    if (not np.isclose(theta0, path["reference_inclination_deg"], atol=1e-7) or
            not np.isclose(phase, path["start_phase_rad"], atol=1e-9) or
            not np.isclose(np.median(np.linalg.norm(vectors, axis=1)), path["original_circle_distance_pca"], rtol=1e-9)):
        raise ValueError("adaptive sweep reference differs from circle geometry")
    direction = math.cos(math.radians(theta0)) * normal + math.sin(math.radians(theta0)) * (
        math.cos(phase) * u + math.sin(phase) * v)
    reference_position = center + direction * path["reference_distance_pca"] / path["distance_scale"]
    reference_rotation = _look_at(reference_position, center, arrays["scene_up_pca"])
    if reference_rotation is None:
        raise ValueError("adaptive sweep reference roll is undefined")
    from scipy.spatial import ConvexHull
    pixels, depths, _ = _project(arrays["roi_points_pca"], reference_position, reference_rotation,
                                 np.tan(np.asarray(value["planning_fov_rad"]) / 2))
    area = float(ConvexHull(pixels).volume / (96 * 64))
    if (np.any(depths <= 0) or not np.isclose(area, cfg["target_roi_fraction"], rtol=1e-7, atol=1e-9)):
        raise ValueError("adaptive sweep reference distance differs from framing policy")
    parameter, theta, phi, distance = sweep_layout(count, cfg, path["start_phase_rad"],
        path["reference_distance_pca"], path["reference_inclination_deg"])
    expected = center + distance[:, None] * (np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * (
        np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v))
    if positions is not None and not np.allclose(positions, expected, atol=1e-8, rtol=1e-7):
        raise ValueError("adaptive cameras differ from the declared open sweep")
    if not np.isclose(value["radius_pca"], np.median(distance), rtol=1e-9):
        raise ValueError("adaptive sweep distance differs from camera metadata")
    for i, frame in enumerate(frames):
        for key, expected_value in (("path_parameter", parameter[i]), ("azimuth_rad", phi[i]),
                                    ("distance_pca", distance[i]), ("incidence_deg", math.degrees(theta[i]))):
            x = frame.get(key)
            if (isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or
                    not np.isclose(x, expected_value, atol=1e-7, rtol=1e-9)):
                raise ValueError("adaptive sweep frame parameter differs from geometry")
    planner = value.get("planner", {})
    mesh = planner.get("mesh", {})
    if (planner.get("sweep_views") != count or
            planner.get("visibility_kind") != "mesh_raycast_and_sampled_point_center_zbuffer" or
            type(mesh.get("queried_triangles")) is not int or type(mesh.get("source_triangles")) is not int or
            not 0 <= mesh["queried_triangles"] <= mesh["source_triangles"] or mesh["source_triangles"] <= 0 or
            mesh.get("crop") != "triangle_aabb_intersection" or mesh.get("decimated") is not False):
        raise ValueError("adaptive sweep mesh query provenance is missing")
    bounds = np.asarray(mesh.get("crop_bounds_pca"), dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[0] < bounds[1]).all():
        raise ValueError("adaptive mesh crop is invalid")
    if not ((expected >= bounds[0]).all() and (expected <= bounds[1]).all()):
        raise ValueError("adaptive sweep leaves the queried mesh volume")


def validate_metadata(value, count, cameras=None):
    """Validate the policy/graph in the shared camera identity, without torch."""
    cfg = options(value.get("options"), revision=value.get("algorithm_revision"))
    ring = cfg.get("path_strategy") == "continuous_ring"
    sweep = value.get("algorithm_revision") == 3
    digest = value.get("planner_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("adaptive planner implementation identity is missing")
    if value.get("options") != cfg or not 6 <= count <= 512:
        raise ValueError("adaptive options/count mismatch")
    sampling = ("repair_surface_open_sweep" if sweep else
                "repair_surface_continuous_ring" if ring else "repair_surface_train_prior")
    if value.get("sampling") != sampling or value.get("orientation") != "scene_up":
        raise ValueError("invalid adaptive sampling/orientation")
    arrays = {}
    for key, shape in (("scene_up_pca", (3,)), ("surface_normal_pca", (3,)),
                        ("roi_radii_pca", (2,)), ("roi_points_pca", (cfg["roi_samples"], 3))):
        a = np.asarray(value.get(key), dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid adaptive {key}")
        arrays[key] = a
    for key in ("scene_up_pca", "surface_normal_pca"):
        if not np.isclose(np.linalg.norm(arrays[key]), 1, atol=1e-8):
            raise ValueError(f"adaptive {key} must be unit length")
    if np.any(arrays["roi_radii_pca"] <= 0):
        raise ValueError("adaptive ROI radii must be positive")
    if value.get("surface_fit", {}).get("kind") not in ("support_plane", "target_proxy"):
        raise ValueError("adaptive surface fit must declare its reliability")
    inputs = value.get("geometry_inputs")
    if not isinstance(inputs, list) or len(inputs) < 2:
        raise ValueError("adaptive geometry inputs are missing")
    for record in inputs:
        digest = record.get("sha256", "")
        if (not isinstance(record.get("path"), str) or not record["path"] or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("adaptive geometry input identity is invalid")
    if sweep and not any(r.get("kind") == "retained_mesh" for r in inputs):
        raise ValueError("adaptive sweep requires retained mesh identity")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError("adaptive frame count mismatch")
    fovs = np.asarray(value.get("planning_fov_rad"), dtype=float)
    if fovs.shape != (2,) or not np.isfinite(fovs).all() or not ((fovs > 0) & (fovs < math.pi)).all():
        raise ValueError("invalid adaptive planning FoV")
    visibility, candidates = [], set()
    weights, quality = [], []
    for i, f in enumerate(frames):
        roles = ("sweep",) if sweep else ("ring",) if ring else ("train", "texture")
        if f.get("image_name") != f"{i:05d}" or f.get("role") not in roles:
            raise ValueError("adaptive frame identity/role mismatch")
        candidate = f.get("candidate_id")
        if type(candidate) is not int or candidate < 0 or candidate in candidates:
            raise ValueError("adaptive candidate identities must be unique")
        candidates.add(candidate)
        for key in ("roi_fraction", "visible_fraction", "context_support", "quality", "weight", "incidence_deg"):
            x = f.get(key)
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
                raise ValueError(f"adaptive frame {key} must be finite")
        if (not cfg["min_roi_fraction"] <= f["roi_fraction"] <= cfg["max_roi_fraction"] or
                not cfg["min_visible_fraction"] <= f["visible_fraction"] <= 1 or
                not cfg.get("min_context_support", 0) <= f["context_support"] <= 1 or not 0 <= f["quality"] <= 1 or
                not 0 <= f["incidence_deg"] <= cfg["max_incidence_deg"] + 1e-6 or f["weight"] <= 0):
            raise ValueError("adaptive frame quality/weight outside policy")
        ids = f.get("visible_roi_indices")
        if (not isinstance(ids, list) or any(type(x) is not int or not 0 <= x < cfg["roi_samples"] for x in ids)
                or sorted(set(ids)) != ids or not np.isclose(len(ids) / cfg["roi_samples"], f["visible_fraction"])):
            raise ValueError("invalid adaptive frame surface visibility")
        mask = np.zeros(cfg["roi_samples"], bool)
        mask[ids] = True
        visibility.append(mask)
        weights.append(f["weight"])
        quality.append(max(.1, f["quality"]))
    expected = np.asarray(quality) / np.mean(quality) if cfg["use_view_weights"] else np.ones(count)
    if not np.allclose(weights, expected, rtol=1e-10, atol=1e-10):
        raise ValueError("adaptive frame weights differ from quality policy")
    ntrain = max(1, min(count - 1, int(math.floor(count * cfg.get("train_fraction", 0) + .5))))
    if not ring and not sweep and sum(f["role"] == "train" for f in frames) != ntrain:
        raise ValueError("adaptive train/texture budget mismatch")
    positions = None
    if cameras is not None:
        matrices = []
        for c in cameras:
            if not np.allclose([c["FoVx"], c["FoVy"]], fovs, atol=1e-12, rtol=0):
                raise ValueError("adaptive render FoV differs from planning FoV")
            w = np.eye(4)
            w[:3, :3], w[:3, 3] = np.asarray(c["R"]).T, c["T"]
            matrices.append(np.asarray(value["world_to_pca"]) @ np.linalg.inv(w))
        matrices = np.stack(matrices)
        positions = matrices[:, :3, 3]
        directions = _unit(positions - value["focus_pca"])
        forward = _unit(matrices[:, :3, 2])
        if not np.allclose(forward, -directions, atol=1e-7):
            raise ValueError("adaptive cameras do not look at the declared repair region")
        incidence = np.degrees(np.arccos(np.clip(directions @ arrays["surface_normal_pca"], -1, 1)))
        if not np.allclose(incidence, [f["incidence_deg"] for f in frames], atol=1e-6):
            raise ValueError("adaptive camera incidence differs from planning metadata")
    if ring:
        path = value.get("continuous_path", {})
        for key in ("inclination_deg", "radius_pca", "start_phase_rad", "azimuth_step_rad", "sweep_deg"):
            x = path.get(key)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError("invalid adaptive continuous ring geometry")
        if (path["inclination_deg"] not in cfg["ring_angles_deg"] or path["radius_pca"] <= 0 or
                not 0 <= path["start_phase_rad"] < 2 * math.pi or path["sweep_deg"] != 360 or
                not np.isclose(path["azimuth_step_rad"], 2 * math.pi / count, rtol=0, atol=1e-12) or
                not np.isclose(value["radius_pca"], path["radius_pca"], rtol=1e-10)):
            raise ValueError("adaptive continuous ring geometry differs from policy")
        if positions is not None:
            normal = arrays["surface_normal_pca"]
            u, v = _basis(normal)
            phases = path["start_phase_rad"] + np.arange(count) * path["azimuth_step_rad"]
            theta = math.radians(path["inclination_deg"])
            expected = np.asarray(value["center_pca"]) + path["radius_pca"] * (
                math.cos(theta) * normal + math.sin(theta) * (
                    np.cos(phases)[:, None] * u + np.sin(phases)[:, None] * v))
            if not np.allclose(positions, expected, rtol=1e-7, atol=1e-8):
                raise ValueError("adaptive cameras differ from the declared uniform ring")
    if sweep:
        _validate_sweep(value, count, cfg, arrays, frames, positions)
    if (ring or sweep) and positions is not None:
        # Reconstruct GL rotations from exact CV cameras, never trusting edge statistics alone.
        actual = []
        from scipy.spatial import ConvexHull
        for i, m in enumerate(matrices):
            rotation = _unit(m[:3, :3].T).T @ np.diag([1., -1., -1.])
            if sweep and not np.allclose(rotation, _look_at(positions[i], np.asarray(value["center_pca"]),
                                                          arrays["scene_up_pca"]), atol=1e-7):
                raise ValueError("adaptive sweep camera roll differs from scene up")
            pixels = _project(arrays["roi_points_pca"], positions[i], rotation, np.tan(fovs / 2))[0]
            area = float(ConvexHull(pixels).volume / (96 * 64))
            if not np.isclose(area, frames[i]["roi_fraction"], atol=1e-9, rtol=1e-8):
                raise ValueError("adaptive ROI scale differs from exact cameras")
            actual.append(dict(position=positions[i], rotation=rotation, direction=directions[i],
                               visible=visibility[i], roi_fraction=area))
        actual_edges = tracking_edges(actual, arrays["roi_points_pca"], fovs, closed=not sweep)
    tracking = value.get("tracking_edges")
    if not isinstance(tracking, list) or len(tracking) != count - int(sweep):
        raise ValueError("adaptive tracking path is incomplete")
    for index, edge in enumerate(tracking):
        i = index + int(sweep)
        if edge.get("i") != (i - 1) % count or edge.get("j") != i:
            raise ValueError("adaptive tracking order differs from frame order")
        for key in ("overlap", "angle_deg", "rotation_deg", *(("scale_ratio", "motion_fraction") if (ring or sweep) else ())):
            x = edge.get(key)
            if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
                raise ValueError("invalid adaptive tracking statistics")
        if ring or sweep:
            if (edge["scale_ratio"] < 1 or edge["motion_fraction"] < 0 or _tracking_violation(edge, cfg)):
                raise ValueError("adaptive tracking edge violates continuity limits")
            if positions is not None and any(not np.isclose(edge[k], actual_edges[index][k], atol=1e-7, rtol=1e-8)
                                             for k in ("scale_ratio", "motion_fraction")):
                raise ValueError("adaptive tracking scale/motion differs from exact cameras")
        overlap = (visibility[i - 1] & visibility[i]).sum() / min(visibility[i - 1].sum(), visibility[i].sum())
        if (not np.isclose(overlap, edge["overlap"], atol=1e-9) or not 0 <= edge["angle_deg"] <= 180 or
                not 0 <= edge["rotation_deg"] <= 180):
            raise ValueError("adaptive tracking overlap/angle mismatch")
        if positions is not None:
            angle = math.degrees(math.acos(np.clip(directions[i - 1] @ directions[i], -1, 1)))
            if not np.isclose(angle, edge["angle_deg"], atol=1e-6):
                raise ValueError("adaptive tracking angle differs from exact cameras")
            a, b = matrices[i - 1, :3, :3], matrices[i, :3, :3]
            a, b = _unit(a.T).T, _unit(b.T).T
            rotation = math.degrees(math.acos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
            if not np.isclose(rotation, edge["rotation_deg"], atol=1e-5):
                raise ValueError("adaptive tracking rotation differs from exact cameras")
    pairs = value.get("geometry_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("adaptive geometry pairs are missing")
    seen = set()
    for p in pairs:
        i, j = p.get("i"), p.get("j")
        if type(i) is not int or type(j) is not int or not 0 <= i < j < count or (i, j) in seen:
            raise ValueError("invalid adaptive geometry pair indices")
        seen.add((i, j))
        for key in ("overlap", "angle_deg", "score"):
            x = p.get(key)
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
                raise ValueError("adaptive pair statistics must be finite")
        if (not cfg["pair_min_overlap"] <= p["overlap"] <= 1 or
                not cfg["pair_min_angle_deg"] <= p["angle_deg"] <= cfg["pair_max_angle_deg"] or
                not 0 < p["score"] <= 1):
            raise ValueError("adaptive geometry pair violates overlap/baseline policy")
        shared = visibility[i] & visibility[j]
        overlap = shared.sum() / min(visibility[i].sum(), visibility[j].sum())
        if not np.isclose(p["overlap"], overlap, atol=1e-9):
            raise ValueError("adaptive pair overlap differs from surface visibility")
        score = overlap / (1 + abs(p["angle_deg"] - cfg["pair_target_angle_deg"]) / cfg["pair_target_angle_deg"])
        if not np.isclose(score, p["score"], atol=1e-9):
            raise ValueError("adaptive pair score differs from geometry policy")
        if positions is not None:
            a = _unit(positions[i] - arrays["roi_points_pca"][shared])
            b = _unit(positions[j] - arrays["roi_points_pca"][shared])
            angle = np.median(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))))
            if not np.isclose(angle, p["angle_deg"], atol=1e-6):
                raise ValueError("adaptive pair angle differs from exact cameras")
