"""Object-centred camera candidates, target surface probes and render checks."""
from __future__ import annotations

import copy
import math
import numpy as np

from view_planning_geometry import camera_center, world_to_camera, project, MeshScene


def look_at(center, focus, up, template, index):
    forward = np.asarray(focus)-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("look-at direction is parallel to scene up")
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(forward, right), forward))
    linear_scale = np.linalg.norm(np.asarray(template["R"])[:, 0])
    linear = rotation.T * linear_scale
    raw_center = np.asarray(center)/template["scale"]-template["trans"]
    record = copy.deepcopy(template)
    record.pop("source_image_name", None)
    record.update(image_name=f"{index:05d}", R=linear.T.tolist(), T=(-linear@raw_center).tolist())
    return record


def scene_up(cameras):
    ups = []
    for camera in cameras:
        inverse = np.linalg.inv(world_to_camera(camera)[:3, :3])
        up = -inverse[:, 1]
        ups.append(up/np.linalg.norm(up))
    up = np.sum(ups, axis=0)
    if np.linalg.norm(up) < .1*len(ups):
        raise ValueError("ambiguous scene up; real-camera rolls are inconsistent")
    return up/np.linalg.norm(up)


def sample_target(vertices, triangles, gaussian_positions, count, seed):
    points = np.asarray(vertices)[np.asarray(triangles)]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    areas = np.linalg.norm(cross, axis=1)/2
    good = areas > 1e-14
    points, areas, cross = points[good], areas[good], cross[good]
    if not len(areas) or not np.isfinite(areas).all():
        raise ValueError("insufficient target semantic surface")
    envelope = np.vstack((points.reshape(-1, 3), gaussian_positions))
    low, high = envelope.min(0), envelope.max(0)
    focus = (low+high)/2
    radius = float(np.linalg.norm(high-low)/2)
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("degenerate target envelope")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(points), size=count, p=areas/areas.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1-uv[uv.sum(1) > 1]
    probes = points[chosen, 0] + uv[:, :1]*(points[chosen, 1]-points[chosen, 0]) + uv[:, 1:]*(points[chosen, 2]-points[chosen, 0])
    return dict(focus=focus, radius=radius, probes=probes,
        bounds=np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]),
        area=float(areas.sum()), target_scene=MeshScene(vertices, triangles),
        gaussian_bounds=[np.min(gaussian_positions, axis=0).tolist(), np.max(gaussian_positions, axis=0).tolist()])


def generate_candidates(roi, training, config):
    template = training[0]
    up = scene_up(training)
    # FoV only proposes distances; exact projected envelope is checked later.
    distance = roi["radius"] / math.sin(min(template["FoVx"], template["FoVy"])/2) * 1.15
    positions = []
    n = config["candidate_directions"]
    # Dense orbit bands add short, certified-able graph steps. The final route
    # need not complete an orbit and is never interpolated after selection.
    helper = np.eye(3)[np.argmin(np.abs(up))]
    right = np.cross(helper, up); right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    step = config["selection"]["max_neighbor_translation_over_target_radius"]*roi["radius"]*.8
    for multiplier in config["radius_multipliers"]:
        radius = distance*multiplier
        directions = []
        heights = (np.arange(n)+.5)/n*2-1
        for i, height in enumerate(heights):
            angle = i * (math.pi*(3-math.sqrt(5)))
            directions.append(up*height + math.sqrt(1-height*height)*(math.cos(angle)*right+math.sin(angle)*forward))
        orbit_count = min(1024, max(n, math.ceil(2*math.pi*radius/step)))
        for elevation in (0., math.radians(25), math.radians(50)):
            for angle in np.linspace(0, 2*math.pi, orbit_count, endpoint=False):
                directions.append(up*math.sin(elevation) + math.cos(elevation)*(math.cos(angle)*right+math.sin(angle)*forward))
        for direction in directions:
            if abs(np.dot(direction, up)) < math.sin(math.radians(config["view_quality"]["max_elevation_deg"])):
                positions.append(roi["focus"]+radius*direction)
    # Include real origins and local neighborhoods as anchors for paths that
    # cannot safely orbit a wall-adjacent object.
    for camera in training:
        origin = camera_center(camera)
        positions.append(origin)
        for axis in (right, forward, up):
            for offset in (-step, step):
                positions.append(origin+offset*axis)
    unique, records = [], []
    seen = set()
    for position in positions:
        key = tuple(np.round(position/roi["radius"], 8))
        if key in seen or np.linalg.norm(position-roi["focus"]) <= roi["radius"]:
            continue
        seen.add(key)
        try:
            record = look_at(position, roi["focus"], up, template, len(records))
        except ValueError:
            continue
        unique.append(position)
        records.append(record)
        if len(records) > config["max_candidates"]:
            raise ValueError("candidate resource limit exceeded; increase explicit max_candidates")
    return np.asarray(unique), records, up


def evaluate_candidate(scene, free, roi, position, camera, config, safe_radius, margin):
    distance = float(scene.distance([position])[0])
    if distance < safe_radius+margin:
        return None, "camera_collision"
    if not free.contains_ball(position, safe_radius+margin):
        return None, "unknown_space"
    q = config["view_quality"]
    uv, z = project(camera, roi["bounds"])
    border = q["image_border_fraction"]
    if (z <= camera["znear"]).any() or (z >= camera["zfar"]).any() or (uv < border).any() or (uv > 1-border).any():
        return None, "target_clipped"
    area = float(np.prod(uv.max(0)-uv.min(0)))
    if not q["target_area_fraction"][0] <= area <= q["target_area_fraction"][1]:
        return None, "target_framing"
    vectors = roi["probes"]-position
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors/lengths[:, None]
    hit = scene.cast(position, directions)
    isolated = roi["target_scene"].cast(position, directions)
    tolerance = roi["radius"]*q["surface_match_over_target_radius"]
    potential = np.abs(isolated-lengths) <= tolerance
    visible = potential & (np.abs(hit-lengths) <= tolerance)
    ratio = float(visible.sum()/max(potential.sum(), 1))
    if not potential.any() or ratio < q["min_target_visible_ratio"]:
        return None, "target_occluded"
    return dict(visible=visible, visible_ratio=ratio, area_fraction=area,
        mesh_distance=distance, score=ratio + .25*visible.mean()), None


def preview_metrics(arrays, config, target_radius, camera):
    """Visibility relative to isolated target; context outside the removal hole."""
    from scipy.ndimage import binary_dilation
    q = config["view_quality"]
    target = arrays["target_alpha"] >= q["preview_alpha_min"]
    tolerance = target_radius*q["surface_match_over_target_radius"]
    # Convert world uncertainty to camera-z units (uniform scale preserved).
    tolerance *= np.linalg.norm(world_to_camera(camera)[2, :3])
    visible = (target & (arrays["full_alpha"] >= q["preview_alpha_min"])
        & (arrays["full_depth"] > 0) & (arrays["target_depth"] > 0)
        & (np.abs(arrays["full_depth"]-arrays["target_depth"]) <= tolerance))
    visibility = float(visible.sum()/max(target.sum(), 1))
    # The actual all-selected hole includes temporary surrounding occluders.
    changed = ((arrays["full_alpha"]-arrays["removed_alpha"] > .1)
        | ((arrays["full_depth"] > 0) & (arrays["removed_depth"] > 0)
           & (np.abs(arrays["full_depth"]-arrays["removed_depth"]) > tolerance)))
    hole = target | changed
    ring = binary_dilation(hole, iterations=q["context_ring_pixels"]) & ~hole
    valid = (np.isfinite(arrays["removed_depth"]) & (arrays["removed_depth"] > 0)
             & (arrays["removed_alpha"] >= q["preview_alpha_min"]))
    context = float((valid & ring).sum()/max(ring.sum(), 1))
    ys, xs = np.nonzero(target)
    border = q["image_border_fraction"]
    h, w = target.shape
    framed = bool(len(xs) and xs.min() >= border*w and xs.max() < (1-border)*w
                  and ys.min() >= border*h and ys.max() < (1-border)*h)
    area = float(target.mean())
    passed = bool(target.any() and ring.any() and framed
        and q["target_area_fraction"][0] <= area <= q["target_area_fraction"][1]
        and visibility >= q["min_target_visible_ratio"] and context >= q["min_context_valid_ratio"])
    return dict(passed=passed, visible_ratio=visibility, context_valid_ratio=context,
                target_area_fraction=area, target_framed=framed)
