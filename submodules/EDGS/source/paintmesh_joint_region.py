"""Fixed local ownership from training masks and 3D support; no GT access."""
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def select_local_rows(vertex, new_mask, cameras, masks, support, spacing, config):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("local selection needs positive initialization spacing")
    if len(cameras) != len(masks) or config["min_views"] > len(cameras):
        raise ValueError("local selection min_views exceeds camera count")
    support = np.asarray(support, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or not len(support) or not np.isfinite(support).all():
        raise ValueError("invalid local selection support")
    xyz = np.column_stack([vertex[k] for k in "xyz"]).astype(np.float64)
    old = np.flatnonzero(~new_mask)
    sigma = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(np.float64)).max(1)
    if not np.isfinite(sigma).all():
        raise ValueError("nonfinite local Gaussian support")
    distance = cKDTree(support).query(xyz[old], workers=4)[0]
    # Spatial gate has a capped 3-sigma footprint: a giant distant Gaussian
    # must not unfreeze unrelated scenery merely by overlapping a 2D mask.
    reach = spacing * config["support_distance_factor"] + np.minimum(
        3 * sigma[old], spacing * config["gaussian_radius_cap_factor"])
    candidates = old[distance <= reach]
    votes = np.zeros(len(vertex), dtype=np.uint16)
    per_view = []
    for cam, raw_mask in zip(cameras, masks):
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (cam["h"], cam["w"]):
            raise ValueError("local mask/camera shape mismatch")
        if not mask.any():
            per_view.append(0)
            continue
        im = Image.fromarray(mask.astype(np.uint8) * 255)
        radius = config["mask_dilation"]
        if radius:
            im = im.filter(ImageFilter.MaxFilter(2 * radius + 1))
        expanded = np.asarray(im) > 0
        distances = distance_transform_edt(~expanded)
        w2c, k = cam["w2c"], cam["k"]
        p = xyz[candidates] @ w2c[:3, :3].T + w2c[:3, 3]
        front = p[:, 2] > 1e-8
        q = p @ k.T
        uv = q[:, :2] / np.maximum(p[:, 2:3], 1e-8)
        # Conservative screen-space bound for a 3-sigma enclosing sphere.
        # ||R|| also handles the uniform scale in the camera manifest.
        r = 3 * sigma[candidates] * np.linalg.norm(w2c[:3, :3], ord=2)
        z = np.maximum(p[:, 2], 1e-8)
        footprint = max(k[0, 0], k[1, 1]) * r / np.maximum(z - r, 1e-8)
        footprint *= 1 + np.linalg.norm(p[:, :2], axis=1) / z
        clipped = np.clip(uv, [0, 0], [cam["w"] - 1, cam["h"] - 1])
        pixel = np.rint(clipped).astype(np.int64)
        delta = np.linalg.norm(uv - clipped, axis=1)
        nearest = distances[pixel[:, 1], pixel[:, 0]] + delta
        hit = front & (nearest <= footprint + .71)
        votes[candidates[hit]] += 1
        per_view.append(int(hit.sum()))
    trainable = new_mask.copy()
    trainable[old] = votes[old] >= config["min_views"]
    selected_old = int((trainable & ~new_mask).sum())
    if not selected_old or trainable.all():
        raise ValueError("local selection must update some old rows and freeze far rows")
    stats = dict(initial_new=int(new_mask.sum()), initial_old=len(old),
        candidate_old=len(candidates), trainable_old=selected_old,
        frozen_old=int((~trainable).sum()), spacing=float(spacing),
        spatial_margin=float(spacing * config["support_distance_factor"]),
        gaussian_spatial_radius_cap=float(spacing * config["gaussian_radius_cap_factor"]),
        per_view_overlap_old=per_view, config=config)
    return trainable, votes, stats
