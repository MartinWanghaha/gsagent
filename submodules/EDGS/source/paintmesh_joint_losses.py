"""PGSR joint losses; external modalities are strictly optional."""
import torch
import torch.nn.functional as F

from source.paintmesh_local_losses import _mean, unit_normal, interior_depth_mask
from source.pgsr_geometry import camera_intrinsics, camera_rays


def weighted_mean(values, mask, weight=None):
    if weight is None:
        return _mean(values, mask)
    selected = values[..., mask]
    weights = weight.detach()[mask]
    if not weights.numel():
        return selected.sum()
    channels = selected.numel() // weights.numel()
    return (selected * weights).sum() / (weights.sum().clamp_min(1e-8) * channels)


def geometry_ramp(step, cfg):
    start, duration = cfg["geometry_from_iter"], cfg["geometry_ramp_iters"]
    return float(step >= start) if duration == 0 else max(0., min(1., (step - start) / duration))


def joint_losses(package, target, baseline, step, cfg):
    rgb, alpha, depth = package["render"], package["rendered_alpha"].squeeze(0), package["plane_depth"].squeeze(0)
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError("nonfinite PGSR RGB/alpha")
    mask = target["mask"].bool()
    valid = mask & torch.isfinite(depth) & (depth > 0) & (alpha.detach() >= cfg["validity"]["render_alpha_min"])
    normal, nv = unit_normal(package["rendered_normal"])
    zero = rgb[..., :0].sum()
    weights, ramp = cfg["loss"], geometry_ramp(step, cfg)
    terms = dict(rgb_hole=weighted_mean((rgb - target["rgb"]).abs(), mask, target.get("rgb_weight")),
        rgb_known_preserve=_mean((rgb - baseline["rgb"]).abs(), ~mask),
        alpha_preserve=_mean(F.relu(torch.maximum(baseline["alpha"], alpha.new_tensor(cfg["validity"]["coverage_alpha_floor"])) - alpha), mask),
        depth=zero, lama_normal=zero, depth_normal_consistency=zero)
    depth_pixels = normal_pixels = consistency_pixels = alpha.new_tensor(0)
    if ramp > 0:
        if cfg["supervision"]["use_depth"] and weights["depth"]:
            reference = target["depth"].detach()
            accepted = valid & torch.isfinite(reference) & (reference > 0)
            a, b = depth[accepted], reference[accepted]
            if a.numel():
                residual = F.smooth_l1_loss(a.log(), b.log(), reduction="none")
                weight = target.get("depth_weight", torch.ones_like(depth)).detach()[accepted]
                terms["depth"] = (residual * weight).sum() / weight.sum().clamp_min(1e-8)
            else:
                terms["depth"] = a.sum()
            depth_pixels = accepted.sum()
        if cfg["supervision"]["use_normal"] and weights["lama_normal"]:
            reference, rv = unit_normal(target["normal"].detach())
            accepted = valid & nv & rv & target["normal_valid"].bool()
            terms["lama_normal"] = weighted_mean(1 - (normal * reference).sum(0).clamp(-1, 1), accepted, target.get("normal_weight"))
            normal_pixels = accepted.sum()
        if weights["depth_normal_consistency"]:
            derived, dv = unit_normal(package["depth_normal"])
            accepted = interior_depth_mask(depth, valid, cfg["validity"]["depth_jump_relative"]) & nv & dv
            terms["depth_normal_consistency"] = _mean(1 - (normal * derived).sum(0).clamp(-1, 1), accepted)
            consistency_pixels = accepted.sum()
    total = sum(value * weights[key] * (ramp if key in ("depth", "lama_normal", "depth_normal_consistency") else 1) for key, value in terms.items())
    metrics = {k: v.detach() for k, v in terms.items()}
    metrics.update(total=total.detach(), ramp=alpha.new_tensor(ramp), coverage=valid.sum() / mask.sum().clamp_min(1),
                   depth_pixels=depth_pixels, normal_pixels=normal_pixels, consistency_pixels=consistency_pixels)
    for key in ("depth", "lama_normal", "depth_normal_consistency", "multiview"):
        metrics["weight_" + key] = alpha.new_tensor(weights[key] * ramp)
    return total, metrics


def multiview_loss(source, target, source_view, target_view, mask, cfg):
    """Visibility-aware rendered z reprojection. No completed geometry is read."""
    depth = source["plane_depth"].squeeze(0)
    other_depth = target["plane_depth"].squeeze(0)
    alpha = source["rendered_alpha"].squeeze(0)
    # Select before unprojection to avoid invalid arithmetic in autograd.
    valid = mask & torch.isfinite(depth) & (depth > 0) & (alpha.detach() >= cfg["validity"]["render_alpha_min"])
    # Deterministic sparse pixels keep multi-view work bounded and RNG-neutral.
    pixels = torch.nonzero(valid, as_tuple=False)[::4]
    if not len(pixels):
        return depth[valid].sum()
    y, x = pixels.T
    rays = camera_rays(source_view, device=depth.device)[y, x]
    pc = rays * depth[y, x, None]
    c2w = source_view.world_view_transform.inverse()
    world = pc @ c2w[:3, :3] + c2w[3, :3]
    matrix = target_view.world_view_transform
    p = world @ matrix[:3, :3] + matrix[3, :3]
    k = camera_intrinsics(target_view, device=depth.device)
    uv = p[:, :2] / p[:, 2:3].clamp_min(1e-6) * k.diag()[:2] + k[:2, 2]
    h, w = other_depth.shape
    grid = (uv + .5) / uv.new_tensor([w, h]) * 2 - 1
    d = F.grid_sample(other_depth[None, None], grid[None, None], align_corners=False).reshape(-1)
    a = F.grid_sample(target["rendered_alpha"][None], grid.detach()[None, None], align_corners=False).reshape(-1)
    accepted = (grid.detach().abs() <= 1).all(1) & (p[:, 2].detach() > 0) & (d.detach() > 0)
    accepted &= (a.detach() >= cfg["validity"]["render_alpha_min"])
    accepted &= ((p[:, 2].detach() - d.detach()).abs() <= cfg["validity"]["depth_jump_relative"] * d.detach())
    return F.smooth_l1_loss(p[accepted, 2].log(), d[accepted].log()) if accepted.any() else p[:0].sum()
