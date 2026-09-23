"""PaintMesh-only losses. Intentionally independent of PGSRLossComposer."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def geometry_ramp(step: int, config: dict) -> float:
    return max(0.0, min(1.0, (step - config["geometry_from_iter"]) /
                        config["geometry_ramp_iters"]))


def _mean(values, mask):
    # Empty selection is differentiable, without ever evaluating 0 * NaN.
    selected = values[..., mask]
    return selected.mean() if selected.numel() else selected.sum()


def unit_normal(value):
    finite = torch.isfinite(value).all(dim=0)
    safe = torch.where(finite[None], value, torch.zeros_like(value))
    length = torch.linalg.vector_norm(safe, dim=0)
    return F.normalize(safe, dim=0, eps=1e-6), finite & (length > 1e-6)


def local_depth_normal(camera, depth):
    """Sanitize before cross products, not after NaNs entered autograd."""
    from source.pgsr_geometry import depth_to_normal
    safe = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.zeros_like(depth))
    return depth_to_normal(camera, safe)


def interior_depth_mask(depth, valid, jump):
    """Exclude hole edges, invalid neighbors and relative depth discontinuities."""
    result = torch.zeros_like(valid)
    if min(depth.shape) < 3:
        return result
    center = depth[1:-1, 1:-1].detach()
    accepted = valid[1:-1, 1:-1].clone()
    for neighbor, neighbor_valid in (
        (depth[:-2, 1:-1], valid[:-2, 1:-1]),
        (depth[2:, 1:-1], valid[2:, 1:-1]),
        (depth[1:-1, :-2], valid[1:-1, :-2]),
        (depth[1:-1, 2:], valid[1:-1, 2:]),
    ):
        accepted &= neighbor_valid & ((neighbor.detach() - center).abs() <=
                                       jump * center.abs().clamp_min(1e-6))
    result[1:-1, 1:-1] = accepted
    return result


def local_losses(package, target, baseline, step, config):
    """Return scalar total and scalar diagnostics, retaining geometry gradients.

    Targets are CHW RGB/normal and HW depth/mask/normal_valid tensors.
    No removed alpha is used to invalidate LaMa predictions inside the hole.
    """
    rgb = package["render"]
    alpha = package["rendered_alpha"].squeeze(0)
    depth = package["plane_depth"].squeeze(0)
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError("non-finite PGSR RGB/alpha")
    mask = target["mask"].bool().detach()
    target_depth = target["depth"].detach()
    target_valid = mask & torch.isfinite(target_depth) & (target_depth > 0)
    prediction_valid = (torch.isfinite(depth) & (depth > 0) &
                        (alpha.detach() >= config["validity"]["render_alpha_min"]))
    depth_valid = target_valid & prediction_valid
    # Normalizing the alpha-weighted numerator is equivalent to dividing by
    # positive alpha and then normalizing. Avoid a redundant unstable division.
    normal, normal_valid = unit_normal(package["rendered_normal"])
    target_normal, target_normal_valid = unit_normal(target["normal"].detach())
    valid_normal = (depth_valid & normal_valid & target_normal_valid &
                    target["normal_valid"].bool().detach())
    weights = config["loss"]
    ramp = geometry_ramp(step, config)
    zero = rgb[..., :0].sum()
    consistency_valid = torch.zeros_like(mask)
    terms = {
        "rgb_hole": _mean((rgb - target["rgb"].detach()).abs(), mask),
        "rgb_known_preserve": _mean((rgb - baseline["rgb"].detach()).abs(), ~mask),
        "alpha_preserve": _mean(F.relu(torch.maximum(
            baseline["alpha"].detach(), alpha.new_tensor(
                config["validity"]["coverage_alpha_floor"])) - alpha), target_valid),
        "depth": zero, "lama_normal": zero, "depth_normal_consistency": zero,
    }
    if ramp > 0:
        if weights["depth"]:
            # Select first: NaN or nonpositive values never enter log/backward.
            d, reference = depth[depth_valid], target_depth[depth_valid]
            terms["depth"] = (F.smooth_l1_loss(d.log(), reference.log())
                              if d.numel() else d.sum())
        if weights["lama_normal"]:
            cosine = (normal * target_normal).sum(0).clamp(-1, 1)
            terms["lama_normal"] = _mean(1 - cosine, valid_normal)
        if weights["depth_normal_consistency"]:
            derived, derived_valid = unit_normal(package["depth_normal"])
            interior = interior_depth_mask(
                depth, depth_valid, config["validity"]["depth_jump_relative"])
            cosine = (normal * derived).sum(0).clamp(-1, 1)
            consistency_valid = interior & normal_valid & derived_valid
            terms["depth_normal_consistency"] = _mean(1 - cosine, consistency_valid)
    total = sum(weights[name] * value * (ramp if name in (
        "depth", "lama_normal", "depth_normal_consistency") else 1)
        for name, value in terms.items())
    denominator = target_valid.sum().clamp_min(1)
    diagnostics = {name: value.detach() for name, value in terms.items()}
    diagnostics.update(
        ramp=rgb.new_tensor(ramp),
        depth_pixels=depth_valid.sum().detach(),
        normal_pixels=valid_normal.sum().detach(),
        consistency_pixels=consistency_valid.sum().detach(),
        coverage=depth_valid.sum().detach() / denominator,
        total=total.detach(),
    )
    for name in ("depth", "lama_normal", "depth_normal_consistency"):
        diagnostics["weight_" + name] = rgb.new_tensor(weights[name] * ramp)
    return total, diagnostics
