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


class MaskedLPIPS(torch.nn.Module):
    """Frozen VGG v0.1; perceptual input gradients are confined to the hole."""
    def __init__(self, cfg, device, *, spatial=True):
        super().__init__()
        import lpips
        device = torch.device(device)
        # Loading a feature extractor must not perturb training/checkpoint RNG.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.network = lpips.LPIPS(net="vgg", version="0.1", spatial=spatial,
                                      verbose=False).to(device).eval().requires_grad_(False)
        self.padding = cfg["appearance"]["lpips_padding"]
        self.max_side = cfg["appearance"]["lpips_max_side"]

    def forward(self, rgb, reference, mask):
        pixels = torch.nonzero(mask, as_tuple=False)
        if not len(pixels):
            return rgb[..., :0].sum()
        h, w = mask.shape
        lo, hi = pixels.amin(0).tolist(), pixels.amax(0).tolist()
        y0, x0 = max(0, lo[0] - self.padding), max(0, lo[1] - self.padding)
        y1, x1 = min(h, hi[0] + self.padding + 1), min(w, hi[1] + self.padding + 1)
        # Keep the same context on both sides and prevent gradients outside M.
        composite = torch.where(mask[None], rgb, reference.detach())
        a, b = composite[None, :, y0:y1, x0:x1], reference.detach()[None, :, y0:y1, x0:x1]
        selected = mask[None, None, y0:y1, x0:x1].to(rgb)
        factor = min(1., self.max_side / max(a.shape[-2:]))
        size = tuple(max(32, round(v * factor)) for v in a.shape[-2:])
        if size != a.shape[-2:]:
            a, b = [F.interpolate(v, size=size, mode="bilinear", align_corners=False) for v in (a, b)]
            selected = F.interpolate(selected, size=size, mode="nearest")
        distance = self.network(a * 2 - 1, b * 2 - 1)
        return (distance * selected).sum() / selected.sum().clamp_min(1)


class OfficialPatchLPIPS(MaskedLPIPS):
    """Released Inpaint360GS: native mask bbox, 2x2 patches, no spatial mask.

    Odd trailing rows/columns are omitted by unfold, matching the release.
    Small patches (<32 in either dimension) contribute zero, also as released.
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device, spatial=False)

    def forward(self, rgb, reference, mask):
        pixels = torch.nonzero(mask, as_tuple=False)
        if not len(pixels):
            return rgb[..., :0].sum()
        lo, hi = pixels.amin(0).tolist(), pixels.amax(0).tolist()
        y0, x0 = lo
        y1, x1 = hi[0] + 1, hi[1] + 1
        a = rgb[None, :, y0:y1, x0:x1]
        b = reference.detach()[None, :, y0:y1, x0:x1]
        ph, pw = a.shape[-2] // 2, a.shape[-1] // 2
        if ph < 32 or pw < 32:
            return rgb[..., :0].sum()

        def patches(image):
            batch, channels = image.shape[:2]
            values = F.unfold(image, (ph, pw), stride=(ph, pw))
            return values.view(batch, channels, ph, pw, -1).permute(0, 4, 1, 2, 3).squeeze(0)

        return self.network(patches(a) * 2 - 1, patches(b) * 2 - 1).mean()


def build_perceptual(cfg, device):
    if not cfg["loss"].get("lpips", 0):
        return None
    cls = OfficialPatchLPIPS if cfg.get("appearance", {}).get("mode", "masked") == "inpaint360gs" else MaskedLPIPS
    return cls(cfg, device)


def scale_flatten_loss(package, gaussians):
    """PGSR scale prior: mean minimum activated axis over visible Gaussians.

    The denominator includes frozen visible rows, exactly as a full-scene
    PGSR render. Ownership still prevents any update to frozen background.
    """
    scales = gaussians.get_scaling
    visible = package["visibility_filter"].to(device=scales.device, dtype=torch.bool).reshape(-1)
    if scales.ndim != 2 or scales.shape[1] != 3 or len(visible) != len(scales):
        raise ValueError("scale flattening visibility/Gaussian shape mismatch")
    # Match upstream PGSR's sort(...)[..., 0], including tied axes. amin()
    # shares gradients across ties and would shrink isotropic seeds uniformly
    # instead of selecting one axis for flattening.
    return scales[visible].sort(dim=-1).values[:, 0].mean() if visible.any() else scales.sum() * 0


def joint_losses(package, target, baseline, step, cfg, perceptual=None, *, gaussians=None):
    rgb, alpha, depth = package["render"], package["rendered_alpha"].squeeze(0), package["plane_depth"].squeeze(0)
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError("nonfinite PGSR RGB/alpha")
    mask = target["mask"].bool()
    valid = mask & torch.isfinite(depth) & (depth > 0) & (alpha.detach() >= cfg["validity"]["render_alpha_min"])
    normal, nv = unit_normal(package["rendered_normal"])
    zero = rgb[..., :0].sum()
    weights, ramp = cfg["loss"], geometry_ramp(step, cfg)
    official = cfg.get("appearance", {}).get("mode", "masked") == "inpaint360gs"
    rgb_weight_mode = cfg.get("appearance", {}).get("rgb_weight_mode", "continuity")
    if rgb_weight_mode not in ("continuity", "uniform"):
        raise ValueError("invalid appearance.rgb_weight_mode")
    rgb_weight = target.get("rgb_weight") if rgb_weight_mode == "continuity" else None
    outside_target = target["rgb"].detach() if official else baseline["rgb"]
    terms = dict(rgb_hole=zero if official else weighted_mean((rgb - target["rgb"]).abs(), mask, rgb_weight),
        rgb_known_preserve=_mean((rgb - outside_target).abs(), ~mask),
        alpha_preserve=_mean(F.relu(torch.maximum(baseline["alpha"], alpha.new_tensor(cfg["validity"]["coverage_alpha_floor"])) - alpha), mask),
        depth=zero, lama_normal=zero, depth_normal_consistency=zero)
    if weights.get("scale_flatten", 0):
        if gaussians is None:
            raise ValueError("scale flattening enabled without Gaussian parameters")
        terms["scale_flatten"] = scale_flatten_loss(package, gaussians)
    if weights.get("ssim", 0):
        from source.losses import ssim
        terms["ssim"] = 1 - ssim(rgb[None], target["rgb"].detach()[None])
    if weights.get("lpips", 0):
        if perceptual is None:
            raise ValueError("LPIPS loss enabled without its frozen feature extractor")
        terms["lpips"] = perceptual(rgb, target["rgb"], mask)
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
    metrics["weight_scale_flatten"] = alpha.new_tensor(weights.get("scale_flatten", 0.))
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
