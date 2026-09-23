"""Loss composition for the EDGS/PGSR hybrid trainer.

The rasterizer and camera/model implementations remain owned by EDGS.  This
module only implements the additional geometric priors introduced by PGSR and
keeps every returned value *already weighted*.  Consequently a trainer can
simply sum the dictionaries returned by :meth:`photo_terms` and
:meth:`pgsr_terms` without accidentally counting the photometric objective
twice.

PGSR uses metric plane z-depth throughout this module.  Passing the inverse
depth produced by the native Gaussian-Splatting renderer is invalid.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Callable

import torch
import torch.nn.functional as F

from source.losses import l1_loss, ssim
from source.pgsr_geometry import (
    camera_intrinsics,
    image_gray,
    patch_offsets,
    patch_warp,
    points_from_depth,
    sample_depth,
)
from source.renderers import visible_mask

_MISSING = object()


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Read one option from a dict, OmegaConf node, or namespace."""

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(config, key, default)


def _loss_config(config: Any) -> Any:
    """Accept either ``pgsr_loss`` itself, ``opt``, or the root config."""

    nested = _get(config, "pgsr_loss", _MISSING)
    if nested is not _MISSING:
        return nested
    opt = _get(config, "opt", _MISSING)
    if opt is not _MISSING:
        nested = _get(opt, "pgsr_loss", _MISSING)
        if nested is not _MISSING:
            return nested
    return config


def _first_present(config: Any, keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        value = _get(config, key, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _as_tensor_value(owner: Any, name: str) -> torch.Tensor:
    value = getattr(owner, name)
    value = value() if callable(value) else value
    if not torch.is_tensor(value):
        raise TypeError(f"{type(owner).__name__}.{name} must be a tensor")
    return value


def _zero_term(pkg: Mapping[str, Any], gaussians: Any) -> torch.Tensor:
    """Return a scalar zero on the active graph's device and dtype."""

    for key in ("render", "plane_depth", "rendered_normal", "radii"):
        value = pkg.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    if gaussians is not None and hasattr(gaussians, "get_scaling"):
        return _as_tensor_value(gaussians, "get_scaling").sum() * 0.0
    return torch.zeros((), dtype=torch.float32)


def _safe_mean(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Mean over finite values, preserving a differentiable zero if empty."""

    finite = torch.isfinite(values)
    if values.numel() == 0 or not bool(finite.any().item()):
        return zero
    return values[finite].mean()


def _depth_map(depth: torch.Tensor, name: str) -> torch.Tensor:
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(
            f"{name} must have shape [H,W] or [1,H,W], got {tuple(depth.shape)}"
        )
    return depth


def _require(pkg: Mapping[str, Any], key: str) -> torch.Tensor:
    value = pkg.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"PGSR renderer output is missing tensor '{key}'")
    return value


def _camera_image(
    camera: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    image = getattr(camera, "original_image", None)
    if image is None and hasattr(camera, "get_image"):
        image = camera.get_image()
        if isinstance(image, tuple):
            image = image[0]
    if not torch.is_tensor(image):
        raise AttributeError("camera must expose original_image or get_image()")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(
            f"camera image must have shape [C,H,W], got {tuple(image.shape)}"
        )
    return image[:3].to(device=device, dtype=dtype)


def _image_gradient_weight(image: torch.Tensor, eps: float) -> torch.Tensor:
    """PGSR edge-aware normal weight with a stable constant-image path."""

    _, height, width = image.shape
    if height < 3 or width < 3:
        return image.new_ones((height, width))

    horizontal = (image[:, 1:-1, 2:] - image[:, 1:-1, :-2]).abs().mean(0)
    vertical = (image[:, :-2, 1:-1] - image[:, 2:, 1:-1]).abs().mean(0)
    gradient = torch.maximum(horizontal, vertical)
    span = gradient.amax() - gradient.amin()
    if bool((span <= eps).item()):
        gradient = torch.zeros_like(gradient)
    else:
        gradient = (gradient - gradient.amin()) / span.clamp_min(eps)
    gradient = F.pad(gradient[None, None], (1, 1, 1, 1), value=1.0)[0, 0]
    return (1.0 - gradient).clamp_(0.0, 1.0).square().detach()


def _world_view(camera: Any, reference: torch.Tensor) -> torch.Tensor:
    transform = getattr(camera, "world_view_transform", None)
    if not torch.is_tensor(transform) or transform.shape != (4, 4):
        raise ValueError("camera.world_view_transform must be a [4,4] tensor")
    return transform.to(device=reference.device, dtype=reference.dtype)


def _transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a transposed Gaussian-Splatting transform to row-vector points."""

    return points @ transform[:3, :3] + transform[3, :3]


def _pixel_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    x, y = torch.meshgrid(
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        indexing="xy",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def _project(
    points_camera: torch.Tensor, intrinsic: torch.Tensor, eps: float
) -> torch.Tensor:
    homogeneous = points_camera @ intrinsic.transpose(0, 1)
    z = homogeneous[:, 2:3]
    safe_z = torch.where(z >= 0, z.clamp_min(eps), z.clamp_max(-eps))
    return homogeneous[:, :2] / safe_z


def _normalize_grid(pixels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    result = pixels.clone()
    result[..., 0] = 2.0 * result[..., 0] / max(width - 1, 1) - 1.0
    result[..., 1] = 2.0 * result[..., 1] / max(height - 1, 1) - 1.0
    return result


def _lncc(
    reference: torch.Tensor, neighbor: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local normalized cross-correlation loss for flattened image patches."""

    ref_centered = reference - reference.mean(dim=-1, keepdim=True)
    neighbor_centered = neighbor - neighbor.mean(dim=-1, keepdim=True)
    ref_variance = ref_centered.square().sum(dim=-1)
    neighbor_variance = neighbor_centered.square().sum(dim=-1)
    cross = (ref_centered * neighbor_centered).sum(dim=-1)
    correlation = cross.square() / (ref_variance * neighbor_variance + eps)
    loss = (1.0 - correlation).clamp(0.0, 2.0)
    valid = (
        (ref_variance > eps)
        & (neighbor_variance > eps)
        & torch.isfinite(loss)
        & (loss < 0.9)
    )
    return loss, valid


class PGSRLossComposer:
    """Compose EDGS photometric and PGSR geometric loss terms.

    Args:
        config: The ``pgsr_loss`` config node (or a parent containing it).
        lambda_dssim: EDGS DSSIM interpolation weight.

    The public methods return scalar tensors with their configured weights
    already applied.  ``pgsr_terms`` always returns four stable keys, using a
    graph-connected zero for disabled or inapplicable terms.
    """

    _TERM_KEYS = ("pgsr_scale", "pgsr_normal", "pgsr_geo", "pgsr_ncc")

    def __init__(self, config: Any, lambda_dssim: float) -> None:
        config = _loss_config(config)
        self.enabled = bool(_get(config, "enabled", True))
        self.lambda_dssim = float(lambda_dssim)
        if not 0.0 <= self.lambda_dssim <= 1.0:
            raise ValueError("lambda_dssim must be in [0, 1]")

        self.scale_weight = float(
            _first_present(config, ("scale_weight", "scale_loss_weight"), 0.0)
        )
        self.single_view_weight = float(_get(config, "single_view_weight", 0.0))
        self.single_view_from_iter = int(
            _first_present(
                config,
                ("single_view_from_iter", "single_view_weight_from_iter"),
                7000,
            )
        )
        image_weight = _get(config, "image_weight", _MISSING)
        self.use_image_weight = (
            bool(image_weight)
            if image_weight is not _MISSING
            else not bool(_get(config, "wo_image_weight", False))
        )

        self.multi_view_from_iter = int(
            _first_present(
                config,
                ("multi_view_from_iter", "multi_view_weight_from_iter"),
                7000,
            )
        )
        self.geo_weight = float(
            _first_present(config, ("geo_weight", "multi_view_geo_weight"), 0.0)
        )
        self.ncc_weight = float(
            _first_present(config, ("ncc_weight", "multi_view_ncc_weight"), 0.0)
        )
        self.patch_size = int(
            _first_present(config, ("patch_size", "multi_view_patch_size"), 3)
        )
        self.sample_num = int(
            _first_present(config, ("sample_num", "multi_view_sample_num"), 102400)
        )
        self.pixel_noise_threshold = float(
            _first_present(
                config,
                ("pixel_noise_th", "multi_view_pixel_noise_th"),
                1.0,
            )
        )
        self.use_geo_occlusion = not bool(_get(config, "wo_use_geo_occ_aware", False))
        self.ncc_scale = float(_get(config, "ncc_scale", 1.0))
        self.eps = float(_get(config, "eps", 1.0e-8))

        if self.patch_size < 0:
            raise ValueError("patch_size is a radius and must be non-negative")
        if self.sample_num < 0:
            raise ValueError("sample_num must be non-negative")
        if self.ncc_scale <= 0:
            raise ValueError("ncc_scale must be positive")
        if bool(_get(config, "virtual_camera", False)):
            raise NotImplementedError(
                "virtual_camera is not supported by the EDGS/PGSR hybrid"
            )

    def required_outputs(self, step: int) -> tuple[bool, bool]:
        """Return ``(need_plane, need_depth_normal)`` for this iteration."""

        if not self.enabled:
            return False, False
        single_active = (
            self.single_view_weight != 0.0 and step > self.single_view_from_iter
        )
        multi_active = self.multi_view_active(step)
        return single_active or multi_active, single_active

    def multi_view_active(self, step: int) -> bool:
        """Whether this step computes PGSR neighbor-view geometry."""

        return (
            self.enabled
            and (self.geo_weight != 0.0 or self.ncc_weight != 0.0)
            and step > self.multi_view_from_iter
        )

    def photo_terms(
        self, image: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the two weighted EDGS photometric terms."""

        l1 = l1_loss(image, gt)
        dssim = 1.0 - ssim(image, gt)
        return {
            "photo_l1": (1.0 - self.lambda_dssim) * l1,
            "photo_dssim": self.lambda_dssim * dssim,
        }

    def pgsr_terms(
        self,
        step: int,
        camera: Any,
        pkg: Mapping[str, Any],
        gaussians: Any,
        neighbors: Any,
        render_neighbor: Callable[[Any], Mapping[str, Any]],
        diagnostics: MutableMapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return weighted PGSR scale, normal, geometry, and LNCC terms.

        ``diagnostics`` is an opt-in side channel for training visualizations.
        It is deliberately separate from the scalar loss dictionary so callers
        can continue summing the returned values without special cases.  No
        diagnostic tensors are retained unless a mutable mapping is supplied.
        """

        zero = _zero_term(pkg, gaussians)
        terms = {key: zero for key in self._TERM_KEYS}
        if not self.enabled:
            return terms

        if self.scale_weight != 0.0:
            terms["pgsr_scale"] = self.scale_weight * self._scale_term(
                pkg, gaussians, zero
            )

        if self.single_view_weight != 0.0 and step > self.single_view_from_iter:
            terms["pgsr_normal"] = self.single_view_weight * self._normal_term(
                camera, pkg, zero, diagnostics
            )
        multi_active = self.multi_view_active(step)
        if (
            diagnostics is not None
            and multi_active
            and "image_weight" not in diagnostics
        ):
            rendered = _require(pkg, "rendered_normal")
            image = _camera_image(
                camera,
                device=rendered.device,
                dtype=rendered.dtype,
            )
            image_weight = _image_gradient_weight(image, self.eps)
            if image_weight.shape != rendered.shape[-2:]:
                image_weight = F.interpolate(
                    image_weight[None, None],
                    size=rendered.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            diagnostics["image_weight"] = image_weight.detach()

        if multi_active:
            neighbor = self._choose_neighbor(camera, neighbors)
            if neighbor is not None:
                geo, ncc = self._multi_view_terms(
                    camera,
                    neighbor,
                    pkg,
                    render_neighbor,
                    zero,
                    diagnostics,
                )
                terms["pgsr_geo"] = self.geo_weight * geo
                terms["pgsr_ncc"] = self.ncc_weight * ncc
        return terms

    def _scale_term(
        self, pkg: Mapping[str, Any], gaussians: Any, zero: torch.Tensor
    ) -> torch.Tensor:
        scales = _as_tensor_value(gaussians, "get_scaling")
        mask = visible_mask(pkg).to(device=scales.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != scales.shape[0]:
            raise ValueError(
                "visibility mask and Gaussian scale count differ: "
                f"{mask.numel()} != {scales.shape[0]}"
            )
        if mask.numel() == 0 or not bool(mask.any().item()):
            return zero
        minimum_scale = scales[mask].amin(dim=-1)
        return _safe_mean(minimum_scale, zero)

    def _normal_term(
        self,
        camera: Any,
        pkg: Mapping[str, Any],
        zero: torch.Tensor,
        diagnostics: MutableMapping[str, Any] | None = None,
    ) -> torch.Tensor:
        rendered = _require(pkg, "rendered_normal")
        depth_normal = _require(pkg, "depth_normal").to(
            device=rendered.device, dtype=rendered.dtype
        )
        if rendered.shape != depth_normal.shape:
            raise ValueError(
                "rendered_normal and depth_normal must have identical shapes, got "
                f"{tuple(rendered.shape)} and {tuple(depth_normal.shape)}"
            )
        difference = (depth_normal - rendered).abs().sum(dim=0)
        image_weight = None
        if self.use_image_weight or diagnostics is not None:
            image = _camera_image(camera, device=rendered.device, dtype=rendered.dtype)
            image_weight = _image_gradient_weight(image, self.eps)
            if image_weight.shape != difference.shape:
                image_weight = F.interpolate(
                    image_weight[None, None],
                    size=difference.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            if diagnostics is not None:
                diagnostics["image_weight"] = image_weight.detach()
        if self.use_image_weight:
            difference = difference * image_weight
        return _safe_mean(difference, zero)

    @staticmethod
    def _choose_neighbor(camera: Any, neighbors: Any) -> Any | None:
        if neighbors is None:
            return None
        candidates = neighbors
        if isinstance(neighbors, Mapping):
            candidates = None
            for key in (
                getattr(camera, "image_name", _MISSING),
                getattr(camera, "uid", _MISSING),
                getattr(camera, "colmap_id", _MISSING),
            ):
                if key is not _MISSING and key in neighbors:
                    candidates = neighbors[key]
                    break
        if candidates is None:
            return None
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            candidates = [candidates]
        candidates = [candidate for candidate in candidates if candidate is not None]
        return random.choice(candidates) if candidates else None

    def _multi_view_terms(
        self,
        camera: Any,
        neighbor: Any,
        pkg: Mapping[str, Any],
        render_neighbor: Callable[[Any], Mapping[str, Any]],
        zero: torch.Tensor,
        diagnostics: MutableMapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_depth = _depth_map(_require(pkg, "plane_depth"), "plane_depth")
        neighbor_pkg = render_neighbor(neighbor)
        neighbor_depth = _depth_map(
            _require(neighbor_pkg, "plane_depth"), "neighbor plane_depth"
        ).to(device=reference_depth.device, dtype=reference_depth.dtype)

        height, width = reference_depth.shape
        world_points = points_from_depth(camera, reference_depth)
        if world_points.shape != (height * width, 3):
            raise ValueError(
                "points_from_depth must return [H*W,3], got "
                f"{tuple(world_points.shape)}"
            )
        world_points = world_points.to(
            device=reference_depth.device, dtype=reference_depth.dtype
        )

        neighbor_world_view = _world_view(neighbor, world_points)
        points_neighbor = _transform_points(world_points, neighbor_world_view)
        sampled_neighbor_depth, valid = sample_depth(
            neighbor, neighbor_depth, points_neighbor
        )
        sampled_neighbor_depth = sampled_neighbor_depth.reshape(-1).to(
            device=world_points.device, dtype=world_points.dtype
        )
        valid = valid.reshape(-1).to(device=world_points.device, dtype=torch.bool)

        neighbor_z = points_neighbor[:, 2]
        valid &= (
            torch.isfinite(points_neighbor).all(dim=-1)
            & torch.isfinite(sampled_neighbor_depth)
            & (neighbor_z > self.eps)
            & (sampled_neighbor_depth > self.eps)
        )
        safe_neighbor_z = neighbor_z.clamp_min(self.eps)
        reconstructed_neighbor = (
            points_neighbor / safe_neighbor_z[:, None] * sampled_neighbor_depth[:, None]
        )

        neighbor_camera_to_world = torch.linalg.inv(neighbor_world_view)
        reconstructed_world = _transform_points(
            reconstructed_neighbor, neighbor_camera_to_world
        )
        reference_world_view = _world_view(camera, world_points)
        reconstructed_reference = _transform_points(
            reconstructed_world, reference_world_view
        )
        reference_k = camera_intrinsics(
            camera,
            device=world_points.device,
            dtype=world_points.dtype,
        )
        reprojected = _project(reconstructed_reference, reference_k, self.eps)
        pixels = _pixel_grid(height, width, reference_depth)
        pixel_noise = torch.linalg.vector_norm(reprojected - pixels, dim=-1)
        valid &= (
            torch.isfinite(reconstructed_reference).all(dim=-1)
            & (reconstructed_reference[:, 2] > self.eps)
            & torch.isfinite(pixel_noise)
        )

        if self.use_geo_occlusion:
            valid &= pixel_noise < self.pixel_noise_threshold
            weights = torch.exp(-pixel_noise).detach()
        else:
            weights = torch.ones_like(pixel_noise)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        if diagnostics is not None:
            diagnostics["reprojection_weight"] = weights.detach().reshape(height, width)
            diagnostics["reprojection_valid"] = valid.detach().reshape(height, width)
            diagnostics["pixel_noise"] = pixel_noise.detach().reshape(height, width)
            diagnostics["neighbor_image_name"] = str(
                getattr(neighbor, "image_name", "unknown")
            )

        geo = zero
        if self.geo_weight != 0.0 and bool(valid.any().item()):
            geo = _safe_mean((weights * pixel_noise)[valid], zero)

        ncc = zero
        if self.ncc_weight != 0.0 and bool(valid.any().item()):
            ncc = self._ncc_term(
                camera,
                neighbor,
                pkg,
                pixels,
                weights,
                valid,
                reference_world_view,
                neighbor_world_view,
                zero,
            )
        return geo, ncc

    def _ncc_term(
        self,
        camera: Any,
        neighbor: Any,
        pkg: Mapping[str, Any],
        pixels: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        reference_world_view: torch.Tensor,
        neighbor_world_view: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() > self.sample_num:
            order = torch.randperm(valid_indices.numel(), device=valid_indices.device)
            valid_indices = valid_indices[order[: self.sample_num]]
        if valid_indices.numel() == 0:
            return zero

        normal = _require(pkg, "rendered_normal")
        distance = _depth_map(_require(pkg, "rendered_distance"), "rendered_distance")
        normal_flat = normal.permute(1, 2, 0).reshape(-1, 3)[valid_indices]
        distance_flat = distance.reshape(-1)[valid_indices]
        plane_valid = (
            torch.isfinite(normal_flat).all(dim=-1)
            & torch.isfinite(distance_flat)
            & (distance_flat.abs() > self.eps)
        )
        if not bool(plane_valid.any().item()):
            return zero
        valid_indices = valid_indices[plane_valid]
        normal_flat = normal_flat[plane_valid]
        distance_flat = distance_flat[plane_valid].clamp_min(self.eps)

        # Convert transposed row-vector W2C matrices back to conventional
        # column-vector extrinsics, then form C_ref -> C_neighbor.
        ref_extrinsic = reference_world_view.transpose(0, 1)
        neighbor_extrinsic = neighbor_world_view.transpose(0, 1)
        relative = neighbor_extrinsic @ torch.linalg.inv(ref_extrinsic)
        rotation = relative[:3, :3]
        translation = relative[:3, 3]
        plane_homography = (
            rotation[None]
            - (translation[None, :, None] * normal_flat[:, None, :])
            / distance_flat[:, None, None]
        )

        ref_k = camera_intrinsics(
            camera,
            scale=self.ncc_scale,
            device=normal.device,
            dtype=normal.dtype,
        )
        neighbor_k = camera_intrinsics(
            neighbor,
            scale=self.ncc_scale,
            device=normal.device,
            dtype=normal.dtype,
        )
        homography = neighbor_k[None] @ plane_homography @ torch.linalg.inv(ref_k)

        centers = pixels[valid_indices] / self.ncc_scale
        offsets = patch_offsets(self.patch_size, centers.device, dtype=centers.dtype)
        reference_pixels = centers[:, None, :] + offsets
        neighbor_pixels = patch_warp(homography, reference_pixels)

        reference_gray = image_gray(
            camera, scale=self.ncc_scale, device=normal.device
        ).to(dtype=normal.dtype)
        neighbor_gray = image_gray(
            neighbor, scale=self.ncc_scale, device=normal.device
        ).to(dtype=normal.dtype)
        ref_height, ref_width = reference_gray.shape[-2:]
        neighbor_height, neighbor_width = neighbor_gray.shape[-2:]
        reference_grid = _normalize_grid(
            reference_pixels, ref_height, ref_width
        ).reshape(1, -1, 1, 2)
        neighbor_grid = _normalize_grid(
            neighbor_pixels, neighbor_height, neighbor_width
        ).reshape(1, -1, 1, 2)

        reference_values = F.grid_sample(
            reference_gray[None], reference_grid, align_corners=True
        ).reshape(valid_indices.numel(), -1)
        neighbor_values = F.grid_sample(
            neighbor_gray[None], neighbor_grid, align_corners=True
        ).reshape(valid_indices.numel(), -1)
        ncc, texture_valid = _lncc(reference_values, neighbor_values, self.eps)
        if not bool(texture_valid.any().item()):
            return zero
        weighted = ncc * weights[valid_indices]
        return _safe_mean(weighted[texture_valid], zero)


__all__ = ["PGSRLossComposer"]
