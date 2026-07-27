"""Forward-only renderer-consistent opacity queries at arbitrary 3D points."""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, TYPE_CHECKING

import torch
from torch import Tensor

from . import _C

if TYPE_CHECKING:
    from . import GaussianRasterizationSettings


class PointIntegrationResult(NamedTuple):
    """Opacity accumulated from the camera to each query point.

    ``inside`` is the calibrated camera-frustum predicate. ``visibility`` also
    requires non-negligible line-of-sight transmittance, so a point behind an
    already opaque surface is not treated as a valid observation.
    """

    alpha: Tensor
    transmittance: Tensor
    inside: Tensor
    visibility: Tensor


def has_point_integration_extension() -> bool:
    """Return whether the loaded CUDA extension contains the point-query ABI."""

    return (
        _C is not None
        and hasattr(_C, "prepare_point_integration")
        and hasattr(_C, "integrate_points_forward")
    )


class PointIntegrationContext:
    """Camera-specific projected Gaussian state shared by many point queries."""

    __slots__ = (
        "_projected_means",
        "_conics",
        "_precisions",
        "_precision_means",
        "_opacities",
        "_radii",
        "_sorted_gaussian_ids",
        "_tile_ranges",
        "_viewmatrix",
        "_image_height",
        "_image_width",
        "_tanfovx",
        "_tanfovy",
        "_cx",
        "_cy",
        "_query_chunk_size",
    )

    def __init__(
        self,
        *,
        projected_means: Tensor,
        conics: Tensor,
        precisions: Tensor,
        precision_means: Tensor,
        opacities: Tensor,
        radii: Tensor,
        sorted_gaussian_ids: Tensor,
        tile_ranges: Tensor,
        viewmatrix: Tensor,
        image_height: int,
        image_width: int,
        tanfovx: float,
        tanfovy: float,
        cx: float,
        cy: float,
        query_chunk_size: int,
    ) -> None:
        self._projected_means = projected_means
        self._conics = conics
        self._precisions = precisions
        self._precision_means = precision_means
        self._opacities = opacities
        self._radii = radii
        self._sorted_gaussian_ids = sorted_gaussian_ids
        self._tile_ranges = tile_ranges
        self._viewmatrix = viewmatrix
        self._image_height = image_height
        self._image_width = image_width
        self._tanfovx = tanfovx
        self._tanfovy = tanfovy
        self._cx = cx
        self._cy = cy
        self._query_chunk_size = query_chunk_size

    @property
    def device(self) -> torch.device:
        return self._projected_means.device

    @property
    def gaussian_count(self) -> int:
        return int(self._projected_means.shape[0])

    @property
    def radii(self) -> Tensor:
        """Projected Gaussian radii for this camera."""

        return self._radii

    @property
    def gaussian_visibility(self) -> Tensor:
        """Gaussians with a non-empty footprint in the calibrated view."""

        return self._radii > 0

    @torch.no_grad()
    def query(
        self,
        points3d: Tensor,
        *,
        chunk_size: Optional[int] = None,
        visibility_threshold: float = 1e-3,
    ) -> PointIntegrationResult:
        """Query points shaped ``[..., 3]`` while preserving leading axes."""

        if not has_point_integration_extension():
            raise RuntimeError(
                "the installed semantic rasterizer predates point integration; "
                "reinstall diff-semantic-gaussian-rasterization"
            )
        if points3d.ndim < 1 or points3d.shape[-1] != 3:
            raise ValueError("points3d must have shape [...,3]")
        if points3d.device != self.device:
            raise ValueError(
                f"points3d is on {points3d.device}, expected {self.device}"
            )
        if points3d.dtype != torch.float32:
            raise ValueError("point integration requires float32 query points")
        selected_chunk_size = (
            self._query_chunk_size if chunk_size is None else int(chunk_size)
        )
        if selected_chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must lie in [0,1]")

        output_shape = points3d.shape[:-1]
        flat_points = points3d.detach().reshape(-1, 3)
        transmittance_chunks: list[Tensor] = []
        inside_chunks: list[Tensor] = []
        for start in range(0, flat_points.shape[0], selected_chunk_size):
            transmittance, inside = _C.integrate_points_forward(
                flat_points[start : start + selected_chunk_size].contiguous(),
                self._projected_means,
                self._conics,
                self._precisions,
                self._precision_means,
                self._opacities,
                self._radii,
                self._sorted_gaussian_ids,
                self._tile_ranges,
                self._viewmatrix,
                self._image_height,
                self._image_width,
                self._tanfovx,
                self._tanfovy,
                self._cx,
                self._cy,
            )
            transmittance_chunks.append(transmittance)
            inside_chunks.append(inside)

        if transmittance_chunks:
            transmittance = torch.cat(transmittance_chunks).reshape(output_shape)
            inside = torch.cat(inside_chunks).reshape(output_shape)
        else:
            transmittance = points3d.new_empty(output_shape)
            inside = torch.empty(output_shape, dtype=torch.bool, device=self.device)
        alpha = (1.0 - transmittance).clamp_(0.0, 1.0)
        visibility = inside & (transmittance >= visibility_threshold)
        return PointIntegrationResult(
            alpha=alpha,
            transmittance=transmittance,
            inside=inside,
            visibility=visibility,
        )


@torch.no_grad()
def prepare_point_integration(
    means3d: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
    raster_settings: GaussianRasterizationSettings,
    *,
    query_chunk_size: int = 65_536,
) -> PointIntegrationContext:
    """Project Gaussians once and return a reusable camera query context."""

    if not has_point_integration_extension():
        raise RuntimeError(
            "renderer-consistent point integration requires the rebuilt CUDA "
            "semantic rasterizer extension"
        )
    if not means3d.is_cuda:
        raise RuntimeError("point integration requires CUDA Gaussian tensors")
    if means3d.dtype != torch.float32:
        raise RuntimeError("point integration requires float32 Gaussian tensors")
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError("means3d must have shape [N,3]")
    count = means3d.shape[0]
    if scales.shape != means3d.shape:
        raise ValueError("scales must have shape [N,3]")
    if rotations.shape != (count, 4):
        raise ValueError("rotations must have shape [N,4]")
    if opacities.numel() != count:
        raise ValueError("opacities must contain N elements")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    tensors = (opacities, scales, rotations)
    if any(tensor.device != means3d.device for tensor in tensors):
        raise ValueError("all Gaussian tensors must be on the same CUDA device")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("all Gaussian tensors must be float32")

    viewmatrix = raster_settings.viewmatrix.detach().to(means3d).contiguous()
    cx = float(raster_settings.cx)
    cy = float(raster_settings.cy)
    if not math.isfinite(cx):
        cx = 0.5 * float(raster_settings.image_width)
    if not math.isfinite(cy):
        cy = 0.5 * float(raster_settings.image_height)
    state = _C.prepare_point_integration(
        means3d.detach().contiguous(),
        scales.detach().contiguous(),
        rotations.detach().contiguous(),
        viewmatrix,
        int(raster_settings.image_height),
        int(raster_settings.image_width),
        float(raster_settings.tanfovx),
        float(raster_settings.tanfovy),
        cx,
        cy,
        float(raster_settings.scale_modifier),
        float(raster_settings.antialias_sigma),
    )
    return PointIntegrationContext(
        projected_means=state[0],
        conics=state[1],
        precisions=state[2],
        precision_means=state[3],
        opacities=opacities.detach().reshape(-1).contiguous(),
        radii=state[4],
        sorted_gaussian_ids=state[5],
        tile_ranges=state[6],
        viewmatrix=viewmatrix,
        image_height=int(raster_settings.image_height),
        image_width=int(raster_settings.image_width),
        tanfovx=float(raster_settings.tanfovx),
        tanfovy=float(raster_settings.tanfovy),
        cx=cx,
        cy=cy,
        query_chunk_size=int(query_chunk_size),
    )


__all__ = [
    "PointIntegrationContext",
    "PointIntegrationResult",
    "has_point_integration_extension",
    "prepare_point_integration",
]
