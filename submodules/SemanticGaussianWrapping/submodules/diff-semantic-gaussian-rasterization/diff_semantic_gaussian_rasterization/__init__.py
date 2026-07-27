"""Semantic Gaussian rasterization with a CUDA/reference dual backend.

The public classes mirror the original Graphdeco extension closely, but this
package has an independent module name and a richer, fixed output contract.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .reference import (
    SEMANTIC_DIM,
    ReferenceRasterizationResult,
    project_gaussians,
    rasterize_reference,
)

try:  # A source checkout must remain usable on machines without CUDA/NVCC.
    from . import _C
except (ImportError, OSError):
    _C = None

from .point_integration import (
    PointIntegrationContext,
    PointIntegrationResult,
    has_point_integration_extension,
    prepare_point_integration,
)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: Tensor
    scale_modifier: float
    viewmatrix: Tensor
    projmatrix: Tensor
    sh_degree: int
    campos: Tensor
    prefiltered: bool = False
    debug: bool = False
    backend: str = "auto"
    chunk_size: int = 32
    antialias_sigma: float = 0.3
    # NaN preserves the legacy constructor while resolving to the centered
    # pinhole principal point at dispatch time.  Unlike a negative sentinel it
    # still permits calibrated principal points outside the image bounds.
    cx: float = math.nan
    cy: float = math.nan


class RasterizationResult(NamedTuple):
    color: Tensor
    semantic: Tensor
    expected_depth: Tensor
    alpha: Tensor
    normal: Tensor
    radii: Tensor
    dominant_index: Tensor


def has_cuda_extension() -> bool:
    """Return whether the independently named compiled extension is loaded."""

    return (
        _C is not None
        and hasattr(_C, "rasterize_forward")
        and hasattr(_C, "rasterize_backward")
    )


def _principal_point(settings: GaussianRasterizationSettings) -> tuple[float, float]:
    cx = float(settings.cx)
    cy = float(settings.cy)
    return (
        cx if math.isfinite(cx) else 0.5 * float(settings.image_width),
        cy if math.isfinite(cy) else 0.5 * float(settings.image_height),
    )


_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh(degree: int, sh: Tensor, directions: Tensor) -> Tensor:
    """Evaluate real spherical harmonics up to degree three.

    Args:
        sh: coefficients shaped ``[N,3,K]``.
        directions: normalized directions shaped ``[N,3]``.
    """

    if not 0 <= degree <= 3:
        raise ValueError("the semantic rasterizer currently supports SH degree 0..3")
    required = (degree + 1) ** 2
    if sh.ndim != 3 or sh.shape[1] != 3 or sh.shape[2] < required:
        raise ValueError(f"sh must have shape [N,3,K] with K >= {required}")
    result = _C0 * sh[..., 0]
    if degree == 0:
        return result
    x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
    result = result - _C1 * y * sh[..., 1] + _C1 * z * sh[..., 2] - _C1 * x * sh[..., 3]
    if degree == 1:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + _C2[0] * xy * sh[..., 4]
        + _C2[1] * yz * sh[..., 5]
        + _C2[2] * (2 * zz - xx - yy) * sh[..., 6]
        + _C2[3] * xz * sh[..., 7]
        + _C2[4] * (xx - yy) * sh[..., 8]
    )
    if degree == 2:
        return result
    return (
        result
        + _C3[0] * y * (3 * xx - yy) * sh[..., 9]
        + _C3[1] * xy * z * sh[..., 10]
        + _C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
        + _C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
        + _C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
        + _C3[5] * z * (xx - yy) * sh[..., 14]
        + _C3[6] * x * (xx - 3 * yy) * sh[..., 15]
    )


def _reference_from_settings(
    means3d: Tensor,
    means2d: Tensor,
    colors: Tensor,
    semantic_features: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
    settings: GaussianRasterizationSettings,
    depth_order: Optional[Tensor] = None,
) -> ReferenceRasterizationResult:
    cx, cy = _principal_point(settings)
    return rasterize_reference(
        means3d,
        means2d,
        colors,
        semantic_features,
        opacities,
        scales,
        rotations,
        viewmatrix=settings.viewmatrix,
        image_height=settings.image_height,
        image_width=settings.image_width,
        tanfovx=settings.tanfovx,
        tanfovy=settings.tanfovy,
        background=settings.bg,
        scale_modifier=settings.scale_modifier,
        antialias_sigma=settings.antialias_sigma,
        chunk_size=settings.chunk_size,
        depth_order=depth_order,
        cx=cx,
        cy=cy,
    )


class _CudaRasterize(torch.autograd.Function):
    """Native CUDA forward/backward for all continuous Gaussian attributes.

    Visibility sorting, finite footprint support, alpha thresholding, normal
    axis selection and dominant IDs are discrete by design, as in ordinary
    3DGS.  The extension saves only projected Gaussians and compact tile
    overlaps; backward does not construct a pixel-by-Gaussian PyTorch graph.
    """

    @staticmethod
    def forward(ctx, means3d, means2d, colors, semantics, opacities, scales, rotations, settings):
        if not has_cuda_extension():
            raise RuntimeError("CUDA rasterizer extension is not installed")
        inputs = tuple(tensor.contiguous() for tensor in (
            means3d,
            means2d,
            colors,
            semantics,
            opacities,
            scales,
            rotations,
        ))
        viewmatrix = settings.viewmatrix.to(means3d).contiguous()
        background = settings.bg.to(means3d).contiguous()
        cx, cy = _principal_point(settings)
        outputs = _C.rasterize_forward(
            *inputs,
            viewmatrix,
            background,
            int(settings.image_height),
            int(settings.image_width),
            float(settings.tanfovx),
            float(settings.tanfovy),
            cx,
            cy,
            float(settings.scale_modifier),
            float(settings.antialias_sigma),
        )
        ctx.settings = settings
        color, semantic, depth, alpha, normal, radii, dominant = outputs[:7]
        projected_means, conics, depths, projected_normals, sorted_ids, tile_ranges = outputs[7:]
        ctx.save_for_backward(
            *inputs,
            viewmatrix,
            background,
            projected_means,
            conics,
            depths,
            projected_normals,
            radii,
            sorted_ids,
            tile_ranges,
            semantic,
            depth,
            alpha,
            normal,
        )
        ctx.mark_non_differentiable(radii, dominant)
        return color, semantic, depth, alpha, normal, radii, dominant

    @staticmethod
    def backward(ctx, grad_color, grad_semantic, grad_depth, grad_alpha, grad_normal, _grad_radii, _grad_dominant):
        (
            means3d,
            means2d,
            colors,
            semantics,
            opacities,
            scales,
            rotations,
            viewmatrix,
            background,
            projected_means,
            conics,
            depths,
            projected_normals,
            radii,
            sorted_ids,
            tile_ranges,
            output_semantic,
            output_depth,
            output_alpha,
            output_normal,
        ) = ctx.saved_tensors

        def contiguous_or_zeros(gradient, reference):
            return torch.zeros_like(reference) if gradient is None else gradient.contiguous()

        gradients = _C.rasterize_backward(
            means3d,
            means2d,
            colors,
            semantics,
            opacities,
            scales,
            rotations,
            viewmatrix,
            background,
            projected_means,
            conics,
            depths,
            projected_normals,
            radii,
            sorted_ids,
            tile_ranges,
            output_semantic,
            output_depth,
            output_alpha,
            output_normal,
            contiguous_or_zeros(grad_color, torch.empty_like(output_normal)),
            contiguous_or_zeros(grad_semantic, output_semantic),
            contiguous_or_zeros(grad_depth, output_depth),
            contiguous_or_zeros(grad_alpha, output_alpha),
            contiguous_or_zeros(grad_normal, output_normal),
            int(ctx.settings.image_height),
            int(ctx.settings.image_width),
            float(ctx.settings.tanfovx),
            float(ctx.settings.tanfovy),
            *_principal_point(ctx.settings),
            float(ctx.settings.scale_modifier),
            float(ctx.settings.antialias_sigma),
        )
        return (*(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad[:7])
        ), None)


def rasterize_gaussians(
    means3d: Tensor,
    means2d: Tensor,
    colors: Tensor,
    semantic_features: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
    raster_settings: GaussianRasterizationSettings,
) -> RasterizationResult:
    """Dispatch to CUDA when possible and otherwise use the reference backend."""

    backend = raster_settings.backend.lower()
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be one of: auto, cuda, reference")
    use_cuda = backend == "cuda" or (
        backend == "auto" and means3d.is_cuda and means3d.dtype == torch.float32 and has_cuda_extension()
    )
    if backend == "cuda" and not means3d.is_cuda:
        raise RuntimeError("the CUDA backend requires CUDA tensors")
    if backend == "cuda" and means3d.dtype != torch.float32:
        raise RuntimeError("the CUDA backend currently requires float32 tensors")
    if backend == "cuda" and not has_cuda_extension():
        raise RuntimeError("the CUDA extension is unavailable; use backend='auto' or 'reference'")
    if use_cuda:
        return RasterizationResult(*_CudaRasterize.apply(
            means3d,
            means2d,
            colors,
            semantic_features,
            opacities,
            scales,
            rotations,
            raster_settings,
        ))
    return RasterizationResult(*_reference_from_settings(
        means3d,
        means2d,
        colors,
        semantic_features,
        opacities,
        scales,
        rotations,
        raster_settings,
    ))


class GaussianRasterizer(nn.Module):
    """Graphdeco-style module exposing the semantic multi-output contract."""

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    @torch.no_grad()
    def markVisible(self, positions: Tensor) -> Tensor:  # noqa: N802 - compatibility
        zeros = positions.new_zeros((positions.shape[0], 3))
        unit_scales = positions.new_ones((positions.shape[0], 3)) * 1e-3
        rotations = positions.new_zeros((positions.shape[0], 4))
        rotations[:, 0] = 1
        projected = project_gaussians(
            positions,
            zeros,
            unit_scales,
            rotations,
            self.raster_settings.viewmatrix,
            self.raster_settings.image_height,
            self.raster_settings.image_width,
            self.raster_settings.tanfovx,
            self.raster_settings.tanfovy,
            cx=_principal_point(self.raster_settings)[0],
            cy=_principal_point(self.raster_settings)[1],
        )
        return projected.visible

    def forward(
        self,
        means3D: Tensor,
        means2D: Optional[Tensor],
        opacities: Tensor,
        semantic_features: Tensor,
        shs: Optional[Tensor] = None,
        colors_precomp: Optional[Tensor] = None,
        scales: Optional[Tensor] = None,
        rotations: Optional[Tensor] = None,
        cov3D_precomp: Optional[Tensor] = None,
    ) -> RasterizationResult:
        if (shs is None) == (colors_precomp is None):
            raise ValueError("provide exactly one of shs or colors_precomp")
        if cov3D_precomp is not None:
            raise NotImplementedError("use scales and rotations; this preserves normal and policy gradients")
        if scales is None or rotations is None:
            raise ValueError("scales and rotations are required")
        n = means3D.shape[0]
        if semantic_features.shape != (n, SEMANTIC_DIM):
            raise ValueError(f"semantic_features must have shape [N,{SEMANTIC_DIM}]")
        if means2D is None:
            means2D = torch.zeros_like(means3D, requires_grad=means3D.requires_grad)
        if colors_precomp is None:
            if shs.ndim != 3:
                raise ValueError("shs must have shape [N,K,3] or [N,3,K]")
            sh = shs.transpose(1, 2) if shs.shape[-1] == 3 else shs
            directions = F.normalize(means3D - self.raster_settings.campos.to(means3D), dim=-1, eps=1e-12)
            colors_precomp = torch.clamp_min(eval_sh(self.raster_settings.sh_degree, sh, directions) + 0.5, 0.0)
        return rasterize_gaussians(
            means3D,
            means2D,
            colors_precomp,
            semantic_features,
            opacities,
            scales,
            rotations,
            self.raster_settings,
        )

    def prepare_point_integration(
        self,
        means3D: Tensor,
        opacities: Tensor,
        scales: Tensor,
        rotations: Tensor,
        *,
        query_chunk_size: int = 65_536,
    ) -> PointIntegrationContext:
        """Prepare one camera's projected state for repeated 3D point queries."""

        return prepare_point_integration(
            means3D,
            opacities,
            scales,
            rotations,
            self.raster_settings,
            query_chunk_size=query_chunk_size,
        )

    def integrate(
        self,
        points3D: Tensor,
        means3D: Tensor,
        opacities: Tensor,
        scales: Tensor,
        rotations: Tensor,
        *,
        chunk_size: int = 65_536,
        visibility_threshold: float = 1e-3,
    ) -> PointIntegrationResult:
        """One-shot renderer-consistent integration compatible with GW usage."""

        context = self.prepare_point_integration(
            means3D,
            opacities,
            scales,
            rotations,
            query_chunk_size=chunk_size,
        )
        return context.query(
            points3D,
            visibility_threshold=visibility_threshold,
        )


__all__ = [
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    "PointIntegrationContext",
    "PointIntegrationResult",
    "RasterizationResult",
    "SEMANTIC_DIM",
    "eval_sh",
    "has_cuda_extension",
    "has_point_integration_extension",
    "prepare_point_integration",
    "rasterize_gaussians",
    "rasterize_reference",
]
