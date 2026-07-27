"""Camera/model adapter for renderer-consistent Gaussian opacity queries."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from diff_semantic_gaussian_rasterization import (
    GaussianRasterizer,
    PointIntegrationContext,
    PointIntegrationResult,
)

from ._camera import camera_raster_settings, resolve_value


def prepare_point_integration(
    viewpoint_camera: Any,
    gaussians: Any,
    pipe: Any,
    *,
    scaling_modifier: float = 1.0,
    query_chunk_size: int = 65_536,
) -> PointIntegrationContext:
    """Prepare one camera once, then reuse it across all mesh-field queries."""

    means3d = resolve_value(gaussians, ("get_xyz", "xyz", "_xyz"))
    scales = resolve_value(gaussians, ("get_scaling", "scaling", "_scaling"))
    rotations = resolve_value(
        gaussians, ("get_rotation", "rotation", "_rotation")
    )
    opacities = resolve_value(
        gaussians, ("get_opacity", "opacity", "_opacity")
    )
    settings = camera_raster_settings(
        viewpoint_camera,
        means3d,
        pipe,
        means3d.new_zeros(3),
        scaling_modifier,
        "cuda",
    )
    return GaussianRasterizer(settings).prepare_point_integration(
        means3d,
        opacities,
        scales,
        rotations,
        query_chunk_size=query_chunk_size,
    )


def integrate_points(
    points3d: Tensor,
    viewpoint_camera: Any,
    gaussians: Any,
    pipe: Any,
    *,
    scaling_modifier: float = 1.0,
    chunk_size: int = 65_536,
    visibility_threshold: float = 1e-3,
) -> PointIntegrationResult:
    """One-shot convenience wrapper; shared callers should retain the context."""

    context = prepare_point_integration(
        viewpoint_camera,
        gaussians,
        pipe,
        scaling_modifier=scaling_modifier,
        query_chunk_size=chunk_size,
    )
    return context.query(
        points3d,
        visibility_threshold=visibility_threshold,
    )


__all__ = ["integrate_points", "prepare_point_integration"]
