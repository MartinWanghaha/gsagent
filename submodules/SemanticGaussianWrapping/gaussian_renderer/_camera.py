"""Shared camera calibration and model attribute adapters."""

from __future__ import annotations

import math
from typing import Any, Iterable

from torch import Tensor

from diff_semantic_gaussian_rasterization import GaussianRasterizationSettings


def resolve_value(owner: Any, names: Iterable[str]) -> Any:
    for name in names:
        if hasattr(owner, name):
            value = getattr(owner, name)
            return value() if callable(value) else value
    raise AttributeError(
        f"{type(owner).__name__} does not expose any of {tuple(names)}"
    )


def camera_raster_settings(
    viewpoint_camera: Any,
    means3d: Tensor,
    pipe: Any,
    background: Tensor,
    scaling_modifier: float,
    backend: str,
) -> GaussianRasterizationSettings:
    image_height = int(
        resolve_value(viewpoint_camera, ("image_height", "height"))
    )
    image_width = int(
        resolve_value(viewpoint_camera, ("image_width", "width"))
    )
    # Calibration is stored in pixel-index coordinates; the rasterizer samples
    # at x/y + 0.5, so this conversion is shared by rendering and point queries.
    principal_x = (
        float(getattr(viewpoint_camera, "Cx", (image_width - 1) * 0.5)) + 0.5
    )
    principal_y = (
        float(getattr(viewpoint_camera, "Cy", (image_height - 1) * 0.5)) + 0.5
    )
    return GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=math.tan(
            float(resolve_value(viewpoint_camera, ("FoVx", "fov_x"))) * 0.5
        ),
        tanfovy=math.tan(
            float(resolve_value(viewpoint_camera, ("FoVy", "fov_y"))) * 0.5
        ),
        bg=background.to(means3d),
        scale_modifier=float(scaling_modifier),
        viewmatrix=resolve_value(
            viewpoint_camera, ("world_view_transform", "viewmatrix")
        ).to(means3d),
        projmatrix=resolve_value(
            viewpoint_camera, ("full_proj_transform", "projmatrix")
        ).to(means3d),
        sh_degree=0,
        campos=resolve_value(
            viewpoint_camera, ("camera_center", "campos")
        ).to(means3d),
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),
        backend=backend,
        chunk_size=int(getattr(pipe, "reference_chunk_size", 32)),
        antialias_sigma=float(getattr(pipe, "antialias_sigma", 0.3)),
        cx=principal_x,
        cy=principal_y,
    )


__all__ = ["camera_raster_settings", "resolve_value"]
