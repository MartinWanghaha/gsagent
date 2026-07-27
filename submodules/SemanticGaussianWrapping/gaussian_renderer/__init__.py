"""Public Semantic Gaussian Wrapping renderer.

The signature and returned dictionary follow standard 3D Gaussian Splatting,
with aligned semantic, depth, alpha, normal and dominant-index outputs added.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import torch
from torch import Tensor


try:
    from diff_semantic_gaussian_rasterization import (
        GaussianRasterizer,
        SEMANTIC_DIM,
    )
except ModuleNotFoundError:
    # Keep a source checkout executable before the optional CUDA package has
    # been installed. This path is project-relative and never shadows another
    # module because the package name is unique.
    _extension_source = Path(__file__).resolve().parents[1] / "submodules" / "diff-semantic-gaussian-rasterization"
    sys.path.insert(0, str(_extension_source))
    from diff_semantic_gaussian_rasterization import (  # type: ignore[no-redef]
        GaussianRasterizer,
        SEMANTIC_DIM,
    )

from ._camera import (
    camera_raster_settings as _camera_raster_settings,
    resolve_value as _resolve_value,
)
from .point_integration import integrate_points, prepare_point_integration


def _semantic_embedding(gaussians: Any) -> Tensor:
    try:
        embedding = _resolve_value(
            gaussians,
            (
                "get_semantic",
                "get_semantic_embedding",
                "get_semantic_features",
                "semantic_embedding",
                "_semantic_embedding",
            ),
        )
    except AttributeError:
        registry = getattr(gaussians, "attribute_registry", getattr(gaussians, "registry", None))
        if registry is None:
            raise
        if hasattr(registry, "get"):
            embedding = registry.get("semantic_embedding")
        else:
            embedding = registry["semantic_embedding"]
    if embedding.ndim == 3 and embedding.shape[1] == 1:
        embedding = embedding[:, 0]
    if embedding.ndim != 2 or embedding.shape[1] != SEMANTIC_DIM:
        raise ValueError(f"Gaussian semantic embedding must have shape [N,{SEMANTIC_DIM}]")
    return embedding


def render(
    viewpoint_camera: Any,
    gaussians: Any,
    pipe: Any,
    background: Tensor,
    scaling_modifier: float = 1.0,
    backend: str = "auto",
    override_color: Optional[Tensor] = None,
):
    """Render a camera with shared front-to-back multi-attribute weights.

    Args mirror standard 3DGS. ``backend='auto'`` selects the custom CUDA
    forward when available and transparently falls back to differentiable
    PyTorch on CPU or source-only installations.
    """

    means3d = _resolve_value(gaussians, ("get_xyz", "xyz", "_xyz"))
    scales = _resolve_value(gaussians, ("get_scaling", "scaling", "_scaling"))
    rotations = _resolve_value(gaussians, ("get_rotation", "rotation", "_rotation"))
    opacities = _resolve_value(gaussians, ("get_opacity", "opacity", "_opacity"))
    semantics = _semantic_embedding(gaussians)
    if not all(t.device == means3d.device for t in (scales, rotations, opacities, semantics)):
        raise ValueError("all Gaussian attributes must be on the same device")

    # This explicit non-leaf tensor preserves Graphdeco's densification hook:
    # training reads an NDC/viewport proxy gradient from viewspace_points.grad.
    viewspace_points = torch.zeros_like(means3d, requires_grad=True) + 0
    try:
        viewspace_points.retain_grad()
    except RuntimeError:
        pass

    settings = _camera_raster_settings(
        viewpoint_camera,
        means3d,
        pipe,
        background,
        scaling_modifier,
        backend,
    )._replace(
        sh_degree=int(_resolve_value(gaussians, ("active_sh_degree",)))
    )
    image_height = settings.image_height
    image_width = settings.image_width
    # The rasterizer exposes differentiable offsets in pixel units, whereas the
    # standard 3DGS densification threshold is calibrated for normalized
    # viewport coordinates.  At the zero-valued proxy this conversion leaves
    # every forward output unchanged and applies the required W/2, H/2 factors
    # only to the gradient returned through ``viewspace_points``.
    viewport_to_pixel = viewspace_points.new_tensor(
        (0.5 * image_width, 0.5 * image_height, 0.0)
    )
    pixel_offsets = viewspace_points * viewport_to_pixel
    rasterizer = GaussianRasterizer(settings)
    shs = None if override_color is not None else _resolve_value(gaussians, ("get_features", "features", "_features"))
    result = rasterizer(
        means3D=means3d,
        means2D=pixel_offsets,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacities,
        semantic_features=semantics,
        scales=scales,
        rotations=rotations,
    )
    visibility = result.radii > 0
    return {
        "render": result.color,
        "semantic": result.semantic,
        "expected_depth": result.expected_depth,
        "alpha": result.alpha,
        "normal": result.normal,
        "dominant_index": result.dominant_index,
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility,
        "radii": result.radii,
    }

__all__ = ["integrate_points", "prepare_point_integration", "render"]
