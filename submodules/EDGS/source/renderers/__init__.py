"""Renderer selection and the common EDGS render-package contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def visible_mask(render_package: Mapping[str, Any]) -> torch.Tensor:
    """Return a flat boolean visibility mask for either renderer backend.

    Upstream EDGS exposes non-zero indices while PGSR exposes a boolean mask.
    Normalizing that difference at the renderer boundary keeps densification and
    scale regularization independent of the selected rasterizer.
    """

    existing = render_package.get("visible_mask")
    if isinstance(existing, torch.Tensor):
        if existing.dtype != torch.bool:
            raise TypeError("render_package['visible_mask'] must be boolean")
        return existing.reshape(-1)

    visibility = render_package.get("visibility_filter")
    if not isinstance(visibility, torch.Tensor):
        raise KeyError("render package has no tensor visibility_filter")
    if visibility.dtype == torch.bool:
        return visibility.reshape(-1)

    radii = render_package.get("radii")
    if not isinstance(radii, torch.Tensor):
        raise KeyError("index visibility_filter requires a tensor radii entry")
    mask = torch.zeros(radii.numel(), device=radii.device, dtype=torch.bool)
    indices = visibility.reshape(-1).to(device=radii.device, dtype=torch.long)
    if indices.numel() > 0:
        if indices.min() < 0 or indices.max() >= mask.numel():
            raise IndexError(
                "visibility_filter contains an out-of-range Gaussian index"
            )
        mask[indices] = True
    return mask


def _option(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, str):
        return config if name == "backend" else default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_renderer(config: Any = None, *, clamp_rgb: bool | None = None):
    """Create a renderer from a backend name or Hydra renderer config.

    The returned object always exposes ``backend`` and ``render(...)`` and is
    also callable.  Backend modules are imported lazily so importing EDGS does
    not require either CUDA rasterization extension to be installed.
    """

    backend = str(_option(config, "backend", "native")).strip().lower()
    if clamp_rgb is None:
        clamp_rgb = bool(_option(config, "clamp_rgb", True))
    if backend == "native":
        from .native import NativeRenderer

        return NativeRenderer(clamp_rgb=clamp_rgb)
    if backend == "pgsr":
        from .pgsr import PGSRRenderer

        return PGSRRenderer(clamp_rgb=clamp_rgb)
    raise ValueError(
        f"unknown renderer backend {backend!r}; expected one of: native, pgsr"
    )


__all__ = ["build_renderer", "visible_mask"]
