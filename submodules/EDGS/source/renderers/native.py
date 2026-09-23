"""Adapter for EDGS's original Gaussian Splatting renderer."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

import torch

from . import visible_mask


class NativeRenderer:
    """Expose the native renderer through the shared backend interface."""

    backend = "native"

    def __init__(
        self,
        *,
        clamp_rgb: bool = True,
        render_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.clamp_rgb = bool(clamp_rgb)
        self._render_fn = render_fn

    def _load_render(self) -> Callable[..., dict[str, Any]]:
        if self._render_fn is None:
            # Keeping both operations here avoids loading the CUDA extension
            # during test discovery or config parsing and also makes this
            # adapter safe to use independently of Warper3DGS.
            from source.vendor import bootstrap_gaussian_splatting

            bootstrap_gaussian_splatting()
            self._render_fn = import_module("gaussian_renderer").render
        return self._render_fn

    def render(
        self,
        viewpoint_camera: Any,
        gaussians: Any,
        pipe: Any,
        background: torch.Tensor,
        scaling_modifier: float = 1.0,
        override_color: torch.Tensor | None = None,
        *,
        return_plane: bool = False,
        return_depth_normal: bool = False,
        **native_options: Any,
    ) -> dict[str, Any]:
        """Render RGB and native inverse depth.

        Plane-related flags are accepted to keep call sites backend-agnostic,
        but native rendering cannot produce PGSR geometry.  Asking for either
        output is therefore an explicit error instead of returning mislabeled
        inverse depth.
        """

        if return_plane or return_depth_normal:
            raise ValueError(
                "the native renderer cannot return PGSR plane depth or depth normals"
            )
        package = dict(
            self._load_render()(
                viewpoint_camera,
                gaussians,
                pipe,
                background,
                scaling_modifier=scaling_modifier,
                override_color=override_color,
                **native_options,
            )
        )
        if self.clamp_rgb:
            package["render"] = package["render"].clamp(0.0, 1.0)
        package["visible_mask"] = visible_mask(package)
        # Upstream calls this value "depth", but it is inverse depth.  Keep the
        # legacy key for compatibility while exposing an unambiguous name.
        if "depth" in package:
            package["inverse_depth"] = package["depth"]
        return package

    __call__ = render


__all__ = ["NativeRenderer"]
