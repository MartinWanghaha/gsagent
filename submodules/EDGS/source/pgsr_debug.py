"""Lightweight PGSR-style training debug montage writer.

The visualizer intentionally depends only on tensors already produced by a
renderer or loss computation.  It does not import either EDGS or PGSR renderer
packages, so callers can use it without introducing another CUDA dependency or
module-name collision.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REQUIRED_RENDER_KEYS = (
    "render",
    "rendered_normal",
    "rendered_distance",
    "plane_depth",
    "depth_normal",
)


def _option(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _opencv():
    """Import OpenCV only when an enabled visualizer writes a frame."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "PGSR debug visualization requires OpenCV (the 'cv2' package)"
        ) from error
    return cv2


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _safe_image_stem(camera: Any) -> str:
    image_name = getattr(camera, "image_name", None)
    if image_name is None or not str(image_name).strip():
        raise ValueError("camera.image_name must be a non-empty string")
    basename = Path(str(image_name)).name
    stem = Path(basename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe:
        safe = "camera"
    return safe[:120]


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.detach().float().cpu()


def _rgb_panel(value: Any, name: str) -> np.ndarray:
    tensor = _tensor(value, name)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] < 3:
        raise ValueError(f"{name} must have shape [3,H,W], got {tuple(tensor.shape)}")
    array = tensor[:3].permute(1, 2, 0).numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _normal_panel(value: Any, name: str, height: int, width: int) -> np.ndarray:
    tensor = _tensor(value, name)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"{name} must have shape [3,H,W], got {tuple(tensor.shape)}")
    if tuple(tensor.shape[1:]) != (height, width):
        raise ValueError(
            f"{name} has spatial shape {tuple(tensor.shape[1:])}, "
            f"expected {(height, width)}"
        )
    array = tensor.permute(1, 2, 0).numpy()
    finite = np.isfinite(array).all(axis=-1)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    # Preserve alpha-weighted normal magnitude exactly as upstream PGSR does;
    # normalizing here would make low-confidence/transparent pixels look valid.
    encoded = np.rint(np.clip((array + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
    encoded[~finite] = 0
    # Match upstream PGSR: normal xyz channels are written directly as BGR.
    return np.ascontiguousarray(encoded)


def _scalar_map(value: Any, name: str, height: int, width: int) -> np.ndarray:
    tensor = _tensor(value, name)
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 and tensor.numel() == height * width:
        tensor = tensor.reshape(height, width)
    if tensor.ndim != 2 or tuple(tensor.shape) != (height, width):
        raise ValueError(
            f"{name} must resolve to shape {(height, width)}, got {tuple(tensor.shape)}"
        )
    return tensor.numpy()


def _jet_minmax(
    value: Any,
    name: str,
    height: int,
    width: int,
    *,
    positive_only: bool = False,
) -> np.ndarray:
    array = _scalar_map(value, name, height, width)
    valid = np.isfinite(array)
    if positive_only:
        valid &= array > 0.0
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)
    minimum = float(np.min(array[valid]))
    maximum = float(np.max(array[valid]))
    normalized = np.zeros((height, width), dtype=np.float32)
    if maximum > minimum:
        normalized[valid] = (array[valid] - minimum) / (maximum - minimum)
    encoded = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2 = _opencv()
    color = cv2.applyColorMap(encoded, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


def _jet_unit(value: Any, name: str, height: int, width: int) -> np.ndarray:
    array = _scalar_map(value, name, height, width)
    valid = np.isfinite(array)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)
    encoded = np.zeros((height, width), dtype=np.uint8)
    encoded[valid] = np.rint(np.clip(array[valid], 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2 = _opencv()
    color = cv2.applyColorMap(encoded, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


def _diagnostic_panel(
    diagnostics: Mapping[str, Any], name: str, height: int, width: int
) -> np.ndarray:
    value = diagnostics.get(name)
    if value is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return _jet_unit(value, name, height, width)


class PGSRDebugVisualizer:
    """Write the 2x4 JPEG debug montage used by PGSR training.

    Scheduling is intentionally strict: a frame is captured only when
    ``step > from_iter`` and ``step % interval == 0``.  A disabled instance
    performs no directory creation and :meth:`save` returns ``None``.
    """

    def __init__(
        self,
        config: Any,
        model_path: str | os.PathLike[str],
        default_from_iter: int,
    ) -> None:
        enabled = _option(config, "enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        self.enabled = enabled
        self.interval = _integer(
            _option(config, "interval", 200), "interval", minimum=1
        )

        configured_from_iter = _option(config, "from_iter", None)
        if configured_from_iter is None:
            configured_from_iter = default_from_iter
        self.from_iter = _integer(configured_from_iter, "from_iter", minimum=0)
        self.jpeg_quality = _integer(
            _option(config, "jpeg_quality", 95), "jpeg_quality", minimum=1
        )
        if self.jpeg_quality > 100:
            raise ValueError("jpeg_quality must be at most 100")

        self.model_path = Path(model_path).expanduser().resolve()
        configured_output = Path(
            str(_option(config, "output_dir", "debug"))
        ).expanduser()
        output_dir = (
            configured_output.resolve()
            if configured_output.is_absolute()
            else (self.model_path / configured_output).resolve()
        )
        try:
            relative_output = output_dir.relative_to(self.model_path)
        except ValueError as error:
            raise ValueError("output_dir must stay inside model_path") from error
        if relative_output == Path("."):
            raise ValueError("output_dir must be a child directory of model_path")
        self.output_dir = output_dir

    def _ensure_output_dir(self) -> None:
        """Create the output directory after rechecking symlink containment."""

        resolved = self.output_dir.resolve()
        try:
            resolved.relative_to(self.model_path)
        except ValueError as error:
            raise ValueError("resolved output_dir escapes model_path") from error
        if resolved == self.model_path:
            raise ValueError("output_dir must be a child directory of model_path")
        resolved.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> Path | None:
        """Create the configured directory after trainer validation succeeds."""

        if not self.enabled:
            return None
        _opencv()
        self._ensure_output_dir()
        return self.output_dir

    def should_capture(self, step: int) -> bool:
        """Return whether ``step`` satisfies the configured debug schedule."""

        step = _integer(step, "step", minimum=0)
        return self.enabled and step > self.from_iter and step % self.interval == 0

    def save(
        self,
        step: int,
        camera: Any,
        render_pkg: Mapping[str, Any],
        gt_image: torch.Tensor,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Save one scheduled montage and return its final path.

        Panel order matches upstream PGSR training:

        ``GT | render | rendered normal | rendered distance``
        ``reprojection weight | plane depth | depth normal | image weight``
        """

        if not self.should_capture(step):
            return None
        if not isinstance(render_pkg, Mapping):
            raise TypeError("render_pkg must be a mapping")
        missing = [name for name in _REQUIRED_RENDER_KEYS if name not in render_pkg]
        if missing:
            raise KeyError(
                "render_pkg is missing required tensors: " + ", ".join(missing)
            )
        if diagnostics is None:
            diagnostics = {}
        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping or None")

        gt = _rgb_panel(gt_image, "gt_image")
        height, width = gt.shape[:2]
        rendering = _rgb_panel(render_pkg["render"], "render_pkg['render']")
        if rendering.shape[:2] != (height, width):
            raise ValueError(
                f"render spatial shape {rendering.shape[:2]} does not match GT "
                f"{(height, width)}"
            )
        rendered_normal = _normal_panel(
            render_pkg["rendered_normal"],
            "render_pkg['rendered_normal']",
            height,
            width,
        )
        rendered_distance = _jet_minmax(
            render_pkg["rendered_distance"],
            "render_pkg['rendered_distance']",
            height,
            width,
        )
        reprojection_weight = _diagnostic_panel(
            diagnostics, "reprojection_weight", height, width
        )
        plane_depth = _jet_minmax(
            render_pkg["plane_depth"],
            "render_pkg['plane_depth']",
            height,
            width,
            positive_only=True,
        )
        depth_normal = _normal_panel(
            render_pkg["depth_normal"],
            "render_pkg['depth_normal']",
            height,
            width,
        )
        image_weight = _diagnostic_panel(diagnostics, "image_weight", height, width)

        row_zero = np.concatenate(
            (gt, rendering, rendered_normal, rendered_distance), axis=1
        )
        row_one = np.concatenate(
            (reprojection_weight, plane_depth, depth_normal, image_weight), axis=1
        )
        montage = np.ascontiguousarray(np.concatenate((row_zero, row_one), axis=0))

        self._ensure_output_dir()
        filename = f"{step:05d}_{_safe_image_stem(camera)}.jpg"
        destination = self.output_dir / filename
        try:
            destination.resolve().relative_to(self.model_path)
        except ValueError as error:  # defensive: filename is already sanitized
            raise ValueError("debug output path escapes model_path") from error

        cv2 = _opencv()
        success, encoded = cv2.imencode(
            ".jpg",
            montage,
            (cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality),
        )
        if not success:
            raise OSError(f"OpenCV failed to encode debug montage: {destination}")
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(encoded.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


__all__ = ["PGSRDebugVisualizer"]
