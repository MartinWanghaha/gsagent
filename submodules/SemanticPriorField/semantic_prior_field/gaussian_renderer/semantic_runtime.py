"""Resolve native semantic rasterizers from an install or in-place build."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = {
    "radegs": (
        "diff_gaussian_rasterization_gw_semantic",
        PROJECT_ROOT / "submodules" / "diff-gaussian-rasterization-semantic",
    ),
    "ours": (
        "diff_gaussian_rasterization_gw_ours_semantic",
        PROJECT_ROOT / "submodules" / "diff-gaussian-rasterization_ours-semantic",
    ),
    # SPF rasterizer: the Ours semantic compositor extended with per-Gaussian
    # statistics buffers in the semantic backward (conflict + contribution),
    # exposed through the raster settings' stats_sink.
    "spf": (
        "diff_gaussian_rasterization_spf",
        PROJECT_ROOT / "submodules" / "diff-gaussian-rasterization-spf",
    ),
}

_warned_spf_fallback = False


def load_semantic_rasterizer_with_fallback(rasterizer: str) -> tuple:
    """Prefer the requested backend; fall back from 'spf' to 'ours'.

    Returns (module, resolved_backend_name).
    """
    global _warned_spf_fallback
    try:
        return load_semantic_rasterizer(rasterizer), rasterizer
    except RuntimeError:
        if rasterizer != "spf":
            raise
        if not _warned_spf_fallback:
            print("[WARNING] SPF rasterizer unavailable; falling back to the "
                  "ours semantic rasterizer (no stats channel).")
            _warned_spf_fallback = True
        return load_semantic_rasterizer("ours"), "ours"


def load_semantic_rasterizer(rasterizer: str) -> ModuleType:
    """Load an installed extension, falling back to its in-place build."""
    if rasterizer not in EXTENSIONS:
        raise ValueError(f"Unknown semantic rasterizer: {rasterizer}")
    package, extension_root = EXTENSIONS[rasterizer]
    try:
        return importlib.import_module(package)
    except (ImportError, ModuleNotFoundError):
        pass

    extension_root_string = str(extension_root)
    if extension_root_string not in sys.path:
        sys.path.insert(0, extension_root_string)
    sys.modules.pop(package, None)
    try:
        return importlib.import_module(package)
    except (ImportError, ModuleNotFoundError) as local_error:
        command = (
            f"{sys.executable} -m pip install --no-build-isolation "
            f"{extension_root}"
        )
        raise RuntimeError(
            f"Native {rasterizer} semantic rasterizer is unavailable. "
            f"Build/install it with:\n  {command}"
        ) from local_error
