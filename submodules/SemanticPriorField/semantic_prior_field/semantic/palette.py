"""Deterministic colors shared by semantic image and mesh exporters."""

from __future__ import annotations

import colorsys

import numpy as np


def semantic_palette(num_classes: int) -> np.ndarray:
    """Return a stable RGB palette with black reserved for background."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    generator = np.random.default_rng(0)
    colors = generator.integers(
        0,
        256,
        size=(num_classes, 3),
        dtype=np.uint8,
    )
    colors[0] = 0
    return colors


def gaga_palette(num_classes: int) -> np.ndarray:
    """Return Gaga's instance-color palette with black background."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    golden_ratio = 1.6180339887
    for object_id in range(1, num_classes):
        hue = (object_id * golden_ratio) % 1.0
        saturation = 0.5 + (object_id % 2) * 0.5
        red, green, blue = colorsys.hls_to_rgb(hue, 0.5, saturation)
        colors[object_id] = (
            int(red * 255),
            int(green * 255),
            int(blue * 255),
        )
    return colors
