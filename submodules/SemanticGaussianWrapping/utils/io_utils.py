"""Small, deterministic output helpers used by rendering and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_rgb(path: str | Path, image: torch.Tensor) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output)


def save_gray16(path: str | Path, image: torch.Tensor, maximum: float | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = image.detach().float().squeeze().cpu()
    valid = torch.isfinite(value)
    scale = float(value[valid].max()) if maximum is None and valid.any() else float(maximum or 1.0)
    normalized = torch.where(valid, value / max(scale, 1e-8), torch.zeros_like(value)).clamp(0, 1)
    Image.fromarray(normalized.mul(65535).round().to(torch.uint16).numpy()).save(output)


def label_colors(labels: torch.Tensor, ignore_label: int = -1) -> torch.Tensor:
    """Hash arbitrary integer IDs to stable RGB colors without a palette file."""

    labels = labels.detach().long().cpu()
    unsigned = labels.clamp_min(0)
    red = (unsigned * 37 + 17) % 255
    green = (unsigned * 67 + 71) % 255
    blue = (unsigned * 97 + 131) % 255
    color = torch.stack((red, green, blue), dim=-1).byte()
    color[labels == ignore_label] = 0
    return color


def save_labels(path: str | Path, labels: torch.Tensor, ignore_label: int = -1) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_colors(labels, ignore_label).numpy()).save(output)


def save_array(path: str | Path, value: torch.Tensor) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, value.detach().cpu().numpy(), allow_pickle=False)
