"""Image metrics and deterministic tensor serialization helpers."""

from __future__ import annotations

import torch


def composite_background(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    background: torch.Tensor,
) -> torch.Tensor:
    """Composite a CHW foreground with the exact background used for rendering."""

    if mask is None:
        return image
    if image.ndim != 3 or background.numel() != image.shape[0]:
        raise ValueError("image/background must have shapes [C,H,W] and [C]")
    mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (1, image.shape[1], image.shape[2]):
        raise ValueError("mask must have shape [H,W] or [1,H,W]")
    value = background.to(device=image.device, dtype=image.dtype).reshape(-1, 1, 1)
    return image * mask + value * (1.0 - mask)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().flatten(1).mean(1)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.ndim == 3:
        prediction, target = prediction[None], target[None]
    return 10.0 * torch.log10(1.0 / mse(prediction.clamp(0, 1), target.clamp(0, 1)).clamp_min(1e-12))
