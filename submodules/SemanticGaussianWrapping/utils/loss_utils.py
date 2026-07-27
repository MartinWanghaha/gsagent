"""Image and semantic losses with explicit masks and uncertainty handling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    error = (prediction - target).abs()
    if mask is None:
        return error.mean()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(0)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Differentiable SSIM without a global device-specific window cache."""

    if prediction.ndim == 3:
        prediction, target = prediction[None], target[None]
    channels = prediction.shape[1]
    coords = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] @ kernel[None, :]).expand(channels, 1, -1, -1)
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy
    value = ((2 * mu_xy + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x2 + mu_y2 + 0.01**2) * (sigma_x2 + sigma_y2 + 0.03**2)
    )
    return value.mean() if size_average else value.flatten(1).mean(1)


def soft_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction = prediction.sigmoid().flatten()
    target = target.float().flatten()
    if weight is None:
        weight = torch.ones_like(target)
    else:
        weight = weight.to(device=prediction.device, dtype=prediction.dtype).flatten()
        if weight.shape != target.shape:
            raise ValueError("Dice weight must match prediction/target shape")
    intersection = (weight * prediction * target).sum()
    mass = (weight * prediction).sum() + (weight * target).sum()
    return 1.0 - (2 * intersection + eps) / (mass + eps)


def symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_log = F.log_softmax(first_logits, dim=-1)
    second_log = F.log_softmax(second_logits, dim=-1)
    first, second = first_log.exp(), second_log.exp()
    return 0.5 * (
        F.kl_div(first_log, second, reduction="none").sum(-1)
        + F.kl_div(second_log, first, reduction="none").sum(-1)
    )
