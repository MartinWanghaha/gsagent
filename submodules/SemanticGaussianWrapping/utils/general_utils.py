"""Numerical utilities for Gaussian parameterization and optimization."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
) -> Callable[[int], float]:
    """Return the log-linearly interpolated learning-rate schedule from 3DGS."""

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            ratio = np.clip(step / lr_delay_steps, 0.0, 1.0)
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(0.5 * math.pi * ratio)
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0.0, 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)
        return float(delay_rate * log_lerp)

    return helper


def build_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions ``[w,x,y,z]`` to rotation matrices."""

    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    result = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - w * z)
    result[..., 0, 2] = 2 * (x * z + w * y)
    result[..., 1, 0] = 2 * (x * y + w * z)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - w * x)
    result[..., 2, 0] = 2 * (x * z - w * y)
    result[..., 2, 1] = 2 * (y * z + w * x)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def build_scaling_rotation(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    matrix = build_rotation(rotation)
    return matrix @ torch.diag_embed(scaling)


def strip_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    result = torch.empty((*matrix.shape[:-2], 6), dtype=matrix.dtype, device=matrix.device)
    result[..., 0] = matrix[..., 0, 0]
    result[..., 1] = matrix[..., 0, 1]
    result[..., 2] = matrix[..., 0, 2]
    result[..., 3] = matrix[..., 1, 1]
    result[..., 4] = matrix[..., 1, 2]
    result[..., 5] = matrix[..., 2, 2]
    return result


def covariance_from_scaling_rotation(
    scaling: torch.Tensor, rotation: torch.Tensor, modifier: float = 1.0
) -> torch.Tensor:
    transform = build_scaling_rotation(modifier * scaling, rotation)
    return transform @ transform.transpose(-1, -2)


def safe_state(seed: int = 0, quiet: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not quiet:
        print(f"Random seed: {seed}")

