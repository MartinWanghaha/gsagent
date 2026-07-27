"""Semantic segmentation metrics with optional Hungarian label matching."""

from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> torch.Tensor:
    prediction = prediction.reshape(-1).long().cpu()
    target = target.reshape(-1).long().cpu()
    valid = target != ignore_index
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.double()
    intersection = matrix.diag()
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
    valid = union > 0
    iou = torch.zeros_like(union)
    iou[valid] = intersection[valid] / union[valid]
    accuracy = intersection.sum() / matrix.sum().clamp_min(1)
    class_accuracy = intersection / matrix.sum(dim=1).clamp_min(1)
    return {
        "miou": float(iou[valid].mean()) if valid.any() else 0.0,
        "pixel_accuracy": float(accuracy),
        "mean_class_accuracy": float(class_accuracy[valid].mean()) if valid.any() else 0.0,
    }


def hungarian_permutation(matrix: torch.Tensor) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("scipy is required for Hungarian semantic evaluation") from error
    rows, columns = linear_sum_assignment(-matrix.numpy())
    permutation = np.arange(matrix.shape[1])
    permutation[columns] = rows
    return permutation
