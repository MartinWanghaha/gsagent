"""Dependency-light rendering and semantic metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from utils.image_utils import psnr
from utils.loss_utils import ssim


def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_label: int = -1,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[C, C+1]`` counts; the last column is an unknown prediction.

    A valid ground-truth pixel with an invalid prediction is a false negative,
    not an ignored sample.  The explicit unknown column records that row mass
    without inventing a false positive for an arbitrary known class.
    """

    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_label) & (target >= 0) & (target < num_classes)
    known_prediction = (prediction >= 0) & (prediction < num_classes)
    prediction_column = torch.where(
        known_prediction,
        prediction,
        torch.full_like(prediction, num_classes),
    )
    columns = num_classes + 1
    indices = target[valid] * columns + prediction_column[valid]
    if weight is None:
        values = torch.ones(indices.shape[0], dtype=torch.float64, device=indices.device)
    else:
        values = weight.reshape(-1)[valid].to(dtype=torch.float64, device=indices.device)
    result = torch.zeros(num_classes * columns, dtype=torch.float64, device=indices.device)
    result.scatter_add_(0, indices, values)
    return result.reshape(num_classes, columns)


def mean_iou(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[1] not in {matrix.shape[0], matrix.shape[0] + 1}:
        raise ValueError("confusion matrix must have shape [C,C] or [C,C+1]")
    classes = matrix.shape[0]
    known = matrix[:, :classes]
    intersection = known.diag()
    # All columns contribute to GT row mass.  The optional final column is an
    # invalid/unknown prediction and therefore contributes only false negatives.
    union = known.sum(0) + matrix.sum(1) - intersection
    valid = union > 0
    per_class = torch.where(valid, intersection / union.clamp_min(1e-12), torch.nan)
    mean = per_class[valid].mean() if valid.any() else matrix.new_tensor(float("nan"))
    return mean, per_class


@dataclass
class ImageMetricAccumulator:
    """Accumulate per-view metrics without retaining images."""

    psnr_values: list[float] = field(default_factory=list)
    ssim_values: list[float] = field(default_factory=list)
    l1_values: list[float] = field(default_factory=list)
    semantic_confusion: torch.Tensor | None = None

    @torch.no_grad()
    def update_image(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.float().clamp(0, 1)
        target = target.to(prediction).clamp(0, 1)
        self.psnr_values.append(float(psnr(prediction, target).mean()))
        self.ssim_values.append(float(ssim(prediction, target)))
        self.l1_values.append(float((prediction - target).abs().mean()))

    @torch.no_grad()
    def update_semantic(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_label: int = -1,
        confidence: torch.Tensor | None = None,
    ) -> None:
        matrix = confusion_matrix(prediction, target, num_classes, ignore_label, confidence).cpu()
        if self.semantic_confusion is None:
            self.semantic_confusion = matrix
        else:
            self.semantic_confusion += matrix

    def compute(self) -> dict[str, object]:
        count = len(self.psnr_values)
        result: dict[str, object] = {
            "views": count,
            "psnr": sum(self.psnr_values) / max(count, 1),
            "ssim": sum(self.ssim_values) / max(count, 1),
            "l1": sum(self.l1_values) / max(count, 1),
        }
        if self.semantic_confusion is not None:
            miou, per_class = mean_iou(self.semantic_confusion)
            result["semantic_miou"] = None if miou.isnan() else float(miou)
            result["semantic_iou"] = [None if value.isnan() else float(value) for value in per_class]
        return result
