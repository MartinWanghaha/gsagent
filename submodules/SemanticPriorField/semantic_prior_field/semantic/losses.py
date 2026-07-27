"""Semantic objectives used by lift and joint Gaussian training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def semantic_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    ignore_index: int = -1,
    normalize_by_classes: bool = True,
) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    per_pixel = F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = labels != ignore_index
    weights = valid.float()
    if confidence is not None:
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(0)
        weights = weights * confidence.to(weights)
    denominator = weights.sum().clamp_min(1.0)
    loss = (per_pixel * weights).sum() / denominator
    if normalize_by_classes:
        loss = loss / max(math.log(logits.shape[1]), 1.0)
    return loss


def spatial_consistency_loss(
    semantic_features: torch.Tensor,
    positions: torch.Tensor,
    *,
    sample_size: int = 10_000,
    neighbors: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Scalable sampled neighbour KL consistency.

    The full Gaussian set is never expanded to an N×N matrix. Only sampled
    anchors are compared against a chunked candidate set.
    """

    count = semantic_features.shape[0]
    if count < 2 or sample_size <= 1:
        return semantic_features.sum() * 0.0
    sample_size = min(int(sample_size), count)
    indices = torch.randperm(count, device=positions.device)[:sample_size]
    sample_positions = positions[indices]
    sample_features = semantic_features[indices]
    chunk_size = min(2048, sample_size)
    losses = []
    probabilities = F.log_softmax(sample_features / temperature, dim=-1)
    target_probabilities = probabilities.detach().exp()
    k = min(int(neighbors) + 1, sample_size)
    for start in range(0, sample_size, chunk_size):
        end = min(start + chunk_size, sample_size)
        distances = torch.cdist(sample_positions[start:end], sample_positions)
        neighbour_indices = distances.topk(k=k, largest=False).indices[:, 1:]
        neighbour_targets = target_probabilities[neighbour_indices].mean(dim=1)
        losses.append(
            F.kl_div(
                probabilities[start:end],
                neighbour_targets,
                reduction="batchmean",
            )
        )
    return torch.stack(losses).mean()
