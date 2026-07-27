"""Sparse, lossless-accounting views of semantic region posteriors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


_PROBABILITY_ATOL = 2e-6


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class SparseRegionMembership:
    """Top-k foreground probabilities with explicit omitted probability mass.

    Region zero is background. ``weights`` retain the original softmax
    probabilities for the selected foreground IDs; they are deliberately not
    renormalized. ``tail`` accounts for all omitted foreground classes.
    """

    ids: Tensor
    weights: Tensor
    background: Tensor
    tail: Tensor
    confidence: Tensor

    def __post_init__(self) -> None:
        if self.ids.ndim != 2:
            raise ValueError("region ids must have shape [M,K]")
        if self.ids.dtype != torch.long:
            raise TypeError("region ids must use torch.long")
        if self.weights.shape != self.ids.shape:
            raise ValueError("region weights must have shape [M,K]")
        rows = self.ids.shape[0]
        for name in ("background", "tail", "confidence"):
            value = getattr(self, name)
            if value.shape != (rows, 1):
                raise ValueError(f"region {name} must have shape [M,1]")
        values = (self.weights, self.background, self.tail, self.confidence)
        if any(not value.is_floating_point() for value in values):
            raise TypeError("region probabilities and confidence must be floating point")
        if any(value.dtype != self.weights.dtype for value in values):
            raise TypeError("region probabilities and confidence must share a dtype")
        if any(value.device != self.ids.device for value in values):
            raise ValueError("all region membership tensors must share a device")
        if self.ids.numel() and bool((self.ids < 1).any()):
            raise ValueError("sparse region ids must be foreground IDs greater than zero")
        if self.ids.shape[1] > 1:
            sorted_ids = self.ids.sort(dim=1).values
            if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
                raise ValueError("sparse region ids must be unique within each row")
            if bool((self.weights[:, 1:] > self.weights[:, :-1] + _PROBABILITY_ATOL).any()):
                raise ValueError("sparse region weights must be sorted in descending order")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("region membership contains NaN or infinite values")
        if any(bool(((value < 0) | (value > 1)).any()) for value in values):
            raise ValueError("region probabilities and confidence must lie in [0,1]")
        total = self.background + self.tail + self.weights.sum(dim=1, keepdim=True)
        tolerance = max(_PROBABILITY_ATOL, 4.0 * torch.finfo(total.dtype).eps)
        if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0.0):
            raise ValueError("background, selected weights, and tail must sum to one")

    def __len__(self) -> int:
        return self.ids.shape[0]

    @property
    def foreground_mass(self) -> Tensor:
        """Return total foreground probability, including the omitted tail."""

        return self.weights.sum(dim=1, keepdim=True) + self.tail

    def index_select(self, indices: Tensor | Sequence[int]) -> "SparseRegionMembership":
        if isinstance(indices, Tensor):
            selected = indices.to(device=self.ids.device)
            if selected.dtype != torch.long:
                raise TypeError("region membership indices must use torch.long")
        else:
            selected = torch.as_tensor(indices, device=self.ids.device, dtype=torch.long)
        if selected.ndim != 1:
            raise ValueError("region membership indices must have shape [S]")
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= len(self)
        ):
            raise IndexError("region membership index is out of range")
        return SparseRegionMembership(
            ids=self.ids.index_select(0, selected),
            weights=self.weights.index_select(0, selected),
            background=self.background.index_select(0, selected),
            tail=self.tail.index_select(0, selected),
            confidence=self.confidence.index_select(0, selected),
        )

    def to(self, *args, **kwargs) -> "SparseRegionMembership":
        """Move/cast membership values while preserving integer region IDs."""

        weights = self.weights.to(*args, **kwargs)
        background = self.background.to(*args, **kwargs)
        tail = self.tail.to(*args, **kwargs)
        confidence = self.confidence.to(*args, **kwargs)
        return SparseRegionMembership(
            ids=self.ids.to(device=weights.device, dtype=torch.long),
            weights=weights,
            background=background,
            tail=tail,
            confidence=confidence,
        )

    def probability(self, region_ids: int | Tensor | Sequence[int]) -> Tensor:
        """Return retained probabilities for queried region IDs as ``[M,R]``.

        Background ID zero is exact. An omitted foreground ID has probability
        zero in this sparse view; its aggregate probability remains in ``tail``.
        """

        if isinstance(region_ids, Tensor):
            query = region_ids.to(device=self.ids.device)
            if query.dtype != torch.long:
                raise TypeError("queried region ids must use torch.long")
        else:
            query = torch.as_tensor(region_ids, device=self.ids.device, dtype=torch.long)
        if query.ndim == 0:
            query = query.reshape(1)
        if query.ndim != 1:
            raise ValueError("queried region ids must be a scalar or have shape [R]")
        if query.numel() and bool((query < 0).any()):
            raise ValueError("queried region ids must be non-negative")
        matched = self.ids[:, :, None] == query[None, None, :]
        result = (self.weights[:, :, None] * matched).sum(dim=1)
        return torch.where(query[None, :] == 0, self.background, result)

    @classmethod
    def from_logits(
        cls,
        logits: Tensor,
        *,
        top_k: int,
        confidence: Tensor | None = None,
    ) -> "SparseRegionMembership":
        """Build a foreground top-k view from FP32 softmax probabilities."""

        top_k = _positive_integer(top_k, "top_k")
        if logits.ndim != 2:
            raise ValueError("semantic logits must have shape [M,C]")
        if logits.shape[1] < 1:
            raise ValueError("semantic logits must contain a background class")
        if not logits.is_floating_point():
            raise TypeError("semantic logits must be floating point")
        probabilities = torch.softmax(logits.float(), dim=1)
        rows = logits.shape[0]
        selected_count = min(top_k, logits.shape[1] - 1)
        foreground = probabilities[:, 1:]
        if selected_count:
            weights, local_ids = foreground.topk(
                selected_count, dim=1, largest=True, sorted=True
            )
            ids = local_ids.to(torch.long) + 1
        else:
            ids = torch.empty(rows, 0, dtype=torch.long, device=logits.device)
            weights = torch.empty(rows, 0, dtype=torch.float32, device=logits.device)
        background = probabilities[:, :1]
        tail = (1.0 - background - weights.sum(dim=1, keepdim=True)).clamp_min(0.0)
        if confidence is None:
            confidence_value = torch.ones(rows, 1, dtype=torch.float32, device=logits.device)
        else:
            confidence_value = torch.as_tensor(confidence, device=logits.device)
            if confidence_value.shape != (rows, 1):
                raise ValueError("region confidence must have shape [M,1]")
            if not confidence_value.is_floating_point():
                raise TypeError("region confidence must be floating point")
            confidence_value = confidence_value.float()
        return cls(ids, weights, background, tail, confidence_value)


@torch.no_grad()
def decode_sparse_region_memberships(
    embedding: Tensor,
    indices: Tensor,
    *,
    decoder: Callable[[Tensor], Tensor],
    num_classes: int,
    top_k: int,
    chunk_size: int,
    confidence: Tensor | None = None,
) -> SparseRegionMembership:
    """Decode selected point embeddings without materializing ``N x C`` logits."""

    top_k = _positive_integer(top_k, "top_k")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    num_classes = _positive_integer(num_classes, "num_classes")
    if embedding.ndim != 2:
        raise ValueError("point semantic embedding must have shape [N,D]")
    if indices.ndim != 1:
        raise ValueError("point semantic indices must have shape [M]")
    if indices.dtype != torch.long:
        raise TypeError("point semantic indices must use torch.long")
    if indices.device != embedding.device:
        raise ValueError("point semantic indices and embedding must share a device")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= embedding.shape[0]
    ):
        raise IndexError("point semantic index is out of range")
    if confidence is not None:
        if confidence.shape != (embedding.shape[0], 1):
            raise ValueError("point semantic confidence must have shape [N,1]")
        if confidence.device != embedding.device:
            raise ValueError("point semantic confidence and embedding must share a device")
        if not confidence.is_floating_point():
            raise TypeError("point semantic confidence must be floating point")

    chunks: list[SparseRegionMembership] = []
    for start in range(0, indices.numel(), chunk_size):
        selected_indices = indices[start : start + chunk_size]
        selected = embedding.index_select(0, selected_indices).float()
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            logits = decoder(selected)
        if logits.shape != (selected.shape[0], num_classes):
            raise ValueError(
                "point semantic decoder must return logits with shape [M,C]"
            )
        chunks.append(
            SparseRegionMembership.from_logits(
                logits,
                top_k=top_k,
                confidence=(
                    None
                    if confidence is None
                    else confidence.index_select(0, selected_indices)
                ),
            )
        )
    if not chunks:
        return SparseRegionMembership.from_logits(
            embedding.new_empty((0, num_classes)),
            top_k=top_k,
            confidence=(
                None if confidence is None else confidence.new_empty((0, 1))
            ),
        )
    return SparseRegionMembership(
        ids=torch.cat([value.ids for value in chunks]),
        weights=torch.cat([value.weights for value in chunks]),
        background=torch.cat([value.background for value in chunks]),
        tail=torch.cat([value.tail for value in chunks]),
        confidence=torch.cat([value.confidence for value in chunks]),
    )


__all__ = ["SparseRegionMembership", "decode_sparse_region_memberships"]
