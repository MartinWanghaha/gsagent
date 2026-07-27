"""Coverage-conserving semantic charts for region-conditioned meshing.

The atlas is the topology boundary between a trained semantic 3DGS checkpoint
and local Delaunay extraction.  Decoder labels are identifiers, not object
categories: every class, including class zero, is treated uniformly.  Low
confidence ownership is routed to an explicit residual region so uncertainty
cannot punch holes in the global surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def _float_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.float().contiguous()


def _long_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.long).contiguous()


def _checkpoint_tensor(gaussians: Any, name: str) -> Tensor:
    if not hasattr(gaussians, name):
        raise AttributeError(f"Gaussian checkpoint does not expose {name}")
    value = getattr(gaussians, name)
    value = value() if callable(value) else value
    if not isinstance(value, Tensor):
        raise TypeError(f"Gaussian checkpoint attribute {name} must be a torch.Tensor")
    return value


def _ordered_unique(parts: Sequence[Tensor], device: torch.device) -> Tensor:
    nonempty = [part.to(device=device, dtype=torch.long).reshape(-1) for part in parts if part.numel()]
    if not nonempty:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(nonempty), sorted=True)


@dataclass(frozen=True)
class GaussianEvidence:
    """One renderer-derived observation summary per checkpoint Gaussian."""

    visible_count: Tensor
    normal: Tensor
    confidence: Tensor

    def __post_init__(self) -> None:
        visible_count = _float_tensor(self.visible_count)
        device = visible_count.device
        normal = _float_tensor(self.normal, device=device)
        confidence = _float_tensor(self.confidence, device=device)
        count = visible_count.numel()
        visible_count = visible_count.reshape(-1)
        confidence = confidence.reshape(-1)
        if normal.shape != (count, 3):
            raise ValueError("Gaussian evidence normal must have shape [N,3]")
        if confidence.shape != (count,):
            raise ValueError("Gaussian evidence confidence must have shape [N]")
        if not all(bool(torch.isfinite(value).all()) for value in (visible_count, normal, confidence)):
            raise ValueError("Gaussian evidence must be finite")
        if bool((visible_count < 0).any()):
            raise ValueError("Gaussian evidence visible_count must be non-negative")
        if bool(((confidence < 0) | (confidence > 1)).any()):
            raise ValueError("Gaussian evidence confidence must lie in [0,1]")
        object.__setattr__(self, "visible_count", visible_count)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "confidence", confidence)

    def __len__(self) -> int:
        return self.visible_count.numel()

    def to(self, device: str | torch.device) -> "GaussianEvidence":
        target = torch.device(device)
        return GaussianEvidence(
            visible_count=self.visible_count.to(target),
            normal=self.normal.to(target),
            confidence=self.confidence.to(target),
        )


@dataclass(frozen=True)
class RegionMembership:
    """Full-class top-k posterior without assigning semantics to any class ID."""

    gaussian_indices: Tensor
    ids: Tensor
    weights: Tensor
    tail: Tensor
    confidence: Tensor

    def __post_init__(self) -> None:
        if self.gaussian_indices.ndim != 1 or self.gaussian_indices.dtype != torch.long:
            raise ValueError("membership gaussian_indices must be torch.long with shape [M]")
        rows = self.gaussian_indices.numel()
        if self.ids.ndim != 2 or self.ids.shape[0] != rows or self.ids.dtype != torch.long:
            raise ValueError("membership ids must be torch.long with shape [M,K]")
        if self.ids.shape[1] < 1:
            raise ValueError("membership must retain at least one decoder class")
        if self.weights.shape != self.ids.shape:
            raise ValueError("membership weights must have shape [M,K]")
        if self.tail.shape != (rows, 1) or self.confidence.shape != (rows, 1):
            raise ValueError("membership tail and confidence must have shape [M,1]")
        values = (self.weights, self.tail, self.confidence)
        if any(not value.is_floating_point() for value in values):
            raise TypeError("membership probabilities must be floating point")
        if any(value.device != self.gaussian_indices.device for value in (*values, self.ids)):
            raise ValueError("membership tensors must share one device")
        if self.ids.numel() and bool((self.ids < 0).any()):
            raise ValueError("decoder region IDs must be non-negative")
        if rows > 1 and bool((self.gaussian_indices[1:] <= self.gaussian_indices[:-1]).any()):
            raise ValueError("membership Gaussian indices must be sorted and unique")
        if self.ids.shape[1] > 1:
            if bool((self.weights[:, 1:] > self.weights[:, :-1] + 2e-6).any()):
                raise ValueError("membership weights must be sorted in descending order")
            sorted_ids = self.ids.sort(dim=1).values
            if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
                raise ValueError("membership IDs must be unique in each row")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("membership contains non-finite values")
        if any(bool(((value < 0) | (value > 1)).any()) for value in values):
            raise ValueError("membership probabilities and confidence must lie in [0,1]")
        total = self.weights.sum(dim=1, keepdim=True) + self.tail
        if not torch.allclose(total, torch.ones_like(total), atol=4e-6, rtol=0.0):
            raise ValueError("membership top-k weights and tail must sum to one")

    def __len__(self) -> int:
        return self.gaussian_indices.numel()

    @property
    def primary_ids(self) -> Tensor:
        if not self.ids.shape[1]:
            raise RuntimeError("membership does not contain a primary region")
        return self.ids[:, 0]

    @property
    def primary_weights(self) -> Tensor:
        if not self.weights.shape[1]:
            raise RuntimeError("membership does not contain a primary region")
        return self.weights[:, 0]

    def index_select(self, indices: Tensor | Sequence[int]) -> "RegionMembership":
        selected = _long_tensor(indices, device=self.gaussian_indices.device).reshape(-1)
        if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= len(self)):
            raise IndexError("membership row is out of range")
        return RegionMembership(
            gaussian_indices=self.gaussian_indices.index_select(0, selected),
            ids=self.ids.index_select(0, selected),
            weights=self.weights.index_select(0, selected),
            tail=self.tail.index_select(0, selected),
            confidence=self.confidence.index_select(0, selected),
        )

    def probability(self, region_id: int) -> Tensor:
        query = int(region_id)
        if query < 0:
            raise ValueError("decoder region_id must be non-negative")
        return torch.where(
            self.ids == query,
            self.weights,
            torch.zeros_like(self.weights),
        ).sum(dim=1)


@dataclass(frozen=True)
class RegionChart:
    """A bounded spatial chart with explicit same-region and contact halos."""

    chart_id: int
    region_id: int
    partition_id: int
    core_indices: Tensor
    boundary_indices: Tensor
    overlap_indices: Tensor
    contact_indices: Tensor

    def __post_init__(self) -> None:
        if self.chart_id < 0 or self.partition_id < 0:
            raise ValueError("chart_id and partition_id must be non-negative")
        tensors = (
            self.core_indices,
            self.boundary_indices,
            self.overlap_indices,
            self.contact_indices,
        )
        if any(value.ndim != 1 or value.dtype != torch.long for value in tensors):
            raise ValueError("chart indices must be torch.long vectors")
        if any(value.device != tensors[0].device for value in tensors[1:]):
            raise ValueError("chart index tensors must share one device")
        for value in tensors:
            if value.numel() and bool((value < 0).any()):
                raise ValueError("chart Gaussian indices must be non-negative")
            if value.numel() > 1 and bool((value[1:] <= value[:-1]).any()):
                raise ValueError("chart Gaussian indices must be sorted and unique")
        owned = _ordered_unique(tensors[:2], tensors[0].device)
        if owned.numel() != self.core_indices.numel() + self.boundary_indices.numel():
            raise ValueError("chart core and boundary ownership must be disjoint")
        halo = _ordered_unique(tensors[2:], tensors[0].device)
        if owned.numel() and halo.numel() and bool(torch.isin(owned, halo).any()):
            raise ValueError("owned and halo Gaussian indices must be disjoint")

    @property
    def owned_indices(self) -> Tensor:
        return _ordered_unique(
            (self.core_indices, self.boundary_indices),
            self.core_indices.device,
        )

    @property
    def halo_indices(self) -> Tensor:
        return _ordered_unique(
            (self.overlap_indices, self.contact_indices),
            self.core_indices.device,
        )

    @property
    def gaussian_indices(self) -> Tensor:
        return _ordered_unique(
            (self.owned_indices, self.halo_indices),
            self.core_indices.device,
        )


@dataclass(frozen=True)
class RegionAtlas:
    """Selected Gaussian anchors and their coverage-conserving chart layout."""

    gaussian_count: int
    gaussian_indices: Tensor
    membership: RegionMembership
    owner_region_ids: Tensor
    local_scale: Tensor
    quality: Tensor
    boundary: Tensor
    contact_pairs: Tensor
    charts: tuple[RegionChart, ...]

    def __post_init__(self) -> None:
        if self.gaussian_count < 1:
            raise ValueError("gaussian_count must be positive")
        count = self.gaussian_indices.numel()
        device = self.gaussian_indices.device
        if self.gaussian_indices.ndim != 1 or self.gaussian_indices.dtype != torch.long:
            raise ValueError("atlas gaussian_indices must be torch.long with shape [A]")
        if count and (
            int(self.gaussian_indices.min()) < 0
            or int(self.gaussian_indices.max()) >= self.gaussian_count
        ):
            raise IndexError("atlas Gaussian index is out of checkpoint range")
        if count > 1 and bool((self.gaussian_indices[1:] <= self.gaussian_indices[:-1]).any()):
            raise ValueError("atlas gaussian_indices must be sorted and unique")
        if not torch.equal(self.membership.gaussian_indices, self.gaussian_indices):
            raise ValueError("atlas membership must align with gaussian_indices")
        for name in ("owner_region_ids", "local_scale", "quality", "boundary"):
            value = getattr(self, name)
            if value.shape != (count,) or value.device != device:
                raise ValueError(f"atlas {name} must have shape [A] on the atlas device")
        if self.owner_region_ids.dtype != torch.long or self.boundary.dtype != torch.bool:
            raise TypeError("owner_region_ids must be long and boundary must be bool")
        if not bool(torch.isfinite(self.local_scale).all()) or not bool(torch.isfinite(self.quality).all()):
            raise ValueError("atlas scale and quality must be finite")
        if count and bool((self.local_scale <= 0).any()):
            raise ValueError("atlas local_scale must be positive")
        if bool(((self.quality < 0) | (self.quality > 1)).any()):
            raise ValueError("atlas quality must lie in [0,1]")
        if self.contact_pairs.ndim != 2 or self.contact_pairs.shape[1] != 2:
            raise ValueError("atlas contact_pairs must have shape [E,2]")
        if self.contact_pairs.dtype != torch.long or self.contact_pairs.device != device:
            raise TypeError("atlas contact_pairs must be long on the atlas device")
        if self.contact_pairs.numel() and not bool(torch.isin(self.contact_pairs, self.gaussian_indices).all()):
            raise ValueError("contact pair endpoint is outside the atlas")
        if self.contact_pairs.numel() and bool(
            (self.contact_pairs[:, 0] >= self.contact_pairs[:, 1]).any()
        ):
            raise ValueError("contact pair endpoints must be sorted and distinct")
        if tuple(chart.chart_id for chart in self.charts) != tuple(range(len(self.charts))):
            raise ValueError("chart IDs must be contiguous and stable")
        if any(chart.core_indices.device != device for chart in self.charts):
            raise ValueError("charts and atlas must share one device")
        owned = [chart.owned_indices for chart in self.charts]
        owned_flat = torch.cat(owned) if owned else torch.empty(0, device=device, dtype=torch.long)
        if owned_flat.numel() != count or not torch.equal(
            torch.sort(owned_flat).values,
            self.gaussian_indices,
        ):
            raise ValueError("every atlas Gaussian must be owned by exactly one chart")
        for chart in self.charts:
            if chart.gaussian_indices.numel() and not bool(
                torch.isin(chart.gaussian_indices, self.gaussian_indices).all()
            ):
                raise ValueError("chart references a Gaussian outside the atlas")
            owned_rows = torch.searchsorted(self.gaussian_indices, chart.owned_indices)
            if owned_rows.numel() and not bool(
                (self.owner_region_ids.index_select(0, owned_rows) == chart.region_id).all()
            ):
                raise ValueError("chart ownership disagrees with atlas region IDs")
            overlap_rows = torch.searchsorted(self.gaussian_indices, chart.overlap_indices)
            if overlap_rows.numel() and not bool(
                (self.owner_region_ids.index_select(0, overlap_rows) == chart.region_id).all()
            ):
                raise ValueError("same-region overlap halo has inconsistent ownership")
            contact_rows = torch.searchsorted(self.gaussian_indices, chart.contact_indices)
            if contact_rows.numel() and bool(
                (self.owner_region_ids.index_select(0, contact_rows) == chart.region_id).any()
            ):
                raise ValueError("contact halo must be owned by another region")

    def __len__(self) -> int:
        return self.gaussian_indices.numel()

    def chart(self, chart_id: int) -> RegionChart:
        if chart_id < 0 or chart_id >= len(self.charts):
            raise IndexError("chart_id is out of range")
        return self.charts[chart_id]

    def row_for_gaussians(self, gaussian_indices: Tensor | Sequence[int]) -> Tensor:
        query = _long_tensor(gaussian_indices, device=self.gaussian_indices.device).reshape(-1)
        rows = torch.searchsorted(self.gaussian_indices, query)
        valid = rows < len(self)
        if len(self):
            valid &= self.gaussian_indices[rows.clamp_max(len(self) - 1)] == query
        if not bool(valid.all()):
            raise IndexError("Gaussian index is outside the region atlas")
        return rows


@dataclass(frozen=True)
class RegionAtlasConfig:
    """Single policy for ownership, anchor budget, and spatial charting."""

    max_gaussians: int = 50_000
    max_core_gaussians: int = 20_000
    top_k: int = 3
    decoder_chunk_size: int = 32_768
    min_opacity: float = 0.02
    max_gaussian_extent: Optional[float] = None
    min_region_gaussians: int = 12
    confident_probability: float = 0.5
    confident_semantic: float = 0.5
    boundary_margin: float = 0.15
    boundary_score_threshold: float = 0.5
    boundary_fraction: float = 0.3
    contact_neighbors: int = 8
    contact_radius_factor: float = 2.5
    halo_factor: float = 1.5
    background_id: Optional[int] = None
    residual_region_id: int = -1

    def __post_init__(self) -> None:
        for name in (
            "max_gaussians",
            "max_core_gaussians",
            "top_k",
            "decoder_chunk_size",
            "min_region_gaussians",
            "contact_neighbors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_gaussians < 4 or self.max_core_gaussians < 4:
            raise ValueError("Gaussian and chart budgets must allow a tetrahedron")
        for name in (
            "min_opacity",
            "confident_probability",
            "confident_semantic",
            "boundary_margin",
            "boundary_score_threshold",
            "boundary_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in ("contact_radius_factor", "halo_factor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.background_id is not None and self.background_id < 0:
            raise ValueError("background_id must be non-negative or None")
        if self.residual_region_id >= 0:
            raise ValueError("residual_region_id must not collide with decoder class IDs")
        if self.max_gaussian_extent is not None and (
            not math.isfinite(self.max_gaussian_extent)
            or self.max_gaussian_extent <= 0.0
        ):
            raise ValueError("max_gaussian_extent must be finite and positive")


@dataclass(frozen=True)
class _CheckpointAttributes:
    xyz: Tensor
    opacity: Tensor
    scaling: Tensor
    semantic: Tensor
    semantic_confidence: Tensor
    geometry_confidence: Tensor
    boundary_score: Tensor
    observation_count: Tensor
    decoder: Callable[[Tensor], Tensor]


def _decode_membership(
    semantic: Tensor,
    gaussian_indices: Tensor,
    semantic_confidence: Tensor,
    decoder: Callable[[Tensor], Tensor],
    top_k: int,
    chunk_size: int,
) -> RegionMembership:
    ids: list[Tensor] = []
    weights: list[Tensor] = []
    tails: list[Tensor] = []
    classes: int | None = None
    with torch.no_grad(), torch.autocast(device_type=semantic.device.type, enabled=False):
        for selected in gaussian_indices.split(chunk_size):
            logits = decoder(semantic.index_select(0, selected).float())
            if logits.ndim != 2 or logits.shape[0] != selected.numel() or logits.shape[1] < 1:
                raise ValueError("semantic decoder must return logits with shape [M,C], C >= 1")
            if classes is None:
                classes = logits.shape[1]
            elif logits.shape[1] != classes:
                raise ValueError("semantic decoder class count changed between chunks")
            probability = torch.softmax(logits.float(), dim=1)
            selected_count = min(top_k, probability.shape[1])
            chunk_weights, chunk_ids = probability.topk(
                selected_count,
                dim=1,
                largest=True,
                sorted=True,
            )
            ids.append(chunk_ids.long())
            weights.append(chunk_weights)
            tails.append((1.0 - chunk_weights.sum(dim=1, keepdim=True)).clamp_min(0.0))
    return RegionMembership(
        gaussian_indices=gaussian_indices,
        ids=torch.cat(ids),
        weights=torch.cat(weights),
        tail=torch.cat(tails),
        confidence=semantic_confidence.index_select(0, gaussian_indices)[:, None],
    )


def _nearest_region_scale(
    xyz: np.ndarray,
    region_ids: np.ndarray,
    intrinsic_scale: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("region atlas construction requires scipy") from error
    result = np.empty(len(xyz), dtype=np.float32)
    for region_id in np.unique(region_ids):
        rows = np.flatnonzero(region_ids == region_id)
        points = xyz[rows]
        fallback = np.maximum(intrinsic_scale[rows], 1e-7)
        if len(rows) == 1:
            result[rows] = fallback
            continue
        neighbors = min(5, len(rows))
        distances, _ = cKDTree(points).query(points, k=neighbors, workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
        positive = np.where(distances > 1e-8, distances, np.inf).min(axis=1)
        finite = positive[np.isfinite(positive)]
        replacement = float(np.median(finite)) if len(finite) else float(np.median(fallback))
        positive = np.where(np.isfinite(positive), positive, replacement)
        low, high = np.quantile(positive, (0.05, 0.95))
        low = max(float(low), min(replacement, float(np.median(fallback))) * 0.1, 1e-7)
        high = max(float(high), low)
        result[rows] = np.clip(positive, low, high)
    return result


def _contact_pairs(
    xyz: np.ndarray,
    region_ids: np.ndarray,
    local_scale: np.ndarray,
    neighbors: int,
    radius_factor: float,
) -> np.ndarray:
    if len(xyz) < 2:
        return np.empty((0, 2), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("region atlas construction requires scipy") from error
    count = min(neighbors + 1, len(xyz))
    distance, index = cKDTree(xyz).query(xyz, k=count, workers=-1)
    if distance.ndim == 1:
        distance = distance[:, None]
        index = index[:, None]
    source = np.arange(len(xyz), dtype=np.int64)[:, None]
    cross = region_ids[index] != region_ids[:, None]
    radius = radius_factor * np.maximum(local_scale[:, None], local_scale[index])
    valid = cross & (index != source) & np.isfinite(distance) & (distance <= radius)
    pairs: list[np.ndarray] = []
    for rank in range(1, valid.shape[1]):
        rows = np.flatnonzero(valid[:, rank])
        if len(rows):
            pairs.append(np.stack((rows, index[rows, rank]), axis=1))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    result = np.sort(np.concatenate(pairs, axis=0), axis=1)
    return np.unique(result, axis=0)


def _allocate_region_budgets(
    region_ids: np.ndarray,
    local_scale: np.ndarray,
    quality: np.ndarray,
    budget: int,
    minimum: int,
) -> dict[int, int]:
    regions, inverse, counts = np.unique(region_ids, return_inverse=True, return_counts=True)
    if len(region_ids) <= budget:
        return {int(region): int(count) for region, count in zip(regions, counts)}
    mean_quality = np.zeros(len(regions), dtype=np.float64)
    median_scale = np.zeros(len(regions), dtype=np.float64)
    for row in range(len(regions)):
        selected = inverse == row
        mean_quality[row] = max(float(np.mean(quality[selected])), 1e-6)
        median_scale[row] = max(float(np.median(local_scale[selected])), 1e-8)
    global_scale = max(float(np.median(local_scale)), 1e-8)
    demand = np.sqrt(counts) * np.sqrt(mean_quality) * np.sqrt(global_scale / median_scale)
    quota = np.minimum(counts, minimum).astype(np.int64)
    if int(quota.sum()) > budget:
        quota.fill(0)
        priority = np.argsort(-demand, kind="stable")
        quota[priority[:budget]] = 1
    while int(quota.sum()) < budget:
        capacity = counts - quota
        active = capacity > 0
        if not active.any():
            break
        remaining = budget - int(quota.sum())
        active_demand = np.where(active, demand, 0.0)
        ideal = remaining * active_demand / active_demand.sum()
        addition = np.minimum(capacity, np.floor(ideal).astype(np.int64))
        if not int(addition.sum()):
            fractional = np.where(active, ideal - np.floor(ideal), -np.inf)
            addition[int(np.argmax(fractional))] = 1
        quota += addition
    return {int(region): int(value) for region, value in zip(regions, quota) if value}


def _spatial_quality_sample(
    rows: np.ndarray,
    xyz: np.ndarray,
    local_scale: np.ndarray,
    quality: np.ndarray,
    budget: int,
) -> np.ndarray:
    if len(rows) <= budget:
        return np.sort(rows)
    points = xyz[rows]
    extent = np.ptp(points, axis=0)
    maximum = max(float(extent.max()), 1e-8)
    active = extent > maximum * 0.02
    dimension = max(int(active.sum()), 1)
    bins = max(int(math.ceil(budget ** (1.0 / dimension))), 1)
    origin = points.min(axis=0)
    normalized = np.zeros_like(points)
    normalized[:, active] = (points[:, active] - origin[active]) / np.maximum(extent[active], 1e-8)
    voxel = np.floor(normalized * bins).astype(np.int64)
    voxel[:, ~active] = 0
    median_scale = max(float(np.median(local_scale[rows])), 1e-8)
    priority = quality[rows] * np.sqrt(
        np.clip(median_scale / np.maximum(local_scale[rows], 1e-8), 0.25, 4.0)
    )
    order = np.lexsort((rows, -priority))
    _, first = np.unique(voxel[order], axis=0, return_index=True)
    representatives = order[first]
    if len(representatives) > budget:
        representative_order = np.argsort(-priority[representatives], kind="stable")
        chosen = representatives[representative_order[:budget]]
    else:
        chosen = representatives
        remaining = budget - len(chosen)
        if remaining:
            available = np.ones(len(rows), dtype=bool)
            available[chosen] = False
            fill = np.flatnonzero(available)
            fill = fill[np.argsort(-priority[fill], kind="stable")[:remaining]]
            chosen = np.concatenate((chosen, fill))
    return np.sort(rows[chosen])


def _select_anchors(
    region_ids: np.ndarray,
    boundary: np.ndarray,
    xyz: np.ndarray,
    local_scale: np.ndarray,
    quality: np.ndarray,
    quotas: dict[int, int],
    boundary_fraction: float,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for region_id in sorted(quotas):
        rows = np.flatnonzero(region_ids == region_id)
        quota = quotas[region_id]
        boundary_rows = rows[boundary[rows]]
        core_rows = rows[~boundary[rows]]
        boundary_budget = min(len(boundary_rows), int(math.ceil(quota * boundary_fraction)))
        core_budget = min(len(core_rows), quota - boundary_budget)
        boundary_budget = min(len(boundary_rows), quota - core_budget)
        chosen = [
            _spatial_quality_sample(
                core_rows,
                xyz,
                local_scale,
                quality,
                core_budget,
            ),
            _spatial_quality_sample(
                boundary_rows,
                xyz,
                local_scale,
                quality,
                boundary_budget,
            ),
        ]
        region_selected = np.concatenate([value for value in chosen if len(value)])
        if len(region_selected) < quota:
            remaining = np.setdiff1d(rows, region_selected, assume_unique=True)
            region_selected = np.concatenate(
                (
                    region_selected,
                    _spatial_quality_sample(
                        remaining,
                        xyz,
                        local_scale,
                        quality,
                        quota - len(region_selected),
                    ),
                )
            )
        selected.append(region_selected)
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _spatial_partitions(
    rows: np.ndarray,
    xyz: np.ndarray,
    gaussian_indices: np.ndarray,
    maximum: int,
) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []

    def split(current: np.ndarray) -> None:
        if len(current) <= maximum:
            leaves.append(np.sort(current))
            return
        extent = np.ptp(xyz[current], axis=0)
        axis = int(np.argmax(extent))
        order = np.lexsort((gaussian_indices[current], xyz[current, axis]))
        ordered = current[order]
        middle = len(ordered) // 2
        split(ordered[:middle])
        split(ordered[middle:])

    split(rows)
    return leaves


def _build_charts(
    gaussian_indices: np.ndarray,
    xyz: np.ndarray,
    region_ids: np.ndarray,
    boundary: np.ndarray,
    local_scale: np.ndarray,
    contact_pairs: np.ndarray,
    config: RegionAtlasConfig,
    device: torch.device,
) -> tuple[RegionChart, ...]:
    contact_adjacency: dict[int, set[int]] = {}
    for left, right in contact_pairs:
        contact_adjacency.setdefault(int(left), set()).add(int(right))
        contact_adjacency.setdefault(int(right), set()).add(int(left))
    charts: list[RegionChart] = []
    for region_id in sorted(np.unique(region_ids).tolist()):
        region_rows = np.flatnonzero(region_ids == region_id)
        partitions = _spatial_partitions(
            region_rows,
            xyz,
            gaussian_indices,
            config.max_core_gaussians,
        )
        for partition_id, owned_rows in enumerate(partitions):
            owned_set = set(gaussian_indices[owned_rows].tolist())
            outside_rows = np.setdiff1d(region_rows, owned_rows, assume_unique=True)
            overlap_rows = np.empty(0, dtype=np.int64)
            if len(outside_rows):
                from scipy.spatial import cKDTree

                distance, nearest = cKDTree(xyz[owned_rows]).query(
                    xyz[outside_rows],
                    k=1,
                    workers=-1,
                )
                radius = config.halo_factor * np.maximum(
                    local_scale[outside_rows],
                    local_scale[owned_rows[nearest]],
                )
                candidates = outside_rows[np.isfinite(distance) & (distance <= radius)]
                if len(candidates) > config.max_core_gaussians:
                    candidate_distance, _ = cKDTree(xyz[owned_rows]).query(
                        xyz[candidates],
                        k=1,
                        workers=-1,
                    )
                    order = np.lexsort((gaussian_indices[candidates], candidate_distance))
                    candidates = candidates[order[: config.max_core_gaussians]]
                overlap_rows = candidates
            contact_gaussians: set[int] = set()
            for gaussian_id in owned_set:
                contact_gaussians.update(contact_adjacency.get(gaussian_id, ()))
            contact_gaussians.difference_update(owned_set)
            core = gaussian_indices[owned_rows[~boundary[owned_rows]]]
            boundary_values = gaussian_indices[owned_rows[boundary[owned_rows]]]
            overlap = gaussian_indices[overlap_rows]
            contact = np.asarray(sorted(contact_gaussians), dtype=np.int64)
            overlap = np.setdiff1d(overlap, np.concatenate((core, boundary_values)), assume_unique=True)
            contact = np.setdiff1d(
                contact,
                np.concatenate((core, boundary_values, overlap)),
                assume_unique=True,
            )
            charts.append(
                RegionChart(
                    chart_id=len(charts),
                    region_id=int(region_id),
                    partition_id=partition_id,
                    core_indices=_long_tensor(np.sort(core), device=device),
                    boundary_indices=_long_tensor(np.sort(boundary_values), device=device),
                    overlap_indices=_long_tensor(np.sort(overlap), device=device),
                    contact_indices=_long_tensor(np.sort(contact), device=device),
                )
            )
    return tuple(charts)


class RegionAtlasBuilder:
    """Decode a checkpoint and construct bounded, overlapping spatial charts."""

    def __init__(self, config: RegionAtlasConfig | None = None) -> None:
        self.config = config or RegionAtlasConfig()

    @staticmethod
    def _checkpoint(gaussians: Any) -> _CheckpointAttributes:
        xyz = _checkpoint_tensor(gaussians, "get_xyz")
        device = xyz.device
        opacity = _checkpoint_tensor(gaussians, "get_opacity").reshape(-1)
        scaling = _checkpoint_tensor(gaussians, "get_scaling")
        semantic = _checkpoint_tensor(gaussians, "get_semantic")
        semantic_confidence = _checkpoint_tensor(
            gaussians,
            "get_semantic_confidence",
        ).reshape(-1)
        geometry = _checkpoint_tensor(gaussians, "get_geometry_posterior")
        boundary_score = _checkpoint_tensor(gaussians, "get_boundary_score").reshape(-1)
        observation_count = _checkpoint_tensor(gaussians, "observation_count").reshape(-1)
        decoder = getattr(gaussians, "semantic_decoder", None)
        if not callable(decoder):
            raise ValueError("Gaussian checkpoint requires its trained semantic_decoder")
        count = xyz.shape[0]
        if xyz.shape != (count, 3) or scaling.shape != (count, 3):
            raise ValueError("Gaussian xyz and scaling must have shape [N,3]")
        if semantic.ndim != 2 or semantic.shape[0] != count:
            raise ValueError("Gaussian semantic embedding must have shape [N,D]")
        if geometry.ndim != 2 or geometry.shape[0] != count or geometry.shape[1] < 1:
            raise ValueError("Gaussian geometry posterior must have shape [N,E]")
        for name, value in (
            ("opacity", opacity),
            ("semantic confidence", semantic_confidence),
            ("boundary score", boundary_score),
            ("observation count", observation_count),
        ):
            if value.shape != (count,):
                raise ValueError(f"Gaussian {name} must have shape [N]")
        tensors = (
            xyz,
            opacity,
            scaling,
            semantic,
            semantic_confidence,
            geometry,
            boundary_score,
            observation_count,
        )
        if any(value.device != device for value in tensors):
            raise ValueError("Gaussian checkpoint tensors must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Gaussian checkpoint contains non-finite attributes")
        if bool((scaling <= 0).any()):
            raise ValueError("Gaussian activated scaling must be positive")
        return _CheckpointAttributes(
            xyz=xyz,
            opacity=opacity,
            scaling=scaling,
            semantic=semantic,
            semantic_confidence=semantic_confidence.clamp(0, 1),
            geometry_confidence=geometry.max(dim=1).values.clamp(0, 1),
            boundary_score=boundary_score.clamp(0, 1),
            observation_count=observation_count.clamp_min(0),
            decoder=decoder,
        )

    def build(
        self,
        gaussians: Any,
        evidence: GaussianEvidence | None = None,
    ) -> RegionAtlas:
        attributes = self._checkpoint(gaussians)
        count = len(attributes.xyz)
        if count < 1:
            raise ValueError("cannot build a region atlas from an empty checkpoint")
        device = attributes.xyz.device
        if evidence is not None:
            if len(evidence) != count:
                raise ValueError("Gaussian evidence length differs from checkpoint")
            evidence = evidence.to(device)
            visible_count = evidence.visible_count
            observation_confidence = evidence.confidence
        else:
            visible_count = attributes.observation_count
            observation_confidence = torch.ones_like(visible_count)
        eligible = (
            (attributes.opacity >= self.config.min_opacity)
            & torch.isfinite(attributes.xyz).all(dim=1)
            & torch.isfinite(attributes.scaling).all(dim=1)
        )
        if self.config.max_gaussian_extent is not None:
            eligible &= (
                attributes.scaling.max(dim=1).values
                <= float(self.config.max_gaussian_extent)
            )
        gaussian_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if not gaussian_indices.numel():
            raise ValueError("no Gaussian passes the atlas opacity policy")
        membership = _decode_membership(
            attributes.semantic,
            gaussian_indices,
            attributes.semantic_confidence,
            attributes.decoder,
            self.config.top_k,
            self.config.decoder_chunk_size,
        )
        owner_region_ids = membership.primary_ids.clone()
        uncertain = (
            (membership.primary_weights < self.config.confident_probability)
            | (membership.confidence[:, 0] < self.config.confident_semantic)
        )
        if self.config.background_id is not None:
            uncertain |= owner_region_ids == self.config.background_id
        owner_region_ids[uncertain] = self.config.residual_region_id

        selected_opacity = attributes.opacity.index_select(0, gaussian_indices).clamp(0, 1)
        selected_geometry = attributes.geometry_confidence.index_select(0, gaussian_indices)
        selected_visible = visible_count.index_select(0, gaussian_indices)
        selected_observation_confidence = observation_confidence.index_select(0, gaussian_indices)
        visibility = (1.0 - torch.exp(-0.5 * selected_visible)) * selected_observation_confidence
        quality = torch.pow(
            (
                selected_opacity.clamp_min(1e-6)
                * membership.confidence[:, 0].clamp_min(1e-6)
                * selected_geometry.clamp_min(1e-6)
                * visibility.clamp_min(1e-6)
            ),
            0.25,
        ).clamp(0, 1)

        xyz_numpy = attributes.xyz.index_select(0, gaussian_indices).detach().cpu().float().numpy()
        region_numpy = owner_region_ids.detach().cpu().numpy()
        scale_numpy = (
            attributes.scaling.index_select(0, gaussian_indices)
            .max(dim=1)
            .values.detach()
            .cpu()
            .float()
            .numpy()
        )
        local_scale_numpy = _nearest_region_scale(
            xyz_numpy,
            region_numpy,
            scale_numpy,
        )
        margin = membership.weights[:, 0]
        if membership.weights.shape[1] > 1:
            margin = margin - membership.weights[:, 1]
        boundary = (
            (margin <= self.config.boundary_margin)
            | (
                attributes.boundary_score.index_select(0, gaussian_indices)
                >= self.config.boundary_score_threshold
            )
            | uncertain
        )
        quality_numpy = quality.detach().cpu().numpy()
        boundary_numpy = boundary.detach().cpu().numpy()
        quotas = _allocate_region_budgets(
            region_numpy,
            local_scale_numpy,
            quality_numpy,
            min(self.config.max_gaussians, len(gaussian_indices)),
            self.config.min_region_gaussians,
        )
        selected_rows = _select_anchors(
            region_numpy,
            boundary_numpy,
            xyz_numpy,
            local_scale_numpy,
            quality_numpy,
            quotas,
            self.config.boundary_fraction,
        )
        selected_global = gaussian_indices.detach().cpu().numpy()[selected_rows]
        order = np.argsort(selected_global, kind="stable")
        selected_rows = selected_rows[order]
        selected_global = selected_global[order]
        selected_rows_tensor = torch.as_tensor(selected_rows, device=device, dtype=torch.long)

        selected_membership = membership.index_select(selected_rows_tensor)
        selected_regions = region_numpy[selected_rows]
        selected_xyz = xyz_numpy[selected_rows]
        selected_local_scale = local_scale_numpy[selected_rows]
        selected_pair_rows = _contact_pairs(
            selected_xyz,
            selected_regions,
            selected_local_scale,
            self.config.contact_neighbors,
            self.config.contact_radius_factor,
        )
        selected_pairs = (
            selected_global[selected_pair_rows]
            if len(selected_pair_rows)
            else np.empty((0, 2), dtype=np.int64)
        )
        selected_boundary = boundary_numpy[selected_rows].copy()
        if len(selected_pair_rows):
            selected_boundary[np.unique(selected_pair_rows)] = True
        charts = _build_charts(
            selected_global,
            selected_xyz,
            selected_regions,
            selected_boundary,
            selected_local_scale,
            selected_pairs,
            self.config,
            device,
        )
        return RegionAtlas(
            gaussian_count=count,
            gaussian_indices=torch.as_tensor(selected_global, device=device, dtype=torch.long),
            membership=selected_membership,
            owner_region_ids=owner_region_ids.index_select(0, selected_rows_tensor),
            local_scale=torch.as_tensor(
                selected_local_scale,
                device=device,
                dtype=attributes.xyz.dtype,
            ),
            quality=quality.index_select(0, selected_rows_tensor),
            boundary=torch.as_tensor(
                selected_boundary,
                device=device,
                dtype=torch.bool,
            ),
            contact_pairs=torch.as_tensor(selected_pairs, device=device, dtype=torch.long),
            charts=charts,
        )


__all__ = [
    "GaussianEvidence",
    "RegionAtlas",
    "RegionAtlasBuilder",
    "RegionAtlasConfig",
    "RegionChart",
    "RegionMembership",
]
