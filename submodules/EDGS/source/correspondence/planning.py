"""Overlap-aware source/target grouping for MV-RoMa.

The planner is deliberately independent from image matching and COLMAP I/O.
It consumes a directed overlap matrix and implements the grouping policy from
MV-RoMa: overlap-weighted source quotas, greedy target coherence, pair-reuse
penalties, and reciprocal directed-pair completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraGroup:
    """One MV-RoMa source view and its jointly processed target views."""

    source_index: int
    target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not self.target_indices:
            raise ValueError("a camera group requires at least one target")
        if self.source_index in self.target_indices:
            raise ValueError("a camera cannot be both source and target")
        if len(set(self.target_indices)) != len(self.target_indices):
            raise ValueError("target camera indices must be unique")

    @property
    def all_indices(self) -> tuple[int, ...]:
        return (self.source_index, *self.target_indices)


@dataclass(frozen=True)
class OverlapPlannerSettings:
    """Hyperparameters for the deterministic MV-RoMa paper planner."""

    targets_per_group: int = 4
    min_targets_per_group: int = 1
    primary_group_budget: int | None = None
    overlap_threshold: float = 0.05
    source_quota_exponent: float = 0.75
    source_overlap_weight: float = 1.0
    target_overlap_weight: float = 1.0
    pair_reuse_penalty: float = 1.0
    augment_reciprocity: bool = True
    allow_partial_groups: bool = False

    def validate(self, camera_count: int) -> None:
        if camera_count < 2:
            raise ValueError("overlap-aware planning requires at least two cameras")
        if not 1 <= self.targets_per_group < camera_count:
            raise ValueError("targets_per_group must be in [1, camera_count - 1]")
        if not 1 <= self.min_targets_per_group <= self.targets_per_group:
            raise ValueError(
                "min_targets_per_group must be in [1, targets_per_group]"
            )
        if (
            self.primary_group_budget is not None
            and self.primary_group_budget < camera_count
        ):
            raise ValueError(
                "primary_group_budget must be at least the camera count so every "
                "view is used as a source"
            )
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if self.source_quota_exponent < 0.0:
            raise ValueError("source_quota_exponent must be non-negative")
        if self.source_overlap_weight < 0.0:
            raise ValueError("source_overlap_weight must be non-negative")
        if self.target_overlap_weight < 0.0:
            raise ValueError("target_overlap_weight must be non-negative")
        if self.pair_reuse_penalty < 0.0:
            raise ValueError("pair_reuse_penalty must be non-negative")
        if self.source_overlap_weight == self.target_overlap_weight == 0.0:
            raise ValueError("at least one overlap score weight must be positive")


@dataclass(frozen=True)
class CameraGroupPlan:
    """Immutable planner result and compact quality diagnostics."""

    groups: tuple[CameraGroup, ...]
    primary_group_count: int
    source_quotas: tuple[int, ...]
    pair_counts: np.ndarray
    mean_primary_source_overlap: float
    minimum_primary_source_overlap: float

    @property
    def reciprocal_group_count(self) -> int:
        return len(self.groups) - self.primary_group_count

    @property
    def reciprocity_coverage(self) -> float:
        selected = (self.pair_counts > 0).copy()
        np.fill_diagonal(selected, False)
        directed_count = int(selected.sum())
        if directed_count == 0:
            return 1.0
        reciprocal = selected & selected.T
        return float(reciprocal.sum() / directed_count)

    def neighbor_table(self) -> np.ndarray:
        """Return the first planned target set for each source camera.

        This preserves EDGS's historical ``[camera, target]`` return contract
        without exposing transform-space kNN. Every source has a primary group.
        """

        camera_count = len(self.source_quotas)
        target_count = max(len(group.target_indices) for group in self.groups)
        table = np.full((camera_count, target_count), -1, dtype=np.int64)
        for group in self.groups[: self.primary_group_count]:
            if table[group.source_index, 0] < 0:
                table[group.source_index, : len(group.target_indices)] = group.target_indices
        if np.any(table[:, 0] < 0):
            missing = np.flatnonzero(table[:, 0] < 0).tolist()
            raise RuntimeError(f"planner omitted primary groups for sources {missing}")
        return table


def _validated_overlap_matrix(overlap_matrix: np.ndarray) -> np.ndarray:
    overlap = np.asarray(overlap_matrix, dtype=np.float64)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap_matrix must be square")
    if overlap.shape[0] < 2:
        raise ValueError("overlap_matrix must contain at least two cameras")
    if not np.isfinite(overlap).all():
        raise ValueError("overlap_matrix contains non-finite values")
    tolerance = 1e-8
    if np.any(overlap < -tolerance) or np.any(overlap > 1.0 + tolerance):
        raise ValueError("overlap_matrix values must lie in [0, 1]")
    overlap = np.clip(overlap, 0.0, 1.0).copy()
    np.fill_diagonal(overlap, 0.0)
    return overlap


def _allocate_source_quotas(
    overlap: np.ndarray,
    *,
    budget: int,
    threshold: float,
    exponent: float,
) -> np.ndarray:
    """Allocate integer quotas with a deterministic largest-remainder rule."""

    camera_count = overlap.shape[0]
    quotas = np.ones(camera_count, dtype=np.int64)
    remaining = budget - camera_count
    if remaining == 0:
        return quotas

    neighbor_count = (overlap > threshold).sum(axis=1)
    weights = np.power(neighbor_count + 1.0, exponent)
    raw_extra = remaining * weights / weights.sum()
    integer_extra = np.floor(raw_extra).astype(np.int64)
    quotas += integer_extra

    unassigned = remaining - int(integer_extra.sum())
    remainder = raw_extra - integer_extra
    order = sorted(
        range(camera_count),
        key=lambda index: (-remainder[index], -weights[index], index),
    )
    quotas[order[:unassigned]] += 1
    return quotas


def _target_score(
    source: int,
    target: int,
    selected_targets: Iterable[int],
    *,
    overlap: np.ndarray,
    pair_counts: np.ndarray,
    settings: OverlapPlannerSettings,
) -> float:
    target_coherence = sum(overlap[neighbor, target] for neighbor in selected_targets)
    numerator = (
        settings.source_overlap_weight * overlap[source, target]
        + settings.target_overlap_weight * target_coherence
    )
    denominator = 1.0 + settings.pair_reuse_penalty * pair_counts[source, target]
    return float(numerator / denominator)


def _select_targets(
    source: int,
    candidates: Iterable[int],
    *,
    overlap: np.ndarray,
    pair_counts: np.ndarray,
    settings: OverlapPlannerSettings,
    initial_targets: Iterable[int] = (),
) -> tuple[int, ...]:
    selected = list(initial_targets)
    minimum_overlap = (
        settings.overlap_threshold if settings.allow_partial_groups else 0.0
    )
    available = {
        int(index)
        for index in candidates
        if int(index) != source and overlap[source, int(index)] > minimum_overlap
    }
    available.discard(source)
    available.difference_update(selected)

    while available and len(selected) < settings.targets_per_group:
        target = max(
            available,
            key=lambda index: (
                _target_score(
                    source,
                    index,
                    selected,
                    overlap=overlap,
                    pair_counts=pair_counts,
                    settings=settings,
                ),
                overlap[source, index],
                -index,
            ),
        )
        selected.append(target)
        available.remove(target)
    return tuple(selected)


def _record_group(group: CameraGroup, pair_counts: np.ndarray) -> None:
    for target in group.target_indices:
        pair_counts[group.source_index, target] += 1


def _primary_groups(
    overlap: np.ndarray,
    settings: OverlapPlannerSettings,
    quotas: np.ndarray,
    pair_counts: np.ndarray,
) -> list[CameraGroup]:
    camera_count = overlap.shape[0]
    candidates = tuple(range(camera_count))
    groups: list[CameraGroup] = []

    # Round-robin scheduling keeps repeated high-quota sources distributed
    # through the plan while remaining deterministic.
    for quota_round in range(int(quotas.max())):
        for source in range(camera_count):
            if quotas[source] <= quota_round:
                continue
            targets = _select_targets(
                source,
                candidates,
                overlap=overlap,
                pair_counts=pair_counts,
                settings=settings,
            )
            minimum = (
                settings.min_targets_per_group
                if settings.allow_partial_groups
                else settings.targets_per_group
            )
            if len(targets) < minimum:
                raise RuntimeError(f"could not fill camera group for source {source}")
            group = CameraGroup(source, targets)
            groups.append(group)
            _record_group(group, pair_counts)
    return groups


def _reciprocal_groups(
    primary_groups: Iterable[CameraGroup],
    *,
    overlap: np.ndarray,
    pair_counts: np.ndarray,
    settings: OverlapPlannerSettings,
) -> list[CameraGroup]:
    """Complete every selected directed pair without opening unpaired edges."""

    if not settings.augment_reciprocity:
        return []

    missing_by_source: dict[int, set[int]] = {}
    for group in primary_groups:
        for target in group.target_indices:
            if pair_counts[target, group.source_index] == 0:
                missing_by_source.setdefault(target, set()).add(group.source_index)

    groups: list[CameraGroup] = []
    for source in sorted(missing_by_source):
        required = missing_by_source[source]
        while required:
            # An earlier filler group may already have closed one of these
            # pairs. Avoid emitting a redundant reciprocal group for it.
            required = {
                target for target in required if pair_counts[source, target] == 0
            }
            if not required:
                break
            selected: list[int] = []
            while required and len(selected) < settings.targets_per_group:
                target = max(
                    required,
                    key=lambda index: (
                        _target_score(
                            source,
                            index,
                            selected,
                            overlap=overlap,
                            pair_counts=pair_counts,
                            settings=settings,
                        ),
                        overlap[source, index],
                        -index,
                    ),
                )
                selected.append(target)
                required.remove(target)

            # Filling only with an already selected undirected pair cannot
            # create a new one-sided correspondence that needs another group.
            paired_candidates = np.flatnonzero(
                (pair_counts[source] + pair_counts[:, source]) > 0
            ).tolist()
            targets = _select_targets(
                source,
                paired_candidates,
                overlap=overlap,
                pair_counts=pair_counts,
                settings=settings,
                initial_targets=selected,
            )
            minimum = (
                settings.min_targets_per_group
                if settings.allow_partial_groups
                else settings.targets_per_group
            )
            if len(targets) < minimum:
                raise RuntimeError(
                    f"could not fill reciprocal camera group for source {source}"
                )
            group = CameraGroup(source, targets)
            groups.append(group)
            _record_group(group, pair_counts)
    return groups


def plan_overlap_aware_groups(
    overlap_matrix: np.ndarray,
    settings: OverlapPlannerSettings,
) -> CameraGroupPlan:
    """Build the complete deterministic overlap-aware MV-RoMa group plan."""

    overlap = _validated_overlap_matrix(overlap_matrix)
    camera_count = overlap.shape[0]
    settings.validate(camera_count)
    budget = settings.primary_group_budget or camera_count
    quotas = _allocate_source_quotas(
        overlap,
        budget=budget,
        threshold=settings.overlap_threshold,
        exponent=settings.source_quota_exponent,
    )
    pair_counts = np.zeros((camera_count, camera_count), dtype=np.int32)
    primary = _primary_groups(overlap, settings, quotas, pair_counts)
    reciprocal = _reciprocal_groups(
        primary,
        overlap=overlap,
        pair_counts=pair_counts,
        settings=settings,
    )

    primary_overlap = np.asarray(
        [
            overlap[group.source_index, target]
            for group in primary
            for target in group.target_indices
        ],
        dtype=np.float64,
    )
    return CameraGroupPlan(
        groups=tuple((*primary, *reciprocal)),
        primary_group_count=len(primary),
        source_quotas=tuple(int(value) for value in quotas),
        pair_counts=pair_counts,
        mean_primary_source_overlap=float(primary_overlap.mean()),
        minimum_primary_source_overlap=float(primary_overlap.min()),
    )
