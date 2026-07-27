"""Adaptive octree probes and extraction-oriented blocked grids."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np

from .field import SurfaceFieldAdapter, as_field_adapter
from .types import SurfaceSamples


@dataclass(frozen=True)
class Bounds:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float32).reshape(3)
        maximum = np.asarray(self.maximum, dtype=np.float32).reshape(3)
        if np.any(maximum <= minimum):
            raise ValueError("bounds maximum must be greater than minimum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_array(cls, value: Sequence[float] | np.ndarray) -> "Bounds":
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (2, 3):
            return cls(array[0], array[1])
        if array.size == 6:
            flat = array.reshape(-1)
            return cls(flat[:3], flat[3:])
        raise ValueError("bounds must be [xmin,ymin,zmin,xmax,ymax,zmax] or [2,3]")

    @property
    def extent(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    def corners(self, include_center: bool = False) -> np.ndarray:
        points = np.asarray(
            [
                [self.minimum[0] if bit[0] == 0 else self.maximum[0],
                 self.minimum[1] if bit[1] == 0 else self.maximum[1],
                 self.minimum[2] if bit[2] == 0 else self.maximum[2]]
                for bit in product((0, 1), repeat=3)
            ],
            dtype=np.float32,
        )
        if include_center:
            points = np.concatenate([points, self.center[None]], axis=0)
        return points

    def split(self) -> list["Bounds"]:
        center = self.center
        children = []
        for bit in product((0, 1), repeat=3):
            lower = np.where(np.asarray(bit, dtype=bool), center, self.minimum)
            upper = np.where(np.asarray(bit, dtype=bool), self.maximum, center)
            children.append(Bounds(lower, upper))
        return children


@dataclass
class AdaptiveSamplingConfig:
    sdf_level: float = 0.0
    occupancy_level: float = 0.5
    occupancy_support: float = 0.05
    near_surface_factor: float = 0.35
    semantic_cosine_threshold: float = 0.85
    detail_posterior_threshold: float = 0.35
    uncertainty_threshold: float = 0.25
    thin_posterior_indices: tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class RefinementDecision:
    active: bool
    surface: bool
    near_surface: bool
    semantic_boundary: bool
    thin_or_boundary: bool
    uncertain: bool

    @property
    def detail_count(self) -> int:
        return sum((self.semantic_boundary, self.thin_or_boundary, self.uncertain))


def _semantic_boundary(semantic: np.ndarray, threshold: float) -> bool:
    if len(semantic) < 2:
        return False
    length = np.linalg.norm(semantic, axis=1, keepdims=True)
    normalized = semantic / np.maximum(length, 1e-8)
    similarity = normalized @ normalized.T
    return bool(np.min(similarity) < threshold)


def refinement_decision(
    samples: SurfaceSamples,
    cell_diagonal: float,
    config: AdaptiveSamplingConfig,
) -> RefinementDecision:
    sdf_min, sdf_max = float(samples.sdf.min()), float(samples.sdf.max())
    occ_min = float(samples.occupancy.min())
    occ_max = float(samples.occupancy.max())
    surface = (
        sdf_min <= config.sdf_level <= sdf_max
        or occ_min <= config.occupancy_level <= occ_max
    )
    near = bool(
        np.min(np.abs(samples.sdf - config.sdf_level))
        <= config.near_surface_factor * cell_diagonal
    )
    support = occ_max >= config.occupancy_support or near or surface
    semantic_boundary = support and _semantic_boundary(
        samples.semantic, config.semantic_cosine_threshold
    )
    indices = [
        index
        for index in config.thin_posterior_indices
        if 0 <= index < samples.geometry_posterior.shape[1]
    ]
    detail = bool(
        support
        and indices
        and np.max(samples.geometry_posterior[:, indices])
        >= config.detail_posterior_threshold
    )
    uncertain = bool(
        support and np.max(samples.uncertainty) >= config.uncertainty_threshold
    )
    return RefinementDecision(
        active=bool(surface or near or semantic_boundary or detail or uncertain),
        surface=bool(surface),
        near_surface=near,
        semantic_boundary=semantic_boundary,
        thin_or_boundary=detail,
        uncertain=uncertain,
    )


@dataclass
class OctreeLeaf:
    bounds: Bounds
    depth: int
    decision: RefinementDecision


class AdaptiveOctreeSampler:
    """Recursive field sampler useful for Delaunay/tetrahedral extraction."""

    def __init__(
        self,
        field: SurfaceFieldAdapter | object,
        bounds: Bounds | Sequence[float],
        *,
        max_depth: int = 6,
        min_depth: int = 1,
        config: AdaptiveSamplingConfig | None = None,
    ) -> None:
        self.field = as_field_adapter(field)
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds.from_array(bounds)
        if max_depth < 0 or min_depth < 0 or min_depth > max_depth:
            raise ValueError("require 0 <= min_depth <= max_depth")
        self.max_depth = int(max_depth)
        self.min_depth = int(min_depth)
        self.config = config or AdaptiveSamplingConfig()

    def leaves(self) -> list[OctreeLeaf]:
        leaves: list[OctreeLeaf] = []
        current = [self.bounds]
        for depth in range(self.max_depth + 1):
            if not current:
                break
            probes = np.concatenate(
                [bounds.corners(include_center=True) for bounds in current], axis=0
            )
            unique_points, inverse = np.unique(probes, axis=0, return_inverse=True)
            unique_samples = self.field.query(unique_points)
            next_level: list[Bounds] = []
            for cell_index, bounds in enumerate(current):
                indices = inverse[cell_index * 9 : (cell_index + 1) * 9]
                samples = unique_samples.take(indices)
                decision = refinement_decision(samples, bounds.diagonal, self.config)
                should_split = depth < self.min_depth or (
                    decision.active and depth < self.max_depth
                )
                if should_split:
                    next_level.extend(bounds.split())
                else:
                    leaves.append(OctreeLeaf(bounds, depth, decision))
            current = next_level
        return leaves

    def sample(self, *, active_only: bool = True) -> SurfaceSamples:
        leaves = self.leaves()
        selected = [leaf for leaf in leaves if leaf.decision.active or not active_only]
        if not selected:
            selected = leaves
        points = np.concatenate(
            [leaf.bounds.corners(include_center=True) for leaf in selected], axis=0
        )
        scale = max(self.bounds.diagonal, 1.0)
        quantized = np.round(points / (scale * 1e-8)).astype(np.int64)
        _, indices = np.unique(quantized, axis=0, return_index=True)
        return self.field.query(points[np.sort(indices)])


@dataclass
class GridBlock:
    bounds: Bounds
    shape: tuple[int, int, int]
    spacing: np.ndarray
    samples: SurfaceSamples
    block_index: tuple[int, int, int]
    refinement_level: int
    decision: RefinementDecision

    def values(self, name: str) -> np.ndarray:
        value = getattr(self.samples, name)
        return value.reshape(self.shape + value.shape[1:])


class BlockedGridSampler:
    """Select coarse blocks, then densely sample only surface/detail blocks."""

    def __init__(
        self,
        field: SurfaceFieldAdapter | object,
        bounds: Bounds | Sequence[float],
        *,
        decision_field: SurfaceFieldAdapter | object | None = None,
        blocks_per_axis: int | Sequence[int] = 4,
        block_cells: int = 8,
        max_refinement: int = 2,
        config: AdaptiveSamplingConfig | None = None,
        fallback_all_blocks: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.field = as_field_adapter(field)
        # Coarse selection may use semantics, while dense lattice values use
        # the scalar geometry view.  This keeps semantic seams as refinement
        # evidence without decoding per-voxel region memberships.
        self.decision_field = (
            self.field
            if decision_field is None
            else as_field_adapter(decision_field)
        )
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds.from_array(bounds)
        if isinstance(blocks_per_axis, int):
            blocks = (blocks_per_axis,) * 3
        else:
            blocks = tuple(int(value) for value in blocks_per_axis)
        if len(blocks) != 3 or min(blocks) < 1:
            raise ValueError("blocks_per_axis must contain three positive values")
        if block_cells < 1 or max_refinement < 0:
            raise ValueError("block_cells must be positive and max_refinement non-negative")
        self.blocks_per_axis = blocks
        self.block_cells = int(block_cells)
        self.max_refinement = int(max_refinement)
        self.config = config or AdaptiveSamplingConfig()
        self.fallback_all_blocks = fallback_all_blocks
        self.progress_callback = progress_callback
        self.last_halo_blocks = 0

    def _emit_progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _coarse_blocks(self) -> Iterable[tuple[tuple[int, int, int], Bounds]]:
        step = self.bounds.extent / np.asarray(self.blocks_per_axis, dtype=np.float32)
        for index in product(*(range(value) for value in self.blocks_per_axis)):
            lower = self.bounds.minimum + step * np.asarray(index, dtype=np.float32)
            upper = lower + step
            yield tuple(index), Bounds(lower, upper)

    def _sample_dense(
        self,
        index: tuple[int, int, int],
        bounds: Bounds,
        level: int,
        decision: RefinementDecision,
    ) -> GridBlock:
        cells = self.block_cells * (2**level)
        shape = (cells + 1,) * 3
        axes = [
            np.linspace(bounds.minimum[axis], bounds.maximum[axis], shape[axis], dtype=np.float32)
            for axis in range(3)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grid, axis=-1).reshape(-1, 3)
        self._emit_progress(
            f"[mesh] sampling block {index} at level {level} ({len(points):,} points)"
        )
        reported_quarters = 0

        def report(completed: int, total: int) -> None:
            nonlocal reported_quarters
            quarter = min(4, (4 * completed) // max(total, 1))
            if quarter > reported_quarters:
                reported_quarters = quarter
                self._emit_progress(
                    f"[mesh] block {index}: {100 * completed // max(total, 1)}%"
                )

        samples = self.field.query(points, progress=report)
        spacing = bounds.extent / cells
        return GridBlock(bounds, shape, spacing, samples, index, level, decision)

    def _conforming_refinement_levels(
        self,
        candidates: Sequence[tuple[tuple[int, int, int], Bounds, RefinementDecision]],
    ) -> dict[tuple[int, int, int], int]:
        """Use one lattice resolution for every touching active component.

        Independent marching-cubes blocks with different face/edge sampling
        create T-junctions that coordinate welding cannot repair.  Propagating
        the maximum requested level over each face-connected sampled component
        is conservative, deterministic, and gives adjacent blocks identical
        boundary grids without introducing a second transition-cell mesher.
        """

        requested = {
            index: min(
                self.max_refinement,
                int(decision.surface or decision.near_surface) + decision.detail_count,
            )
            for index, _, decision in candidates
        }
        remaining = set(requested)
        resolved: dict[tuple[int, int, int], int] = {}
        offsets = (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        )
        while remaining:
            seed = remaining.pop()
            component = [seed]
            stack = [seed]
            while stack:
                current = stack.pop()
                for offset in offsets:
                    neighbor = tuple(current[axis] + offset[axis] for axis in range(3))
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            level = max(requested[index] for index in component)
            resolved.update({index: level for index in component})
        return resolved

    def sample_blocks(self) -> list[GridBlock]:
        self.last_halo_blocks = 0
        candidates: list[
            tuple[tuple[int, int, int], Bounds, RefinementDecision]
        ] = []
        all_decisions: list[
            tuple[tuple[int, int, int], Bounds, RefinementDecision]
        ] = []
        coarse = list(self._coarse_blocks())
        probe_points = np.concatenate(
            [bounds.corners(include_center=True) for _, bounds in coarse], axis=0
        )
        unique_points, inverse = np.unique(probe_points, axis=0, return_inverse=True)
        self._emit_progress(
            f"[mesh] selecting {len(coarse)} coarse blocks ({len(unique_points)} probes)"
        )
        unique_samples = self.decision_field.query(unique_points)
        for block_number, (index, bounds) in enumerate(coarse):
            probes = unique_samples.take(inverse[block_number * 9 : (block_number + 1) * 9])
            decision = refinement_decision(probes, bounds.diagonal, self.config)
            item = (index, bounds, decision)
            all_decisions.append(item)
            if decision.active:
                candidates.append(item)
        if not candidates and self.fallback_all_blocks:
            candidates = all_decisions
        elif candidates:
            # A one-block face halo prevents marching cubes from being clipped
            # exactly at the adaptive selection boundary.  It also connects
            # diagonal detail blocks through conforming face lattices.
            active = {index for index, _, _ in candidates}
            selected = set(active)
            available = {index for index, _, _ in all_decisions}
            for index in tuple(selected):
                for axis in range(3):
                    for direction in (-1, 1):
                        neighbor = list(index)
                        neighbor[axis] += direction
                        neighbor = tuple(neighbor)
                        if neighbor in available:
                            selected.add(neighbor)
            self.last_halo_blocks = len(selected - active)
            candidates = [item for item in all_decisions if item[0] in selected]

        levels = self._conforming_refinement_levels(candidates)
        estimated_points = sum(
            (self.block_cells * (2 ** levels[index]) + 1) ** 3
            for index, _, _ in candidates
        )
        self._emit_progress(
            f"[mesh] selected {len(candidates)} blocks "
            f"({self.last_halo_blocks} halo), {estimated_points:,} dense samples"
        )
        blocks = []
        for index, bounds, decision in candidates:
            blocks.append(self._sample_dense(index, bounds, levels[index], decision))
        return blocks

    def sample(self) -> SurfaceSamples:
        blocks = self.sample_blocks()
        if not blocks:
            raise RuntimeError("adaptive blocked sampler selected no blocks")
        samples = SurfaceSamples.concatenate([block.samples for block in blocks])
        scale = max(self.bounds.diagonal, 1.0)
        quantized = np.round(samples.points / (scale * 1e-8)).astype(np.int64)
        _, indices = np.unique(quantized, axis=0, return_index=True)
        return samples.take(np.sort(indices))
