"""High-precision extraction of the exact surface field used during training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .bounds import (
    MeshSupportPolicy,
    _quaternion_matrix,
    gaussian_support_bounds,
)
from .sampling import Bounds
from .types import TriangleMesh


ALGORITHM = "training_consistent_surface_field_marching_cubes"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingFieldMeshConfig:
    """Quality and memory policy for sparse high-resolution marching cubes."""

    resolution: int = 512
    block_cells: int = 32
    support_halo: str = "face"
    support_sigma: float = 3.0
    relative_padding: float = 0.02
    trim_quantile: float = 0.001
    min_opacity: float = 0.05
    min_semantic_confidence: float = 0.35
    require_observation: bool = True
    level: float = 0.0
    scout_resolution: int = 0
    scout_near_surface_voxels: float = 0.75
    complete_boundary_neighbors: bool = True
    query_chunk_size: int = 2_048
    projection_steps: int = 4
    projection_step_voxels: float = 0.5
    projection_tolerance_voxels: float = 0.01
    semantic_decode_chunk_size: int = 8_192
    min_component_faces: int = 64
    weld_tolerance_voxels: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "resolution",
            "block_cells",
            "query_chunk_size",
            "semantic_decode_chunk_size",
        ):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.resolution < self.block_cells:
            raise ValueError("resolution must be at least block_cells")
        if self.scout_resolution < 0:
            raise ValueError("scout_resolution must be non-negative")
        if self.scout_resolution and self.scout_resolution >= self.resolution:
            raise ValueError("scout_resolution must be smaller than resolution")
        if self.support_halo not in {"none", "face", "full"}:
            raise ValueError("support_halo must be none, face, or full")
        for name in (
            "support_sigma",
            "projection_step_voxels",
            "projection_tolerance_voxels",
            "weld_tolerance_voxels",
            "scout_near_surface_voxels",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "relative_padding",
            "trim_quantile",
            "min_opacity",
            "min_semantic_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.trim_quantile >= 0.5:
            raise ValueError("trim_quantile must be smaller than 0.5")
        if self.min_opacity > 1.0 or self.min_semantic_confidence > 1.0:
            raise ValueError("opacity and confidence thresholds must not exceed one")
        if not math.isfinite(float(self.level)):
            raise ValueError("level must be finite")
        if self.projection_steps < 0:
            raise ValueError("projection_steps must be non-negative")
        if self.min_component_faces < 0:
            raise ValueError("min_component_faces must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SparseBlockLayout:
    """One isotropic-resolution block layout over a rectangular scene."""

    bounds: Bounds
    blocks_per_axis: np.ndarray
    block_cells: int
    spacing: np.ndarray
    active_blocks: np.ndarray
    trusted_gaussians: int
    scout_stats: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        blocks = np.asarray(self.blocks_per_axis, dtype=np.int64).reshape(3)
        spacing = np.asarray(self.spacing, dtype=np.float32).reshape(3)
        active = np.asarray(self.active_blocks, dtype=np.int64).reshape(-1, 3)
        if np.any(blocks < 1) or self.block_cells < 1:
            raise ValueError("block layout dimensions must be positive")
        if np.any(spacing <= 0.0) or not np.isfinite(spacing).all():
            raise ValueError("block spacing must be finite and positive")
        if len(active) and (
            np.any(active < 0) or np.any(active >= blocks[None])
        ):
            raise ValueError("active block index lies outside the layout")
        object.__setattr__(self, "blocks_per_axis", blocks)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "active_blocks", active)

    @property
    def block_extent(self) -> np.ndarray:
        return self.bounds.extent / self.blocks_per_axis

    @property
    def nominal_resolution(self) -> np.ndarray:
        return self.blocks_per_axis * int(self.block_cells)

    @property
    def voxel_size(self) -> float:
        return float(np.max(self.spacing))

    @property
    def dense_sample_count(self) -> int:
        return int(len(self.active_blocks) * (self.block_cells + 1) ** 3)

    def block_bounds(self, index: np.ndarray) -> Bounds:
        index = np.asarray(index, dtype=np.int64).reshape(3)
        lower = self.bounds.minimum + self.block_extent * index
        upper = lower + self.block_extent
        return Bounds(lower.astype(np.float32), upper.astype(np.float32))

    def as_dict(self) -> dict[str, Any]:
        result = {
            "bounds": [
                *self.bounds.minimum.astype(float).tolist(),
                *self.bounds.maximum.astype(float).tolist(),
            ],
            "blocks_per_axis": self.blocks_per_axis.astype(int).tolist(),
            "block_cells": int(self.block_cells),
            "nominal_resolution": self.nominal_resolution.astype(int).tolist(),
            "spacing": self.spacing.astype(float).tolist(),
            "voxel_size": self.voxel_size,
            "active_blocks": int(len(self.active_blocks)),
            "dense_sample_count": self.dense_sample_count,
            "trusted_gaussians": int(self.trusted_gaussians),
        }
        if self.scout_stats is not None:
            result["scout"] = dict(self.scout_stats)
        return result


def _halo_offsets(mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros((1, 3), dtype=np.int64)
    if mode == "face":
        return np.asarray(
            (
                (0, 0, 0),
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ),
            dtype=np.int64,
        )
    return np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int64)


def _clean_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return faces
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    faces = faces[distinct]
    if not len(faces):
        return faces
    triangles = vertices[faces]
    area_twice = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    faces = faces[np.isfinite(area_twice) & (area_twice > 1e-12)]
    if not len(faces):
        return faces
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    return np.ascontiguousarray(faces[np.sort(first)], dtype=np.int64)


def _compact(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(faces):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (
        np.ascontiguousarray(vertices[used], dtype=np.float32),
        np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64),
    )


def _filter_small_components(
    faces: np.ndarray,
    *,
    minimum_faces: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter edge-connected components using a sparse face graph."""

    if minimum_faces <= 1 or not len(faces):
        return faces, {
            "components_before": 0 if not len(faces) else 1,
            "components_removed": 0,
        }
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    face_count = len(faces)
    edges = np.concatenate(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    owners = np.tile(np.arange(face_count, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    ordered_owners = owners[order]
    shared = np.all(ordered_edges[1:] == ordered_edges[:-1], axis=1)
    left = ordered_owners[:-1][shared]
    right = ordered_owners[1:][shared]
    if len(left):
        rows = np.concatenate((left, right))
        columns = np.concatenate((right, left))
        graph = coo_matrix(
            (
                np.ones(len(rows), dtype=np.uint8),
                (rows, columns),
            ),
            shape=(face_count, face_count),
        ).tocsr()
        component_count, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )
    else:
        component_count = face_count
        labels = np.arange(face_count, dtype=np.int32)
    counts = np.bincount(labels, minlength=component_count)
    keep_component = counts >= int(minimum_faces)
    if len(counts):
        keep_component[int(np.argmax(counts))] = True
    keep = keep_component[labels]
    return (
        np.ascontiguousarray(faces[keep], dtype=np.int64),
        {
            "components_before": int(component_count),
            "components_removed": int(np.count_nonzero(~keep_component)),
            "faces_removed_with_components": int(np.count_nonzero(~keep)),
        },
    )


class TrainingFieldMeshExtractor:
    """Extract one global mesh from ``SemanticSurfaceField.sdf == 0``."""

    def __init__(
        self,
        surface_field: Any,
        gaussians: Any,
        semantic_decoder: Any,
        *,
        config: Optional[TrainingFieldMeshConfig] = None,
        bounds: Bounds | Sequence[float] | None = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not callable(getattr(surface_field, "query_geometry", None)):
            raise TypeError(
                "training-consistent extraction requires "
                "SemanticSurfaceField.query_geometry"
            )
        if not callable(getattr(surface_field, "query", None)):
            raise TypeError("surface field must define query")
        if semantic_decoder is None or not callable(semantic_decoder):
            raise TypeError("semantic decoder must be restored from a checkpoint")
        self.surface_field = surface_field
        self.gaussians = gaussians
        self.semantic_decoder = semantic_decoder
        self.config = config or TrainingFieldMeshConfig()
        self.explicit_bounds = (
            None
            if bounds is None
            else bounds
            if isinstance(bounds, Bounds)
            else Bounds.from_array(bounds)
        )
        self.progress_callback = progress_callback
        self._trusted_indices: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.gaussians.get_xyz.device

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    @staticmethod
    def _block_mask(
        blocks: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros(tuple(blocks.astype(int).tolist()), dtype=bool)
        if len(indices):
            mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return mask

    def _scout_active_blocks(
        self,
        *,
        bounds: Bounds,
        blocks: np.ndarray,
        support_active: np.ndarray,
        trusted_xyz: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select target blocks from a coarse exact-field pass.

        Sign-changing coarse cells find ordinary surface sheets. Near-zero
        probes retain sheets thinner than one coarse cell, while exact queries
        at one trusted Gaussian center per occupied coarse cell seed small
        enclosed negative components that corner-only scouting could miss.
        """

        scout_resolution = int(self.config.scout_resolution)
        nominal_voxel = float(np.max(bounds.extent)) / scout_resolution
        cells = np.maximum(
            np.ceil(bounds.extent / nominal_voxel).astype(np.int64),
            1,
        )
        spacing = bounds.extent / cells
        axes = [
            np.linspace(
                bounds.minimum[axis],
                bounds.maximum[axis],
                int(cells[axis]) + 1,
                dtype=np.float32,
            )
            for axis in range(3)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grid, axis=-1).reshape(-1, 3)
        self._progress(
            "[mesh] scouting training field "
            f"{cells.tolist()} ({len(points):,} grid samples)"
        )
        volume = self._query_sdf(points).reshape(
            tuple((cells + 1).astype(int).tolist())
        )
        del points, grid

        minimum = np.full(tuple(cells.astype(int).tolist()), np.inf, dtype=np.float32)
        maximum = np.full(tuple(cells.astype(int).tolist()), -np.inf, dtype=np.float32)
        minimum_absolute = np.full_like(minimum, np.inf)
        for corner in product((0, 1), repeat=3):
            slices = tuple(
                slice(corner[axis], corner[axis] + int(cells[axis]))
                for axis in range(3)
            )
            values = volume[slices]
            np.minimum(minimum, values, out=minimum)
            np.maximum(maximum, values, out=maximum)
            np.minimum(
                minimum_absolute,
                np.abs(values - float(self.config.level)),
                out=minimum_absolute,
            )
        finite = np.isfinite(minimum) & np.isfinite(maximum)
        crossing = (
            finite
            & (minimum <= float(self.config.level))
            & (maximum >= float(self.config.level))
        )
        near_distance = (
            float(self.config.scout_near_surface_voxels)
            * float(np.max(spacing))
        )
        near = finite & (
            minimum_absolute
            <= near_distance
        )

        inside = np.all(
            (trusted_xyz >= bounds.minimum[None])
            & (trusted_xyz <= bounds.maximum[None]),
            axis=1,
        )
        center_xyz = trusted_xyz[inside]
        center_cells = np.floor(
            (center_xyz - bounds.minimum[None]) / spacing[None]
        ).astype(np.int64)
        center_cells = np.clip(center_cells, 0, cells - 1)
        flat_cells = np.ravel_multi_index(
            center_cells.T,
            tuple(cells.astype(int).tolist()),
        )
        _, first = np.unique(flat_cells, return_index=True)
        representative_cells = center_cells[first]
        representative_xyz = center_xyz[first]
        center_sdf = self._query_sdf(representative_xyz)
        negative_cells = representative_cells[
            np.isfinite(center_sdf)
            & (center_sdf <= float(self.config.level))
        ]
        seeded = np.zeros_like(crossing)
        if len(negative_cells):
            seeded[
                negative_cells[:, 0],
                negative_cells[:, 1],
                negative_cells[:, 2],
            ] = True

        selected_cells = np.argwhere(crossing | near | seeded)
        if not len(selected_cells):
            raise RuntimeError("coarse training-field scout found no surface support")
        target_extent = bounds.extent / blocks
        cell_minimum = (
            bounds.minimum[None] + selected_cells * spacing[None]
        )
        cell_maximum = cell_minimum + spacing[None]
        lower = np.floor(
            (cell_minimum - bounds.minimum[None]) / target_extent[None]
        ).astype(np.int64)
        upper = np.floor(
            (
                np.nextafter(cell_maximum, cell_minimum)
                - bounds.minimum[None]
            )
            / target_extent[None]
        ).astype(np.int64)
        lower = np.clip(lower, 0, blocks - 1)
        upper = np.clip(upper, 0, blocks - 1)
        block_parts = []
        for corner in product((0, 1), repeat=3):
            block_parts.append(
                np.stack(
                    [
                        upper[:, axis] if corner[axis] else lower[:, axis]
                        for axis in range(3)
                    ],
                    axis=1,
                )
            )
        narrow = np.unique(np.concatenate(block_parts, axis=0), axis=0)
        candidates = narrow[:, None, :] + _halo_offsets(
            self.config.support_halo
        )[None, :, :]
        candidates = candidates.reshape(-1, 3)
        valid = np.all(
            (candidates >= 0) & (candidates < blocks[None]),
            axis=1,
        )
        narrow = np.unique(candidates[valid], axis=0)
        active_mask = self._block_mask(blocks, support_active)
        narrow_mask = self._block_mask(blocks, narrow)
        active = np.argwhere(narrow_mask)
        if not len(active):
            raise RuntimeError("training-field scout found no target blocks")
        return active, {
            "resolution": scout_resolution,
            "cells": cells.astype(int).tolist(),
            "grid_samples": int(volume.size),
            "near_surface_distance": float(near_distance),
            "crossing_cells": int(np.count_nonzero(crossing)),
            "near_cells": int(np.count_nonzero(near)),
            "negative_seed_cells": int(len(negative_cells)),
            "selected_cells": int(len(selected_cells)),
            "candidate_target_blocks": int(len(narrow)),
            "active_target_blocks": int(len(active)),
            "support_overlap_blocks": int(
                np.count_nonzero(active_mask & narrow_mask)
            ),
        }

    @torch.no_grad()
    def _trusted_support(self) -> Tensor:
        if self._trusted_indices is None:
            policy = MeshSupportPolicy(
                min_opacity=self.config.min_opacity,
                min_semantic_confidence=self.config.min_semantic_confidence,
                require_observation=self.config.require_observation,
                trim_quantile=self.config.trim_quantile,
                chunk_size=max(self.config.query_chunk_size, 65_536),
            )
            selected = policy.selected_indices(self.gaussians)
            if len(selected) < 3:
                raise RuntimeError("fewer than three trusted Gaussian supports")
            self._trusted_indices = selected
        return self._trusted_indices

    @torch.no_grad()
    def build_layout(self) -> SparseBlockLayout:
        selected = self._trusted_support()
        bounds = self.explicit_bounds
        if bounds is None:
            bounds = gaussian_support_bounds(
                self.gaussians,
                sigma=self.config.support_sigma,
                relative_padding=self.config.relative_padding,
                selection=selected,
                trim_quantile=self.config.trim_quantile,
            )
        maximum_extent = float(np.max(bounds.extent))
        nominal_voxel = maximum_extent / float(self.config.resolution)
        target_cells = np.maximum(
            np.ceil(bounds.extent / nominal_voxel).astype(np.int64),
            1,
        )
        blocks = np.maximum(
            np.ceil(target_cells / int(self.config.block_cells)).astype(np.int64),
            1,
        )
        spacing = bounds.extent / (blocks * int(self.config.block_cells))

        model_indices = selected.to(self.device)
        xyz = (
            self.gaussians.get_xyz.index_select(0, model_indices)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        scales = (
            self.gaussians.get_scaling.index_select(0, model_indices)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        rotations = (
            self.gaussians.get_rotation.index_select(0, model_indices)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        matrix = _quaternion_matrix(rotations)
        support_extent = float(self.config.support_sigma) * np.einsum(
            "nij,nj->ni",
            np.abs(matrix),
            scales,
        )
        support_minimum = xyz - support_extent
        support_maximum = xyz + support_extent
        finite = (
            np.isfinite(support_minimum).all(axis=1)
            & np.isfinite(support_maximum).all(axis=1)
        )
        intersects = (
            finite
            & np.all(support_maximum >= bounds.minimum[None], axis=1)
            & np.all(support_minimum <= bounds.maximum[None], axis=1)
        )
        support_minimum = support_minimum[intersects]
        support_maximum = support_maximum[intersects]
        if not len(support_minimum):
            raise RuntimeError("trusted Gaussian supports do not intersect mesh bounds")

        block_extent = bounds.extent / blocks
        lower = np.floor(
            (support_minimum - bounds.minimum[None]) / block_extent[None]
        ).astype(np.int64)
        upper = np.floor(
            (support_maximum - bounds.minimum[None]) / block_extent[None]
        ).astype(np.int64)
        lower = np.clip(lower, 0, blocks - 1)
        upper = np.clip(upper, 0, blocks - 1)
        # Rasterize every rotated 3-sigma AABB with a 3-D difference array.
        # The lattice contains only a few thousand blocks even for a high
        # resolution scene, so this remains bounded while preventing large or
        # anisotropic splats from being truncated by center-only activation.
        difference_shape = tuple((blocks + 1).astype(int).tolist())
        difference = np.zeros(difference_shape, dtype=np.int64)
        for corner in product((0, 1), repeat=3):
            indices = np.stack(
                [
                    upper[:, axis] + 1 if corner[axis] else lower[:, axis]
                    for axis in range(3)
                ],
                axis=1,
            )
            sign = -1 if sum(corner) % 2 else 1
            np.add.at(
                difference,
                (indices[:, 0], indices[:, 1], indices[:, 2]),
                sign,
            )
        occupancy = difference.cumsum(0).cumsum(1).cumsum(2)
        base = np.argwhere(
            occupancy[: blocks[0], : blocks[1], : blocks[2]] > 0
        )
        scout_stats = None
        if self.config.scout_resolution:
            active, scout_stats = self._scout_active_blocks(
                bounds=bounds,
                blocks=blocks,
                support_active=base,
                trusted_xyz=xyz,
            )
        else:
            candidates = base[:, None, :] + _halo_offsets(
                self.config.support_halo
            )[None, :, :]
            candidates = candidates.reshape(-1, 3)
            valid = np.all(
                (candidates >= 0) & (candidates < blocks[None]),
                axis=1,
            )
            active = np.unique(candidates[valid], axis=0)
        order = np.lexsort((active[:, 2], active[:, 1], active[:, 0]))
        active = np.ascontiguousarray(active[order], dtype=np.int64)
        layout = SparseBlockLayout(
            bounds=bounds,
            blocks_per_axis=blocks,
            block_cells=int(self.config.block_cells),
            spacing=spacing,
            active_blocks=active,
            trusted_gaussians=int(len(selected)),
            scout_stats=scout_stats,
        )
        self._progress(
            "[mesh] layout "
            f"{layout.nominal_resolution.tolist()}, "
            f"{len(layout.active_blocks):,} active blocks, "
            f"{layout.dense_sample_count:,} field samples"
        )
        return layout

    def _query_sdf(self, points: np.ndarray) -> np.ndarray:
        """Query only SDF values with a hard outer memory bound."""

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(points), self.config.query_chunk_size):
                end = min(start + self.config.query_chunk_size, len(points))
                tensor = torch.as_tensor(
                    points[start:end],
                    device=self.device,
                    dtype=torch.float32,
                )
                query = self.surface_field.query_geometry(
                    tensor,
                    chunk_size=self.config.query_chunk_size,
                )
                outputs.append(query.sdf.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    @torch.no_grad()
    def _sample_zero_set(
        self,
        layout: SparseBlockLayout,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        from skimage.measure import marching_cubes

        vertex_parts: list[np.ndarray] = []
        face_parts: list[np.ndarray] = []
        crossing_blocks = 0
        vertex_offset = 0
        cells = int(layout.block_cells)
        shape = (cells + 1,) * 3
        pending = [
            tuple(int(value) for value in row)
            for row in layout.active_blocks.tolist()
        ]
        scheduled = set(pending)
        initial_blocks = len(pending)
        dynamically_added = 0
        block_number = 0
        while block_number < len(pending):
            block_index = np.asarray(pending[block_number], dtype=np.int64)
            block_number += 1
            bounds = layout.block_bounds(block_index)
            axes = [
                np.linspace(
                    bounds.minimum[axis],
                    bounds.maximum[axis],
                    cells + 1,
                    dtype=np.float32,
                )
                for axis in range(3)
            ]
            grid = np.meshgrid(*axes, indexing="ij")
            points = np.stack(grid, axis=-1).reshape(-1, 3)
            volume = self._query_sdf(points).reshape(shape)
            finite = np.isfinite(volume)
            if not bool(finite.any()):
                continue
            if not bool(finite.all()):
                finite_values = volume[finite]
                below = min(
                    float(finite_values.min()),
                    float(self.config.level) - layout.voxel_size,
                )
                above = max(
                    float(finite_values.max()),
                    float(self.config.level) + layout.voxel_size,
                )
                volume = np.nan_to_num(
                    volume,
                    nan=above,
                    posinf=above,
                    neginf=below,
                )

            if self.config.complete_boundary_neighbors:
                for axis in range(3):
                    for side, position in ((-1, 0), (1, -1)):
                        face = np.take(volume, position, axis=axis)
                        if not (
                            float(face.min()) <= float(self.config.level)
                            <= float(face.max())
                        ):
                            continue
                        neighbor = block_index.copy()
                        neighbor[axis] += side
                        if np.any(neighbor < 0) or np.any(
                            neighbor >= layout.blocks_per_axis
                        ):
                            continue
                        key = tuple(int(value) for value in neighbor)
                        if key not in scheduled:
                            scheduled.add(key)
                            pending.append(key)
                            dynamically_added += 1
            if (
                float(volume.min()) > self.config.level
                or float(volume.max()) < self.config.level
                or np.isclose(float(volume.min()), float(volume.max()))
            ):
                continue
            vertices, faces, _, _ = marching_cubes(
                volume,
                level=float(self.config.level),
                spacing=tuple(float(value) for value in layout.spacing),
                allow_degenerate=False,
            )
            if not len(faces):
                continue
            vertices = (
                vertices.astype(np.float32, copy=False)
                + bounds.minimum[None]
            )
            vertex_parts.append(vertices)
            face_parts.append(
                faces.astype(np.int64, copy=False) + vertex_offset
            )
            vertex_offset += len(vertices)
            crossing_blocks += 1
            if (
                block_number == len(pending)
                or block_number % 100 == 0
            ):
                self._progress(
                    f"[mesh] sampled blocks {block_number:,}/"
                    f"{len(pending):,}; crossing {crossing_blocks:,}; "
                    f"boundary-added {dynamically_added:,}"
                )
        if not face_parts:
            raise RuntimeError("training surface field produced no crossing blocks")
        return (
            np.concatenate(vertex_parts, axis=0),
            np.concatenate(face_parts, axis=0),
            {
                "initial_blocks": int(initial_blocks),
                "sampled_blocks": int(len(pending)),
                "crossing_blocks": int(crossing_blocks),
                "boundary_added_blocks": int(dynamically_added),
            },
        )

    def _weld(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        tolerance = max(
            float(voxel_size) * self.config.weld_tolerance_voxels,
            8.0
            * np.finfo(np.float32).eps
            * max(float(np.abs(vertices).max()), 1.0),
        )
        origin = vertices.min(axis=0, keepdims=True)
        keys = np.rint((vertices - origin) / tolerance).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        weights = np.bincount(inverse, minlength=count).astype(np.float64)
        welded = np.zeros((count, 3), dtype=np.float64)
        for axis in range(3):
            welded[:, axis] = np.bincount(
                inverse,
                weights=vertices[:, axis],
                minlength=count,
            )
        welded = (welded / weights[:, None]).astype(np.float32)
        clean_faces = _clean_faces(welded, inverse[faces])
        return _compact(welded, clean_faces)

    @torch.no_grad()
    def _project_vertices(
        self,
        vertices: np.ndarray,
        *,
        voxel_size: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        result = np.ascontiguousarray(vertices, dtype=np.float32).copy()
        maximum_step = (
            float(voxel_size) * self.config.projection_step_voxels
        )
        tolerance = (
            float(voxel_size) * self.config.projection_tolerance_voxels
        )
        before_values: list[np.ndarray] = []
        after_values: list[np.ndarray] = []
        for start in range(0, len(result), self.config.query_chunk_size):
            end = min(start + self.config.query_chunk_size, len(result))
            points = torch.as_tensor(
                result[start:end],
                device=self.device,
                dtype=torch.float32,
            )
            query = self.surface_field.query_geometry(
                points,
                chunk_size=self.config.query_chunk_size,
            )
            residual = query.sdf - float(self.config.level)
            normal = query.normal
            initial = residual.abs().detach().cpu().numpy()
            movable = torch.ones_like(residual, dtype=torch.bool)
            for _ in range(self.config.projection_steps):
                finite = torch.isfinite(residual) & torch.isfinite(normal).all(dim=1)
                normal_length = normal.norm(dim=1)
                active = (
                    movable
                    & finite
                    & (normal_length > 1e-6)
                    & (residual.abs() > tolerance)
                )
                if not bool(active.any()):
                    break
                unit_normal = normal / normal_length.clamp_min(1e-8)[:, None]
                step = residual.clamp(-maximum_step, maximum_step)
                candidate_points = torch.where(
                    active[:, None],
                    points - step[:, None] * unit_normal,
                    points,
                )
                candidate = self.surface_field.query_geometry(
                    candidate_points,
                    chunk_size=self.config.query_chunk_size,
                )
                candidate_residual = candidate.sdf - float(self.config.level)
                accepted = (
                    active
                    & torch.isfinite(candidate_residual)
                    & (candidate_residual.abs() <= residual.abs())
                )
                points = torch.where(
                    accepted[:, None],
                    candidate_points,
                    points,
                )
                residual = torch.where(
                    accepted,
                    candidate_residual,
                    residual,
                )
                normal = torch.where(
                    accepted[:, None],
                    candidate.normal,
                    normal,
                )
                # A rejected bounded Newton step is frozen rather than being
                # allowed to oscillate or increase the training-field residual.
                movable &= accepted
            result[start:end] = points.detach().cpu().numpy()
            before_values.append(initial)
            after_values.append(residual.abs().detach().cpu().numpy())
            if end == len(result) or end % (self.config.query_chunk_size * 25) == 0:
                self._progress(
                    f"[mesh] projected vertices {end:,}/{len(result):,}"
                )
        before = np.concatenate(before_values)
        after = np.concatenate(after_values)
        return result, {
            "sdf_abs_before_mean": float(before.mean()),
            "sdf_abs_before_p90": float(np.quantile(before, 0.90)),
            "sdf_abs_before_p99": float(np.quantile(before, 0.99)),
            "sdf_abs_after_mean": float(after.mean()),
            "sdf_abs_after_p90": float(np.quantile(after, 0.90)),
            "sdf_abs_after_p99": float(np.quantile(after, 0.99)),
        }

    @torch.no_grad()
    def _vertex_attributes(
        self,
        vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        normals: list[np.ndarray] = []
        semantics: list[np.ndarray] = []
        uncertainty: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        for start in range(0, len(vertices), self.config.query_chunk_size):
            end = min(start + self.config.query_chunk_size, len(vertices))
            points = torch.as_tensor(
                vertices[start:end],
                device=self.device,
                dtype=torch.float32,
            )
            query = self.surface_field.query(
                points,
                chunk_size=self.config.query_chunk_size,
            )
            normals.append(query.normal.detach().float().cpu().numpy())
            semantics.append(query.semantic.detach().float().cpu().numpy())
            uncertainty.append(
                query.uncertainty.detach().float().cpu().numpy()
            )
            residuals.append(
                (query.sdf - float(self.config.level))
                .abs()
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            if end == len(vertices) or end % (self.config.query_chunk_size * 25) == 0:
                self._progress(
                    f"[mesh] queried attributes {end:,}/{len(vertices):,}"
                )
        normal = np.concatenate(normals, axis=0)
        normal /= np.maximum(
            np.linalg.norm(normal, axis=1, keepdims=True),
            1e-8,
        )
        semantic = np.concatenate(semantics, axis=0)
        uncertainty_array = np.concatenate(uncertainty, axis=0)
        residual = np.concatenate(residuals, axis=0)

        semantic_ids: list[np.ndarray] = []
        parameters = getattr(self.semantic_decoder, "parameters", None)
        decoder_parameter = (
            next(iter(parameters()), None) if callable(parameters) else None
        )
        decoder_device = (
            self.device
            if decoder_parameter is None
            else decoder_parameter.device
        )
        for start in range(
            0,
            len(semantic),
            self.config.semantic_decode_chunk_size,
        ):
            chunk = torch.as_tensor(
                semantic[start : start + self.config.semantic_decode_chunk_size],
                device=decoder_device,
                dtype=torch.float32,
            )
            logits = self.semantic_decoder(chunk)
            if (
                not isinstance(logits, Tensor)
                or logits.ndim != 2
                or logits.shape[0] != len(chunk)
                or logits.shape[1] < 1
            ):
                raise ValueError(
                    "semantic decoder must return logits with shape [N,C]"
                )
            semantic_ids.append(
                logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
            )
        return (
            normal.astype(np.float32, copy=False),
            semantic.astype(np.float32, copy=False),
            np.concatenate(semantic_ids),
            uncertainty_array.astype(np.float32, copy=False),
            {
                "vertex_sdf_abs_mean": float(residual.mean()),
                "vertex_sdf_abs_p90": float(np.quantile(residual, 0.90)),
                "vertex_sdf_abs_p99": float(np.quantile(residual, 0.99)),
                "vertex_uncertainty_mean": float(uncertainty_array.mean()),
                "vertex_uncertainty_median": float(
                    np.median(uncertainty_array)
                ),
            },
        )

    @staticmethod
    def _orient_faces(
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        triangles = vertices[faces]
        face_normal = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        target = normals[faces].sum(axis=1)
        alignment = np.sum(face_normal * target, axis=1)
        finite = np.isfinite(alignment)
        flip_all = bool(
            np.count_nonzero(finite)
            and float(np.median(alignment[finite])) < 0.0
        )
        result = faces.copy()
        if flip_all:
            result = result[:, (0, 2, 1)]
        return result, int(len(result) if flip_all else 0)

    def extract(
        self,
        *,
        layout_only: bool = False,
    ) -> tuple[TriangleMesh | None, SparseBlockLayout]:
        layout = self.build_layout()
        if layout_only:
            return None, layout

        vertices, faces, sampling_stats = self._sample_zero_set(layout)
        raw_vertices = len(vertices)
        raw_faces = len(faces)
        vertices, faces = self._weld(
            vertices,
            faces,
            layout.voxel_size,
        )
        welded_vertices = len(vertices)
        faces, component_stats = _filter_small_components(
            faces,
            minimum_faces=self.config.min_component_faces,
        )
        vertices, faces = _compact(vertices, faces)
        if not len(faces):
            raise RuntimeError("mesh cleanup removed every surface face")

        projection_stats: dict[str, float] = {}
        if self.config.projection_steps:
            vertices, projection_stats = self._project_vertices(
                vertices,
                voxel_size=layout.voxel_size,
            )
            faces = _clean_faces(vertices, faces)
            vertices, faces = _compact(vertices, faces)
            if not len(faces):
                raise RuntimeError(
                    "training-field projection removed every surface face"
                )

        (
            normals,
            semantic,
            semantic_id,
            uncertainty,
            field_stats,
        ) = self._vertex_attributes(vertices)
        faces, flipped_faces = self._orient_faces(
            vertices,
            faces,
            normals,
        )
        vertex_regions = semantic_id[faces]
        unanimous = np.all(
            vertex_regions == vertex_regions[:, :1],
            axis=1,
        )
        face_region_id = np.where(
            unanimous,
            vertex_regions[:, 0],
            -2,
        ).astype(np.int32)
        metadata: dict[str, Any] = {
            "algorithm": ALGORITHM,
            "field": "SemanticSurfaceField.query_geometry().sdf",
            "field_level": float(self.config.level),
            "raw_vertices": int(raw_vertices),
            "raw_faces": int(raw_faces),
            "welded_vertices": int(welded_vertices),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "flipped_faces": int(flipped_faces),
            "mixed_semantic_faces": int(np.count_nonzero(~unanimous)),
            "layout": layout.as_dict(),
            **sampling_stats,
            **component_stats,
            **projection_stats,
            **field_stats,
        }
        mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            semantic=semantic,
            semantic_id=semantic_id,
            uncertainty=uncertainty,
            face_region_id=face_region_id,
            metadata=metadata,
        )
        self._progress(
            f"[mesh] final training-field mesh: "
            f"{len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces"
        )
        return mesh, layout


__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "SparseBlockLayout",
    "TrainingFieldMeshConfig",
    "TrainingFieldMeshExtractor",
]
