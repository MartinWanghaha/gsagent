"""Per-chart Delaunay topology and shared marching-tetrahedra roots.

Semantic regions decide which pivots may compete for local topology.  The
scalar field remains global, so every chart cuts the same renderer-consistent
zero set and overlapping charts share exactly the same edge roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor


_TETRA_EDGES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    dtype=np.int64,
)
_TRIANGLE_TABLE = np.asarray(
    (
        (-1, -1, -1, -1, -1, -1),
        (1, 0, 2, -1, -1, -1),
        (4, 0, 3, -1, -1, -1),
        (1, 4, 2, 1, 3, 4),
        (3, 1, 5, -1, -1, -1),
        (2, 3, 0, 2, 5, 3),
        (1, 4, 0, 1, 5, 4),
        (4, 2, 5, -1, -1, -1),
        (4, 5, 2, -1, -1, -1),
        (4, 1, 0, 4, 5, 1),
        (3, 2, 0, 3, 5, 2),
        (1, 3, 5, -1, -1, -1),
        (4, 1, 2, 4, 3, 1),
        (3, 0, 4, -1, -1, -1),
        (2, 0, 1, -1, -1, -1),
        (-1, -1, -1, -1, -1, -1),
    ),
    dtype=np.int64,
)
_TRIANGLE_COUNT = np.asarray(
    (0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0),
    dtype=np.int64,
)


@dataclass(frozen=True)
class RegionTetrahedralConfig:
    """Numerical policy for local topology and global roots."""

    level: float = 0.0
    max_crossing_edge_factor: float = 2.0
    binary_steps: int = 10
    query_chunk_size: int = 65_536
    qhull_options: str = "Qbb Qc Q12 QJ"
    deduplication_fraction: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.level):
            raise ValueError("level must be finite")
        if (
            not math.isfinite(self.max_crossing_edge_factor)
            or self.max_crossing_edge_factor <= 0.0
        ):
            raise ValueError("max_crossing_edge_factor must be positive")
        if self.binary_steps < 0:
            raise ValueError("binary_steps must be non-negative")
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        if (
            not math.isfinite(self.deduplication_fraction)
            or self.deduplication_fraction <= 0.0
        ):
            raise ValueError("deduplication_fraction must be positive")


@dataclass(frozen=True)
class ChartSurface:
    """A chart surface whose vertices are canonical global pivot edges."""

    edge_keys: np.ndarray
    faces: np.ndarray
    region_id: int
    chart_id: int
    tetrahedra: int

    def __post_init__(self) -> None:
        edges = np.ascontiguousarray(np.asarray(self.edge_keys, dtype=np.int64))
        faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64))
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edge_keys must have shape [E,2]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [F,3]")
        if len(edges) and np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("edge_keys must be canonical increasing pairs")
        if faces.size and (faces.min() < 0 or faces.max() >= len(edges)):
            raise ValueError("face index is outside edge_keys")
        object.__setattr__(self, "edge_keys", edges)
        object.__setattr__(self, "faces", faces)

    @classmethod
    def empty(cls, region_id: int, chart_id: int) -> "ChartSurface":
        return cls(
            edge_keys=np.empty((0, 2), dtype=np.int64),
            faces=np.empty((0, 3), dtype=np.int64),
            region_id=int(region_id),
            chart_id=int(chart_id),
            tetrahedra=0,
        )


@dataclass(frozen=True)
class SharedTopology:
    """All charts expressed on one deduplicated edge-root index."""

    edge_keys: np.ndarray
    faces: np.ndarray
    face_region_id: np.ndarray
    chart_face_count: dict[int, int]

    def __post_init__(self) -> None:
        edges = np.ascontiguousarray(np.asarray(self.edge_keys, dtype=np.int64))
        faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64))
        regions = np.ascontiguousarray(
            np.asarray(self.face_region_id, dtype=np.int32).reshape(-1)
        )
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edge_keys must have shape [E,2]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [F,3]")
        if len(regions) != len(faces):
            raise ValueError("face_region_id must have shape [F]")
        if faces.size and (faces.min() < 0 or faces.max() >= len(edges)):
            raise ValueError("face index is outside edge_keys")
        object.__setattr__(self, "edge_keys", edges)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "face_region_id", regions)


@dataclass(frozen=True)
class RefinedRoots:
    """Renderer-consistent roots of all unique sign-changing pivot edges."""

    vertices: Tensor
    interpolation: Tensor
    valid: Tensor
    confidence: Tensor

    def __post_init__(self) -> None:
        count = len(self.vertices)
        if self.vertices.shape != (count, 3):
            raise ValueError("vertices must have shape [V,3]")
        if self.interpolation.shape != (count,):
            raise ValueError("interpolation must have shape [V]")
        if self.valid.shape != (count,) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [V]")
        if self.confidence.shape != (count,):
            raise ValueError("confidence must have shape [V]")
        if not bool(torch.isfinite(self.vertices).all()):
            raise ValueError("root vertices must be finite")


def _deduplicated_chart_pivots(
    global_indices: np.ndarray,
    points: np.ndarray,
    *,
    scene_extent: float,
    fraction: float,
) -> np.ndarray:
    """Keep one stable global pivot for each numerically identical position."""

    if not len(global_indices):
        return global_indices
    tolerance = max(float(scene_extent) * float(fraction), np.finfo(np.float64).eps)
    selected = np.asarray(points[global_indices], dtype=np.float64)
    origin = selected.min(axis=0, keepdims=True)
    keys = np.rint((selected - origin) / tolerance).astype(np.int64)
    order = np.lexsort(
        (global_indices, keys[:, 2], keys[:, 1], keys[:, 0])
    )
    ordered_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    return np.ascontiguousarray(global_indices[order[first]], dtype=np.int64)


def delaunay_chart(
    *,
    chart_id: int,
    region_id: int,
    pivot_indices: np.ndarray,
    pivot_owner_chart: np.ndarray,
    points: np.ndarray,
    phi: np.ndarray,
    valid: np.ndarray,
    radius: np.ndarray,
    scene_extent: float,
    config: RegionTetrahedralConfig,
) -> ChartSurface:
    """Delaunay one semantic chart, then cut and filter only crossing roots."""

    from scipy.spatial import Delaunay, QhullError

    requested = np.ascontiguousarray(
        np.asarray(pivot_indices, dtype=np.int64).reshape(-1)
    )
    if len(requested) < 4:
        return ChartSurface.empty(region_id, chart_id)
    unique_global = _deduplicated_chart_pivots(
        requested,
        points,
        scene_extent=scene_extent,
        fraction=config.deduplication_fraction,
    )
    if len(unique_global) < 4:
        return ChartSurface.empty(region_id, chart_id)
    local_points = np.asarray(points[unique_global], dtype=np.float64)
    if np.linalg.matrix_rank(local_points - local_points.mean(axis=0)) < 3:
        return ChartSurface.empty(region_id, chart_id)
    try:
        tetrahedra = Delaunay(
            local_points,
            qhull_options=config.qhull_options,
        ).simplices
    except QhullError:
        return ChartSurface.empty(region_id, chart_id)
    tetrahedra = np.ascontiguousarray(tetrahedra, dtype=np.int64)
    tetrahedra = tetrahedra[
        (tetrahedra >= 0).all(axis=1)
        & (tetrahedra < len(local_points)).all(axis=1)
    ]
    if not len(tetrahedra):
        return ChartSurface.empty(region_id, chart_id)

    local_phi = np.asarray(phi[unique_global], dtype=np.float64)
    local_valid = np.asarray(valid[unique_global], dtype=bool)
    inside = local_phi <= float(config.level)
    case = np.sum(
        inside[tetrahedra] * (1 << np.arange(4, dtype=np.int64))[None],
        axis=1,
    )
    active = (
        (case != 0)
        & (case != 15)
        & local_valid[tetrahedra].all(axis=1)
    )
    tetrahedra = tetrahedra[active]
    case = case[active]
    if not len(tetrahedra):
        return ChartSurface.empty(region_id, chart_id)

    tetra_edges = tetrahedra[:, _TETRA_EDGES]
    canonical_local = np.sort(tetra_edges, axis=-1)
    flat_edges = canonical_local.reshape(-1, 2)
    unique_edges, inverse = np.unique(flat_edges, axis=0, return_inverse=True)
    crossing = inside[unique_edges[:, 0]] != inside[unique_edges[:, 1]]
    edge_to_root = np.full(len(unique_edges), -1, dtype=np.int64)
    edge_to_root[crossing] = np.arange(int(crossing.sum()), dtype=np.int64)
    tet_roots = edge_to_root[inverse].reshape(-1, 6)
    root_local_edges = unique_edges[crossing]

    one = _TRIANGLE_COUNT[case] == 1
    two = _TRIANGLE_COUNT[case] == 2
    faces: list[np.ndarray] = []
    if np.any(one):
        faces.append(
            np.take_along_axis(
                tet_roots[one],
                _TRIANGLE_TABLE[case[one], :3],
                axis=1,
            )
        )
    if np.any(two):
        faces.append(
            np.take_along_axis(
                tet_roots[two],
                _TRIANGLE_TABLE[case[two], :6],
                axis=1,
            ).reshape(-1, 3)
        )
    if not faces:
        return ChartSurface.empty(region_id, chart_id)
    root_faces = np.concatenate(faces, axis=0)
    root_global_edges = np.sort(unique_global[root_local_edges], axis=1)

    distances = np.linalg.norm(
        points[root_global_edges[:, 0]] - points[root_global_edges[:, 1]],
        axis=1,
    )
    support = (
        radius[root_global_edges[:, 0]]
        + radius[root_global_edges[:, 1]]
    )
    root_allowed = (
        np.isfinite(distances)
        & np.isfinite(support)
        & (support > 0.0)
        & (
            distances
            <= float(config.max_crossing_edge_factor) * support
        )
    )
    face_allowed = root_allowed[root_faces].all(axis=1)

    owners = np.asarray(pivot_owner_chart, dtype=np.int64).reshape(-1)
    if len(owners) != len(points):
        raise ValueError("pivot_owner_chart must align with global pivots")
    root_touches_core = (owners[root_global_edges] == int(chart_id)).any(axis=1)
    face_allowed &= root_touches_core[root_faces].any(axis=1)
    root_faces = root_faces[face_allowed]
    if not len(root_faces):
        return ChartSurface.empty(region_id, chart_id)

    used, remap = np.unique(root_faces.reshape(-1), return_inverse=True)
    return ChartSurface(
        edge_keys=root_global_edges[used],
        faces=remap.reshape(-1, 3),
        region_id=int(region_id),
        chart_id=int(chart_id),
        tetrahedra=int(len(tetrahedra)),
    )


def merge_chart_surfaces(charts: list[ChartSurface]) -> SharedTopology:
    """Weld overlap at canonical pivot edges and remove duplicate triangles."""

    populated = [chart for chart in charts if len(chart.faces)]
    if not populated:
        return SharedTopology(
            edge_keys=np.empty((0, 2), dtype=np.int64),
            faces=np.empty((0, 3), dtype=np.int64),
            face_region_id=np.empty((0,), dtype=np.int32),
            chart_face_count={},
        )
    all_edges = np.concatenate([chart.edge_keys for chart in populated], axis=0)
    edge_keys, edge_inverse = np.unique(all_edges, axis=0, return_inverse=True)
    faces: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    chart_face_count: dict[int, int] = {}
    offset = 0
    for chart in populated:
        mapping = edge_inverse[offset : offset + len(chart.edge_keys)]
        chart_faces = mapping[chart.faces]
        faces.append(chart_faces)
        regions.append(
            np.full(len(chart_faces), chart.region_id, dtype=np.int32)
        )
        chart_face_count[chart.chart_id] = len(chart_faces)
        offset += len(chart.edge_keys)
    merged_faces = np.concatenate(faces, axis=0)
    merged_regions = np.concatenate(regions, axis=0)

    canonical = np.sort(merged_faces, axis=1)
    _, first, inverse = np.unique(
        canonical,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first)
    minimum = np.full(len(first), np.iinfo(np.int32).max, dtype=np.int32)
    maximum = np.full(len(first), np.iinfo(np.int32).min, dtype=np.int32)
    np.minimum.at(minimum, inverse, merged_regions)
    np.maximum.at(maximum, inverse, merged_regions)
    ownership = np.where(minimum == maximum, minimum, -2).astype(np.int32)
    return SharedTopology(
        edge_keys=edge_keys,
        faces=merged_faces[first[order]],
        face_region_id=ownership[order],
        chart_face_count=chart_face_count,
    )


@torch.no_grad()
def refine_shared_roots(
    topology: SharedTopology,
    *,
    pivot_points: Tensor,
    pivot_view_ids: Optional[Tensor],
    field: Any,
    config: RegionTetrahedralConfig,
) -> RefinedRoots:
    """Candidate-first refine every canonical edge exactly once."""

    device = pivot_points.device
    dtype = pivot_points.dtype
    edges = torch.as_tensor(topology.edge_keys, device=device, dtype=torch.long)
    if not len(edges):
        return RefinedRoots(
            vertices=torch.empty((0, 3), device=device, dtype=dtype),
            interpolation=torch.empty((0,), device=device, dtype=dtype),
            valid=torch.empty((0,), device=device, dtype=torch.bool),
            confidence=torch.empty((0,), device=device, dtype=dtype),
        )
    endpoints = pivot_points[edges]
    if pivot_view_ids is None or not pivot_view_ids.numel():
        raise ValueError("shared roots require controlling pivot view IDs")
    candidate_views = pivot_view_ids[edges].reshape(len(edges), -1)
    refined = field.refine_edges(
        endpoints,
        candidate_view_ids=candidate_views,
        binary_steps=config.binary_steps,
        chunk_size=config.query_chunk_size,
    )
    return RefinedRoots(
        vertices=torch.as_tensor(
            refined.vertices,
            device=device,
            dtype=dtype,
        ),
        interpolation=torch.as_tensor(
            refined.interpolation,
            device=device,
            dtype=dtype,
        ),
        valid=torch.as_tensor(
            refined.valid,
            device=device,
            dtype=torch.bool,
        ),
        confidence=torch.as_tensor(
            refined.confidence,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0),
    )


def filter_invalid_root_faces(
    topology: SharedTopology,
    roots: RefinedRoots,
) -> SharedTopology:
    """Drop only faces whose renderer-consistent crossing root is invalid."""

    valid = roots.valid.detach().cpu().numpy()
    keep = valid[topology.faces].all(axis=1)
    return SharedTopology(
        edge_keys=topology.edge_keys,
        faces=topology.faces[keep],
        face_region_id=topology.face_region_id[keep],
        chart_face_count=dict(topology.chart_face_count),
    )


__all__ = [
    "ChartSurface",
    "RefinedRoots",
    "RegionTetrahedralConfig",
    "SharedTopology",
    "delaunay_chart",
    "filter_invalid_root_faces",
    "merge_chart_surfaces",
    "refine_shared_roots",
]
