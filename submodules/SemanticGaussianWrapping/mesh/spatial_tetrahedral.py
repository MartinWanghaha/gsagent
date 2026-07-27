"""Spatially local Gaussian-Wrapping topology.

The Delaunay scaffold is determined only by Euclidean support.  Semantic
posteriors are intentionally absent from this module: semantics may allocate
anchors and label the extracted surface, but may never create spatial
adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


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
class SpatialTetrahedralConfig:
    """Physical and numerical quality policy for local Delaunay cells."""

    level: float = 0.0
    max_edge_support_factor: float = 1.0
    max_edge_spacing_factor: float = 3.0
    max_tetra_edge_ratio: float = 100.0
    min_tetra_volume_ratio: float = 1e-6
    max_circumradius_to_edge: float = 8.0
    max_face_edge_support_factor: float = 1.0
    max_face_edge_spacing_factor: float = 3.0
    max_face_aspect_ratio: float = 100.0
    min_face_area_fraction: float = 1e-12
    qhull_options: str = "Qbb Qc Q12 QJ"
    deduplication_fraction: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.level):
            raise ValueError("level must be finite")
        positive = (
            "max_edge_support_factor",
            "max_edge_spacing_factor",
            "max_tetra_edge_ratio",
            "min_tetra_volume_ratio",
            "max_circumradius_to_edge",
            "max_face_edge_support_factor",
            "max_face_edge_spacing_factor",
            "max_face_aspect_ratio",
            "min_face_area_fraction",
            "deduplication_fraction",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ChartSurface:
    """One spatial chart expressed through canonical global pivot edges."""

    edge_keys: np.ndarray
    faces: np.ndarray
    chart_id: int
    tetrahedra: int
    accepted_tetrahedra: int

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
    def empty(
        cls,
        chart_id: int,
        *,
        tetrahedra: int = 0,
        accepted_tetrahedra: int = 0,
    ) -> "ChartSurface":
        return cls(
            edge_keys=np.empty((0, 2), dtype=np.int64),
            faces=np.empty((0, 3), dtype=np.int64),
            chart_id=int(chart_id),
            tetrahedra=int(tetrahedra),
            accepted_tetrahedra=int(accepted_tetrahedra),
        )


@dataclass(frozen=True)
class SharedTopology:
    """All chart faces welded at canonical global pivot edges."""

    edge_keys: np.ndarray
    faces: np.ndarray
    chart_face_count: dict[int, int]

    def __post_init__(self) -> None:
        edges = np.ascontiguousarray(np.asarray(self.edge_keys, dtype=np.int64))
        faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64))
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edge_keys must have shape [E,2]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [F,3]")
        if faces.size and (faces.min() < 0 or faces.max() >= len(edges)):
            raise ValueError("face index is outside edge_keys")
        object.__setattr__(self, "edge_keys", edges)
        object.__setattr__(self, "faces", faces)


def _deduplicate_pivots(
    global_indices: np.ndarray,
    points: np.ndarray,
    *,
    scene_extent: float,
    fraction: float,
) -> np.ndarray:
    if not len(global_indices):
        return np.ascontiguousarray(global_indices, dtype=np.int64)
    tolerance = max(float(scene_extent) * float(fraction), np.finfo(np.float64).eps)
    selected = np.asarray(points[global_indices], dtype=np.float64)
    origin = selected.min(axis=0, keepdims=True)
    keys = np.rint((selected - origin) / tolerance).astype(np.int64)
    order = np.lexsort((global_indices, keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return np.ascontiguousarray(global_indices[order[first]], dtype=np.int64)


def _tetra_quality_mask(
    tetrahedra: np.ndarray,
    points: np.ndarray,
    support: np.ndarray,
    spacing: np.ndarray,
    config: SpatialTetrahedralConfig,
) -> np.ndarray:
    """Reject non-local and ill-conditioned cells before marching."""

    tetra_points = points[tetrahedra]
    edge_points = tetra_points[:, _TETRA_EDGES]
    edge_vectors = edge_points[:, :, 1] - edge_points[:, :, 0]
    lengths = np.linalg.norm(edge_vectors, axis=2)
    finite = np.isfinite(lengths).all(axis=1)

    tetra_support = support[tetrahedra]
    support_limit = float(config.max_edge_support_factor) * (
        tetra_support[:, _TETRA_EDGES[:, 0]]
        + tetra_support[:, _TETRA_EDGES[:, 1]]
    )
    tetra_spacing = spacing[tetrahedra]
    spacing_limit = float(config.max_edge_spacing_factor) * np.maximum(
        tetra_spacing[:, _TETRA_EDGES[:, 0]],
        tetra_spacing[:, _TETRA_EDGES[:, 1]],
    )
    local = (
        (lengths <= support_limit)
        & (lengths <= spacing_limit)
        & (support_limit > 0.0)
        & (spacing_limit > 0.0)
    ).all(axis=1)

    shortest = lengths.min(axis=1)
    longest = lengths.max(axis=1)
    ratio = longest / np.maximum(shortest, np.finfo(np.float64).eps)
    conditioned = ratio <= float(config.max_tetra_edge_ratio)

    basis = tetra_points[:, 1:] - tetra_points[:, :1]
    determinant = np.linalg.det(basis)
    volume = np.abs(determinant) / 6.0
    volume_ratio = volume / np.maximum(
        longest**3,
        np.finfo(np.float64).eps,
    )
    volumetric = volume_ratio >= float(config.min_tetra_volume_ratio)

    circumradius = np.full(len(tetrahedra), np.inf, dtype=np.float64)
    nonsingular = np.isfinite(determinant) & (
        np.abs(determinant)
        > np.finfo(np.float64).eps * np.maximum(longest, 1.0) ** 3
    )
    rows = np.flatnonzero(nonsingular)
    if len(rows):
        rhs = 0.5 * np.sum(basis[rows] * basis[rows], axis=2)
        try:
            center = np.linalg.solve(basis[rows], rhs)
            circumradius[rows] = np.linalg.norm(center, axis=1)
        except np.linalg.LinAlgError:
            for row in rows:
                try:
                    center = np.linalg.solve(basis[row], 0.5 * np.sum(basis[row] ** 2, axis=1))
                except np.linalg.LinAlgError:
                    continue
                circumradius[row] = np.linalg.norm(center)
    bounded_sphere = (
        circumradius
        <= float(config.max_circumradius_to_edge)
        * np.maximum(longest, np.finfo(np.float64).eps)
    )
    return finite & local & conditioned & volumetric & bounded_sphere


def delaunay_chart(
    *,
    chart_id: int,
    pivot_indices: np.ndarray,
    pivot_owner_chart: np.ndarray,
    points: np.ndarray,
    phi: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    spacing: np.ndarray,
    scene_extent: float,
    config: SpatialTetrahedralConfig,
) -> ChartSurface:
    """Delaunay a bounded spatial chart and cut its global zero set."""

    from scipy.spatial import Delaunay, QhullError

    requested = np.ascontiguousarray(
        np.asarray(pivot_indices, dtype=np.int64).reshape(-1)
    )
    if len(requested) < 4:
        return ChartSurface.empty(chart_id)
    unique_global = _deduplicate_pivots(
        requested,
        points,
        scene_extent=scene_extent,
        fraction=config.deduplication_fraction,
    )
    if len(unique_global) < 4:
        return ChartSurface.empty(chart_id)
    local_points = np.asarray(points[unique_global], dtype=np.float64)
    if np.linalg.matrix_rank(local_points - local_points.mean(axis=0)) < 3:
        return ChartSurface.empty(chart_id)
    try:
        raw_tetrahedra = Delaunay(
            local_points,
            qhull_options=config.qhull_options,
        ).simplices
    except QhullError:
        return ChartSurface.empty(chart_id)
    raw_tetrahedra = np.ascontiguousarray(raw_tetrahedra, dtype=np.int64)
    raw_tetrahedra = raw_tetrahedra[
        (raw_tetrahedra >= 0).all(axis=1)
        & (raw_tetrahedra < len(local_points)).all(axis=1)
    ]
    tetrahedra_count = len(raw_tetrahedra)
    if not tetrahedra_count:
        return ChartSurface.empty(chart_id)

    owners = np.asarray(pivot_owner_chart, dtype=np.int64).reshape(-1)
    if len(owners) != len(points):
        raise ValueError("pivot_owner_chart must align with global pivots")
    global_tetrahedra = unique_global[raw_tetrahedra]
    touches_core = (owners[global_tetrahedra] == int(chart_id)).any(axis=1)
    quality = _tetra_quality_mask(
        raw_tetrahedra,
        local_points,
        np.asarray(support[unique_global], dtype=np.float64),
        np.asarray(spacing[unique_global], dtype=np.float64),
        config,
    )
    tetrahedra = raw_tetrahedra[touches_core & quality]
    if not len(tetrahedra):
        return ChartSurface.empty(
            chart_id,
            tetrahedra=tetrahedra_count,
        )

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
    accepted_tetrahedra = len(tetrahedra)
    if not accepted_tetrahedra:
        return ChartSurface.empty(
            chart_id,
            tetrahedra=tetrahedra_count,
        )

    tetra_edges = tetrahedra[:, _TETRA_EDGES]
    canonical_local = np.sort(tetra_edges, axis=-1)
    flat_edges = canonical_local.reshape(-1, 2)
    unique_edges, inverse = np.unique(flat_edges, axis=0, return_inverse=True)
    crossing = inside[unique_edges[:, 0]] != inside[unique_edges[:, 1]]
    edge_to_root = np.full(len(unique_edges), -1, dtype=np.int64)
    edge_to_root[crossing] = np.arange(int(crossing.sum()), dtype=np.int64)
    tetra_roots = edge_to_root[inverse].reshape(-1, 6)
    root_local_edges = unique_edges[crossing]

    faces: list[np.ndarray] = []
    one = _TRIANGLE_COUNT[case] == 1
    two = _TRIANGLE_COUNT[case] == 2
    if np.any(one):
        faces.append(
            np.take_along_axis(
                tetra_roots[one],
                _TRIANGLE_TABLE[case[one], :3],
                axis=1,
            )
        )
    if np.any(two):
        faces.append(
            np.take_along_axis(
                tetra_roots[two],
                _TRIANGLE_TABLE[case[two], :6],
                axis=1,
            ).reshape(-1, 3)
        )
    if not faces:
        return ChartSurface.empty(
            chart_id,
            tetrahedra=tetrahedra_count,
            accepted_tetrahedra=accepted_tetrahedra,
        )
    root_faces = np.concatenate(faces, axis=0)
    root_global_edges = np.sort(unique_global[root_local_edges], axis=1)
    used, remap = np.unique(root_faces.reshape(-1), return_inverse=True)
    return ChartSurface(
        edge_keys=root_global_edges[used],
        faces=remap.reshape(-1, 3),
        chart_id=int(chart_id),
        tetrahedra=int(tetrahedra_count),
        accepted_tetrahedra=int(accepted_tetrahedra),
    )


def merge_chart_surfaces(charts: list[ChartSurface]) -> SharedTopology:
    """Weld chart overlap and remove duplicate triangles."""

    populated = [chart for chart in charts if len(chart.faces)]
    if not populated:
        return SharedTopology(
            edge_keys=np.empty((0, 2), dtype=np.int64),
            faces=np.empty((0, 3), dtype=np.int64),
            chart_face_count={},
        )
    all_edges = np.concatenate([chart.edge_keys for chart in populated], axis=0)
    edge_keys, inverse = np.unique(all_edges, axis=0, return_inverse=True)
    face_parts: list[np.ndarray] = []
    chart_face_count: dict[int, int] = {}
    offset = 0
    for chart in populated:
        mapping = inverse[offset : offset + len(chart.edge_keys)]
        chart_faces = mapping[chart.faces]
        face_parts.append(chart_faces)
        chart_face_count[int(chart.chart_id)] = int(len(chart_faces))
        offset += len(chart.edge_keys)
    faces = np.concatenate(face_parts, axis=0)
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    first.sort()
    return SharedTopology(
        edge_keys=edge_keys,
        faces=faces[first],
        chart_face_count=chart_face_count,
    )


def filter_refined_faces(
    topology: SharedTopology,
    *,
    root_vertices: np.ndarray,
    root_valid: np.ndarray,
    root_interpolation: np.ndarray,
    pivot_support: np.ndarray,
    pivot_spacing: np.ndarray,
    scene_extent: float,
    config: SpatialTetrahedralConfig,
) -> tuple[SharedTopology, dict[str, int]]:
    """Apply the final physical quality gate to refined mesh triangles."""

    vertices = np.asarray(root_vertices, dtype=np.float64)
    valid = np.asarray(root_valid, dtype=bool).reshape(-1)
    interpolation = np.asarray(root_interpolation, dtype=np.float64).reshape(-1)
    if vertices.shape != (len(topology.edge_keys), 3):
        raise ValueError("root_vertices must align with topology.edge_keys")
    if valid.shape != (len(vertices),) or interpolation.shape != (len(vertices),):
        raise ValueError("root validity/interpolation must align with roots")
    edges = topology.edge_keys
    support = np.asarray(pivot_support, dtype=np.float64)
    spacing = np.asarray(pivot_spacing, dtype=np.float64)
    root_support = (
        support[edges[:, 0]] * (1.0 - interpolation)
        + support[edges[:, 1]] * interpolation
    )
    root_spacing = (
        spacing[edges[:, 0]] * (1.0 - interpolation)
        + spacing[edges[:, 1]] * interpolation
    )

    faces = topology.faces
    initial = len(faces)
    keep = valid[faces].all(axis=1)
    keep &= (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    candidate_rows = np.flatnonzero(keep)
    if len(candidate_rows):
        candidate_faces = faces[candidate_rows]
        triangles = vertices[candidate_faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 1]
        edge_c = triangles[:, 0] - triangles[:, 2]
        lengths = np.stack(
            (
                np.linalg.norm(edge_a, axis=1),
                np.linalg.norm(edge_b, axis=1),
                np.linalg.norm(edge_c, axis=1),
            ),
            axis=1,
        )
        face_pairs = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
        face_support = root_support[candidate_faces]
        support_limit = float(config.max_face_edge_support_factor) * (
            face_support[:, face_pairs[:, 0]]
            + face_support[:, face_pairs[:, 1]]
        )
        face_spacing = root_spacing[candidate_faces]
        spacing_limit = float(config.max_face_edge_spacing_factor) * np.maximum(
            face_spacing[:, face_pairs[:, 0]],
            face_spacing[:, face_pairs[:, 1]],
        )
        local = (
            np.isfinite(lengths).all(axis=1)
            & (lengths <= support_limit).all(axis=1)
            & (lengths <= spacing_limit).all(axis=1)
        )
        area = 0.5 * np.linalg.norm(np.cross(edge_a, -edge_c), axis=1)
        longest = lengths.max(axis=1)
        minimum_area = max(
            float(scene_extent) ** 2 * float(config.min_face_area_fraction),
            np.finfo(np.float64).eps,
        )
        aspect = longest**2 / np.maximum(
            2.0 * area,
            np.finfo(np.float64).eps,
        )
        geometry = (
            np.isfinite(area)
            & (area >= minimum_area)
            & np.isfinite(aspect)
            & (aspect <= float(config.max_face_aspect_ratio))
        )
        keep[candidate_rows] &= local & geometry

    filtered = faces[keep]
    statistics = {
        "faces_before_quality_gate": int(initial),
        "faces_rejected_by_quality_gate": int(initial - len(filtered)),
        "faces_after_quality_gate": int(len(filtered)),
    }
    return (
        SharedTopology(
            edge_keys=topology.edge_keys,
            faces=filtered,
            chart_face_count=dict(topology.chart_face_count),
        ),
        statistics,
    )


__all__ = [
    "ChartSurface",
    "SharedTopology",
    "SpatialTetrahedralConfig",
    "delaunay_chart",
    "filter_refined_faces",
    "merge_chart_surfaces",
]
