from __future__ import annotations

import numpy as np

from mesh.spatial_tetrahedral import (
    ChartSurface,
    SharedTopology,
    SpatialTetrahedralConfig,
    delaunay_chart,
    filter_refined_faces,
    merge_chart_surfaces,
)


def _two_tetrahedra() -> np.ndarray:
    first = np.asarray(
        (
            (-0.2, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.0, 0.3, 0.0),
            (0.0, 0.0, 0.3),
        ),
        dtype=np.float64,
    )
    return np.concatenate((first, first + np.asarray((10.0, 0.0, 0.0))), axis=0)


def test_spatial_delaunay_never_bridges_disconnected_support() -> None:
    points = _two_tetrahedra()
    phi = np.tile(np.asarray((-1.0, 1.0, 1.0, 1.0)), 2)
    surface = delaunay_chart(
        chart_id=0,
        pivot_indices=np.arange(len(points)),
        pivot_owner_chart=np.zeros(len(points), dtype=np.int64),
        points=points,
        phi=phi,
        valid=np.ones(len(points), dtype=bool),
        support=np.full(len(points), 0.4),
        spacing=np.full(len(points), 0.25),
        scene_extent=10.0,
        config=SpatialTetrahedralConfig(
            max_tetra_edge_ratio=20.0,
            min_tetra_volume_ratio=1e-5,
            max_circumradius_to_edge=4.0,
        ),
    )

    assert len(surface.faces) > 0
    endpoint_distance = np.linalg.norm(
        points[surface.edge_keys[:, 0]] - points[surface.edge_keys[:, 1]],
        axis=1,
    )
    assert float(endpoint_distance.max()) < 1.0
    assert not np.any(
        (surface.edge_keys[:, 0] < 4) & (surface.edge_keys[:, 1] >= 4)
    )


def test_degenerate_sliver_tetrahedron_is_rejected_before_marching() -> None:
    points = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1e-8),
        ),
        dtype=np.float64,
    )
    surface = delaunay_chart(
        chart_id=0,
        pivot_indices=np.arange(4),
        pivot_owner_chart=np.zeros(4, dtype=np.int64),
        points=points,
        phi=np.asarray((-1.0, 1.0, 1.0, 1.0)),
        valid=np.ones(4, dtype=bool),
        support=np.full(4, 10.0),
        spacing=np.full(4, 10.0),
        scene_extent=2.0,
        config=SpatialTetrahedralConfig(min_tetra_volume_ratio=1e-4),
    )

    assert len(surface.faces) == 0
    assert surface.tetrahedra >= 0
    assert surface.accepted_tetrahedra == 0


def test_chart_overlap_welds_only_canonical_global_roots() -> None:
    first = ChartSurface(
        edge_keys=np.asarray(((0, 1), (0, 2), (1, 2))),
        faces=np.asarray(((0, 1, 2),)),
        chart_id=0,
        tetrahedra=1,
        accepted_tetrahedra=1,
    )
    second = ChartSurface(
        edge_keys=np.asarray(((0, 1), (0, 2), (1, 2))),
        faces=np.asarray(((2, 1, 0),)),
        chart_id=1,
        tetrahedra=1,
        accepted_tetrahedra=1,
    )

    merged = merge_chart_surfaces([first, second])

    assert len(merged.edge_keys) == 3
    assert len(merged.faces) == 1
    assert merged.chart_face_count == {0: 1, 1: 1}


def test_final_face_gate_rejects_long_or_skinny_root_triangles() -> None:
    topology = SharedTopology(
        edge_keys=np.asarray(((0, 1), (2, 3), (4, 5))),
        faces=np.asarray(((0, 1, 2),)),
        chart_face_count={0: 1},
    )
    filtered, statistics = filter_refined_faces(
        topology,
        root_vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
                (0.0, 1e-6, 0.0),
            )
        ),
        root_valid=np.ones(3, dtype=bool),
        root_interpolation=np.full(3, 0.5),
        pivot_support=np.full(6, 0.1),
        pivot_spacing=np.full(6, 0.1),
        scene_extent=5.0,
        config=SpatialTetrahedralConfig(),
    )

    assert len(filtered.faces) == 0
    assert statistics["faces_rejected_by_quality_gate"] == 1

