from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mesh.region_tetrahedral import (
    RegionTetrahedralConfig,
    delaunay_chart,
    merge_chart_surfaces,
    refine_shared_roots,
)


class _PlaneField:
    """Analytic ``x=0`` field with the same candidate-first root interface."""

    def __init__(self) -> None:
        self.calls = 0
        self.edge_count = 0

    def refine_edges(
        self,
        endpoints: torch.Tensor,
        *,
        candidate_view_ids: torch.Tensor,
        binary_steps: int,
        chunk_size: int,
    ):
        del chunk_size
        self.calls += 1
        self.edge_count = len(endpoints)
        assert candidate_view_ids.shape[0] == len(endpoints)

        start = endpoints[:, 0].clone()
        end = endpoints[:, 1].clone()
        start_inside = start[:, 0] <= 0.0
        crossing = start_inside != (end[:, 0] <= 0.0)
        for _ in range(binary_steps):
            midpoint = 0.5 * (start + end)
            midpoint_inside = midpoint[:, 0] <= 0.0
            move_start = midpoint_inside == start_inside
            start = torch.where(move_start[:, None], midpoint, start)
            end = torch.where(move_start[:, None], end, midpoint)

        vertices = 0.5 * (start + end)
        original = endpoints[:, 1] - endpoints[:, 0]
        denominator = torch.sum(original * original, dim=1).clamp_min(1e-12)
        interpolation = torch.sum(
            (vertices - endpoints[:, 0]) * original,
            dim=1,
        ) / denominator
        return SimpleNamespace(
            vertices=vertices,
            interpolation=interpolation,
            valid=crossing,
            confidence=torch.ones(
                len(endpoints),
                dtype=endpoints.dtype,
                device=endpoints.device,
            ),
        )


def test_region_charts_share_canonical_plane_roots() -> None:
    points = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    phi = points[:, 0]
    valid = np.ones(len(points), dtype=bool)
    radius = np.full(len(points), 2.0, dtype=np.float64)
    owner_chart = np.asarray([11, 12, 11, 12, 11, 12, 11, 12], dtype=np.int64)
    config = RegionTetrahedralConfig(
        max_crossing_edge_factor=10.0,
        binary_steps=12,
        qhull_options="Qbb Qc Q12 QJ",
    )

    charts = [
        delaunay_chart(
            chart_id=chart_id,
            region_id=7,
            pivot_indices=np.arange(len(points), dtype=np.int64),
            pivot_owner_chart=owner_chart,
            points=points,
            phi=phi,
            valid=valid,
            radius=radius,
            scene_extent=2.0,
            config=config,
        )
        for chart_id in (11, 12)
    ]
    assert all(len(chart.faces) > 0 for chart in charts)
    shared_keys = {
        tuple(edge)
        for edge in charts[0].edge_keys
    } & {
        tuple(edge)
        for edge in charts[1].edge_keys
    }
    assert shared_keys

    topology = merge_chart_surfaces(charts)
    assert len(topology.edge_keys) < sum(len(chart.edge_keys) for chart in charts)
    assert len(np.unique(topology.edge_keys, axis=0)) == len(topology.edge_keys)
    assert set(topology.chart_face_count) == {11, 12}

    field = _PlaneField()
    pivot_points = torch.as_tensor(points, dtype=torch.float32)
    pivot_view_ids = torch.zeros((len(points), 2), dtype=torch.long)
    roots = refine_shared_roots(
        topology,
        pivot_points=pivot_points,
        pivot_view_ids=pivot_view_ids,
        field=field,
        config=config,
    )

    assert field.calls == 1
    assert field.edge_count == len(topology.edge_keys)
    assert bool(roots.valid.all())
    assert float(roots.vertices[:, 0].abs().max()) < 2.0 ** -11
    assert bool(((roots.interpolation > 0.0) & (roots.interpolation < 1.0)).all())
