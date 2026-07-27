from types import SimpleNamespace

import torch
from torch import nn

from mesh.region_atlas import (
    GaussianEvidence,
    RegionAtlasBuilder,
    RegionAtlasConfig,
)


class _IdentityDecoder(nn.Module):
    def forward(self, embedding):
        return embedding


def _gaussians(points, logits, *, confidence=None, opacity=None, boundary=None):
    points = torch.as_tensor(points, dtype=torch.float32)
    logits = torch.as_tensor(logits, dtype=torch.float32)
    count = len(points)
    if confidence is None:
        confidence = torch.ones(count)
    if opacity is None:
        opacity = torch.full((count,), 0.9)
    if boundary is None:
        boundary = torch.zeros(count)
    return SimpleNamespace(
        get_xyz=points,
        get_opacity=torch.as_tensor(opacity, dtype=torch.float32)[:, None],
        get_scaling=torch.full((count, 3), 0.08),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        get_semantic=logits,
        get_semantic_confidence=torch.as_tensor(confidence, dtype=torch.float32)[:, None],
        get_geometry_posterior=torch.full((count, 4), 0.25),
        get_boundary_score=torch.as_tensor(boundary, dtype=torch.float32)[:, None],
        observation_count=torch.full((count, 1), 4.0),
        semantic_decoder=_IdentityDecoder(),
    )


def _owned_indices(atlas):
    return torch.cat([chart.owned_indices for chart in atlas.charts]).sort().values


def test_atlas_keeps_class_zero_and_routes_uncertain_gaussians_to_residual():
    gaussians = _gaussians(
        [[float(index), 0.0, 0.0] for index in range(6)],
        [
            [8.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        confidence=[0.9, 0.9, 0.9, 0.9, 0.2, 0.2],
    )
    atlas = RegionAtlasBuilder(
        RegionAtlasConfig(
            max_gaussians=6,
            max_core_gaussians=4,
            min_region_gaussians=1,
            contact_radius_factor=0.5,
        )
    ).build(gaussians)

    assert set(atlas.owner_region_ids.tolist()) == {-1, 0, 1}
    assert atlas.owner_region_ids[:2].tolist() == [0, 0]
    assert atlas.owner_region_ids[-2:].tolist() == [-1, -1]
    assert torch.equal(_owned_indices(atlas), atlas.gaussian_indices)
    assert sum(chart.owned_indices.numel() for chart in atlas.charts) == len(atlas)


def test_large_region_is_split_into_stable_spatial_charts_with_overlap():
    count = 11
    gaussians = _gaussians(
        [[float(index), 0.0, 0.0] for index in range(count)],
        [[9.0, 0.0]] * count,
    )
    config = RegionAtlasConfig(
        max_gaussians=count,
        max_core_gaussians=4,
        min_region_gaussians=1,
        halo_factor=1.1,
        contact_radius_factor=0.5,
    )
    first = RegionAtlasBuilder(config).build(gaussians)
    second = RegionAtlasBuilder(config).build(gaussians)

    assert len(first.charts) > 1
    assert all(chart.owned_indices.numel() <= 4 for chart in first.charts)
    assert any(chart.overlap_indices.numel() for chart in first.charts)
    assert torch.equal(_owned_indices(first), first.gaussian_indices)
    assert [
        (chart.chart_id, chart.region_id, chart.partition_id, chart.owned_indices.tolist())
        for chart in first.charts
    ] == [
        (chart.chart_id, chart.region_id, chart.partition_id, chart.owned_indices.tolist())
        for chart in second.charts
    ]


def test_contact_halo_overlaps_regions_without_duplicating_ownership():
    points = [
        [-0.30, 0.0, 0.0],
        [-0.10, 0.0, 0.0],
        [-0.30, 0.2, 0.0],
        [-0.10, 0.2, 0.0],
        [0.10, 0.0, 0.0],
        [0.30, 0.0, 0.0],
        [0.10, 0.2, 0.0],
        [0.30, 0.2, 0.0],
    ]
    gaussians = _gaussians(
        points,
        [[8.0, 0.0]] * 4 + [[0.0, 8.0]] * 4,
    )
    atlas = RegionAtlasBuilder(
        RegionAtlasConfig(
            max_gaussians=8,
            max_core_gaussians=8,
            min_region_gaussians=1,
            contact_radius_factor=1.2,
        )
    ).build(
        gaussians,
        GaussianEvidence(
            visible_count=torch.full((8,), 3.0),
            normal=torch.tensor([[0.0, 0.0, 1.0]]).repeat(8, 1),
            confidence=torch.ones(8),
        ),
    )

    assert atlas.contact_pairs.numel()
    chart_zero = next(chart for chart in atlas.charts if chart.region_id == 0)
    chart_one = next(chart for chart in atlas.charts if chart.region_id == 1)
    assert chart_zero.contact_indices.numel()
    assert chart_one.contact_indices.numel()
    assert not torch.isin(chart_zero.core_indices, chart_one.core_indices).any()
    assert torch.equal(_owned_indices(atlas), atlas.gaussian_indices)


def test_region_budget_preserves_a_small_high_quality_region():
    points = [[float(index), 0.0, 0.0] for index in range(20)]
    points += [[0.0, 3.0, 0.0], [1.0, 3.0, 0.0]]
    gaussians = _gaussians(
        points,
        [[8.0, 0.0]] * 20 + [[0.0, 8.0]] * 2,
    )
    atlas = RegionAtlasBuilder(
        RegionAtlasConfig(
            max_gaussians=8,
            max_core_gaussians=8,
            min_region_gaussians=2,
            contact_radius_factor=0.5,
        )
    ).build(gaussians)

    assert len(atlas) == 8
    assert int((atlas.owner_region_ids == 1).sum()) == 2

