from types import SimpleNamespace

import torch
from torch import nn

from mesh.gaussian_pivots import (
    GaussianAdaptivePivotBuilder,
    GaussianPivotConfig,
)
from mesh.region_atlas import (
    GaussianEvidence,
    RegionAtlasBuilder,
    RegionAtlasConfig,
)


class _IdentityDecoder(nn.Module):
    def forward(self, embedding):
        return embedding


def _gaussians():
    return SimpleNamespace(
        get_xyz=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        get_opacity=torch.full((2, 1), 0.9),
        get_scaling=torch.tensor([[0.4, 0.2, 0.1], [0.4, 0.2, 0.1]]),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        get_semantic=torch.tensor([[8.0, 0.0], [8.0, 0.0]]),
        get_semantic_confidence=torch.full((2, 1), 0.9),
        get_geometry_posterior=torch.full((2, 4), 0.25),
        get_boundary_score=torch.zeros(2, 1),
        observation_count=torch.full((2, 1), 4.0),
        semantic_decoder=_IdentityDecoder(),
    )


def test_pivots_use_covariance_minimum_axis_oriented_by_observed_normal():
    gaussians = _gaussians()
    evidence = GaussianEvidence(
        visible_count=torch.full((2,), 3.0),
        normal=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
        confidence=torch.ones(2),
    )
    atlas = RegionAtlasBuilder(
        RegionAtlasConfig(
            max_gaussians=4,
            max_core_gaussians=4,
            min_region_gaussians=1,
            contact_radius_factor=0.5,
        )
    ).build(gaussians, evidence)
    pivots = GaussianAdaptivePivotBuilder(
        GaussianPivotConfig(
            min_sigma_to_local=0.01,
            max_sigma_to_local=1.0,
        )
    ).build(gaussians, atlas, evidence)

    assert pivots.gaussian_count == 2
    assert torch.equal(
        pivots.roles,
        torch.tensor([-1, 0, 1, -1, 0, 1], dtype=torch.int8),
    )
    assert torch.allclose(
        pivots.normals[:3],
        torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        atol=1e-6,
    )
    assert torch.allclose(
        pivots.normals[3:],
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1),
        atol=1e-6,
    )
    assert torch.allclose(pivots.normal_sigma, torch.full((6,), 0.1), atol=1e-6)
    assert torch.allclose(pivots.points[1], gaussians.get_xyz[0])
    assert pivots.points[0, 2] > pivots.points[1, 2] > pivots.points[2, 2]


def test_chart_and_gaussian_lookup_return_canonical_complete_triplets():
    gaussians = _gaussians()
    atlas = RegionAtlasBuilder(
        RegionAtlasConfig(
            max_gaussians=4,
            max_core_gaussians=4,
            min_region_gaussians=1,
            contact_radius_factor=0.5,
        )
    ).build(gaussians)
    pivots = GaussianAdaptivePivotBuilder().build(gaussians, atlas)

    assert torch.equal(
        pivots.indices_for_gaussians(torch.tensor([1])),
        torch.tensor([3, 4, 5]),
    )
    chart_pivots = pivots.for_chart(atlas.charts[0])
    assert chart_pivots.gaussian_count == atlas.charts[0].gaussian_indices.numel()
    assert torch.equal(
        chart_pivots.membership_ids.reshape(-1, 3, 2)[:, 0],
        atlas.membership.ids,
    )
    assert torch.equal(
        chart_pivots.membership_confidence.reshape(-1, 3)[:, 0],
        atlas.membership.confidence[:, 0],
    )
