from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from regularization.surface import (
    gaussian_surface_consistency,
    prepare_gaussian_surface_consistency,
)
from semantic.region_membership import SparseRegionMembership


def _memberships(indices: torch.Tensor, top_k: int) -> SparseRegionMembership:
    rows = indices.shape[0]
    logits = torch.full((rows, 4), -2.0, device=indices.device)
    logits[:, 1] = 2.0
    logits[:, 2] = 1.0
    return SparseRegionMembership.from_logits(
        logits,
        top_k=top_k,
        confidence=torch.full((rows, 1), 0.8, device=indices.device),
    )


class _Gaussians:
    def __init__(self) -> None:
        self.xyz = torch.tensor(
            [
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.1],
                [0.0, 0.3, -0.1],
                [0.1, -0.2, 0.05],
            ],
            requires_grad=True,
        )
        self.scaling = torch.full((4, 3), 0.1, requires_grad=True)
        self.rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        self.confidence = torch.tensor([[0.2], [0.4], [0.6], [0.8]])

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scaling(self):
        return self.scaling

    @property
    def get_rotation(self):
        return self.rotation

    @property
    def get_semantic_confidence(self):
        return self.confidence

    def point_region_memberships(self, indices, *, top_k, chunk_size):
        del chunk_size
        return _memberships(indices, top_k)


class _RecordingField:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def query_point_regions(self, points: torch.Tensor, region_ids: torch.Tensor):
        self.calls.append(points)
        sdf = (points[:, 2] + 0.15 * points[:, 0])[:, None].expand_as(region_ids)
        normal = F.normalize(
            points + points.new_tensor([0.1, 0.2, 1.0]),
            dim=-1,
        )[:, None, :].expand(-1, region_ids.shape[1], -1)
        return SimpleNamespace(
            region_ids=region_ids,
            valid=torch.ones_like(region_ids, dtype=torch.bool),
            sdf=sdf,
            normal=normal,
        )


def test_surface_consistency_combines_center_and_pivots_into_one_query():
    gaussians = _Gaussians()
    field = _RecordingField()
    loss, components = gaussian_surface_consistency(
        gaussians,
        field,
        sample_points=3,
    )

    assert len(field.calls) == 1
    # Both foreground fields reuse the same nine spatial probes.
    assert field.calls[0].shape == (9, 3)
    assert set(components) == {"center", "crossing", "normal"}
    loss.backward()
    assert gaussians.xyz.grad is not None
    assert gaussians.scaling.grad is not None
    assert torch.isfinite(gaussians.xyz.grad).all()
    assert torch.isfinite(gaussians.scaling.grad).all()


def test_zero_surface_samples_return_finite_zero_without_querying():
    gaussians = _Gaussians()
    field = _RecordingField()

    loss, components = gaussian_surface_consistency(
        gaussians,
        field,
        sample_points=0,
    )

    assert not field.calls
    assert loss.shape == () and loss.item() == 0.0
    assert all(value.item() == 0.0 for value in components.values())


def test_surface_sampling_gathers_raw_candidates_before_activation(monkeypatch):
    count = 10_000

    class CandidateFirstGaussians:
        def __init__(self):
            self.xyz = torch.nn.Parameter(torch.randn(count, 3) * 0.01)
            self.registry = {
                "scaling": torch.nn.Parameter(torch.full((count, 3), -2.0)),
                "rotation": torch.nn.Parameter(
                    torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, -1).clone()
                ),
                "semantic_confidence": torch.full((count, 1), 0.8),
                "propagated_semantic_confidence": torch.full((count, 1), 0.2),
            }

        @property
        def get_xyz(self):
            return self.xyz

        @property
        def get_scaling(self):
            raise AssertionError("full-model scaling activation is forbidden")

        @property
        def get_rotation(self):
            raise AssertionError("full-model rotation activation is forbidden")

        @property
        def get_semantic_confidence(self):
            raise AssertionError("full-model confidence activation is forbidden")

        def point_region_memberships(self, indices, *, top_k, chunk_size):
            del chunk_size
            return _memberships(indices, top_k)

    monkeypatch.setattr(
        torch,
        "randperm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full-model permutation is forbidden")
        ),
    )
    gaussians = CandidateFirstGaussians()
    prepared = prepare_gaussian_surface_consistency(gaussians, sample_points=128)

    assert prepared is not None
    assert prepared.indices.shape == (128,)
    assert prepared.indices.unique().numel() == 128
    assert prepared.region_ids.shape == (128, 3)
    assert prepared.query_points.shape == (384, 3)
