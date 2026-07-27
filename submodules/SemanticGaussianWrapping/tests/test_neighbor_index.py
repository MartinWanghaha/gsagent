from __future__ import annotations

import math

import pytest
import torch

from semantic.neighbor_index import GaussianNeighborIndex


class _Points:
    def __init__(self, xyz: torch.Tensor) -> None:
        self.xyz = xyz

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    def __len__(self) -> int:
        return self.xyz.shape[0]


class _SupportPoints(_Points):
    def __init__(self, xyz: torch.Tensor, scaling: torch.Tensor) -> None:
        super().__init__(xyz)
        self.scaling = scaling
        self.rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(self), -1).clone()
        self.opacity = torch.full((len(self), 1), 0.9)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity


class _CountingSupportPoints(_SupportPoints):
    def __init__(self, xyz: torch.Tensor, scaling: torch.Tensor) -> None:
        super().__init__(xyz, scaling)
        self.getter_calls = {"scaling": 0, "rotation": 0, "opacity": 0}

    @property
    def get_scaling(self) -> torch.Tensor:
        self.getter_calls["scaling"] += 1
        return self.scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        self.getter_calls["rotation"] += 1
        return self.rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        self.getter_calls["opacity"] += 1
        return self.opacity


def _brute_points(xyz: torch.Tensor, points: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cdist(points, xyz).topk(k, largest=False, sorted=True).indices


def _brute_indices(xyz: torch.Tensor, indices: torch.Tensor, k: int) -> torch.Tensor:
    distance = torch.cdist(xyz[indices], xyz)
    distance[torch.arange(indices.numel()), indices] = math.inf
    return distance.topk(k, largest=False, sorted=True).indices


def test_exact_query_points_and_indices_match_brute_force() -> None:
    torch.manual_seed(4)
    xyz = torch.randn(17, 3)
    points = torch.randn(7, 3)
    source = torch.tensor([0, 3, 9, 16])
    model = _Points(xyz)
    index = GaussianNeighborIndex(
        model,
        backend="exact",
        gaussian_chunk_size=4,
        query_chunk_size=3,
    )

    assert torch.equal(index.query_points(points, 5), _brute_points(xyz, points, 5))
    assert torch.equal(index.query_indices(source, 6), _brute_indices(xyz, source, 6))
    assert not index.query_points(points.requires_grad_(), 2).requires_grad


def test_exact_fallback_bounds_both_distance_dimensions(monkeypatch) -> None:
    xyz = torch.randn(23, 3)
    points = torch.randn(11, 3)
    model = _Points(xyz)
    index = GaussianNeighborIndex(
        model,
        backend="exact",
        gaussian_chunk_size=5,
        query_chunk_size=9,
        max_distance_bytes=40,
    )
    observed: list[tuple[int, int]] = []
    original = torch.cdist

    def recording_cdist(left, right, *args, **kwargs):
        observed.append((left.shape[0], right.shape[0]))
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    result = index.query_points(points, 3)
    assert result.shape == (11, 3)
    assert observed
    assert max(rows for rows, _ in observed) <= 2
    assert max(columns for _, columns in observed) <= 5


def test_topology_invalidates_automatically_and_motion_refreshes_explicitly() -> None:
    pytest.importorskip("scipy")
    model = _Points(torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    index = GaussianNeighborIndex(model, backend="scipy")
    query = torch.tensor([[0.1, 0.0, 0.0]])

    assert index.query_points(query, 1).item() == 0
    first_signature, first_tree = index.signature, index.tree
    with torch.no_grad():
        model.xyz[0, 0] = 100.0
    # Optimizer-style in-place movement preserves the topology signature.
    assert index.query_points(query, 1).item() == 0
    assert index.tree is first_tree
    index.refresh(force=True)
    assert index.query_points(query, 1).item() == 1

    model.xyz = torch.cat((model.xyz, torch.tensor([[0.05, 0.0, 0.0]])), dim=0)
    assert index.query_points(query, 1).item() == 2
    assert index.signature != first_signature
    assert index.tree is not first_tree


def test_scipy_and_exact_selection_agree_without_distance_ties() -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(17)
    xyz = torch.randn(31, 3, generator=generator)
    points = torch.randn(13, 3, generator=generator)
    source = torch.tensor([1, 7, 15, 23])
    model = _Points(xyz)
    exact = GaussianNeighborIndex(model, backend="exact", gaussian_chunk_size=7)
    scipy = GaussianNeighborIndex(model, backend="scipy")

    assert torch.equal(scipy.query_points(points, 8), exact.query_points(points, 8))
    assert torch.equal(scipy.query_indices(source, 8), exact.query_indices(source, 8))


def test_support_query_keeps_far_large_gaussian_when_k_is_one() -> None:
    pytest.importorskip("scipy")
    model = _SupportPoints(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([[0.01, 0.01, 0.01], [2.0, 2.0, 2.0]]),
    )
    point = torch.tensor([[0.05, 0.0, 0.0]])
    exact = GaussianNeighborIndex(model, backend="exact", gaussian_chunk_size=1)
    scipy = GaussianNeighborIndex(model, backend="scipy")

    assert scipy.query_points(point, 1).item() == 0
    assert exact.query_support(point, 1).item() == 1
    assert scipy.query_support(point, 1).item() == 1


def test_exact_support_ranking_batches_queries_and_gaussian_blocks(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(23)
    model = _SupportPoints(
        torch.randn(257, 3, generator=generator),
        0.1 + torch.rand(257, 3, generator=generator),
    )
    points = torch.randn(64, 3, generator=generator)
    index = GaussianNeighborIndex(
        model,
        backend="exact",
        gaussian_chunk_size=64,
        query_chunk_size=32,
    )
    calls: list[tuple[int, int]] = []
    original = index._support_score_shared

    def recording_score(query, candidates, density_scale):
        calls.append((query.shape[0], candidates.shape[0]))
        return original(query, candidates, density_scale)

    monkeypatch.setattr(index, "_support_score_shared", recording_score)
    result = index.query_support(points, 8)
    assert result.shape == (64, 8)
    assert calls == [(32, 64), (32, 64)] * 4 + [(32, 1), (32, 1)]
    assert all(query_count > 1 for query_count, _ in calls)


def test_exact_support_activates_foreign_model_attributes_once_per_query():
    generator = torch.Generator().manual_seed(41)
    model = _CountingSupportPoints(
        torch.randn(129, 3, generator=generator),
        0.1 + torch.rand(129, 3, generator=generator),
    )
    points = torch.randn(17, 3, generator=generator)
    index = GaussianNeighborIndex(
        model,
        backend="exact",
        gaussian_chunk_size=16,
        query_chunk_size=4,
    )

    assert index.query_support(points, 8).shape == (17, 8)
    assert model.getter_calls == {"scaling": 1, "rotation": 1, "opacity": 1}


def test_scipy_support_ranking_batches_ragged_candidates(monkeypatch) -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(29)
    model = _SupportPoints(
        torch.randn(257, 3, generator=generator),
        torch.full((257, 3), 10.0),
    )
    points = torch.randn(64, 3, generator=generator)
    index = GaussianNeighborIndex(model, backend="scipy", query_chunk_size=128)
    calls: list[tuple[int, int]] = []
    original = index._support_score_ragged

    def recording_score(query, candidates, density_scale):
        calls.append((query.shape[0], candidates.shape[1]))
        return original(query, candidates, density_scale)

    monkeypatch.setattr(index, "_support_score_ragged", recording_score)
    result = index.query_support(points, 8)
    assert result.shape == (64, 8)
    assert calls == [(64, index.support_candidate_budget)]


def test_scipy_multiscale_shortlist_obeys_hard_candidate_budget(monkeypatch) -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(31)
    count = 4096
    model = _SupportPoints(
        torch.randn(count, 3, generator=generator),
        torch.exp(torch.linspace(-5.0, 2.0, count))[:, None].expand(-1, 3),
    )
    points = torch.randn(37, 3, generator=generator)
    index = GaussianNeighborIndex(
        model,
        backend="scipy",
        support_candidate_budget=64,
    )
    widths: list[tuple[int, int]] = []
    original = index._batch_dense_support

    def recording(points, candidates, valid, k, density_scale):
        widths.append((candidates.shape[1], int(valid.sum(dim=1).max())))
        return original(points, candidates, valid, k, density_scale)

    monkeypatch.setattr(index, "_batch_dense_support", recording)
    result = index.query_support(points, 8)

    assert result.shape == (37, 8)
    assert len(widths) == 1
    assert widths[0][0] == 64
    assert 8 <= widths[0][1] <= 64
    assert all(torch.unique(row).numel() == 8 for row in result)


def test_scipy_shortlist_workspace_is_query_block_bounded(monkeypatch) -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(37)
    model = _SupportPoints(
        torch.randn(513, 3, generator=generator),
        0.05 + torch.rand(513, 3, generator=generator),
    )
    points = torch.randn(11, 3, generator=generator)
    width = 64
    rows_per_block = 3
    workspace_bytes = 32 * 4 * width * rows_per_block
    index = GaussianNeighborIndex(
        model,
        backend="scipy",
        query_chunk_size=32,
        max_distance_bytes=workspace_bytes,
        support_candidate_budget=width,
    )
    observed: list[tuple[int, int]] = []
    original = index._batch_dense_support

    def recording(points, candidates, valid, k, density_scale):
        observed.append((points.shape[0], candidates.shape[1]))
        return original(points, candidates, valid, k, density_scale)

    monkeypatch.setattr(index, "_batch_dense_support", recording)
    bounded = index.query_support(points, 8)
    roomy = GaussianNeighborIndex(
        model,
        backend="scipy",
        support_candidate_budget=width,
    ).query_support(points, 8)

    assert torch.equal(bounded, roomy)
    assert len(observed) == math.ceil(points.shape[0] / rows_per_block)
    assert max(rows for rows, _ in observed) <= rows_per_block
    assert all(candidate_width == width for _, candidate_width in observed)
    assert all(
        rows * 32 * max(points.element_size(), 4) * candidate_width
        <= workspace_bytes
        for rows, candidate_width in observed
    )


def test_scipy_routing_chunk_is_decoupled_from_gpu_rerank_workspace(
    monkeypatch,
) -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(43)
    model = _SupportPoints(
        torch.randn(513, 3, generator=generator),
        torch.exp(torch.linspace(-4.0, 1.0, 513))[:, None].expand(-1, 3),
    )
    points = torch.randn(11, 3, generator=generator)
    width = 64
    rerank_rows = 3
    index = GaussianNeighborIndex(
        model,
        backend="scipy",
        query_chunk_size=32,
        max_distance_bytes=32 * 4 * width * rerank_rows,
        support_candidate_budget=width,
        support_routing_query_chunk=points.shape[0],
        scipy_workers=1,
    )
    tree_calls: list[int] = []
    rerank_calls: list[int] = []
    original_query_tree = index._query_tree
    original_rerank = index._batch_dense_support

    def recording_tree(tree, query, **kwargs):
        tree_calls.append(query.shape[0])
        return original_query_tree(tree, query, **kwargs)

    def recording_rerank(query, candidates, valid, k, density_scale):
        rerank_calls.append(query.shape[0])
        return original_rerank(query, candidates, valid, k, density_scale)

    monkeypatch.setattr(index, "_query_tree", recording_tree)
    monkeypatch.setattr(index, "_batch_dense_support", recording_rerank)
    result = index.query_support(points, 8)

    assert result.shape == (11, 8)
    # One global-center query plus one query per active scale bucket. Every
    # tree sees the complete CPU routing block exactly once.
    assert len(tree_calls) == 1 + len(index._support_buckets)
    assert set(tree_calls) == {points.shape[0]}
    # Exact reranking remains independently bounded by the 3-row GPU budget.
    assert rerank_calls == [3, 3, 3, 2]


def test_scipy_routing_chunk_size_does_not_change_support_selection() -> None:
    pytest.importorskip("scipy")
    generator = torch.Generator().manual_seed(47)
    model = _SupportPoints(
        torch.randn(1025, 3, generator=generator),
        0.01 + torch.rand(1025, 3, generator=generator),
    )
    points = torch.randn(29, 3, generator=generator)
    common = {
        "backend": "scipy",
        "query_chunk_size": 16,
        "max_distance_bytes": 32 * 4 * 64 * 2,
        "support_candidate_budget": 64,
        "scipy_workers": 1,
    }
    small = GaussianNeighborIndex(
        model,
        support_routing_query_chunk=3,
        **common,
    ).query_support(points, 8)
    large = GaussianNeighborIndex(
        model,
        support_routing_query_chunk=points.shape[0],
        **common,
    ).query_support(points, 8)

    assert torch.equal(small, large)


def test_scipy_shortlist_rejects_budget_smaller_than_one_query_row() -> None:
    pytest.importorskip("scipy")
    model = _SupportPoints(torch.randn(17, 3), torch.ones(17, 3))
    width = 16
    index = GaussianNeighborIndex(
        model,
        backend="scipy",
        max_distance_bytes=32 * 4 * width - 1,
        support_candidate_budget=width,
    )

    with pytest.raises(ValueError, match="too small for one support shortlist row"):
        index.query_support(torch.randn(2, 3), 4)
