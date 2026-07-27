from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.dataset_readers import BasicPointCloud
from scene.gaussian_model import GaussianModel
from semantic.geometry_policy import GeometryEvidenceProjector, SoftGeometryPolicyBank
from semantic.neighbor_index import GaussianNeighborIndex
from semantic.surface_field import SemanticSurfaceField, SurfaceQueryContext, SurfaceQueryResult


def _model() -> GaussianModel:
    cloud = BasicPointCloud(
        points=np.array([[-0.2, 0, 0], [0.2, 0, 0], [0, 0.2, 0]], dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
    )
    model = GaussianModel(sh_degree=1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.configure_semantic_decoder(4, temperature=0.2)
    with torch.no_grad():
        model.registry["scaling"].fill_(math.log(0.25))
        model.registry["opacity"].fill_(0.0)
        model.registry["semantic_confidence"].fill_(0.9)
        model.registry["geometry_logits"][:, 0] = 2.0
    return model


class _CountingGaussianModel(GaussianModel):
    def __init__(self, *args, **kwargs) -> None:
        self.activation_getter_calls = {"scaling": 0, "rotation": 0, "opacity": 0}
        super().__init__(*args, **kwargs)

    @property
    def get_scaling(self) -> torch.Tensor:
        self.activation_getter_calls["scaling"] += 1
        return torch.exp(self.scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        self.activation_getter_calls["rotation"] += 1
        return torch.nn.functional.normalize(self.rotation, dim=-1, eps=1e-8)

    @property
    def get_opacity(self) -> torch.Tensor:
        self.activation_getter_calls["opacity"] += 1
        return torch.sigmoid(self.opacity)


def _random_model(count: int, model_type=GaussianModel) -> GaussianModel:
    generator = np.random.default_rng(31)
    cloud = BasicPointCloud(
        points=generator.normal(size=(count, 3)).astype(np.float32),
        colors=generator.random(size=(count, 3)).astype(np.float32),
        normals=np.zeros((count, 3), dtype=np.float32),
    )
    model = model_type(sh_degree=1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.configure_semantic_decoder(4, temperature=0.2)
    with torch.no_grad():
        model.registry["scaling"].clamp_(max=math.log(0.2))
        model.registry["semantic_confidence"].fill_(0.7)
        model.registry["geometry_logits"].normal_(0.0, 0.2)
    return model


def _region_model() -> GaussianModel:
    model = _model()
    with torch.no_grad():
        model.registry["semantic_embedding"].zero_()
        model.registry["semantic_embedding"][0, 0] = 1.0
        model.registry["semantic_embedding"][1:, 1] = 1.0
        decoder = model.semantic_decoder
        assert decoder is not None
        decoder.linear.weight.zero_()
        decoder.linear.bias.fill_(-8.0)
        decoder.linear.weight[1, 0] = 8.0
        decoder.linear.weight[2, 1] = 8.0
        decoder.linear.bias[1:3] = 0.0
    return model


def test_surface_inference_snapshot_is_minimal_restorable_and_query_exact() -> None:
    generator = np.random.default_rng(47)
    count = 64
    cloud = BasicPointCloud(
        points=generator.normal(size=(count, 3)).astype(np.float32),
        colors=generator.random(size=(count, 3)).astype(np.float32),
        normals=np.zeros((count, 3), dtype=np.float32),
    )
    model = GaussianModel(sh_degree=3, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.configure_semantic_decoder(3, temperature=0.2)
    with torch.no_grad():
        model.registry["scaling"].clamp_(max=math.log(0.3))
        model.registry["semantic_confidence"].uniform_(0.4, 0.9)
        model.registry["propagated_semantic_confidence"].uniform_(0.0, 0.7)
        model.registry["boundary_score"].uniform_(0.0, 1.0)
        model.registry["geometry_error"].uniform_(0.0, 0.5)
        model.registry["geometry_logits"].normal_(0.0, 0.4)
        model.policy_bank.profiles.add_(0.03125)
        model.semantic_decoder.linear.weight.normal_(0.0, 0.2)
        model.semantic_decoder.linear.bias.normal_(0.0, 0.1)
    model.clone(torch.arange(count) == 0)

    full_snapshot = model.capture_inference("cpu")
    surface_snapshot = model.capture_surface_inference("cpu")
    expected_names = {
        "xyz",
        "opacity",
        "scaling",
        "rotation",
        "semantic_embedding",
        "geometry_logits",
        "semantic_confidence",
        "propagated_semantic_confidence",
        "boundary_score",
        "geometry_error",
        "observation_count",
    }
    assert set(surface_snapshot["registry"]["tensors"]) == expected_names
    assert "features_dc" not in surface_snapshot["registry"]["tensors"]
    assert "features_rest" not in surface_snapshot["registry"]["tensors"]
    assert "max_radii2D" not in surface_snapshot["registry"]["tensors"]

    def tensor_bytes(snapshot) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in snapshot["registry"]["tensors"].values()
        )

    assert tensor_bytes(surface_snapshot) < 0.5 * tensor_bytes(full_snapshot)

    full_model = GaussianModel(sh_degree=3, semantic_dim=4, device="cpu")
    surface_model = GaussianModel(sh_degree=3, semantic_dim=4, device="cpu")
    full_model.restore(full_snapshot)
    surface_model.restore(surface_snapshot)
    assert surface_model.topology_stamp == full_model.topology_stamp
    assert torch.equal(
        surface_model.policy_bank.profiles,
        full_model.policy_bank.profiles,
    )
    assert surface_model.semantic_decoder is not None
    assert full_model.semantic_decoder is not None
    for name, value in full_model.semantic_decoder.state_dict().items():
        assert torch.equal(surface_model.semantic_decoder.state_dict()[name], value)

    points = torch.tensor(
        [[-0.2, 0.1, 0.3], [0.4, -0.3, 0.2], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    full_result = SemanticSurfaceField(
        full_model,
        k_neighbors=6,
        neighbor_backend="exact",
    ).query(points)
    surface_result = SemanticSurfaceField(
        surface_model,
        k_neighbors=6,
        neighbor_backend="exact",
    ).query(points)
    for name in full_result:
        assert torch.allclose(
            surface_result[name],
            full_result[name],
            atol=1e-7,
            rtol=1e-6,
        )

    frozen_xyz = surface_snapshot["registry"]["tensors"]["xyz"].clone()
    with torch.no_grad():
        model.registry["xyz"].add_(10.0)
    assert torch.equal(
        surface_snapshot["registry"]["tensors"]["xyz"],
        frozen_xyz,
    )


def test_surface_query_contract_mapping_and_gradients():
    model = _model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=3,
        query_chunk_size=2,
        gaussian_chunk_size=2,
    )
    points = torch.tensor([[0.0, 0.0, 0.05], [2.0, 2.0, 2.0]], requires_grad=True)
    result = field.query(points)
    assert isinstance(result, SurfaceQueryResult)
    assert result["sdf"] is result.sdf
    assert result.occupancy.shape == (2,)
    assert result.sdf.shape == (2,)
    assert result.normal.shape == (2, 3)
    assert result.semantic.shape == (2, 4)
    assert result.geometry_posterior.shape == (2, 5)
    assert result.uncertainty.shape == (2,)
    assert result.local_scale.shape == (2,)
    assert torch.all(result.local_scale > 0)
    assert result.occupancy[0] > result.occupancy[1]
    assert torch.isfinite(result.sdf).all()
    assert torch.isfinite(result.normal).all()
    result.sdf.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_geometry_query_matches_full_scalar_field_without_semantic_payload():
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        query_chunk_size=4,
        region_candidate_neighbors=3,
        neighbor_backend="exact",
    )
    points = torch.tensor(
        [[-0.2, 0.0, 0.02], [0.0, 0.1, 0.03], [0.2, 0.0, -0.01]]
    )
    full = field.query(points)
    geometry = field.query_geometry(points)

    assert geometry.semantic.shape == (len(points), 1)
    assert torch.count_nonzero(geometry.semantic) == 0
    for name in (
        "occupancy",
        "sdf",
        "normal",
        "geometry_posterior",
        "uncertainty",
        "local_scale",
    ):
        assert torch.allclose(
            getattr(geometry, name),
            getattr(full, name),
            atol=1e-6,
            rtol=1e-6,
        )


def test_compact_query_context_matches_dense_values_and_gradients():
    model = _random_model(17)
    field = SemanticSurfaceField(
        model,
        k_neighbors=4,
        query_chunk_size=2,
        gaussian_chunk_size=5,
        neighbor_backend="exact",
    )
    points = torch.tensor(
        [[-0.1, 0.2, 0.05], [0.3, -0.4, 0.2], [0.7, 0.1, -0.2]],
        requires_grad=True,
    )
    compact_context = field._prepare_query_context(points, field.k_neighbors)
    compact = field._query_chunk(
        points,
        compact_context.neighbor_indices[:, : field.k_neighbors],
        compact_context,
    )

    all_indices = torch.arange(len(model))
    support = field.neighbor_index.gather_support_attributes(all_indices, points, detach=False)
    semantic = field._gather_candidate_tensor(
        model,
        all_indices,
        points,
        raw_name="semantic_embedding",
        getter_name="get_semantic",
    )
    logits = field._gather_candidate_tensor(
        model,
        all_indices,
        points,
        raw_name="geometry_logits",
        getter_name="get_geometry_logits",
    )
    confidence = field._gather_semantic_confidence(all_indices, points).reshape(-1)
    boundary = field._gather_candidate_tensor(
        model,
        all_indices,
        points,
        raw_name="boundary_score",
        getter_name="get_boundary_score",
    )
    error = field._gather_candidate_tensor(
        model,
        all_indices,
        points,
        raw_name="geometry_error",
        getter_name="get_geometry_error",
    )
    policy = field.policy_bank(logits, confidence, boundary, error)
    global_neighbors = field._knn_indices(points, model.get_xyz, field.k_neighbors)
    dense_context = SurfaceQueryContext(
        neighbor_indices=global_neighbors,
        support=support,
        semantic=semantic,
        geometry_posterior=policy.posterior,
        surface_bandwidth=policy.surface_bandwidth,
        semantic_confidence=confidence,
        region_membership=model.point_region_memberships(
            all_indices,
            top_k=3,
            chunk_size=16,
        ),
    )
    dense = field._query_chunk(points, global_neighbors, dense_context)

    for key in compact:
        assert torch.allclose(compact[key], dense[key], atol=1e-6, rtol=1e-6)

    def scalar(result: SurfaceQueryResult) -> torch.Tensor:
        return (
            0.11 * result.occupancy.sum()
            + 0.13 * result.sdf.sum()
            + 0.17 * result.normal.sum()
            + 0.19 * result.semantic.sum()
            + 0.23 * result.geometry_posterior.square().sum()
            + 0.29 * result.uncertainty.sum()
            + 0.31 * result.local_scale.sum()
        )

    targets = (
        points,
        model.xyz,
        model.scaling,
        model.rotation,
        model.opacity,
        model.semantic_embedding,
        model.geometry_logits,
    )
    compact_gradients = torch.autograd.grad(scalar(compact), targets, retain_graph=True)
    dense_gradients = torch.autograd.grad(scalar(dense), targets)
    for compact_gradient, dense_gradient in zip(compact_gradients, dense_gradients):
        assert torch.isfinite(compact_gradient).all()
        assert torch.allclose(compact_gradient, dense_gradient, atol=2e-6, rtol=2e-5)


def test_query_cost_uses_one_compact_live_context_across_many_chunks(monkeypatch):
    model = _random_model(512, _CountingGaussianModel)
    field = SemanticSurfaceField(
        model,
        k_neighbors=4,
        query_chunk_size=1,
        gaussian_chunk_size=64,
        neighbor_backend="exact",
    )
    points = torch.randn(12, 3)
    gathers: list[tuple[bool, int]] = []
    original_gather = field.neighbor_index.gather_support_attributes

    def recording_gather(indices, reference, *, detach=False):
        gathers.append((detach, indices.numel()))
        return original_gather(indices, reference, detach=detach)

    monkeypatch.setattr(field.neighbor_index, "gather_support_attributes", recording_gather)
    policy_rows: list[int] = []
    hook = field.policy_bank.register_forward_hook(
        lambda _module, inputs, _output: policy_rows.append(inputs[0].shape[0])
    )
    try:
        result = field.query(points)
    finally:
        hook.remove()

    assert result.sdf.shape == (12,)
    assert model.activation_getter_calls == {"scaling": 0, "rotation": 0, "opacity": 0}
    detached = [entry for entry in gathers if entry[0]]
    assert sum(rows for _, rows in detached) == len(model)
    assert all(rows <= field.gaussian_chunk_size for _, rows in detached)
    assert len(policy_rows) == 1
    assert [entry for entry in gathers if not entry[0]] == [(False, policy_rows[0])]
    assert policy_rows[0] <= min(
        len(model),
        points.shape[0] * field.region_candidate_neighbors,
    )


def test_scipy_query_blocks_still_share_one_live_surface_context(monkeypatch):
    pytest.importorskip("scipy")
    model = _random_model(128)
    width = 16
    neighbor_index = GaussianNeighborIndex(
        model,
        backend="scipy",
        query_chunk_size=32,
        max_distance_bytes=32 * 4 * width * 2,
        support_candidate_budget=width,
    )
    field = SemanticSurfaceField(
        model,
        k_neighbors=4,
        query_chunk_size=2,
        neighbor_backend="scipy",
        neighbor_index=neighbor_index,
        max_distance_bytes=neighbor_index.max_distance_bytes,
        support_candidate_budget=width,
        region_candidate_neighbors=width,
    )
    points = torch.randn(9, 3)
    gathers: list[tuple[bool, int]] = []
    original_gather = neighbor_index.gather_support_attributes

    def recording_gather(indices, reference, *, detach=False):
        gathers.append((detach, indices.numel()))
        return original_gather(indices, reference, detach=detach)

    monkeypatch.setattr(neighbor_index, "gather_support_attributes", recording_gather)
    policy_rows: list[int] = []
    hook = field.policy_bank.register_forward_hook(
        lambda _module, inputs, _output: policy_rows.append(inputs[0].shape[0])
    )
    try:
        result = field.query(points)
    finally:
        hook.remove()

    assert result.sdf.shape == (points.shape[0],)
    assert len([entry for entry in gathers if entry[0]]) > 1
    live_gathers = [entry for entry in gathers if not entry[0]]
    assert live_gathers == [(False, policy_rows[0])]
    assert len(policy_rows) == 1
    assert policy_rows[0] <= min(
        len(model),
        points.shape[0] * field.region_candidate_neighbors,
    )


def test_surface_context_uses_propagated_semantic_confidence():
    model = _model()
    with torch.no_grad():
        model.registry["semantic_confidence"].zero_()
        model.registry["propagated_semantic_confidence"].copy_(
            torch.tensor([[0.2], [0.8], [0.4]])
        )
    field = SemanticSurfaceField(model, k_neighbors=2, neighbor_backend="exact")
    points = torch.tensor([[0.1, 0.0, 0.02]])
    context = field._prepare_query_context(points, 2)
    expected = model.get_semantic_confidence.index_select(0, context.support.indices)
    assert torch.equal(context.semantic_confidence, expected.reshape(-1))


def test_surface_sdf_grows_in_empty_far_field():
    model = _model()
    field = SemanticSurfaceField(model, k_neighbors=3, neighbor_backend="exact")
    points = torch.tensor([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]])
    result = field.query(points)
    assert result.sdf[0] > 0
    assert result.sdf[1] > result.sdf[0] * 100
    assert result.occupancy[1] < 1e-8


def test_low_confidence_policy_reduces_to_baseline():
    policy = SoftGeometryPolicyBank()
    logits = torch.tensor([[20.0, -5.0, -5.0, -5.0, -5.0]])
    output = policy(logits, torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1))
    assert torch.allclose(output.density_multiplier, torch.ones(1))
    assert torch.allclose(output.target_flatness, torch.ones(1))
    assert torch.allclose(output.sh_high_order_retention, torch.ones(1))
    assert torch.count_nonzero(output.normal_alignment_weight) == 0


def test_geometry_evidence_targets_are_soft_and_normalized():
    model = _model()
    projector = GeometryEvidenceProjector()
    indices, target = projector.sample_targets(model, max_points=2, k=2, search_chunk=2)
    assert indices.shape == (2,)
    assert target.shape == (2, 5)
    assert torch.all(target > 0)
    assert torch.allclose(target.sum(-1), torch.ones(2), atol=1e-5)


def test_geometry_evidence_reuses_index_and_matches_exact_small_scene():
    pytest.importorskip("scipy")
    model = _model()
    selected = torch.arange(len(model))
    exact_index = GaussianNeighborIndex(model, backend="exact", gaussian_chunk_size=2)
    scipy_index = GaussianNeighborIndex(model, backend="scipy")
    exact_projector = GeometryEvidenceProjector(neighbor_index=exact_index)
    scipy_projector = GeometryEvidenceProjector(neighbor_index=scipy_index)

    _, exact_target = exact_projector.target_distribution(model, selected, k=2)
    _, scipy_target = scipy_projector.target_distribution(model, selected, k=2)
    assert exact_projector.neighbor_index is exact_index
    assert scipy_projector.neighbor_index is scipy_index
    assert torch.allclose(scipy_target, exact_target, atol=1e-6)


def test_empty_surface_has_well_defined_shapes():
    model = GaussianModel(sh_degree=1, semantic_dim=6, device="cpu")
    field = SemanticSurfaceField(model)
    result = field(torch.empty(0, 3))
    assert result.semantic.shape == (0, 6)
    assert result.geometry_posterior.shape == (0, 5)


def test_surface_field_selects_by_anisotropic_support_not_center_distance():
    model = _model()
    with torch.no_grad():
        model.registry["scaling"].fill_(math.log(0.01))
        model.registry["scaling"][1].fill_(math.log(2.0))
    index = GaussianNeighborIndex(model, backend="exact", gaussian_chunk_size=1)
    field = SemanticSurfaceField(model, k_neighbors=1, neighbor_index=index)
    point = torch.tensor([[-0.19, 0.0, 0.0]])

    assert index.query_points(point, 1).item() == 0
    assert field._knn_indices(point, model.get_xyz, 1).item() == 1


def test_spatial_index_matches_exact_and_invalidates_on_topology_change():
    pytest.importorskip("scipy")
    model = _model()
    exact = SemanticSurfaceField(model, k_neighbors=2, neighbor_backend="exact")
    indexed = SemanticSurfaceField(model, k_neighbors=2, neighbor_backend="scipy")
    points = torch.tensor([[0.05, 0.02, 0.03], [-0.1, 0.1, 0.0]])
    exact_result = exact(points)
    indexed_result = indexed(points)
    for key in exact_result:
        assert torch.allclose(indexed_result[key], exact_result[key], atol=1e-6)
    old_signature = indexed.spatial_index_signature
    old_index = indexed._spatial_index
    model.clone(torch.tensor([True, False, False]))
    indexed(points)
    assert indexed.spatial_index_signature != old_signature
    assert indexed._spatial_index is not old_index


def test_region_ownership_separates_regions():
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        region_candidate_neighbors=3,
        region_min_membership=0.1,
        neighbor_backend="exact",
    )
    points = torch.tensor([[-0.2, 0.0, 0.02], [0.2, 0.0, 0.02]])
    ownership = field.query_region_ownership(
        points,
        region_ids=torch.tensor([1, 2], dtype=torch.long),
    )
    assert ownership.valid.tolist() == [True, True]
    assert ownership.region_id.tolist() == [1, 2]
    assert bool((ownership.confidence > 0).all())


def test_absent_region_is_explicitly_invalid_without_global_fallback():
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        region_candidate_neighbors=3,
        region_min_membership=0.1,
        neighbor_backend="exact",
    )
    result = field.query_point_regions(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[3]], dtype=torch.long),
    )

    assert not bool(result.valid.item())
    assert result.occupancy.item() == 0.0
    assert math.isinf(result.sdf.item())
    assert result.support_fraction.item() == 0.0
    assert result.uncertainty.item() == 1.0


def test_region_ownership_is_chunk_invariant():
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        region_candidate_neighbors=3,
        region_min_membership=0.1,
        neighbor_backend="exact",
    )
    points = torch.tensor(
        [[-0.2, 0.0, 0.02], [0.2, 0.0, 0.02], [0.0, 0.0, 0.02]]
    )
    region_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    whole = field.query_region_ownership(points, region_ids=region_ids)
    chunked = field.query_region_ownership(
        points,
        region_ids=region_ids,
        chunk_size=1,
    )
    assert torch.equal(whole.requested_region_ids, chunked.requested_region_ids)
    assert torch.equal(whole.region_id, chunked.region_id)
    assert torch.equal(whole.valid, chunked.valid)
    assert torch.allclose(whole.confidence, chunked.confidence, atol=1e-6, rtol=1e-6)


def test_point_region_query_is_chunk_invariant_and_differentiable():
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        query_chunk_size=4,
        region_candidate_neighbors=3,
        region_min_membership=0.1,
        neighbor_backend="exact",
    )
    points = torch.tensor(
        [[-0.2, 0.0, 0.02], [0.2, 0.0, 0.02]],
        requires_grad=True,
    )
    ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    whole = field.query_point_regions(points, ids, chunk_size=4)
    chunked = field.query_point_regions(points, ids, chunk_size=1)
    for key in whole:
        if whole[key].dtype == torch.bool or whole[key].dtype == torch.long:
            assert torch.equal(whole[key], chunked[key])
        else:
            assert torch.allclose(whole[key], chunked[key], atol=1e-6, rtol=1e-6)
    whole.sdf.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_partitioned_query_shares_routing_and_matches_separate_fields(monkeypatch):
    model = _region_model()
    field = SemanticSurfaceField(
        model,
        k_neighbors=2,
        query_chunk_size=4,
        region_candidate_neighbors=3,
        region_min_membership=0.1,
        neighbor_backend="exact",
    )
    global_points = torch.tensor([[0.0, 0.0, 0.03]])
    regional_points = torch.tensor(
        [[-0.2, 0.0, 0.02], [0.2, 0.0, 0.02]]
    )
    region_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    calls = 0
    prepare = field._prepare_query_context

    def record(points, k):
        nonlocal calls
        calls += 1
        return prepare(points, k)

    monkeypatch.setattr(field, "_prepare_query_context", record)
    shared = field.query_partitioned(
        global_points,
        regional_points,
        region_ids,
    )
    assert calls == 1

    expected_global = field.query(global_points)
    expected_regions = field.query_point_regions(regional_points, region_ids)
    for key in shared.global_field:
        assert torch.allclose(
            shared.global_field[key],
            expected_global[key],
            atol=1e-6,
            rtol=1e-6,
        )
    for key in shared.point_regions:
        actual = shared.point_regions[key]
        expected = expected_regions[key]
        if actual.dtype in (torch.bool, torch.long):
            assert torch.equal(actual, expected)
        else:
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "region_ids",
    [
        torch.tensor([2, 1], dtype=torch.long),
        torch.tensor([1, 1], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    ],
)
def test_region_ownership_rejects_invalid_region_domain(region_ids):
    field = SemanticSurfaceField(
        _region_model(),
        region_candidate_neighbors=8,
        neighbor_backend="exact",
    )
    with pytest.raises(ValueError):
        field.query_region_ownership(torch.zeros(1, 3), region_ids=region_ids)
