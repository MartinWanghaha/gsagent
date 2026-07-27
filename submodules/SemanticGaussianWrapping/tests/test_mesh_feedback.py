import threading

import numpy as np
import pytest
import torch

from mesh import TriangleMesh
from regularization import Phase
from regularization.mesh_feedback import (
    MeshCache,
    MeshFeedbackRegularizer,
    MeshSnapshotStamp,
)
from semantic.region_membership import SparseRegionMembership
from semantic.surface_field import (
    PartitionedSurfaceQueryResult,
    PointRegionSurfaceQueryResult,
    SurfaceQueryResult,
)
from training.engine import SemanticGaussianTrainer


def QueryResult(points):
    sdf = points.norm(dim=-1) - 1.0
    return SurfaceQueryResult(
        occupancy=torch.exp(-sdf.abs()),
        sdf=sdf,
        normal=torch.nn.functional.normalize(points, dim=-1),
        semantic=points.new_zeros((points.shape[0], 16)),
        geometry_posterior=points.new_full((points.shape[0], 5), 0.2),
        uncertainty=points.new_zeros(points.shape[0]),
        local_scale=points.new_ones(points.shape[0]),
    )


class Field(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_calls = 0

    def query(self, points, chunk_size=None):
        self.query_calls += 1
        return QueryResult(points)

    def query_partitioned(
        self,
        global_points,
        regional_points,
        regional_ids,
        chunk_size=None,
    ):
        del chunk_size
        self.query_calls += 1
        global_field = QueryResult(global_points)
        regional = QueryResult(regional_points)
        count = regional_ids.shape[1]
        point_regions = PointRegionSurfaceQueryResult(
            region_ids=regional_ids,
            valid=torch.ones_like(regional_ids, dtype=torch.bool),
            occupancy=regional.occupancy[:, None].expand(-1, count),
            sdf=regional.sdf[:, None].expand(-1, count),
            normal=regional.normal[:, None, :].expand(-1, count, -1),
            semantic=regional.semantic[:, None, :].expand(-1, count, -1),
            geometry_posterior=regional.geometry_posterior[:, None, :].expand(
                -1, count, -1
            ),
            uncertainty=regional.uncertainty[:, None].expand(-1, count),
            local_scale=regional.local_scale[:, None].expand(-1, count),
            support_fraction=regional_points.new_ones(regional_ids.shape),
        )
        return PartitionedSurfaceQueryResult(
            global_field=global_field,
            point_regions=point_regions,
        )


class Gaussians:
    def __init__(self):
        self.xyz = torch.nn.Parameter(torch.tensor([[0.9, 0.0, 0.0], [0.0, 0.8, 0.0]]))
        self.semantic = torch.zeros(2, 16)
        self.confidence = torch.ones(2, 1)
        self.scaling = torch.full((2, 3), 0.1)
        self.rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_semantic(self):
        return self.semantic

    @property
    def get_semantic_confidence(self):
        return self.confidence

    @property
    def get_scaling(self):
        return self.scaling

    @property
    def get_rotation(self):
        return self.rotation

    def point_region_memberships(self, indices, *, top_k, chunk_size):
        del chunk_size
        logits = torch.full((indices.numel(), 3), -2.0, device=indices.device)
        logits[:, 1] = 2.0
        logits[:, 2] = 1.0
        return SparseRegionMembership.from_logits(
            logits,
            top_k=top_k,
            confidence=self.confidence.index_select(0, indices),
        )


def test_cached_mesh_feedback_is_differentiable() -> None:
    field = Field()
    regularizer = MeshFeedbackRegularizer(field, enabled=True, sample_points=8)
    vertices = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    regularizer.cache = MeshCache(
        iteration=5,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    gaussians = Gaussians()
    loss = regularizer.loss(6, gaussians, field)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert gaussians.xyz.grad is not None
    assert torch.isfinite(gaussians.xyz.grad).all()


def test_one_point_feedback_sample_keeps_finite_tangent_loss() -> None:
    field = Field()
    regularizer = MeshFeedbackRegularizer(field, enabled=True, sample_points=1)
    vertices = torch.eye(3)
    regularizer.cache = MeshCache(
        iteration=5,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    regularizer.last_attempt_iteration = 5
    gaussians = Gaussians()

    prepared = regularizer.prepare(6, gaussians)
    assert prepared is not None and prepared.gaussian_centers.shape == (1, 3)
    loss = regularizer.loss(6, gaussians, field, prepared=prepared)

    assert loss is not None and torch.isfinite(loss)


def test_mesh_feedback_scipy_worker_budget_is_forwarded() -> None:
    regularizer = MeshFeedbackRegularizer(Field(), scipy_workers=1)
    regularizer.cache = MeshCache(
        iteration=1,
        vertices=torch.eye(3),
        normals=torch.eye(3),
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    calls = []

    class Tree:
        def query(self, points, *, k, workers):
            calls.append((points.shape, k, workers))
            return np.zeros(len(points)), np.zeros(len(points), dtype=np.int64)

    regularizer._mesh_tree = Tree()

    indices = regularizer._nearest_mesh_indices(torch.eye(3))

    assert torch.equal(indices, torch.zeros(3, dtype=torch.long))
    assert calls == [((3, 3), 1, 1)]


def test_real_surface_field_requires_minimal_surface_snapshot() -> None:
    class RealSurfaceField(Field):
        k_neighbors = 8

    class FullSnapshotOnly(Gaussians):
        def capture_inference(self, _device):
            raise AssertionError("full inference snapshot must not be used")

    regularizer = MeshFeedbackRegularizer(RealSurfaceField())

    with pytest.raises(RuntimeError, match="capture_surface_inference"):
        regularizer._frozen_job(1, FullSnapshotOnly())


def test_empty_live_model_skips_stale_mesh_feedback() -> None:
    field = Field()
    regularizer = MeshFeedbackRegularizer(field, enabled=True)
    vertices = torch.eye(3)
    regularizer.cache = MeshCache(
        iteration=5,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    regularizer.last_attempt_iteration = 5
    gaussians = Gaussians()
    gaussians.xyz = torch.nn.Parameter(gaussians.xyz[:0].detach().clone())
    gaussians.semantic = gaussians.semantic[:0]
    gaussians.confidence = gaussians.confidence[:0]
    gaussians.scaling = gaussians.scaling[:0]
    gaussians.rotation = gaussians.rotation[:0]

    assert regularizer.prepare(6, gaussians) is None
    assert regularizer.loss(6, gaussians, field) is None


def test_surface_and_mesh_losses_share_one_field_query() -> None:
    field = Field()
    gaussians = Gaussians()
    regularizer = MeshFeedbackRegularizer(
        field,
        enabled=True,
        refresh_interval=500,
        sample_points=8,
    )
    vertices = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    regularizer.cache = MeshCache(
        iteration=5,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    regularizer.last_attempt_iteration = 5

    trainer = object.__new__(SemanticGaussianTrainer)
    trainer.gaussians = gaussians
    trainer.surface_field = field
    trainer.mesh_feedback = regularizer
    trainer.region_top_k = 2
    trainer.region_decode_chunk_size = 32
    trainer.config = {
        "surface": {
            "enabled": True,
            "consistency_interval": 1,
            "sample_points": 2,
            "region_surface_min_weight": 0.05,
        }
    }

    surface_loss, mesh_loss = trainer._surface_losses(6, Phase.SURFACE_REFINE)

    assert surface_loss is not None and torch.isfinite(surface_loss)
    assert mesh_loss is not None and torch.isfinite(mesh_loss)
    assert field.query_calls == 1
    (surface_loss + mesh_loss).backward()
    assert gaussians.xyz.grad is not None
    regularizer.close()


def test_final_surface_step_does_not_launch_mesh_refresh(monkeypatch) -> None:
    field = Field()
    gaussians = Gaussians()
    regularizer = MeshFeedbackRegularizer(
        field,
        enabled=True,
        refresh_interval=5,
        sample_points=8,
    )
    vertices = torch.eye(3)
    regularizer.cache = MeshCache(
        iteration=1,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    regularizer.last_attempt_iteration = 1
    starts = []
    monkeypatch.setattr(
        regularizer,
        "_start_refresh",
        lambda iteration, model: starts.append(iteration),
    )
    trainer = object.__new__(SemanticGaussianTrainer)
    trainer.gaussians = gaussians
    trainer.surface_field = field
    trainer.mesh_feedback = regularizer
    trainer.region_top_k = 2
    trainer.region_decode_chunk_size = 32
    trainer.config = {
        "optimization": {"iterations": 6},
        "surface": {
            "enabled": True,
            "consistency_interval": 1,
            "sample_points": 2,
            "region_surface_min_weight": 0.05,
        },
    }

    surface_loss, mesh_loss = trainer._surface_losses(6, Phase.SURFACE_REFINE)

    assert surface_loss is not None
    assert mesh_loss is not None
    assert starts == []


def test_empty_mesh_refresh_obeys_attempt_interval() -> None:
    class EmptyExtractor:
        calls = 0

        def extract(self, _bounds):
            self.calls += 1
            return TriangleMesh.empty(16)

    field = Field()
    regularizer = MeshFeedbackRegularizer(field, enabled=True, refresh_interval=5)
    extractor = EmptyExtractor()
    regularizer.extractor = extractor
    gaussians = Gaussians()

    # Loss preparation is read-only with respect to extraction scheduling.
    assert regularizer.loss(1, gaussians, field) is None
    regularizer.after_optimizer_step(1, gaussians)
    assert extractor.calls == 1
    regularizer.after_optimizer_step(2, gaussians)
    regularizer.after_optimizer_step(5, gaussians)
    assert extractor.calls == 1
    regularizer.after_optimizer_step(1 + regularizer.retry_interval, gaussians)
    assert extractor.calls == 2
    assert regularizer.enabled


def test_mesh_feedback_cache_round_trips() -> None:
    field = Field()
    original = MeshFeedbackRegularizer(field)
    original.cache = MeshCache(
        iteration=9,
        vertices=torch.eye(3),
        normals=torch.eye(3),
        semantic=torch.ones(3, 16),
        uncertainty=torch.tensor([0.1, 0.2, 0.3]),
        faces=torch.tensor([[0, 1, 2]]),
    )
    original.last_attempt_iteration = 9
    restored = MeshFeedbackRegularizer(field)
    restored.load_state_dict(
        original.state_dict(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert restored.cache is not None and restored.cache.iteration == 9
    assert torch.equal(restored.cache.vertices, original.cache.vertices)
    assert restored.last_attempt_iteration == 9


@pytest.mark.parametrize("version", [1, 2, 3])
def test_mesh_feedback_rejects_pre_region_schema(version) -> None:
    regularizer = MeshFeedbackRegularizer(Field())
    with pytest.raises(ValueError, match="region-conditioned schema"):
        regularizer.load_state_dict(
            {"version": version},
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_mesh_feedback_rejects_unknown_future_checkpoint_schema() -> None:
    regularizer = MeshFeedbackRegularizer(Field())

    with pytest.raises(ValueError, match="newer schema"):
        regularizer.load_state_dict(
            {"version": 5},
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_nonfinite_current_checkpoint_cache_is_rejected() -> None:
    regularizer = MeshFeedbackRegularizer(Field())
    regularizer.cache = MeshCache(
        iteration=9,
        vertices=torch.eye(3),
        normals=torch.eye(3),
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    state = regularizer.state_dict()
    state["cache"]["uncertainty"][1] = float("nan")

    with pytest.raises(ValueError, match="checkpoint cache is invalid"):
        regularizer.load_state_dict(
            state,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_prepare_can_use_cache_without_starting_final_refresh(monkeypatch) -> None:
    field = Field()
    regularizer = MeshFeedbackRegularizer(field, refresh_interval=5)
    vertices = torch.eye(3)
    regularizer.cache = MeshCache(
        iteration=1,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    regularizer.last_attempt_iteration = 1
    starts = []
    monkeypatch.setattr(
        regularizer,
        "_start_refresh",
        lambda iteration, gaussians: starts.append(iteration),
    )

    prepared = regularizer.prepare(6, Gaussians())

    assert prepared is not None
    assert starts == []


def test_async_refresh_keeps_old_cache_until_atomic_install(monkeypatch) -> None:
    field = Field()
    regularizer = MeshFeedbackRegularizer(
        field,
        async_refresh=True,
        refresh_interval=5,
        min_component_faces=0,
    )
    vertices = torch.eye(3)
    regularizer.cache = MeshCache(
        iteration=1,
        vertices=vertices,
        normals=vertices,
        semantic=torch.zeros(3, 16),
        uncertainty=torch.zeros(3),
        faces=torch.tensor([[0, 1, 2]]),
    )
    mesh = TriangleMesh(
        vertices=(2.0 * torch.eye(3)).numpy(),
        faces=torch.tensor([[0, 1, 2]]).numpy(),
        normals=torch.eye(3).numpy(),
        semantic=torch.zeros(3, 16).numpy(),
        uncertainty=torch.zeros(3).numpy(),
    )
    release = threading.Event()
    monkeypatch.setattr(regularizer, "_frozen_job", lambda iteration, _: object())

    def extract(_):
        assert release.wait(timeout=2)
        return 6, MeshSnapshotStamp(6, 0, 0, 2), mesh

    monkeypatch.setattr(regularizer, "_extract_job", extract)

    gaussians = Gaussians()
    loss = regularizer.loss(6, gaussians, field)
    assert loss is not None
    assert regularizer.cache.iteration == 1
    regularizer.after_optimizer_step(6, gaussians)
    assert regularizer._pending is not None
    release.set()
    regularizer._pending.result(timeout=2)

    # Worker completion creates a candidate. It cannot replace the active
    # target until the next live-model quality gate, and accepted targets are
    # blended instead of causing an abrupt loss jump.
    regularizer._collect_completed(7, gaussians)
    assert regularizer.cache.iteration == 1
    assert regularizer._candidate is not None
    loss = regularizer.loss(7, gaussians, field)
    assert loss is not None
    assert regularizer.cache.iteration == 1
    assert regularizer._candidate is None
    if regularizer._incoming is not None:
        regularizer.prepare(7 + regularizer.blend_iterations, gaussians)
        assert regularizer.cache.iteration == 6
        assert torch.allclose(regularizer.cache.vertices, 2.0 * vertices)
    regularizer.close()


def test_invalid_mesh_is_rejected_before_feedback_installation() -> None:
    with pytest.raises(ValueError, match="vertices must be finite"):
        TriangleMesh(
            vertices=np.asarray(
                [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            normals=np.eye(3, dtype=np.float32),
            semantic=np.zeros((3, 16), dtype=np.float32),
            uncertainty=np.zeros(3, dtype=np.float32),
        )


def test_feedback_cleanup_removes_fragment_components() -> None:
    mesh = TriangleMesh(
        vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        ).numpy(),
        faces=torch.tensor([[0, 1, 2], [3, 4, 5]]).numpy(),
        uncertainty=torch.ones(6).numpy(),
    )
    regularizer = MeshFeedbackRegularizer(
        Field(),
        min_component_faces=2,
    )

    cleaned = regularizer._clean_mesh(mesh)

    assert len(cleaned.faces) == 0
    assert cleaned.metadata["components_removed"] == 2
