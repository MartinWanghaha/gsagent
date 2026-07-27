from concurrent.futures import Future
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from mesh import TriangleMesh
from regularization.mesh_feedback import (
    MeshCache,
    MeshFeedbackBatch,
    MeshFeedbackRegularizer,
    MeshQualityReport,
    MeshSnapshotStamp,
)


class Field:
    def query(self, points, chunk_size=None):
        del chunk_size
        return SimpleNamespace(
            sdf=points[:, 2],
            local_scale=torch.ones_like(points[:, 2]),
            normal=torch.tensor([0.0, 0.0, 1.0], device=points.device).expand_as(points),
            semantic=torch.tensor([1.0, 0.0], device=points.device).expand(len(points), -1),
            uncertainty=torch.zeros(len(points), device=points.device),
        )


class Gaussians:
    def __init__(self) -> None:
        self.xyz = torch.tensor(
            [[0.2, 0.2, 0.01], [0.35, 0.25, 0.02]],
            dtype=torch.float32,
        )
        self.semantic = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.scaling = torch.full((2, 3), 0.1)
        self.opacity = torch.full((2, 1), 0.9)
        self.confidence = torch.full((2, 1), 0.9)
        self.posterior = torch.tensor([[0.95, 0.05], [0.95, 0.05]])
        self.normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        self.generation = 0
        self.churn = 0

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_semantic(self):
        return self.semantic

    @property
    def get_scaling(self):
        return self.scaling

    @property
    def get_opacity(self):
        return self.opacity

    @property
    def get_semantic_confidence(self):
        return self.confidence

    @property
    def get_geometry_posterior(self):
        return self.posterior

    @property
    def get_normal(self):
        return self.normal

    def topology_stamp(self):
        return SimpleNamespace(
            generation=self.generation,
            cumulative_topology_churn=self.churn,
            gaussian_count=len(self.xyz),
        )


def _mesh(z: float = 0.0) -> TriangleMesh:
    vertices = np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
        dtype=np.float32,
    )
    return TriangleMesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        normals=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
        semantic=np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1)),
        uncertainty=np.zeros(3, dtype=np.float32),
    )


def _cache(
    source_iteration: int,
    *,
    generation: int = 0,
    churn: int = 0,
    gaussian_count: int = 2,
    z: float = 0.0,
) -> MeshCache:
    mesh = _mesh(z)
    return MeshCache(
        iteration=source_iteration,
        vertices=torch.from_numpy(mesh.vertices),
        normals=torch.from_numpy(mesh.normals),
        semantic=torch.from_numpy(mesh.semantic),
        uncertainty=torch.from_numpy(mesh.uncertainty),
        faces=torch.from_numpy(mesh.faces),
        source_iteration=source_iteration,
        topology_generation=generation,
        cumulative_topology_churn=churn,
        gaussian_count=gaussian_count,
    )


def _quality(score: float, *, accepted: bool, reason: str) -> MeshQualityReport:
    return MeshQualityReport(
        score=score,
        sdf_p90=0.1 if accepted else 10.0,
        normal_alignment=1.0 if accepted else 0.0,
        semantic_alignment=1.0 if accepted else 0.0,
        match_coverage=1.0 if accepted else 0.0,
        matches=2 if accepted else 0,
        probes=2,
        accepted=accepted,
        reason=reason,
    )


def test_refresh_starts_only_after_optimizer_and_captures_postcommit_stamp(monkeypatch) -> None:
    regularizer = MeshFeedbackRegularizer(Field(), sample_points=2, gate_probes=2, min_matches=1)
    gaussians = Gaussians()
    captured = []

    def record_start(iteration, model):
        captured.append(regularizer._topology_stamp(iteration, model))

    monkeypatch.setattr(regularizer, "_start_refresh", record_start)

    assert regularizer.prepare(10, gaussians) is None
    assert captured == []

    # This models the topology transaction committed between loss/backward and
    # the post-optimizer scheduling boundary.
    gaussians.generation = 3
    gaussians.churn = 7
    density_report = SimpleNamespace(cloned=0, split_parents=0, pruned=2)
    regularizer.after_optimizer_step(10, gaussians, density_report)

    assert len(captured) == 1
    assert captured[0] == MeshSnapshotStamp(10, 3, 7, 2)


def test_completed_worker_becomes_candidate_without_replacing_active() -> None:
    regularizer = MeshFeedbackRegularizer(Field(), min_component_faces=0)
    gaussians = Gaussians()
    active = _cache(10)
    regularizer.cache = active
    stamp = MeshSnapshotStamp(20, 0, 0, 2)
    completed = Future()
    completed.set_result((20, stamp, _mesh(z=0.01)))
    regularizer._pending = completed
    regularizer._pending_iteration = 20

    candidate = regularizer._collect_completed(21, gaussians)

    assert candidate is not None
    assert regularizer._candidate is candidate
    assert regularizer.cache is active
    assert regularizer._incoming is None
    assert regularizer._pending is None


def test_freshness_rejects_age_topology_events_and_churn() -> None:
    regularizer = MeshFeedbackRegularizer(
        Field(),
        max_candidate_age=10,
        max_topology_events=1,
        max_churn_ratio=0.015,
    )
    gaussians = Gaussians()

    assert regularizer._freshness_reason(_cache(0, gaussian_count=100), 11, gaussians).startswith("age:")

    gaussians.generation = 2
    assert regularizer._freshness_reason(_cache(0, gaussian_count=100), 5, gaussians).startswith(
        "topology_events:"
    )

    gaussians.generation = 1
    gaussians.churn = 2
    assert regularizer._freshness_reason(_cache(0, gaussian_count=100), 5, gaussians).startswith(
        "churn_ratio:"
    )


def test_quality_gate_rejects_bad_candidate_and_preserves_active(monkeypatch) -> None:
    regularizer = MeshFeedbackRegularizer(Field(), retry_interval=5)
    gaussians = Gaussians()
    active = _cache(10)
    candidate = _cache(20, z=0.1)
    regularizer.cache = active
    regularizer._candidate = candidate
    monkeypatch.setattr(
        regularizer,
        "_measure_quality",
        lambda cache, iteration, model, render_package: _quality(
            0.1,
            accepted=False,
            reason="sdf_p90:10.0>2.5",
        ),
    )

    regularizer._accept_or_reject_candidate(21, gaussians, None)

    assert regularizer.cache is active
    assert regularizer._candidate is None
    assert regularizer._incoming is None
    assert regularizer.rejected_candidates == 1
    assert regularizer.next_refresh_iteration == 26
    assert regularizer.last_rejection_reason.startswith("sdf_p90:")


def test_accepted_candidate_blends_with_smoothstep_then_promotes(monkeypatch) -> None:
    regularizer = MeshFeedbackRegularizer(Field(), blend_iterations=10)
    gaussians = Gaussians()
    active = _cache(100)
    candidate = _cache(100, z=0.01)
    regularizer.cache = active
    regularizer._candidate = candidate

    def measure(cache, iteration, model, render_package, **kwargs):
        del iteration, model, render_package, kwargs
        return _quality(
            0.9 if cache is candidate else 0.8,
            accepted=True,
            reason="accepted",
        )

    monkeypatch.setattr(regularizer, "_measure_quality", measure)
    regularizer._accept_or_reject_candidate(100, gaussians, None)

    assert regularizer.cache is active
    assert regularizer._incoming is candidate
    assert regularizer._blend_start == 100

    halfway = regularizer._blend_parts(105, gaussians)
    assert halfway == [(active, pytest.approx(0.5)), (candidate, pytest.approx(0.5))]
    assert regularizer.cache is active

    promoted = regularizer._blend_parts(110, gaussians)
    assert promoted == [(candidate, 1.0)]
    assert regularizer.cache is candidate
    assert regularizer._incoming is None
    # Freshness starts at live validation, not at the end of the transition.
    assert candidate.accepted_iteration == 100


class LossQuery:
    def __init__(self, scale: float) -> None:
        self.sdf = torch.tensor([1e6 * scale])
        self.local_scale = torch.tensor([scale])
        self.normal = torch.tensor([[0.0, 0.0, 1.0]])
        self.semantic = torch.tensor([[1.0, 0.0]])
        self.uncertainty = torch.zeros(1)


def _loss_at_scale(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    regularizer = MeshFeedbackRegularizer(Field(), robust_delta=1.5)
    center = torch.tensor([[0.0, 0.0, 1e6 * scale]], requires_grad=True)
    prepared = MeshFeedbackBatch(
        query_points=torch.tensor([[0.0, 0.0, 0.0]]),
        mesh_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        mesh_semantic=torch.tensor([[1.0, 0.0]]),
        mesh_confidence=torch.ones(1),
        gaussian_centers=center,
        gaussian_center_indices=torch.zeros(1, dtype=torch.long),
        nearest_vertices=torch.zeros(1, 3),
        nearest_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        mesh_local_spacing=torch.tensor([scale]),
        mesh_weights=torch.ones(1),
        gaussian_local_scale=torch.tensor([scale]),
        gaussian_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        nearest_local_spacing=torch.tensor([scale]),
        match_valid=torch.ones(1, dtype=torch.bool),
        match_weights=torch.ones(1),
    )
    gaussians = SimpleNamespace(get_scaling=torch.full((1, 3), scale))
    loss = regularizer.loss(
        1,
        gaussians,
        prepared=prepared,
        query_result=LossQuery(scale),
    )
    assert loss is not None
    loss.backward()
    assert center.grad is not None and torch.isfinite(center.grad).all()
    return loss.detach(), center.grad.detach()


def test_mesh_loss_is_bounded_and_scale_normalized() -> None:
    unit_loss, unit_gradient = _loss_at_scale(1.0)
    scaled_loss, scaled_gradient = _loss_at_scale(100.0)

    assert torch.isfinite(unit_loss)
    assert unit_loss.item() <= 1.975
    assert torch.allclose(unit_loss, scaled_loss, atol=1e-6)
    assert torch.isfinite(unit_gradient).all()
    assert torch.isfinite(scaled_gradient).all()


def test_prepare_publishes_bounded_mesh_coverage_signal() -> None:
    regularizer = MeshFeedbackRegularizer(
        Field(),
        sample_points=2,
        gate_probes=2,
        min_matches=1,
        match_k=1,
    )
    gaussians = Gaussians()
    regularizer.cache = _cache(10)

    prepared = regularizer.prepare(
        11,
        gaussians,
        render_package={"visibility_filter": torch.ones(2, dtype=torch.bool)},
    )
    signal = regularizer.pop_coverage_signal()

    assert prepared is not None
    assert signal is not None
    indices, residual, valid = signal
    assert indices.shape == residual.shape == valid.shape
    assert indices.numel() == 1  # bounded by sample_points // 2
    assert torch.isfinite(residual).all()
    assert ((0.0 <= residual) & (residual <= 1.0)).all()
    assert valid.dtype == torch.bool
    assert regularizer.pop_coverage_signal() is None


def test_checkpoint_drops_half_blend_and_reschedules_from_active(monkeypatch) -> None:
    original = MeshFeedbackRegularizer(Field(), sample_points=2, gate_probes=2, min_matches=1)
    original.cache = _cache(100)
    original._incoming = _cache(110, z=0.01)
    original._blend_start = 111
    state = original.state_dict()

    restored = MeshFeedbackRegularizer(Field(), sample_points=2, gate_probes=2, min_matches=1)
    restored.load_state_dict(state, device=torch.device("cpu"), dtype=torch.float32)

    assert restored.cache is not None and restored.cache.iteration == 100
    assert restored._incoming is None
    assert restored._active_needs_validation
    assert restored._desired_refresh
    assert restored.next_refresh_iteration == 0

    monkeypatch.setattr(
        restored,
        "_measure_quality",
        lambda *args, **kwargs: _quality(0.9, accepted=True, reason="accepted"),
    )
    gaussians = Gaussians()
    assert restored.prepare(112, gaussians) is not None
    starts = []
    monkeypatch.setattr(restored, "_start_refresh", lambda iteration, model: starts.append(iteration))
    restored.after_optimizer_step(112, gaussians)
    assert starts == [112]


def test_rejected_cache_owns_and_releases_its_triangle_projector() -> None:
    regularizer = MeshFeedbackRegularizer(Field(), match_k=1)
    candidate = _cache(10)
    projector = regularizer._projector(candidate)
    cache_reference = weakref.ref(candidate)
    projector_reference = weakref.ref(projector)

    regularizer._candidate = candidate
    regularizer._candidate = None
    del candidate, projector
    gc.collect()

    assert cache_reference() is None
    assert projector_reference() is None
