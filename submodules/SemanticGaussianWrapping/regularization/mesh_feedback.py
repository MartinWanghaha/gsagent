"""Versioned asynchronous mesh feedback for live Gaussian optimization.

Extraction is deliberately non-differentiable.  A worker receives an
immutable *post-topology-commit* snapshot, while the training thread owns the
candidate freshness/quality gate and every differentiable loss.  Consequently
a completed worker can never silently replace the target used by a step.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from mesh import (
    MeshSupportPolicy,
    MeshExtractionConfig,
    SemanticMeshExtractor,
    SurfaceFieldAdapter,
    TriangleMesh,
    gaussian_support_bounds,
    remove_small_components,
)
from utils.general_utils import build_rotation
from .mesh_correspondence import (
    TriangleMeshProjector,
    detached_local_scale,
    geman_mcclure,
)


@dataclass(frozen=True)
class MeshSnapshotStamp:
    """Topology identity captured by one extraction job."""

    source_iteration: int
    topology_generation: int
    cumulative_topology_churn: int
    gaussian_count: int


@dataclass
class MeshCache:
    """One immutable-in-practice mesh generation plus its source identity."""

    iteration: int
    vertices: torch.Tensor
    normals: torch.Tensor
    semantic: torch.Tensor
    uncertainty: torch.Tensor
    faces: torch.Tensor
    source_iteration: int | None = None
    topology_generation: int = 0
    cumulative_topology_churn: int = 0
    gaussian_count: int = 0
    accepted_iteration: int | None = None
    quality_score: float = 0.0
    _triangle_projector: TriangleMeshProjector | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def stamp(self) -> MeshSnapshotStamp:
        return MeshSnapshotStamp(
            int(self.iteration if self.source_iteration is None else self.source_iteration),
            int(self.topology_generation),
            int(self.cumulative_topology_churn),
            int(self.gaussian_count),
        )


@dataclass(frozen=True)
class MeshQualityReport:
    """Bounded live-model measurements used to accept one candidate."""

    score: float
    sdf_p90: float
    normal_alignment: float
    semantic_alignment: float
    match_coverage: float
    matches: int
    probes: int
    accepted: bool
    reason: str


@dataclass(frozen=True)
class MeshFeedbackBatch:
    """One stable, possibly blended feedback batch for a shared field query."""

    # The first eight fields preserve the public v3 batch contract.
    query_points: torch.Tensor
    mesh_normals: torch.Tensor
    mesh_semantic: torch.Tensor
    mesh_confidence: torch.Tensor
    gaussian_centers: torch.Tensor
    gaussian_center_indices: torch.Tensor
    nearest_vertices: torch.Tensor
    nearest_normals: torch.Tensor
    mesh_local_spacing: torch.Tensor | None = None
    mesh_weights: torch.Tensor | None = None
    gaussian_local_scale: torch.Tensor | None = None
    gaussian_normals: torch.Tensor | None = None
    nearest_local_spacing: torch.Tensor | None = None
    match_valid: torch.Tensor | None = None
    match_weights: torch.Tensor | None = None


@dataclass
class _RefreshJob:
    iteration: int
    gaussians: Any
    extractor: SemanticMeshExtractor
    stamp: MeshSnapshotStamp
    ready_event: Any | None = None


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
    weight = weight.detach().clamp_min(0)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class MeshFeedbackRegularizer:
    """Freshness-gated, scale-normalized bidirectional mesh supervision."""

    STATE_VERSION = 4
    MIN_MATCH_COVERAGE = 0.50

    def __init__(
        self,
        surface_field,
        *,
        refresh_interval: int = 500,
        resolution: int = 96,
        sample_points: int = 8192,
        padding: float = 0.05,
        support_sigma: float = 3.0,
        method: str = "marching_cubes",
        isovalue: float = 0.0,
        enabled: bool = True,
        async_refresh: bool = True,
        snapshot_device: str = "auto",
        min_component_faces: int = 64,
        scipy_workers: int = 1,
        max_candidate_age: int = 1_000,
        max_topology_events: int = 1,
        max_churn_ratio: float = 0.015,
        retry_interval: int = 100,
        blend_iterations: int = 125,
        gate_probes: int = 2_048,
        gate_min_score: float = 0.70,
        gate_sdf_p90: float = 2.5,
        gate_normal: float = 0.60,
        gate_semantic: float = 0.50,
        min_opacity: float = 0.05,
        min_confidence: float = 0.35,
        min_expert_certainty: float = 0.55,
        match_k: int = 16,
        match_radius: float = 3.0,
        match_semantic: float = 0.50,
        robust_delta: float = 1.5,
        min_matches: int = 256,
    ) -> None:
        if snapshot_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("snapshot_device must be auto, cpu, or cuda")
        if min_component_faces < 0:
            raise ValueError("min_component_faces must be non-negative")
        if scipy_workers == 0 or scipy_workers < -1:
            raise ValueError("scipy_workers must be -1 or a positive integer")
        integer_values = {
            "max_candidate_age": (max_candidate_age, 1),
            "max_topology_events": (max_topology_events, 0),
            "retry_interval": (retry_interval, 1),
            "blend_iterations": (blend_iterations, 0),
            "gate_probes": (gate_probes, 1),
            "match_k": (match_k, 1),
            "min_matches": (min_matches, 1),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        unit_values = {
            "max_churn_ratio": max_churn_ratio,
            "gate_min_score": gate_min_score,
            "gate_normal": gate_normal,
            "gate_semantic": gate_semantic,
            "min_opacity": min_opacity,
            "min_confidence": min_confidence,
            "min_expert_certainty": min_expert_certainty,
        }
        for name, value in unit_values.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1]")
        if not -1.0 <= float(match_semantic) <= 1.0:
            raise ValueError("match_semantic must lie in [-1,1]")
        for name, value in {
            "gate_sdf_p90": gate_sdf_p90,
            "match_radius": match_radius,
            "robust_delta": robust_delta,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.surface_field = surface_field
        self.refresh_interval = max(int(refresh_interval), 1)
        self.resolution = max(int(resolution), 16)
        self.sample_points = max(int(sample_points), 1)
        self.padding = float(padding)
        self.support_sigma = float(support_sigma)
        self.enabled = bool(enabled)
        self.async_refresh = bool(async_refresh)
        self.snapshot_device = str(snapshot_device)
        self.min_component_faces = int(min_component_faces)
        self.scipy_workers = int(scipy_workers)
        self.method = str(method)
        self.isovalue = float(isovalue)
        self.max_candidate_age = int(max_candidate_age)
        self.max_topology_events = int(max_topology_events)
        self.max_churn_ratio = float(max_churn_ratio)
        self.retry_interval = int(retry_interval)
        self.blend_iterations = int(blend_iterations)
        self.gate_probes = int(gate_probes)
        self.gate_min_score = float(gate_min_score)
        self.gate_sdf_p90 = float(gate_sdf_p90)
        self.gate_normal = float(gate_normal)
        self.gate_semantic = float(gate_semantic)
        self.min_opacity = float(min_opacity)
        self.min_confidence = float(min_confidence)
        self.min_expert_certainty = float(min_expert_certainty)
        self.match_k = int(match_k)
        self.match_radius = float(match_radius)
        self.match_semantic = float(match_semantic)
        self.robust_delta = float(robust_delta)
        self.min_matches = int(min_matches)

        self.extractor = self._make_extractor(surface_field)
        self.cache: MeshCache | None = None
        self._candidate: MeshCache | None = None
        self._incoming: MeshCache | None = None
        self._blend_start: int | None = None
        self._active_needs_validation = False
        self.last_attempt_iteration: int | None = None
        self.last_accepted_iteration: int | None = None
        self.next_refresh_iteration = 0
        self.last_error: str | None = None
        self.last_quality: MeshQualityReport | None = None
        self.last_rejection_reason: str | None = None
        self.accepted_candidates = 0
        self.rejected_candidates = 0
        self._desired_refresh = True
        self._executor: ThreadPoolExecutor | None = None
        self._pending: Future | None = None
        self._pending_iteration: int | None = None
        self._mesh_tree = None
        self._coverage_signal: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._last_blend_weight = 0.0
        self._last_active_freshness_weight = 0.0
        self._feedback_applied = False
        self._last_gate_eligible = 0
        self._last_live_eligible = 0
        self._last_loss_terms: dict[str, float] = {}
        self._last_iteration = 0

    def _make_extractor(self, field) -> SemanticMeshExtractor:
        blocks = 4
        cells = max(self.resolution // blocks, 4)
        semantic_field = (
            field if isinstance(field, SurfaceFieldAdapter) else SurfaceFieldAdapter(field)
        )
        return SemanticMeshExtractor(
            semantic_field,
            config=MeshExtractionConfig(
                method=self.method,
                scalar="sdf",
                level=self.isovalue,
                blocks_per_axis=blocks,
                block_cells=cells,
                max_grid_refinement=0,
                min_component_faces=0,
                preserve_semantic_instances=False,
            ),
        )

    def _clean_mesh(self, mesh: TriangleMesh) -> TriangleMesh:
        if self.min_component_faces <= 1 or not len(mesh.faces):
            return mesh
        return remove_small_components(
            mesh,
            min_faces=self.min_component_faces,
            preserve_semantic_instances=False,
        )

    @staticmethod
    def _topology_stamp(iteration: int, gaussians) -> MeshSnapshotStamp:
        stamp = getattr(gaussians, "topology_stamp", None)
        if callable(stamp):
            stamp = stamp()
        return MeshSnapshotStamp(
            int(iteration),
            int(getattr(stamp, "generation", getattr(gaussians, "topology_generation", 0))),
            int(
                getattr(
                    stamp,
                    "cumulative_topology_churn",
                    getattr(gaussians, "cumulative_topology_churn", 0),
                )
            ),
            int(getattr(stamp, "gaussian_count", gaussians.get_xyz.shape[0])),
        )

    @staticmethod
    def _tree_from_vertices(vertices: torch.Tensor):
        if vertices.shape[0] == 0:
            return None
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("mesh feedback cache contains non-finite vertices")
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return None
        return cKDTree(vertices.detach().float().cpu().numpy().copy())

    def _rebuild_mesh_tree(self) -> None:
        self._mesh_tree = None
        if self.cache is None:
            return
        try:
            self._mesh_tree = self._tree_from_vertices(self.cache.vertices)
        except (RuntimeError, ValueError) as error:
            self.last_error = str(error)
            self.cache = None

    def _projector(self, cache: MeshCache) -> TriangleMeshProjector:
        projector = cache._triangle_projector
        if projector is None:
            projector = TriangleMeshProjector(
                cache.vertices,
                cache.faces,
                normals=cache.normals,
                semantic=cache.semantic,
                uncertainty=cache.uncertainty,
                k_candidates=self.match_k,
                scipy_workers=self.scipy_workers,
            )
            cache._triangle_projector = projector
        return projector

    def _build_cache(
        self,
        iteration: int,
        mesh: TriangleMesh,
        *,
        device: torch.device,
        dtype: torch.dtype,
        semantic_dim: int | None,
        stamp: MeshSnapshotStamp | None = None,
        invalid_prefix: str = "surface extraction returned non-finite ",
    ) -> MeshCache | None:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            self.last_error = "surface extraction returned an empty mesh after cleanup"
            return None
        arrays = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "normals": mesh.normals,
            "semantic": mesh.semantic,
            "uncertainty": mesh.uncertainty,
        }
        invalid = [
            name
            for name, value in arrays.items()
            if value is not None and not np.isfinite(np.asarray(value)).all()
        ]
        if invalid:
            self.last_error = invalid_prefix + ", ".join(invalid)
            return None
        vertices = torch.as_tensor(mesh.vertices, device=device, dtype=dtype)
        faces = torch.as_tensor(mesh.faces, device=device, dtype=torch.long)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            self.last_error = "mesh feedback vertices must have shape [V,3]"
            return None
        if faces.ndim != 2 or faces.shape[-1] != 3:
            self.last_error = "mesh feedback faces must have shape [F,3]"
            return None
        if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
            self.last_error = "mesh feedback faces contain an out-of-range index"
            return None
        normals_np = mesh.normals if mesh.normals is not None else np.zeros_like(mesh.vertices)
        semantics_np = mesh.semantic
        if semantics_np is None:
            semantics_np = np.zeros(
                (len(mesh.vertices), 1 if semantic_dim is None else semantic_dim),
                dtype=np.float32,
            )
        uncertainty_np = mesh.uncertainty
        if uncertainty_np is None:
            uncertainty_np = np.ones(len(mesh.vertices), dtype=np.float32)
        stamp = stamp or MeshSnapshotStamp(int(iteration), 0, 0, 0)
        try:
            cache = MeshCache(
                iteration=int(iteration),
                vertices=vertices,
                normals=F.normalize(
                    torch.as_tensor(normals_np, device=device, dtype=dtype),
                    dim=-1,
                    eps=1e-8,
                ),
                semantic=torch.as_tensor(semantics_np, device=device, dtype=dtype),
                uncertainty=torch.as_tensor(
                    uncertainty_np,
                    device=device,
                    dtype=dtype,
                ).reshape(-1),
                faces=faces,
                source_iteration=stamp.source_iteration,
                topology_generation=stamp.topology_generation,
                cumulative_topology_churn=stamp.cumulative_topology_churn,
                gaussian_count=stamp.gaussian_count,
            )
            self._projector(cache)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self.last_error = f"mesh feedback index build failed: {error}"
            return None
        self.last_error = None
        return cache

    def _install_active(
        self,
        cache: MeshCache,
        iteration: int,
        report: MeshQualityReport | None = None,
    ) -> MeshCache:
        if cache.accepted_iteration is None:
            cache.accepted_iteration = int(iteration)
        if report is not None:
            cache.quality_score = float(report.score)
        self.cache = cache
        self._incoming = None
        self._blend_start = None
        self._active_needs_validation = False
        self.last_accepted_iteration = int(iteration)
        self._rebuild_mesh_tree()
        self.last_error = None
        return cache

    def _cache_from_mesh(self, iteration: int, mesh: TriangleMesh, gaussians) -> MeshCache | None:
        """Validate a synchronous extraction and publish it atomically."""

        xyz = gaussians.get_xyz
        cache = self._build_cache(
            iteration,
            mesh,
            device=xyz.device,
            dtype=xyz.dtype,
            semantic_dim=int(gaussians.get_semantic.shape[-1]),
            stamp=self._topology_stamp(iteration, gaussians),
        )
        return None if cache is None else self._install_active(cache, iteration)

    @staticmethod
    def _serialized_cache_mesh(cache: dict[str, object]) -> TriangleMesh:
        def numpy(name: str, *, required: bool = True):
            value = cache.get(name)
            if value is None:
                if required:
                    raise ValueError(f"mesh feedback cache is missing {name}")
                return None
            return torch.as_tensor(value).detach().float().cpu().numpy()

        return TriangleMesh(
            vertices=numpy("vertices"),
            faces=torch.as_tensor(cache["faces"]).detach().long().cpu().numpy(),
            normals=numpy("normals", required=False),
            semantic=numpy("semantic", required=False),
            uncertainty=numpy("uncertainty", required=False),
        )

    @staticmethod
    def _serialize_cache(cache: MeshCache | None) -> dict[str, object] | None:
        if cache is None:
            return None
        return {
            "iteration": cache.iteration,
            "source_iteration": cache.stamp.source_iteration,
            "topology_generation": cache.topology_generation,
            "cumulative_topology_churn": cache.cumulative_topology_churn,
            "gaussian_count": cache.gaussian_count,
            "accepted_iteration": cache.accepted_iteration,
            "quality_score": cache.quality_score,
            "vertices": cache.vertices.detach().clone(),
            "normals": cache.normals.detach().clone(),
            "semantic": cache.semantic.detach().clone(),
            "uncertainty": cache.uncertainty.detach().clone(),
            "faces": cache.faces.detach().clone(),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "enabled": self.enabled,
            "last_attempt_iteration": self.last_attempt_iteration,
            "last_accepted_iteration": self.last_accepted_iteration,
            "next_refresh_iteration": self.next_refresh_iteration,
            "last_error": self.last_error,
            "last_rejection_reason": self.last_rejection_reason,
            "accepted_candidates": self.accepted_candidates,
            "rejected_candidates": self.rejected_candidates,
            "pending_iteration": self._pending_iteration,
            "cache": self._serialize_cache(self.cache),
            # Candidate/worker/blend state is intentionally coalesced into a
            # fresh post-resume request. Serializing a second full mesh can
            # add hundreds of MiB and still cannot restore the worker context.
            "incoming": None,
            "blend_was_in_flight": self._incoming is not None,
            "candidate_was_pending": self._candidate is not None,
            "desired_refresh": bool(
                self._desired_refresh
                or self._pending is not None
                or self._candidate is not None
                or self._incoming is not None
            ),
        }

    def _deserialize_cache(
        self,
        value: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MeshCache | None:
        if not isinstance(value, dict):
            return None
        mesh = self._serialized_cache_mesh(value)
        stamp = MeshSnapshotStamp(
            int(value["source_iteration"]),
            int(value["topology_generation"]),
            int(value["cumulative_topology_churn"]),
            int(value["gaussian_count"]),
        )
        cache = self._build_cache(
            int(value["iteration"]),
            mesh,
            device=device,
            dtype=dtype,
            semantic_dim=None,
            stamp=stamp,
            invalid_prefix="mesh feedback cache contains non-finite ",
        )
        if cache is None:
            raise ValueError(self.last_error or "mesh feedback cache is empty")
        accepted = value.get("accepted_iteration")
        cache.accepted_iteration = None if accepted is None else int(accepted)
        cache.quality_score = float(value.get("quality_score", 0.0))
        return cache

    def load_state_dict(
        self,
        state: dict[str, object] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not state:
            raise ValueError("mesh feedback checkpoint state is required")
        version = state.get("version")
        if type(version) is not int:
            raise ValueError("mesh feedback checkpoint has an invalid schema version")
        if version != self.STATE_VERSION:
            if version > self.STATE_VERSION:
                raise ValueError(
                    f"mesh feedback checkpoint uses newer schema version {version}"
                )
            raise ValueError(
                "mesh feedback checkpoint does not contain the region-conditioned "
                "schema; start a fresh run"
            )
        self.enabled = bool(state.get("enabled", self.enabled))
        self.last_error = None if state.get("last_error") is None else str(state["last_error"])
        self.last_rejection_reason = (
            None
            if state.get("last_rejection_reason") is None
            else str(state["last_rejection_reason"])
        )
        self.accepted_candidates = int(state.get("accepted_candidates", 0))
        self.rejected_candidates = int(state.get("rejected_candidates", 0))
        attempt = state.get("last_attempt_iteration")
        self.last_attempt_iteration = None if attempt is None else int(attempt)
        accepted = state.get("last_accepted_iteration")
        self.last_accepted_iteration = None if accepted is None else int(accepted)
        self.next_refresh_iteration = int(state.get("next_refresh_iteration", 0))
        self.cache = None
        self._candidate = None
        self._incoming = None
        self._active_needs_validation = False
        try:
            restored = self._deserialize_cache(
                state.get("cache"),
                device=device,
                dtype=dtype,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"mesh feedback checkpoint cache is invalid: {error}"
            ) from error
        self.cache = restored
        # Worker/candidate/blend state is process-local. Resume from the last
        # accepted active mesh and recapture live topology after the next commit.
        self._incoming = None
        self._blend_start = None
        self._active_needs_validation = restored is not None
        # Worker state is process-local and can never cross a checkpoint.
        pending_was_in_flight = state.get("pending_iteration") is not None
        blend_was_in_flight = bool(state.get("blend_was_in_flight", False)) or isinstance(
            state.get("incoming"),
            dict,
        )
        candidate_was_pending = bool(state.get("candidate_was_pending", False))
        self._pending = None
        self._pending_iteration = None
        self._desired_refresh = bool(
            state.get("desired_refresh", False)
            or pending_was_in_flight
            or blend_was_in_flight
            or candidate_was_pending
        )
        if self._desired_refresh:
            # The serialized worker cannot survive process restart. Make the
            # first post-commit scheduling boundary recapture the live model.
            self.next_refresh_iteration = 0
        if self.cache is None and self._candidate is None:
            self._desired_refresh = True

    def _resolve_snapshot_device(self, gaussians) -> torch.device:
        live_device = gaussians.get_xyz.device
        if self.snapshot_device == "cpu" or live_device.type != "cuda":
            return torch.device("cpu")
        if self.snapshot_device == "cuda":
            return live_device
        # Auto mode is conservative for dense scenes: an independent CUDA
        # stream does not isolate allocator pressure from live backward. CPU
        # keeps the 2M-Gaussian counter run safe; users can still explicitly
        # request a CUDA snapshot after measuring their own headroom.
        if int(gaussians.get_xyz.shape[0]) > 500_000:
            return torch.device("cpu")
        surface_names = {
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
        snapshot_bytes = sum(
            value.numel() * value.element_size()
            for name, value in gaussians.registry.named_attributes()
            if name in surface_names
        )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(live_device)
        except (RuntimeError, TypeError):
            return torch.device("cpu")
        reserve = max(4 * 1024 * 1024 * 1024, 4 * snapshot_bytes)
        return live_device if free_bytes > snapshot_bytes + reserve else torch.device("cpu")

    @torch.no_grad()
    def _frozen_job(self, iteration: int, gaussians) -> _RefreshJob | None:
        if not hasattr(self.surface_field, "k_neighbors"):
            return None
        capture = getattr(gaussians, "capture_surface_inference", None)
        if not callable(capture):
            raise RuntimeError(
                "asynchronous mesh feedback requires "
                "gaussians.capture_surface_inference()"
            )
        from scene.gaussian_model import GaussianModel
        from semantic.surface_field import SemanticSurfaceField

        stamp = self._topology_stamp(iteration, gaussians)
        target = self._resolve_snapshot_device(gaussians)
        snapshot = capture(target)
        live_policy = getattr(gaussians, "policy_bank", None)
        frozen = GaussianModel(
            int(snapshot["max_sh_degree"]),
            int(snapshot["semantic_dim"]),
            int(snapshot["geometry_experts"]),
            target,
            confidence_floor=float(
                snapshot.get(
                    "policy_confidence_floor",
                    getattr(live_policy, "confidence_floor", 0.05),
                )
            ),
        )
        frozen.restore(snapshot)
        if live_policy is not None:
            frozen.policy_bank.load_state_dict(
                {
                    name: value.detach().to(device=target, copy=True)
                    for name, value in live_policy.state_dict().items()
                }
            )
        field = SemanticSurfaceField(
            frozen,
            policy_bank=frozen.policy_bank,
            k_neighbors=int(self.surface_field.k_neighbors),
            query_chunk_size=int(self.surface_field.query_chunk_size),
            gaussian_chunk_size=int(self.surface_field.gaussian_chunk_size),
            occupancy_iso=float(self.surface_field.occupancy_iso),
            density_scale=float(self.surface_field.density_scale),
            semantic_decoder=frozen.semantic_decoder,
            max_distance_bytes=int(self.surface_field.max_distance_bytes),
            neighbor_backend=str(self.surface_field.neighbor_backend),
            support_log_cutoff=float(self.surface_field.support_log_cutoff),
            support_candidate_budget=int(self.surface_field.support_candidate_budget),
            support_routing_query_chunk=int(
                self.surface_field.support_routing_query_chunk
            ),
            scipy_workers=self.scipy_workers,
            region_top_k=int(self.surface_field.region_top_k),
            region_decode_chunk_size=int(
                self.surface_field.region_decode_chunk_size
            ),
            region_candidate_neighbors=int(
                self.surface_field.region_candidate_neighbors
            ),
            region_min_membership=float(
                self.surface_field.region_min_membership
            ),
        )
        object.__setattr__(field, "_frozen_gaussians_owner", frozen)
        ready_event = None
        if target.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(target))
        return _RefreshJob(
            int(iteration),
            frozen,
            self._make_extractor(field),
            stamp,
            ready_event,
        )

    def _trusted_support_indices(self, gaussians) -> torch.Tensor:
        return MeshSupportPolicy(
            min_opacity=self.min_opacity,
            min_semantic_confidence=self.min_confidence,
            require_observation=True,
            trim_quantile=0.001,
        ).selected_indices(gaussians)

    def _extract_job(self, job: _RefreshJob):
        if job.ready_event is not None:
            job.ready_event.synchronize()
        device = job.gaussians.get_xyz.device

        def extract():
            trusted = self._trusted_support_indices(job.gaussians)
            if trusted.numel() < 3:
                raise ValueError("fewer than three high-confidence Gaussian supports")
            bounds = gaussian_support_bounds(
                job.gaussians,
                sigma=self.support_sigma,
                relative_padding=self.padding,
                selection=trusted,
                trim_quantile=0.001,
            )
            return self._clean_mesh(job.extractor.extract(bounds))

        if device.type != "cuda":
            return job.iteration, job.stamp, extract()
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device, priority=0)
            with torch.cuda.stream(stream):
                mesh = extract()
            stream.synchronize()
        return job.iteration, job.stamp, mesh

    def _record_refresh_error(self, error: BaseException, iteration: int | None = None) -> None:
        self.last_error = str(error)
        self._desired_refresh = True
        if iteration is not None:
            self.next_refresh_iteration = int(iteration) + self.retry_interval
        if isinstance(error, ImportError):
            self.enabled = False
        warnings.warn(
            f"mesh feedback refresh failed: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    @torch.no_grad()
    def refresh(self, iteration: int, gaussians) -> MeshCache | None:
        """Explicit synchronous refresh for tools; training uses candidates."""

        self.last_attempt_iteration = int(iteration)
        if gaussians.get_xyz.shape[0] == 0:
            return None
        try:
            trusted = self._trusted_support_indices(gaussians)
            if trusted.numel() < 3:
                raise ValueError("fewer than three high-confidence Gaussian supports")
            bounds = gaussian_support_bounds(
                gaussians,
                sigma=self.support_sigma,
                relative_padding=self.padding,
                selection=trusted,
                trim_quantile=0.001,
            )
            mesh = self._clean_mesh(self.extractor.extract(bounds))
        except (RuntimeError, ValueError, ImportError) as error:
            self._record_refresh_error(error, iteration)
            return None
        return self._cache_from_mesh(iteration, mesh, gaussians)

    @torch.no_grad()
    def _start_refresh(self, iteration: int, gaussians) -> None:
        if self._pending is not None or self._candidate is not None:
            self._desired_refresh = True
            return
        self.last_attempt_iteration = int(iteration)
        self.next_refresh_iteration = int(iteration) + self.refresh_interval
        self._desired_refresh = False
        try:
            job = self._frozen_job(iteration, gaussians)
        except (RuntimeError, ValueError, ImportError) as error:
            self._record_refresh_error(error, iteration)
            return
        if job is None:
            # Lightweight external fields have no immutable snapshot contract.
            # Their synchronous result still enters the same live quality gate.
            try:
                trusted = self._trusted_support_indices(gaussians)
                bounds = gaussian_support_bounds(
                    gaussians,
                    sigma=self.support_sigma,
                    relative_padding=self.padding,
                    selection=trusted,
                    trim_quantile=0.001,
                )
                mesh = self._clean_mesh(self.extractor.extract(bounds))
                self._candidate = self._build_cache(
                    iteration,
                    mesh,
                    device=gaussians.get_xyz.device,
                    dtype=gaussians.get_xyz.dtype,
                    semantic_dim=int(gaussians.get_semantic.shape[-1]),
                    stamp=self._topology_stamp(iteration, gaussians),
                )
                if self._candidate is None:
                    self.next_refresh_iteration = int(iteration) + self.retry_interval
                    self._desired_refresh = True
            except (RuntimeError, ValueError, ImportError) as error:
                self._record_refresh_error(error, iteration)
            return
        if not self.async_refresh:
            try:
                source_iteration, stamp, mesh = self._extract_job(job)
                self._candidate = self._build_cache(
                    source_iteration,
                    mesh,
                    device=gaussians.get_xyz.device,
                    dtype=gaussians.get_xyz.dtype,
                    semantic_dim=int(gaussians.get_semantic.shape[-1]),
                    stamp=stamp,
                )
                if self._candidate is None:
                    self.next_refresh_iteration = int(iteration) + self.retry_interval
                    self._desired_refresh = True
            except (RuntimeError, ValueError, ImportError) as error:
                self._record_refresh_error(error, iteration)
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="semantic-mesh-feedback",
            )
        self._pending_iteration = int(iteration)
        self._pending = self._executor.submit(self._extract_job, job)

    @torch.no_grad()
    def _collect_completed(self, iteration: int, gaussians) -> MeshCache | None:
        if self._pending is None or not self._pending.done():
            return None
        future = self._pending
        self._pending = None
        self._pending_iteration = None
        try:
            source_iteration, stamp, mesh = future.result()
            candidate = self._build_cache(
                int(source_iteration),
                mesh,
                device=gaussians.get_xyz.device,
                dtype=gaussians.get_xyz.dtype,
                semantic_dim=int(gaussians.get_semantic.shape[-1]),
                stamp=stamp,
            )
            if candidate is None:
                self.next_refresh_iteration = int(iteration) + self.retry_interval
                self._desired_refresh = True
                return None
            self._candidate = candidate
            return candidate
        except Exception as error:  # worker failures must never kill training
            self._record_refresh_error(error, iteration)
            self._desired_refresh = True
            return None

    def _freshness_reason(
        self,
        cache: MeshCache,
        iteration: int,
        gaussians,
        *,
        include_age: bool = True,
    ) -> str | None:
        current = self._topology_stamp(iteration, gaussians)
        source = cache.stamp
        age = max(int(iteration) - source.source_iteration, 0)
        if include_age and age > self.max_candidate_age:
            return f"age:{age}>{self.max_candidate_age}"
        events = current.topology_generation - source.topology_generation
        if events < 0:
            return "future_topology_generation"
        if events > self.max_topology_events:
            return f"topology_events:{events}>{self.max_topology_events}"
        churn = current.cumulative_topology_churn - source.cumulative_topology_churn
        if churn < 0:
            return "future_topology_churn"
        if source.gaussian_count > 0:
            ratio = churn / max(source.gaussian_count, 1)
            if ratio > self.max_churn_ratio:
                return f"churn_ratio:{ratio:.6f}>{self.max_churn_ratio:.6f}"
        return None

    def _active_freshness_weight(self, cache: MeshCache, iteration: int) -> float:
        """Fade validated targets instead of deleting them on one iteration."""

        validated = (
            cache.stamp.source_iteration
            if cache.accepted_iteration is None
            else cache.accepted_iteration
        )
        age = max(int(iteration) - int(validated), 0)
        full_until = max(self.refresh_interval, self.max_candidate_age // 2)
        fade_until = full_until + self.max_candidate_age
        if age <= full_until:
            return 1.0
        if age >= fade_until:
            return 0.0
        progress = (age - full_until) / max(fade_until - full_until, 1)
        return 1.0 - self._smoothstep(progress)

    @staticmethod
    def _bounded_indices(
        indices: torch.Tensor,
        count: int,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        if indices.numel() <= count:
            return indices
        if deterministic:
            positions = (
                torch.arange(count, device=indices.device, dtype=torch.long)
                * indices.numel()
                // count
            )
            return indices.index_select(0, positions)
        return indices.index_select(
            0,
            torch.randperm(indices.numel(), device=indices.device)[:count],
        )

    @staticmethod
    def _selected_attribute(
        gaussians,
        indices: torch.Tensor,
        *,
        raw_name: str,
        getter_name: str | None,
        activation=None,
    ) -> torch.Tensor | None:
        registry = getattr(gaussians, "registry", None)
        source = None
        if registry is not None and raw_name in registry:
            source = registry[raw_name]
        elif getter_name is None:
            source = getattr(gaussians, raw_name, None)
        if isinstance(source, torch.Tensor) and source.shape[0] == gaussians.get_xyz.shape[0]:
            selected = source.detach().index_select(0, indices.to(source.device))
            selected = selected.to(indices.device)
            return selected if activation is None else activation(selected)
        if getter_name is None:
            return None
        source = getattr(gaussians, getter_name, None)
        source = source() if callable(source) else source
        if not isinstance(source, torch.Tensor) or source.shape[0] != gaussians.get_xyz.shape[0]:
            return None
        return source.detach().index_select(0, indices.to(source.device)).to(indices.device)

    def _selected_opacity(self, gaussians, indices: torch.Tensor) -> torch.Tensor:
        value = self._selected_attribute(
            gaussians,
            indices,
            raw_name="opacity",
            getter_name="get_opacity",
            activation=torch.sigmoid,
        )
        if value is None:
            return torch.ones(len(indices), device=indices.device)
        return value.reshape(len(indices), -1).max(-1).values

    def _selected_semantic_confidence(
        self,
        gaussians,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        direct = self._selected_attribute(
            gaussians,
            indices,
            raw_name="semantic_confidence",
            getter_name=None,
        )
        propagated = self._selected_attribute(
            gaussians,
            indices,
            raw_name="propagated_semantic_confidence",
            getter_name=None,
        )
        if direct is not None:
            value = direct if propagated is None else torch.maximum(direct, propagated)
        else:
            value = self._selected_attribute(
                gaussians,
                indices,
                raw_name="__missing_semantic_confidence__",
                getter_name="get_semantic_confidence",
            )
        if value is None:
            return torch.ones(len(indices), device=indices.device)
        return value.reshape(len(indices), -1).max(-1).values.clamp(0, 1)

    def _selected_expert_certainty(
        self,
        gaussians,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        posterior = self._selected_attribute(
            gaussians,
            indices,
            raw_name="geometry_logits",
            getter_name="get_geometry_posterior",
            activation=lambda value: value.float().softmax(dim=-1).to(value.dtype),
        )
        if posterior is None:
            return torch.ones(len(indices), device=indices.device)
        return posterior.reshape(len(indices), -1).max(-1).values

    def _selected_scaling(self, gaussians, indices: torch.Tensor) -> torch.Tensor:
        scaling = self._selected_attribute(
            gaussians,
            indices,
            raw_name="scaling",
            getter_name="get_scaling",
            activation=torch.exp,
        )
        if scaling is None:
            return torch.ones((len(indices), 3), device=indices.device)
        return scaling

    def _selected_normals(self, gaussians, indices: torch.Tensor) -> torch.Tensor:
        rotation = self._selected_attribute(
            gaussians,
            indices,
            raw_name="rotation",
            getter_name=None,
            activation=lambda value: F.normalize(value, dim=-1, eps=1e-8),
        )
        if rotation is None:
            normal = self._selected_attribute(
                gaussians,
                indices,
                raw_name="__missing_normal__",
                getter_name="get_normal",
            )
            return torch.zeros((len(indices), 3), device=indices.device) if normal is None else normal
        scaling = self._selected_scaling(gaussians, indices)
        matrix = build_rotation(rotation)
        axis = scaling.argmin(dim=-1)
        batch = torch.arange(len(indices), device=indices.device)
        return F.normalize(matrix[batch, :, axis], dim=-1, eps=1e-8)

    @torch.no_grad()
    def _eligible_indices(
        self,
        gaussians,
        render_package: dict[str, torch.Tensor] | None,
        *,
        count: int,
        deterministic: bool,
    ) -> torch.Tensor:
        total = int(gaussians.get_xyz.shape[0])
        device = gaussians.get_xyz.device
        pool_size = min(
            max(int(count) * 16, self.min_matches * 8, 65_536),
            total,
        )
        if render_package is not None:
            visible = render_package.get("visibility_filter")
            if isinstance(visible, torch.Tensor) and visible.numel() == total:
                candidates = visible.detach().to(device=device).reshape(-1).bool().nonzero(as_tuple=False).flatten()
            else:
                candidates = torch.arange(total, device=device)
        else:
            if pool_size < total:
                candidates = (
                    torch.arange(pool_size, device=device, dtype=torch.long)
                    * total
                    // pool_size
                )
            else:
                candidates = torch.arange(total, device=device)
        candidates = self._bounded_indices(
            candidates,
            pool_size,
            deterministic=deterministic,
        )
        keep = self._selected_opacity(gaussians, candidates) >= self.min_opacity
        keep &= (
            self._selected_semantic_confidence(gaussians, candidates)
            >= self.min_confidence
        )
        keep &= (
            self._selected_expert_certainty(gaussians, candidates)
            >= self.min_expert_certainty
        )
        if render_package is None:
            observed = self._selected_attribute(
                gaussians,
                candidates,
                raw_name="observation_count",
                getter_name=None,
            )
            if observed is not None:
                keep &= observed.reshape(len(candidates), -1).max(-1).values > 0
        return self._bounded_indices(
            candidates[keep],
            count,
            deterministic=deterministic,
        )

    def _mesh_sample(
        self,
        cache: MeshCache,
        count: int,
        face_indices: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if face_indices is None:
            face_pool = torch.arange(
                len(cache.faces),
                device=cache.faces.device,
                dtype=torch.long,
            )
        else:
            face_pool = torch.unique(
                face_indices.detach().to(device=cache.faces.device, dtype=torch.long)
            )
            face_pool = face_pool[(face_pool >= 0) & (face_pool < len(cache.faces))]
            if face_pool.numel() == 0:
                raise ValueError("local mesh sampling received no valid faces")
        pooled_faces = cache.faces.index_select(0, face_pool)
        triangles = cache.vertices[pooled_faces]
        edges = torch.stack(
            (
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            ),
            dim=1,
        )
        area = torch.linalg.vector_norm(
            torch.cross(edges[:, 0], -edges[:, 2], dim=-1),
            dim=-1,
        )
        probabilities = torch.where(torch.isfinite(area), area, torch.zeros_like(area))
        if float(probabilities.sum()) <= 0:
            probabilities = torch.ones_like(probabilities)
        selected = torch.multinomial(
            probabilities,
            int(count),
            replacement=True,
            generator=generator,
        )
        chosen_faces = pooled_faces[selected]
        chosen_triangles = triangles[selected]
        random = torch.rand(
            (count, 2),
            device=cache.vertices.device,
            dtype=cache.vertices.dtype,
            generator=generator,
        )
        root = random[:, :1].sqrt()
        barycentric = torch.cat((1.0 - root, root * (1.0 - random[:, 1:]), root * random[:, 1:]), dim=1)
        points = (chosen_triangles * barycentric[..., None]).sum(1)

        vertex_normals = cache.normals[chosen_faces]
        normals = F.normalize((vertex_normals * barycentric[..., None]).sum(1), dim=-1, eps=1e-8)
        face_normals = F.normalize(
            torch.cross(chosen_triangles[:, 1] - chosen_triangles[:, 0], chosen_triangles[:, 2] - chosen_triangles[:, 0], dim=-1),
            dim=-1,
            eps=1e-8,
        )
        normals = torch.where((normals.norm(dim=-1) > 1e-6)[:, None], normals, face_normals)
        semantic = (cache.semantic[chosen_faces] * barycentric[..., None]).sum(1)
        uncertainty = (cache.uncertainty[chosen_faces] * barycentric).sum(1).clamp(0, 1)
        spacing = edges[selected].square().sum(-1).mean(-1).sqrt().clamp_min(1e-8)
        return points, normals, semantic, 1.0 - uncertainty, spacing

    def _required_matches(self, eligible_count: int) -> int:
        coverage_floor = math.ceil(self.MIN_MATCH_COVERAGE * eligible_count)
        return min(
            eligible_count,
            max(self.min_matches, coverage_floor),
        )

    def _semantic_gate_available(
        self,
        cache: MeshCache,
        query_semantic: torch.Tensor,
    ) -> bool:
        return (
            cache.semantic.ndim == 2
            and query_semantic.ndim == 2
            and cache.semantic.shape[-1] == query_semantic.shape[-1]
            and bool((cache.semantic.norm(dim=-1) > 1e-6).any())
            and bool((query_semantic.norm(dim=-1) > 1e-6).any())
        )

    @torch.no_grad()
    def _project_live_centers(
        self,
        cache: MeshCache,
        gaussians,
        indices: torch.Tensor,
    ):
        centers = gaussians.get_xyz.index_select(0, indices).detach()
        semantic = gaussians.get_semantic.index_select(0, indices).detach()
        scale = self._selected_scaling(gaussians, indices).detach().min(-1).values
        use_semantic = self._semantic_gate_available(cache, semantic)
        return self._projector(cache).project(
            centers,
            query_semantic=semantic if use_semantic else None,
            semantic_min_cosine=self.match_semantic if use_semantic else None,
            query_scale=scale,
            radius_factor=self.match_radius,
        )

    @torch.no_grad()
    def _measure_quality(
        self,
        cache: MeshCache,
        iteration: int,
        gaussians,
        render_package: dict[str, torch.Tensor] | None,
        *,
        enforce_candidate_age: bool = True,
    ) -> MeshQualityReport:
        del render_package
        reason = self._freshness_reason(
            cache,
            iteration,
            gaussians,
            include_age=enforce_candidate_age,
        )
        if reason is not None:
            return MeshQualityReport(0.0, math.inf, 0.0, 0.0, 0.0, 0, 0, False, reason)
        generator = torch.Generator(device=cache.vertices.device)
        generator.manual_seed(
            (
                1_000_003 * cache.stamp.source_iteration
                + 97 * cache.topology_generation
                + 17
            )
            % (2**63 - 1)
        )
        points, normals, semantic, confidence, spacing = self._mesh_sample(
            cache,
            self.gate_probes,
            generator=generator,
        )
        query = self.surface_field.query(points)
        local_scale = getattr(query, "local_scale", torch.ones_like(query.sdf))
        scale = detached_local_scale(local_scale, spacing)
        normalized_sdf = (query.sdf.abs() / scale).detach()
        finite = torch.isfinite(normalized_sdf)
        finite_fraction = float(finite.float().mean())
        if finite_fraction < 0.99:
            return MeshQualityReport(
                0.0,
                math.inf,
                0.0,
                0.0,
                0.0,
                0,
                len(points),
                False,
                f"finite_probes:{finite_fraction:.4f}<0.9900",
            )
        sdf_p90 = float(torch.quantile(normalized_sdf[finite].float(), 0.90))
        weights = (0.25 + 0.75 * confidence).detach()
        normal_value = (
            F.normalize(query.normal, dim=-1, eps=1e-8) * normals
        ).sum(-1).abs()
        normal_alignment = float(_weighted_mean(normal_value, weights))

        semantic_alignment = 1.0
        semantic_available = semantic.shape == query.semantic.shape
        if semantic_available:
            semantic_valid = (semantic.norm(dim=-1) > 1e-6) & (query.semantic.norm(dim=-1) > 1e-6)
            semantic_available = bool(semantic_valid.any())
            if semantic_available:
                semantic_alignment = float(
                    _weighted_mean(
                        F.cosine_similarity(query.semantic, semantic, dim=-1, eps=1e-8),
                        weights * semantic_valid,
                    )
                )

        eligible = self._eligible_indices(
            gaussians,
            # Publication is a scene-level decision and must not depend on
            # whichever random training view happened to finish the worker.
            None,
            count=self.gate_probes,
            deterministic=True,
        )
        self._last_gate_eligible = int(eligible.numel())
        if eligible.numel() == 0:
            return MeshQualityReport(0.0, sdf_p90, normal_alignment, semantic_alignment, 0.0, 0, len(points), False, "no_visible_confident_support")
        total_gaussians = int(gaussians.get_xyz.shape[0])
        if total_gaussians >= self.min_matches and eligible.numel() < self.min_matches:
            return MeshQualityReport(
                0.0,
                sdf_p90,
                normal_alignment,
                semantic_alignment,
                0.0,
                0,
                len(points),
                False,
                f"eligible_support:{eligible.numel()}<{self.min_matches}",
            )
        projection = self._project_live_centers(cache, gaussians, eligible)
        matches = int(projection.valid.sum())
        coverage = matches / max(int(eligible.numel()), 1)
        required = self._required_matches(int(eligible.numel()))
        sdf_score = 1.0 / (1.0 + (sdf_p90 / self.gate_sdf_p90) ** 2)
        score = (
            0.40 * sdf_score
            + 0.25 * normal_alignment
            + 0.15 * semantic_alignment
            + 0.20 * coverage
        )
        checks = [
            (matches >= required, f"matches:{matches}<{required}"),
            (
                coverage >= self.MIN_MATCH_COVERAGE,
                f"coverage:{coverage:.4f}<{self.MIN_MATCH_COVERAGE:.4f}",
            ),
            (sdf_p90 <= self.gate_sdf_p90, f"sdf_p90:{sdf_p90:.4f}>{self.gate_sdf_p90:.4f}"),
            (normal_alignment >= self.gate_normal, f"normal:{normal_alignment:.4f}<{self.gate_normal:.4f}"),
            (
                not semantic_available or semantic_alignment >= self.gate_semantic,
                f"semantic:{semantic_alignment:.4f}<{self.gate_semantic:.4f}",
            ),
            (score >= self.gate_min_score, f"score:{score:.4f}<{self.gate_min_score:.4f}"),
        ]
        failure = next((message for passed, message in checks if not passed), None)
        return MeshQualityReport(
            float(score),
            sdf_p90,
            normal_alignment,
            semantic_alignment,
            float(coverage),
            matches,
            len(points),
            failure is None,
            "accepted" if failure is None else failure,
        )

    @torch.no_grad()
    def _accept_or_reject_candidate(
        self,
        iteration: int,
        gaussians,
        render_package: dict[str, torch.Tensor] | None,
    ) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        try:
            report = self._measure_quality(candidate, iteration, gaussians, render_package)
            if report.accepted and self.cache is not None:
                active_report = self._measure_quality(
                    self.cache,
                    iteration,
                    gaussians,
                    render_package,
                    enforce_candidate_age=False,
                )
                if active_report.accepted and report.score + 0.02 < active_report.score:
                    report = MeshQualityReport(
                        report.score,
                        report.sdf_p90,
                        report.normal_alignment,
                        report.semantic_alignment,
                        report.match_coverage,
                        report.matches,
                        report.probes,
                        False,
                        f"worse_than_active:{report.score:.4f}<{active_report.score:.4f}",
                    )
        except (RuntimeError, TypeError, ValueError) as error:
            report = MeshQualityReport(0.0, math.inf, 0.0, 0.0, 0.0, 0, 0, False, f"quality_error:{error}")
        self.last_quality = report
        self._candidate = None
        if not report.accepted:
            self.rejected_candidates += 1
            self.last_rejection_reason = report.reason
            self.next_refresh_iteration = int(iteration) + self.retry_interval
            self._desired_refresh = True
            return
        candidate.quality_score = report.score
        candidate.accepted_iteration = int(iteration)
        self.accepted_candidates += 1
        self.last_rejection_reason = None
        self.last_error = None
        self._desired_refresh = False
        if self.cache is None or self.blend_iterations == 0:
            self._install_active(candidate, iteration, report)
        else:
            self._incoming = candidate
            self._blend_start = int(iteration)

    @torch.no_grad()
    def _validate_restored_active(
        self,
        iteration: int,
        gaussians,
        render_package: dict[str, torch.Tensor] | None,
    ) -> None:
        if not self._active_needs_validation or self.cache is None:
            return
        restored = self.cache
        self._active_needs_validation = False
        try:
            report = self._measure_quality(
                restored,
                iteration,
                gaussians,
                render_package,
                enforce_candidate_age=False,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            report = MeshQualityReport(
                0.0,
                math.inf,
                0.0,
                0.0,
                0.0,
                0,
                0,
                False,
                f"quality_error:{error}",
            )
        self.last_quality = report
        if report.accepted:
            restored.quality_score = report.score
            self.cache = restored
            self.last_error = None
            return
        self.cache = None
        self._incoming = None
        self._blend_start = None
        self.rejected_candidates += 1
        self.last_rejection_reason = "restored_" + report.reason
        self._desired_refresh = True
        self.next_refresh_iteration = int(iteration) + self.retry_interval

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(float(value), 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _blend_parts(
        self,
        iteration: int,
        gaussians,
    ) -> list[tuple[MeshCache, float]]:
        if self.cache is None:
            return []
        active_reason = self._freshness_reason(
            self.cache,
            iteration,
            gaussians,
            include_age=False,
        )
        if active_reason is not None:
            self.last_rejection_reason = "active_" + active_reason
            self._desired_refresh = True
            self.next_refresh_iteration = min(
                self.next_refresh_iteration,
                int(iteration),
            )
            self.cache = None
            self._incoming = None
            self._blend_start = None
            return []
        active_weight = self._active_freshness_weight(self.cache, iteration)
        self._last_active_freshness_weight = active_weight
        if active_weight <= 0.0:
            self.last_rejection_reason = "active_freshness_weight:0"
            self._desired_refresh = True
            self.next_refresh_iteration = min(self.next_refresh_iteration, int(iteration))
        if self._incoming is None or self._blend_start is None:
            self._last_blend_weight = 0.0
            return [] if active_weight <= 0.0 else [(self.cache, active_weight)]
        incoming_reason = self._freshness_reason(self._incoming, iteration, gaussians)
        if incoming_reason is not None:
            self.last_rejection_reason = "incoming_" + incoming_reason
            self.rejected_candidates += 1
            self._incoming = None
            self._blend_start = None
            self._desired_refresh = True
            self.next_refresh_iteration = int(iteration) + self.retry_interval
            self._last_blend_weight = 0.0
            return [] if active_weight <= 0.0 else [(self.cache, active_weight)]
        progress = (
            1.0
            if self.blend_iterations == 0
            else (int(iteration) - self._blend_start) / self.blend_iterations
        )
        weight = self._smoothstep(progress)
        self._last_blend_weight = weight
        if progress >= 1.0:
            incoming = self._incoming
            assert incoming is not None
            self._install_active(incoming, iteration, self.last_quality)
            self._last_blend_weight = 1.0
            return [(incoming, 1.0)]
        parts = []
        if weight < 1.0 and active_weight > 0.0:
            parts.append((self.cache, (1.0 - weight) * active_weight))
        if weight > 0.0:
            parts.append((self._incoming, weight))
        return parts

    @torch.no_grad()
    def after_optimizer_step(
        self,
        iteration: int,
        gaussians,
        density_report=None,
        *,
        allow_refresh: bool = True,
    ) -> None:
        """Schedule only after optimizer and any topology transaction commit."""

        self._last_iteration = int(iteration)
        if not self.enabled:
            return
        self._collect_completed(iteration, gaussians)
        topology_changed = density_report is not None and any(
            int(getattr(density_report, name, 0)) > 0
            for name in ("cloned", "split_parents", "pruned")
        )
        if topology_changed:
            self._desired_refresh = True
            self.next_refresh_iteration = min(
                self.next_refresh_iteration,
                int(iteration),
            )
        if not allow_refresh or gaussians.get_xyz.shape[0] == 0:
            return
        due = (
            topology_changed
            or self.last_attempt_iteration is None
            or int(iteration) >= self.next_refresh_iteration
        )
        if due and self._pending is None and self._candidate is None and self._incoming is None:
            self._start_refresh(iteration, gaussians)

    def _prepare_cache_part(
        self,
        cache: MeshCache,
        blend_weight: float,
        gaussians,
        eligible: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if (
            int(gaussians.get_xyz.shape[0]) >= self.min_matches
            and eligible.numel() < self.min_matches
        ):
            self.last_rejection_reason = (
                f"live_eligible_support:{eligible.numel()}<{self.min_matches}"
            )
            return None
        projection = self._project_live_centers(cache, gaussians, eligible)
        matches = int(projection.valid.sum())
        required = self._required_matches(int(eligible.numel()))
        if matches < required:
            self.last_rejection_reason = f"live_matches:{matches}<{required}"
            return None
        mesh_points, mesh_normals, mesh_semantic, mesh_confidence, mesh_spacing = self._mesh_sample(
            cache,
            self.sample_points,
            projection.face_indices[projection.valid],
        )
        centers = gaussians.get_xyz.index_select(0, eligible)
        local_scale = self._selected_scaling(gaussians, eligible).min(-1).values
        gaussian_normals = self._selected_normals(gaussians, eligible)
        support_weight = self._selected_semantic_confidence(gaussians, eligible)
        target_confidence = (
            centers.new_ones(len(eligible))
            if projection.uncertainty is None
            else 1.0 - projection.uncertainty.to(centers).clamp(0, 1)
        )
        match_weight = (
            float(blend_weight)
            * support_weight.detach().clamp(0, 1)
            * target_confidence.detach()
            * projection.valid.to(centers.dtype)
        )
        return {
            "query_points": mesh_points,
            "mesh_normals": mesh_normals,
            "mesh_semantic": mesh_semantic,
            "mesh_confidence": mesh_confidence,
            "mesh_spacing": mesh_spacing,
            "mesh_weights": mesh_confidence * float(blend_weight),
            "centers": centers,
            "indices": eligible,
            "nearest": projection.closest_points.to(centers),
            "nearest_normals": projection.normals.to(centers),
            "nearest_spacing": projection.local_spacing.to(centers),
            "local_scale": local_scale,
            "gaussian_normals": gaussian_normals,
            "match_valid": projection.valid.to(centers.device),
            "match_weights": match_weight,
            "distance": projection.distance.to(centers),
        }

    def prepare(
        self,
        iteration: int,
        gaussians,
        *,
        render_package: dict[str, torch.Tensor] | None = None,
    ) -> MeshFeedbackBatch | None:
        """Validate candidates and build one stable shared-query context."""
        self._last_iteration = int(iteration)
        self._coverage_signal = None
        self._feedback_applied = False
        self._last_loss_terms = {}
        self._last_live_eligible = 0
        if not self.enabled or gaussians.get_xyz.shape[0] == 0:
            return None
        self._collect_completed(iteration, gaussians)
        self._validate_restored_active(iteration, gaussians, render_package)
        self._accept_or_reject_candidate(iteration, gaussians, render_package)
        parts = self._blend_parts(iteration, gaussians)
        if not parts:
            return None
        eligible = self._eligible_indices(
            gaussians,
            render_package,
            count=min(self.gate_probes, max(self.sample_points // 2, 1)),
            deterministic=False,
        )
        self._last_live_eligible = int(eligible.numel())
        if eligible.numel() == 0:
            self.last_rejection_reason = "no_visible_confident_support"
            return None

        prepared = []
        for cache, weight in parts:
            part = self._prepare_cache_part(cache, weight, gaussians, eligible)
            if part is not None:
                prepared.append(part)
        if not prepared:
            return None

        # Coverage is a topology signal, not a long-range gradient. Prefer the
        # dominant target during a blend so each Gaussian contributes once.
        coverage_part = max(prepared, key=lambda value: float(value["mesh_weights"].mean()))
        coverage_scale = detached_local_scale(
            coverage_part["local_scale"],
            coverage_part["nearest_spacing"],
        )
        valid = coverage_part["match_valid"]
        normalized_distance = coverage_part["distance"] / coverage_scale
        residual = torch.where(
            valid,
            geman_mcclure(normalized_distance, self.robust_delta),
            torch.ones_like(normalized_distance),
        )
        self._coverage_signal = (
            coverage_part["indices"].detach(),
            residual.detach(),
            torch.ones_like(valid),
        )

        def concatenate(name: str) -> torch.Tensor:
            return torch.cat([part[name] for part in prepared], dim=0)

        return MeshFeedbackBatch(
            query_points=concatenate("query_points"),
            mesh_normals=concatenate("mesh_normals"),
            mesh_semantic=concatenate("mesh_semantic"),
            mesh_confidence=concatenate("mesh_confidence"),
            gaussian_centers=concatenate("centers"),
            gaussian_center_indices=concatenate("indices"),
            nearest_vertices=concatenate("nearest").detach(),
            nearest_normals=concatenate("nearest_normals").detach(),
            mesh_local_spacing=concatenate("mesh_spacing").detach(),
            mesh_weights=concatenate("mesh_weights").detach(),
            gaussian_local_scale=concatenate("local_scale"),
            gaussian_normals=concatenate("gaussian_normals"),
            nearest_local_spacing=concatenate("nearest_spacing").detach(),
            match_valid=concatenate("match_valid").detach(),
            match_weights=concatenate("match_weights").detach(),
        )

    def pop_coverage_signal(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        signal = self._coverage_signal
        self._coverage_signal = None
        return signal

    def loss(
        self,
        iteration: int,
        gaussians,
        surface_field=None,
        *,
        prepared: MeshFeedbackBatch | None = None,
        query_result=None,
    ) -> torch.Tensor | None:
        if prepared is None:
            prepared = self.prepare(iteration, gaussians)
        if prepared is None:
            return None
        field = surface_field or self.surface_field
        query = field.query(prepared.query_points) if query_result is None else query_result
        if query.sdf.shape[0] != prepared.query_points.shape[0]:
            raise ValueError("mesh feedback query result has the wrong length")

        mesh_spacing = (
            torch.ones_like(query.sdf)
            if prepared.mesh_local_spacing is None
            else prepared.mesh_local_spacing.to(query.sdf)
        )
        query_scale = getattr(query, "local_scale", torch.ones_like(query.sdf))
        normalized_sdf = query.sdf / detached_local_scale(query_scale, mesh_spacing)
        mesh_weights = (
            prepared.mesh_confidence
            if prepared.mesh_weights is None
            else prepared.mesh_weights
        ).to(query.sdf)
        query_uncertainty = getattr(query, "uncertainty", None)
        if isinstance(query_uncertainty, torch.Tensor):
            mesh_weights = mesh_weights * (1.0 - query_uncertainty.detach().clamp(0, 1))
        surface_loss = _weighted_mean(
            geman_mcclure(normalized_sdf, self.robust_delta),
            mesh_weights,
        )
        normal_error = 1.0 - (
            F.normalize(query.normal, dim=-1, eps=1e-8)
            * prepared.mesh_normals
        ).sum(-1).abs()
        normal_loss = _weighted_mean(normal_error, mesh_weights)

        semantic_loss = surface_loss.new_zeros(())
        semantic_target = prepared.mesh_semantic
        if semantic_target.shape == query.semantic.shape:
            valid_semantic = (
                (semantic_target.norm(dim=-1) > 1e-6)
                & (query.semantic.norm(dim=-1) > 1e-6)
            )
            if bool(valid_semantic.any()):
                semantic_error = 0.5 * (
                    1.0
                    - F.cosine_similarity(
                        query.semantic,
                        semantic_target.detach(),
                        dim=-1,
                        eps=1e-8,
                    )
                )
                semantic_loss = _weighted_mean(
                    semantic_error,
                    mesh_weights * valid_semantic,
                )

        gaussian_scale = (
            gaussians.get_scaling[prepared.gaussian_center_indices].min(-1).values
            if prepared.gaussian_local_scale is None
            else prepared.gaussian_local_scale
        )
        nearest_spacing = (
            torch.ones_like(gaussian_scale)
            if prepared.nearest_local_spacing is None
            else prepared.nearest_local_spacing.to(gaussian_scale)
        )
        tangent_scale = detached_local_scale(gaussian_scale, nearest_spacing)
        tangent_residual = (
            (prepared.gaussian_centers - prepared.nearest_vertices)
            * prepared.nearest_normals
        ).sum(-1) / tangent_scale
        match_weights = (
            torch.ones_like(tangent_residual)
            if prepared.match_weights is None
            else prepared.match_weights.to(tangent_residual)
        )
        if prepared.match_valid is not None:
            match_weights = match_weights * prepared.match_valid.to(match_weights)
        tangent_loss = _weighted_mean(
            geman_mcclure(tangent_residual, self.robust_delta),
            match_weights,
        )
        gaussian_normal_loss = surface_loss.new_zeros(())
        if prepared.gaussian_normals is not None:
            gaussian_normal_error = 1.0 - (
                F.normalize(prepared.gaussian_normals, dim=-1, eps=1e-8)
                * prepared.nearest_normals
            ).sum(-1).abs()
            gaussian_normal_loss = _weighted_mean(gaussian_normal_error, match_weights)

        total = (
            surface_loss
            + 0.25 * normal_loss
            + 0.10 * semantic_loss
            + 0.50 * tangent_loss
            + 0.125 * gaussian_normal_loss
        )
        self._last_loss_terms = {
            "mesh_surface": float(surface_loss.detach()),
            "mesh_normal": float(normal_loss.detach()),
            "mesh_semantic": float(semantic_loss.detach()),
            "mesh_tangent": float(tangent_loss.detach()),
            "mesh_gaussian_normal": float(gaussian_normal_loss.detach()),
            "mesh_total_bounded": float(total.detach()),
        }
        self._feedback_applied = True
        return total

    def _nearest_mesh_indices(self, centers: torch.Tensor) -> torch.Tensor:
        """Retained only for v3 diagnostics/tests; losses use triangles."""

        if self.cache is None:
            raise RuntimeError("mesh cache is unavailable")
        if self._mesh_tree is not None:
            points = centers.detach().float().cpu().numpy()
            try:
                _, indices = self._mesh_tree.query(points, k=1, workers=self.scipy_workers)
            except TypeError:
                _, indices = self._mesh_tree.query(points, k=1)
            return torch.as_tensor(indices, device=centers.device, dtype=torch.long)
        vertices = self.cache.vertices
        outputs = []
        for start in range(0, centers.shape[0], 256):
            outputs.append(torch.cdist(centers[start : start + 256].detach(), vertices).argmin(1))
        return torch.cat(outputs) if outputs else torch.empty(0, device=centers.device, dtype=torch.long)

    def diagnostics(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {
            "mesh_feedback_pending": float(self._pending is not None),
            "mesh_feedback_candidate": float(self._candidate is not None),
            "mesh_feedback_incoming": float(self._incoming is not None),
            "mesh_feedback_blend_weight": float(self._last_blend_weight),
            "mesh_feedback_active_freshness_weight": float(
                self._last_active_freshness_weight
            ),
            "mesh_feedback_applied": float(self._feedback_applied),
            "mesh_feedback_gate_eligible": float(self._last_gate_eligible),
            "mesh_feedback_live_eligible": float(self._last_live_eligible),
            "mesh_feedback_accepted": float(self.accepted_candidates),
            "mesh_feedback_rejected": float(self.rejected_candidates),
        }
        if self.cache is not None:
            values.update(
                mesh_cache_source_iteration=float(self.cache.stamp.source_iteration),
                mesh_cache_age=float(max(self._last_iteration - self.cache.stamp.source_iteration, 0)),
                mesh_cache_topology_generation=float(self.cache.topology_generation),
                mesh_cache_quality=float(self.cache.quality_score),
            )
        if self.last_quality is not None:
            values.update(
                mesh_gate_score=self.last_quality.score,
                mesh_gate_sdf_p90=self.last_quality.sdf_p90,
                mesh_gate_normal=self.last_quality.normal_alignment,
                mesh_gate_semantic=self.last_quality.semantic_alignment,
                mesh_gate_coverage=self.last_quality.match_coverage,
                mesh_gate_matches=float(self.last_quality.matches),
            )
        if self.last_rejection_reason is not None:
            values["mesh_feedback_reason"] = self.last_rejection_reason
        values.update(self._last_loss_terms)
        return values

    def close(self) -> None:
        pending = self._pending
        self._pending = None
        self._pending_iteration = None
        if pending is not None and not pending.done():
            pending.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "MeshCache",
    "MeshFeedbackBatch",
    "MeshFeedbackRegularizer",
    "MeshQualityReport",
    "MeshSnapshotStamp",
]
