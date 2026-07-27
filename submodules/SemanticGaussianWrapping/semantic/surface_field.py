"""One differentiable semantic surface field shared by training and meshing."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry_policy import SoftGeometryPolicyBank
from .neighbor_index import GaussianNeighborIndex, GaussianSupportAttributes
from .region_membership import SparseRegionMembership


class _ResultMapping(Mapping[str, Tensor]):
    def __getitem__(self, key: str) -> Tensor:
        if key not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (field.name for field in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def to(self, *args: Any, **kwargs: Any):
        return type(self)(**{key: value.to(*args, **kwargs) for key, value in self.items()})

    def detach(self):
        return type(self)(**{key: value.detach() for key, value in self.items()})


@dataclass(frozen=True)
class SurfaceQueryResult(_ResultMapping):
    occupancy: Tensor
    sdf: Tensor
    normal: Tensor
    semantic: Tensor
    geometry_posterior: Tensor
    uncertainty: Tensor
    # World-space bandwidth of the local Gaussian mixture. Consumers must
    # detach it when it is used as a residual denominator so the optimizer
    # cannot reduce a normalized loss by merely inflating Gaussian scales.
    local_scale: Tensor


@dataclass(frozen=True)
class PointRegionSurfaceQueryResult(_ResultMapping):
    """Several assigned region fields per point, with shape ``[P,K]``."""

    region_ids: Tensor
    valid: Tensor
    occupancy: Tensor
    sdf: Tensor
    normal: Tensor
    semantic: Tensor
    geometry_posterior: Tensor
    uncertainty: Tensor
    local_scale: Tensor
    support_fraction: Tensor


@dataclass(frozen=True)
class PartitionedSurfaceQueryResult:
    """Global-only and region-only consumers sharing one routing context."""

    global_field: SurfaceQueryResult
    point_regions: PointRegionSurfaceQueryResult


@dataclass(frozen=True)
class RegionOwnershipResult:
    """Best requested soft region support for every query point."""

    requested_region_ids: Tensor
    region_id: Tensor
    confidence: Tensor
    valid: Tensor


@dataclass(frozen=True)
class SurfaceQueryContext:
    """Compact live Gaussian state shared by every chunk of one query.

    Neighbor routing is expressed in the local candidate domain.  Activated
    rendering attributes and geometry policy outputs are evaluated once per
    unique supporting Gaussian, then reused by all points and chunks.
    """

    neighbor_indices: Tensor
    support: GaussianSupportAttributes
    semantic: Tensor
    geometry_posterior: Tensor
    surface_bandwidth: Tensor
    semantic_confidence: Tensor
    region_membership: SparseRegionMembership


@dataclass(frozen=True)
class GeometryQueryContext:
    """Candidate-first state required for an exact geometry-only reduction.

    Marching cubes evaluates millions of voxel samples, but it does not need a
    semantic embedding or sparse region memberships at those samples.  Keeping
    this context separate is important: it preserves the identical support and
    geometry-policy computation while avoiding a scene-classifier pass over
    every routed Gaussian candidate.
    """

    neighbor_indices: Tensor
    support: GaussianSupportAttributes
    geometry_posterior: Tensor
    surface_bandwidth: Tensor
    semantic_confidence: Tensor


@dataclass(frozen=True)
class _ComponentQuery:
    """One candidate-first geometric evaluation shared by all reductions."""

    flat_indices: Tensor
    scales: Tensor
    rotation: Tensor
    local_delta: Tensor
    log_density: Tensor


@dataclass(frozen=True)
class _GeometryReduction:
    """Shared mixture reduction used by full and geometry-only queries."""

    flat_indices: Tensor
    weights: Tensor
    occupancy: Tensor
    sdf: Tensor
    normal: Tensor
    geometry_posterior: Tensor
    uncertainty: Tensor
    local_scale: Tensor


class SemanticSurfaceField(nn.Module):
    """An anisotropic Gaussian mixture interpreted as one surface field.

    Neighbor selection uses a detached cached cKDTree when SciPy is available,
    with a memory-bounded exact scan as fallback. Selection is discrete while
    all gathered field values remain differentiable with respect to Gaussian
    parameters and query points.
    """

    def __init__(
        self,
        gaussians: Any,
        policy_bank: SoftGeometryPolicyBank | None = None,
        k_neighbors: int = 8,
        query_chunk_size: int = 2048,
        gaussian_chunk_size: int = 8192,
        occupancy_iso: float = 0.5,
        density_scale: float = 8.0,
        semantic_decoder: nn.Module | None = None,
        max_distance_bytes: int = 64 * 1024 * 1024,
        neighbor_backend: str = "auto",
        neighbor_index: GaussianNeighborIndex | None = None,
        support_log_cutoff: float = -12.0,
        support_candidate_budget: int = 2_048,
        support_routing_query_chunk: int = 8_192,
        scipy_workers: int = 4,
        region_top_k: int = 3,
        region_decode_chunk_size: int = 32_768,
        region_candidate_neighbors: int = 64,
        region_min_membership: float = 1e-4,
    ) -> None:
        super().__init__()
        if (
            k_neighbors < 1
            or query_chunk_size < 1
            or gaussian_chunk_size < 1
            or support_candidate_budget < 1
            or support_routing_query_chunk < 1
            or scipy_workers == 0
            or scipy_workers < -1
            or region_top_k < 1
            or region_decode_chunk_size < 1
            or region_candidate_neighbors < k_neighbors
        ):
            raise ValueError("neighbor and chunk sizes must be positive")
        if not 0.0 <= float(region_min_membership) < 1.0:
            raise ValueError("region_min_membership must lie in [0,1)")
        if not 0.0 < occupancy_iso < 1.0:
            raise ValueError("occupancy_iso must lie in (0,1)")
        self.k_neighbors = int(k_neighbors)
        self.query_chunk_size = int(query_chunk_size)
        self.gaussian_chunk_size = int(gaussian_chunk_size)
        self.occupancy_iso = float(occupancy_iso)
        self.density_scale = float(density_scale)
        self.max_distance_bytes = int(max_distance_bytes)
        self.neighbor_backend = neighbor_backend
        self.support_log_cutoff = float(support_log_cutoff)
        self.support_candidate_budget = int(support_candidate_budget)
        self.support_routing_query_chunk = int(support_routing_query_chunk)
        self.scipy_workers = int(scipy_workers)
        self.region_top_k = int(region_top_k)
        self.region_decode_chunk_size = int(region_decode_chunk_size)
        self.region_candidate_neighbors = int(region_candidate_neighbors)
        self.region_min_membership = float(region_min_membership)
        self.policy_bank = policy_bank or getattr(gaussians, "policy_bank", None) or SoftGeometryPolicyBank()
        self.semantic_decoder = semantic_decoder
        self.set_gaussians(gaussians)
        self.neighbor_index = neighbor_index or GaussianNeighborIndex(
            gaussians,
            backend=neighbor_backend,
            gaussian_chunk_size=gaussian_chunk_size,
            query_chunk_size=query_chunk_size,
            max_distance_bytes=max_distance_bytes,
            support_candidate_budget=support_candidate_budget,
            support_routing_query_chunk=support_routing_query_chunk,
            scipy_workers=scipy_workers,
        )
        if self.neighbor_index.gaussians is not gaussians:
            self.neighbor_index.set_gaussians(gaussians)

    @classmethod
    def from_gaussian_model(cls, gaussians: Any, **kwargs: Any) -> "SemanticSurfaceField":
        return cls(gaussians, **kwargs)

    def set_gaussians(self, gaussians: Any) -> None:
        # A weak reference prevents the field's state_dict from duplicating the
        # complete Gaussian checkpoint while still allowing live gradients.
        object.__setattr__(self, "_gaussians_ref", weakref.ref(gaussians))
        index = getattr(self, "neighbor_index", None)
        if index is not None and index.gaussians is not gaussians:
            index.set_gaussians(gaussians)

    def set_neighbor_index(self, neighbor_index: GaussianNeighborIndex) -> None:
        if neighbor_index.gaussians is not self.gaussians:
            neighbor_index.set_gaussians(self.gaussians)
        self.neighbor_index = neighbor_index

    @property
    def gaussians(self) -> Any:
        value = self._gaussians_ref()
        if value is None:
            raise RuntimeError("the Gaussian model backing this surface field was released")
        return value

    def decode_semantic(self, result_or_embedding: SurfaceQueryResult | Tensor) -> Tensor:
        embedding = (
            result_or_embedding.semantic
            if isinstance(result_or_embedding, SurfaceQueryResult)
            else result_or_embedding
        )
        decoder = self.semantic_decoder or getattr(self.gaussians, "semantic_decoder", None)
        if decoder is None:
            raise RuntimeError(
                "region-conditioned surface inference requires a scene semantic decoder; "
                "a PLY-only model is insufficient"
            )
        return decoder(embedding)

    @property
    def spatial_index_signature(self) -> tuple[int, tuple[int, ...], str, int | None] | None:
        return self.neighbor_index.signature

    @property
    def _spatial_index(self):
        """Backward-compatible diagnostic alias for the shared cKDTree."""

        return self.neighbor_index.tree

    def refresh_spatial_index(self, force: bool = True) -> str:
        """Build or refresh the detached neighbor index.

        Topology changes replace registry parameters, so the pointer/shape
        signature invalidates automatically.  Call with ``force=True`` after a
        long geometry-only optimization interval when positions have moved but
        topology has not changed.
        """

        return self.neighbor_index.refresh(force=force)

    @staticmethod
    def _empty(points: Tensor, semantic_dim: int, experts: int) -> SurfaceQueryResult:
        count = points.shape[0]
        return SurfaceQueryResult(
            occupancy=points.new_zeros(count),
            sdf=points.new_full((count,), float("inf")),
            normal=points.new_zeros(count, 3),
            semantic=points.new_zeros(count, semantic_dim),
            geometry_posterior=points.new_full((count, experts), 1.0 / experts),
            uncertainty=points.new_ones(count),
            local_scale=points.new_ones(count),
        )

    def _knn_indices(self, points: Tensor, xyz: Tensor, k: int) -> Tensor:
        del xyz
        return self.neighbor_index.query_support(
            points,
            k,
            density_scale=self.density_scale,
            minimum_log_support=self.support_log_cutoff,
        )

    @staticmethod
    def _gather_candidate_tensor(
        gaussians: Any,
        indices: Tensor,
        reference: Tensor,
        *,
        raw_name: str,
        getter_name: str,
    ) -> Tensor:
        registry = getattr(gaussians, "registry", None)
        if registry is not None and raw_name in registry:
            source = registry[raw_name]
        else:
            source = getattr(gaussians, getter_name)
        model_indices = indices.to(device=source.device, dtype=torch.long)
        return source.index_select(0, model_indices).to(device=reference.device, dtype=reference.dtype)

    def _gather_semantic_confidence(
        self,
        indices: Tensor,
        reference: Tensor,
    ) -> Tensor:
        gaussians = self.gaussians
        registry = getattr(gaussians, "registry", None)
        if registry is not None and "semantic_confidence" in registry:
            model_indices = indices.to(device=registry.device, dtype=torch.long)
            direct = registry["semantic_confidence"].index_select(0, model_indices)
            if "propagated_semantic_confidence" in registry:
                propagated = registry["propagated_semantic_confidence"].index_select(0, model_indices)
                confidence = torch.maximum(direct, propagated)
            else:
                confidence = direct
            return confidence.to(device=reference.device, dtype=reference.dtype)
        # Compatibility path for foreign models.  In the project model the
        # branch above is exactly ``get_semantic_confidence`` evaluated after
        # candidate gathering, including propagated neighborhood evidence.
        source = gaussians.get_semantic_confidence
        model_indices = indices.to(device=source.device, dtype=torch.long)
        return source.index_select(0, model_indices).to(device=reference.device, dtype=reference.dtype)

    def _prepare_query_context(self, points: Tensor, k: int) -> SurfaceQueryContext:
        gaussians = self.gaussians
        candidate_count = min(
            max(k, self.region_candidate_neighbors),
            len(gaussians),
        )
        global_neighbors = self._knn_indices(
            points,
            gaussians.get_xyz,
            candidate_count,
        )
        unique_indices, inverse = torch.unique(
            global_neighbors.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        support = self.neighbor_index.gather_support_attributes(
            unique_indices,
            points,
            detach=False,
        )
        semantic = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="semantic_embedding",
            getter_name="get_semantic",
        )
        geometry_logits = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="geometry_logits",
            getter_name="get_geometry_logits",
        )
        confidence = self._gather_semantic_confidence(unique_indices, points).reshape(-1)
        membership_decoder = getattr(gaussians, "point_region_memberships", None)
        if not callable(membership_decoder):
            raise TypeError(
                "Gaussian model must implement point_region_memberships for "
                "region-conditioned surface queries"
            )
        region_membership = membership_decoder(
            unique_indices.to(device=gaussians.get_semantic.device, dtype=torch.long),
            top_k=self.region_top_k,
            chunk_size=self.region_decode_chunk_size,
        ).to(device=points.device)
        boundary = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="boundary_score",
            getter_name="get_boundary_score",
        )
        error = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="geometry_error",
            getter_name="get_geometry_error",
        )
        policy = self.policy_bank(geometry_logits, confidence, boundary, error)
        return SurfaceQueryContext(
            neighbor_indices=inverse.reshape_as(global_neighbors),
            support=support,
            semantic=semantic,
            geometry_posterior=policy.posterior,
            surface_bandwidth=policy.surface_bandwidth,
            semantic_confidence=confidence,
            region_membership=region_membership,
        )

    def _prepare_geometry_query_context(
        self,
        points: Tensor,
        k: int,
    ) -> GeometryQueryContext:
        """Route geometry support without materializing semantic memberships.

        The candidate count intentionally remains the region-aware shortlist
        width.  A full query evaluates its final geometric mixture from the
        first ``k`` entries of that same ordered shortlist, so this preserves
        the scalar field exactly while omitting data that marching cubes only
        needs after it has found surface vertices.
        """

        gaussians = self.gaussians
        candidate_count = min(
            max(k, self.region_candidate_neighbors),
            len(gaussians),
        )
        global_neighbors = self._knn_indices(
            points,
            gaussians.get_xyz,
            candidate_count,
        )
        unique_indices, inverse = torch.unique(
            global_neighbors.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        support = self.neighbor_index.gather_support_attributes(
            unique_indices,
            points,
            detach=False,
        )
        geometry_logits = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="geometry_logits",
            getter_name="get_geometry_logits",
        )
        confidence = self._gather_semantic_confidence(unique_indices, points).reshape(-1)
        boundary = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="boundary_score",
            getter_name="get_boundary_score",
        )
        error = self._gather_candidate_tensor(
            gaussians,
            unique_indices,
            points,
            raw_name="geometry_error",
            getter_name="get_geometry_error",
        )
        policy = self.policy_bank(geometry_logits, confidence, boundary, error)
        return GeometryQueryContext(
            neighbor_indices=inverse.reshape_as(global_neighbors),
            support=support,
            geometry_posterior=policy.posterior,
            surface_bandwidth=policy.surface_bandwidth,
            semantic_confidence=confidence,
        )

    def _component_query(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: SurfaceQueryContext | GeometryQueryContext,
    ) -> _ComponentQuery:
        if local_indices.ndim != 2 or local_indices.shape[0] != points.shape[0]:
            raise ValueError("candidate indices must have shape [P,K]")
        rows, candidates = local_indices.shape
        flat = local_indices.reshape(-1)
        centers = context.support.xyz.index_select(0, flat).reshape(rows, candidates, 3)
        scales = context.support.scaling.index_select(0, flat).reshape(
            rows, candidates, 3
        ).clamp_min(1e-7)
        rotation = context.support.rotation_matrix.index_select(0, flat).reshape(
            rows, candidates, 3, 3
        )
        delta = points[:, None, :] - centers
        local = torch.einsum(
            "qcij,qcj->qci",
            rotation.transpose(-1, -2),
            delta,
        )
        # Work in log density.  Clamping the Mahalanobis distance before the
        # exponential made every sufficiently distant point look equally near
        # to the surface and caused adaptive meshing to refine empty space.
        mahalanobis = (local / scales).square().sum(-1).clamp_max(1e12)
        opacity = context.support.opacity.index_select(0, flat).reshape(
            rows, candidates
        )
        log_component = (
            math.log(self.density_scale)
            + opacity.clamp_min(1e-12).log()
            - 0.5 * mahalanobis
        )
        return _ComponentQuery(flat, scales, rotation, local, log_component)

    def _geometry_reduction(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: SurfaceQueryContext | GeometryQueryContext,
    ) -> _GeometryReduction:
        """Evaluate the shared geometric Gaussian-mixture reduction once."""

        rows, candidates = local_indices.shape
        component = self._component_query(points, local_indices, context)
        flat = component.flat_indices
        scales = component.scales
        rotation = component.rotation
        local = component.local_delta
        log_component = component.log_density
        log_support = torch.logsumexp(log_component, dim=-1)
        support = torch.exp(log_support.clamp_max(20.0))
        occupancy = -torch.expm1(-support)
        # Softmax remains normalized even when every linear-space component
        # underflows; normal and local bandwidth stay well-defined.
        weights = F.softmax(log_component, dim=-1)

        posterior = context.geometry_posterior.index_select(0, flat).reshape(
            rows, candidates, -1
        )
        geometry_posterior = (weights[..., None] * posterior).sum(1)
        posterior_mass = geometry_posterior.sum(-1, keepdim=True)
        geometry_posterior = geometry_posterior + (
            1.0 - posterior_mass
        ).clamp_min(0.0) / geometry_posterior.shape[-1]
        geometry_posterior = geometry_posterior / geometry_posterior.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        bandwidth_policy = context.surface_bandwidth.index_select(0, flat).reshape(
            rows, candidates
        )
        bandwidth = (
            weights * scales.min(-1).values * bandwidth_policy
        ).sum(-1).clamp_min(1e-7)
        iso_density = -math.log1p(-self.occupancy_iso)
        # A log-density residual is sign-consistent at the occupancy isosurface
        # and grows with distance, unlike the bounded linear density residual.
        sdf = bandwidth * (math.log(iso_density) - log_support)

        inverse_local_delta = local / scales.square()
        world_gradient = torch.einsum("qkij,qkj->qki", rotation, inverse_local_delta)
        normal = F.normalize(
            (weights[..., None] * world_gradient).sum(1),
            dim=-1,
            eps=1e-8,
        )
        weak = normal.norm(dim=-1) < 1e-4
        if weak.any():
            axes = scales.argmin(-1)
            gaussian_normals = rotation.gather(
                3, axes[..., None, None].expand(-1, -1, 3, 1)
            ).squeeze(-1)
            dominant = gaussian_normals[:, 0]
            sign = torch.where(
                (gaussian_normals * dominant[:, None]).sum(-1, keepdim=True) < 0,
                -1.0,
                1.0,
            )
            fallback = F.normalize(
                (weights[..., None] * gaussian_normals * sign).sum(1),
                dim=-1,
                eps=1e-8,
            )
            normal = torch.where(weak[:, None], fallback, normal)

        confidence = context.semantic_confidence.index_select(0, flat).reshape(
            rows, candidates
        )
        confidence = (weights * confidence).sum(-1).clamp(0.0, 1.0)
        entropy = -(
            geometry_posterior.clamp_min(1e-8).log() * geometry_posterior
        ).sum(-1)
        entropy = entropy / math.log(max(geometry_posterior.shape[-1], 2))
        uncertainty = (
            0.45 * (1.0 - confidence)
            + 0.35 * entropy
            + 0.20 * torch.exp(-support)
        ).clamp(0.0, 1.0)
        return _GeometryReduction(
            flat_indices=flat,
            weights=weights,
            occupancy=occupancy,
            sdf=sdf,
            normal=normal,
            geometry_posterior=geometry_posterior,
            uncertainty=uncertainty,
            local_scale=bandwidth,
        )

    def _query_chunk(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: SurfaceQueryContext,
    ) -> SurfaceQueryResult:
        rows, candidates = local_indices.shape
        reduction = self._geometry_reduction(points, local_indices, context)
        semantic = context.semantic.index_select(0, reduction.flat_indices).reshape(
            rows, candidates, -1
        )
        semantic = (reduction.weights[..., None] * semantic).sum(1)
        return SurfaceQueryResult(
            reduction.occupancy,
            reduction.sdf,
            reduction.normal,
            semantic,
            reduction.geometry_posterior,
            reduction.uncertainty,
            reduction.local_scale,
        )

    def _geometry_query_chunk(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: GeometryQueryContext,
    ) -> SurfaceQueryResult:
        reduction = self._geometry_reduction(points, local_indices, context)
        # Voxel samples only need a placeholder to satisfy the shared mesh
        # sample contract. Full semantic embeddings are queried once at final
        # mesh vertices, where they can influence topology and region ownership.
        semantic = points.new_zeros((points.shape[0], 1))
        return SurfaceQueryResult(
            reduction.occupancy,
            reduction.sdf,
            reduction.normal,
            semantic,
            reduction.geometry_posterior,
            reduction.uncertainty,
            reduction.local_scale,
        )

    def _query_region_chunk(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: SurfaceQueryContext,
        requested_regions: Tensor,
    ) -> dict[str, Tensor]:
        """Reduce one candidate set into several soft region fields.

        ``requested_regions`` has shape ``[P,R]``.  Geometry support is
        weighted by the decoder posterior and by calibrated semantic evidence;
        missing support remains explicitly invalid and never falls back to the
        global mixture.
        """

        if requested_regions.ndim != 2 or requested_regions.shape[0] != points.shape[0]:
            raise ValueError("requested region IDs must have shape [P,R]")
        rows, candidates = local_indices.shape
        component = self._component_query(points, local_indices, context)
        flat = component.flat_indices
        scales = component.scales
        rotation = component.rotation
        local = component.local_delta
        log_component = component.log_density

        membership = context.region_membership
        member_ids = membership.ids.index_select(0, flat).reshape(
            rows,
            candidates,
            -1,
        )
        member_weights = membership.weights.index_select(0, flat).reshape_as(member_ids)
        member_confidence = membership.confidence.index_select(0, flat).reshape(
            rows,
            candidates,
        )
        member_tail = membership.tail.index_select(0, flat).reshape(rows, candidates)
        matches = member_ids[..., None] == requested_regions[:, None, None, :]
        region_mass = (member_weights[..., None] * matches).sum(dim=2)
        region_mass = region_mass * member_confidence[..., None]
        component_valid = region_mass >= self.region_min_membership
        region_log_component = log_component[..., None] + region_mass.clamp_min(1e-30).log()
        region_log_component = region_log_component.masked_fill(
            ~component_valid,
            -float("inf"),
        )

        maximum = region_log_component.max(dim=1).values
        valid = torch.isfinite(maximum)
        origin = torch.where(valid, maximum, torch.zeros_like(maximum))
        linear = torch.exp(region_log_component - origin[:, None, :])
        linear = torch.where(component_valid, linear, torch.zeros_like(linear))
        linear_sum = linear.sum(dim=1)
        weights = linear / linear_sum[:, None, :].clamp_min(1e-30)
        log_support = origin + linear_sum.clamp_min(1e-30).log()
        support = torch.exp(log_support.clamp_max(20.0))
        occupancy = -torch.expm1(-support)

        candidate_semantic = context.semantic.index_select(0, flat).reshape(
            rows,
            candidates,
            -1,
        )
        semantic = torch.einsum("qcr,qcd->qrd", weights, candidate_semantic)
        candidate_posterior = context.geometry_posterior.index_select(0, flat).reshape(
            rows,
            candidates,
            -1,
        )
        posterior = torch.einsum("qcr,qce->qre", weights, candidate_posterior)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        bandwidth_policy = context.surface_bandwidth.index_select(0, flat).reshape(
            rows,
            candidates,
        )
        component_scale = scales.min(dim=-1).values * bandwidth_policy
        local_scale = torch.einsum("qcr,qc->qr", weights, component_scale).clamp_min(1e-7)
        iso_density = -math.log1p(-self.occupancy_iso)
        sdf = local_scale * (math.log(iso_density) - log_support)

        inverse_local_delta = local / scales.square()
        world_gradient = torch.einsum("qcij,qcj->qci", rotation, inverse_local_delta)
        normal = F.normalize(
            torch.einsum("qcr,qcd->qrd", weights, world_gradient),
            dim=-1,
            eps=1e-8,
        )
        weak = normal.norm(dim=-1) < 1e-4
        if bool(weak.any()):
            axes = scales.argmin(dim=-1)
            gaussian_normals = rotation.gather(
                3,
                axes[..., None, None].expand(-1, -1, 3, 1),
            ).squeeze(-1)
            dominant = weights.argmax(dim=1)
            selected = gaussian_normals.gather(
                1,
                dominant[..., None].expand(-1, -1, 3),
            )
            normal = torch.where(weak[..., None], selected, normal)

        confidence = torch.einsum("qcr,qc->qr", weights, member_confidence).clamp(0.0, 1.0)
        tail = torch.einsum("qcr,qc->qr", weights, member_tail).clamp(0.0, 1.0)
        entropy = -(posterior.clamp_min(1e-8).log() * posterior).sum(dim=-1)
        entropy = entropy / math.log(max(posterior.shape[-1], 2))
        global_log_support = torch.logsumexp(log_component, dim=1)
        support_fraction = torch.exp(log_support - global_log_support[:, None]).clamp(0.0, 1.0)
        uncertainty = (
            0.35 * (1.0 - confidence)
            + 0.25 * entropy
            + 0.25 * (1.0 - support_fraction)
            + 0.15 * tail
        ).clamp(0.0, 1.0)

        uniform = posterior.new_full(
            posterior.shape,
            1.0 / max(posterior.shape[-1], 1),
        )
        return {
            "valid": valid,
            "occupancy": torch.where(valid, occupancy, torch.zeros_like(occupancy)),
            "sdf": torch.where(valid, sdf, torch.full_like(sdf, float("inf"))),
            "normal": torch.where(valid[..., None], normal, torch.zeros_like(normal)),
            "semantic": torch.where(
                valid[..., None],
                semantic,
                torch.zeros_like(semantic),
            ),
            "geometry_posterior": torch.where(valid[..., None], posterior, uniform),
            "uncertainty": torch.where(valid, uncertainty, torch.ones_like(uncertainty)),
            "local_scale": torch.where(valid, local_scale, torch.ones_like(local_scale)),
            "support_fraction": torch.where(
                valid,
                support_fraction,
                torch.zeros_like(support_fraction),
            ),
        }

    def _region_ownership_chunk(
        self,
        points: Tensor,
        local_indices: Tensor,
        context: SurfaceQueryContext,
        requested_regions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reduce sparse candidate memberships directly to one owner per point."""

        rows, candidates = local_indices.shape
        component_query = self._component_query(points, local_indices, context)
        flat = component_query.flat_indices
        log_component = component_query.log_density
        origin = log_component.max(dim=1, keepdim=True).values
        component = torch.exp(log_component - origin)
        global_support = component.sum(dim=1, keepdim=True).clamp_min(1e-30)

        membership = context.region_membership
        member_ids = membership.ids.index_select(0, flat).reshape(
            rows, candidates, -1
        )
        evidence = membership.weights.index_select(0, flat).reshape_as(member_ids)
        evidence = evidence * membership.confidence.index_select(0, flat).reshape(
            rows, candidates, 1
        )
        evidence = torch.where(
            evidence >= self.region_min_membership,
            evidence,
            torch.zeros_like(evidence),
        )

        positions = torch.searchsorted(requested_regions, member_ids)
        safe_positions = positions.clamp_max(requested_regions.numel() - 1)
        requested = requested_regions.index_select(0, safe_positions.reshape(-1))
        requested = requested.reshape_as(member_ids)
        selected = (positions < requested_regions.numel()) & (requested == member_ids)
        contribution = component[..., None] * evidence
        contribution = torch.where(
            selected,
            contribution,
            torch.zeros_like(contribution),
        )
        regional_support = component.new_zeros(
            rows,
            requested_regions.numel(),
        )
        regional_support.scatter_add_(
            1,
            safe_positions.reshape(rows, -1),
            contribution.reshape(rows, -1),
        )
        support_fraction = regional_support / global_support
        confidence, owner_index = support_fraction.max(dim=1)
        valid = confidence > 0
        owner = requested_regions.index_select(0, owner_index)
        owner = torch.where(valid, owner, torch.full_like(owner, -1))
        return owner, confidence.clamp(0.0, 1.0), valid

    def _query_size(self, points: Tensor, requested: int, candidate_count: int) -> int:
        memory_limited = max(
            1,
            self.max_distance_bytes
            // max(32 * max(points.element_size(), 4) * candidate_count, 1),
        )
        return min(requested, self.query_chunk_size, memory_limited)

    def _global_from_context(
        self,
        points: Tensor,
        context: SurfaceQueryContext,
        size: int,
        neighbor_indices: Tensor | None = None,
    ) -> SurfaceQueryResult:
        candidates = (
            context.neighbor_indices
            if neighbor_indices is None
            else neighbor_indices
        )
        if candidates.shape[0] != points.shape[0]:
            raise ValueError("global neighbor rows must match query points")
        global_indices = candidates[:, : min(self.k_neighbors, candidates.shape[1])]
        chunks = [
            self._query_chunk(
                points[start : start + size],
                global_indices[start : start + size],
                context,
            )
            for start in range(0, points.shape[0], size)
        ]
        return SurfaceQueryResult(
            **{
                key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                for key in chunks[0]
            }
        )

    def _geometry_from_context(
        self,
        points: Tensor,
        context: GeometryQueryContext,
        size: int,
    ) -> SurfaceQueryResult:
        if context.neighbor_indices.shape[0] != points.shape[0]:
            raise ValueError("global neighbor rows must match query points")
        global_indices = context.neighbor_indices[
            :, : min(self.k_neighbors, context.neighbor_indices.shape[1])
        ]
        chunks = [
            self._geometry_query_chunk(
                points[start : start + size],
                global_indices[start : start + size],
                context,
            )
            for start in range(0, points.shape[0], size)
        ]
        return SurfaceQueryResult(
            **{
                key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                for key in chunks[0]
            }
        )

    def query(self, points: Tensor, chunk_size: int | None = None) -> SurfaceQueryResult:
        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a torch.Tensor with shape [P,3]")
        if not points.is_floating_point():
            raise TypeError("surface query points must be floating point")
        gaussians = self.gaussians
        semantic_dim = getattr(gaussians, "semantic_dim", None)
        if semantic_dim is None:
            semantic_dim = gaussians.get_semantic.shape[-1]
        experts = getattr(gaussians, "geometry_experts", None)
        if experts is None:
            experts = gaussians.get_geometry_logits.shape[-1]
        semantic_dim, experts = int(semantic_dim), int(experts)
        if points.shape[0] == 0 or len(gaussians) == 0:
            return self._empty(points, semantic_dim, experts)
        requested = int(chunk_size or self.query_chunk_size)
        if requested < 1:
            raise ValueError("surface query chunk size must be positive")
        k = min(self.k_neighbors, len(gaussians))
        context = self._prepare_query_context(points, k)
        size = self._query_size(
            points,
            requested,
            context.neighbor_indices.shape[1],
        )
        return self._global_from_context(points, context, size)

    def query_geometry(
        self,
        points: Tensor,
        chunk_size: int | None = None,
    ) -> SurfaceQueryResult:
        """Evaluate the exact scalar surface field without semantic decoding.

        This is the dense-volume contract used by mesh extraction.  It retains
        the same support routing, policy-conditioned bandwidth, SDF, normals,
        posterior and uncertainty as :meth:`query`; only the unused per-voxel
        semantic embedding is replaced by a one-channel placeholder.
        """

        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a torch.Tensor with shape [P,3]")
        if not points.is_floating_point():
            raise TypeError("surface query points must be floating point")
        gaussians = self.gaussians
        experts = getattr(gaussians, "geometry_experts", None)
        if experts is None:
            experts = gaussians.get_geometry_logits.shape[-1]
        experts = int(experts)
        if points.shape[0] == 0 or len(gaussians) == 0:
            return self._empty(points, 1, experts)
        requested = int(chunk_size or self.query_chunk_size)
        if requested < 1:
            raise ValueError("surface query chunk size must be positive")
        k = min(self.k_neighbors, len(gaussians))
        context = self._prepare_geometry_query_context(points, k)
        size = self._query_size(
            points,
            requested,
            context.neighbor_indices.shape[1],
        )
        return self._geometry_from_context(points, context, size)

    @staticmethod
    def _validate_region_ids(region_ids: Tensor, device: torch.device) -> Tensor:
        if not isinstance(region_ids, Tensor) or region_ids.ndim != 1:
            raise ValueError("region_ids must be a one-dimensional tensor")
        if region_ids.dtype != torch.long:
            raise TypeError("region_ids must use torch.long")
        region_ids = region_ids.to(device=device)
        if region_ids.numel() == 0:
            raise ValueError("at least one foreground region ID is required")
        if bool((region_ids <= 0).any()):
            raise ValueError("region_ids must contain foreground IDs greater than zero")
        if region_ids.numel() > 1 and bool((region_ids[1:] <= region_ids[:-1]).any()):
            raise ValueError("region_ids must be sorted and unique")
        return region_ids

    def query_region_ownership(
        self,
        points: Tensor,
        *,
        region_ids: Tensor,
        chunk_size: int | None = None,
    ) -> RegionOwnershipResult:
        """Assign requested regions without materializing a ``[P,R]`` field."""

        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a torch.Tensor with shape [P,3]")
        if not points.is_floating_point():
            raise TypeError("surface query points must be floating point")
        regions = self._validate_region_ids(region_ids, points.device)
        if points.shape[0] == 0 or len(self.gaussians) == 0:
            return RegionOwnershipResult(
                requested_region_ids=regions,
                region_id=torch.full(
                    (points.shape[0],),
                    -1,
                    dtype=torch.long,
                    device=points.device,
                ),
                confidence=points.new_zeros(points.shape[0]),
                valid=torch.zeros(
                    points.shape[0],
                    dtype=torch.bool,
                    device=points.device,
                ),
            )
        requested = int(chunk_size or self.query_chunk_size)
        if requested < 1:
            raise ValueError("surface query chunk size must be positive")
        context = self._prepare_query_context(
            points,
            min(self.k_neighbors, len(self.gaussians)),
        )
        size = self._query_size(points, requested, context.neighbor_indices.shape[1])
        owners = []
        confidences = []
        valid = []
        for start in range(0, points.shape[0], size):
            stop = min(start + size, points.shape[0])
            owner, confidence, chunk_valid = self._region_ownership_chunk(
                points[start:stop],
                context.neighbor_indices[start:stop],
                context,
                regions,
            )
            owners.append(owner)
            confidences.append(confidence)
            valid.append(chunk_valid)
        return RegionOwnershipResult(
            requested_region_ids=regions,
            region_id=torch.cat(owners),
            confidence=torch.cat(confidences),
            valid=torch.cat(valid),
        )

    @staticmethod
    def _validate_point_region_ids(points: Tensor, region_ids: Tensor) -> Tensor:
        if not isinstance(region_ids, Tensor) or region_ids.ndim != 2:
            raise ValueError("point region_ids must have shape [P,K]")
        if region_ids.shape[0] != points.shape[0] or region_ids.shape[1] < 1:
            raise ValueError("point region_ids must have shape [P,K] with K >= 1")
        if region_ids.dtype != torch.long:
            raise TypeError("point region_ids must use torch.long")
        region_ids = region_ids.to(points.device)
        if bool((region_ids <= 0).any()):
            raise ValueError("point region_ids must contain foreground IDs greater than zero")
        return region_ids

    @staticmethod
    def _empty_point_regions(
        points: Tensor,
        region_ids: Tensor,
        semantic_dim: int,
        experts: int,
    ) -> PointRegionSurfaceQueryResult:
        shape = region_ids.shape
        return PointRegionSurfaceQueryResult(
            region_ids=region_ids,
            valid=torch.zeros(shape, dtype=torch.bool, device=points.device),
            occupancy=points.new_zeros(shape),
            sdf=points.new_full(shape, float("inf")),
            normal=points.new_zeros((*shape, 3)),
            semantic=points.new_zeros((*shape, semantic_dim)),
            geometry_posterior=points.new_full(
                (*shape, experts),
                1.0 / max(experts, 1),
            ),
            uncertainty=points.new_ones(shape),
            local_scale=points.new_ones(shape),
            support_fraction=points.new_zeros(shape),
        )

    def _point_regions_from_context(
        self,
        points: Tensor,
        region_ids: Tensor,
        context: SurfaceQueryContext,
        neighbor_indices: Tensor,
        size: int,
    ) -> PointRegionSurfaceQueryResult:
        chunks = []
        for start in range(0, points.shape[0], size):
            stop = min(start + size, points.shape[0])
            chunks.append(
                self._query_region_chunk(
                    points[start:stop],
                    neighbor_indices[start:stop],
                    context,
                    region_ids[start:stop],
                )
            )
        return PointRegionSurfaceQueryResult(
            region_ids=region_ids,
            **{
                key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                for key in chunks[0]
            },
        )

    def query_point_regions(
        self,
        points: Tensor,
        region_ids: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> PointRegionSurfaceQueryResult:
        """Evaluate K assigned foreground regions per point in one reduction."""

        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a torch.Tensor with shape [P,3]")
        if not points.is_floating_point():
            raise TypeError("surface query points must be floating point")
        region_ids = self._validate_point_region_ids(points, region_ids)
        semantic_dim = int(getattr(self.gaussians, "semantic_dim", 1))
        experts = int(getattr(self.gaussians, "geometry_experts", 5))
        if points.shape[0] == 0 or len(self.gaussians) == 0:
            return self._empty_point_regions(
                points,
                region_ids,
                semantic_dim,
                experts,
            )
        requested = int(chunk_size or self.query_chunk_size)
        if requested < 1:
            raise ValueError("surface query chunk size must be positive")
        context = self._prepare_query_context(
            points,
            min(self.k_neighbors, len(self.gaussians)),
        )
        size = self._query_size(points, requested, context.neighbor_indices.shape[1])
        return self._point_regions_from_context(
            points,
            region_ids,
            context,
            context.neighbor_indices,
            size,
        )

    def query_partitioned(
        self,
        global_points: Tensor,
        regional_points: Tensor,
        regional_ids: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> PartitionedSurfaceQueryResult:
        """Route global-only and regional-only probes through one context."""

        for name, points in (
            ("global_points", global_points),
            ("regional_points", regional_points),
        ):
            if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"{name} must be a torch.Tensor with shape [P,3]")
            if not points.is_floating_point():
                raise TypeError(f"{name} must be floating point")
        if global_points.device != regional_points.device:
            raise ValueError("partitioned query points must share a device")
        if global_points.dtype != regional_points.dtype:
            raise TypeError("partitioned query points must share a dtype")
        regional_ids = self._validate_point_region_ids(regional_points, regional_ids)
        total = global_points.shape[0] + regional_points.shape[0]
        if total == 0:
            raise ValueError("partitioned surface query requires at least one point")
        if len(self.gaussians) == 0:
            raise RuntimeError("partitioned surface query requires non-empty Gaussians")
        requested = int(chunk_size or self.query_chunk_size)
        if requested < 1:
            raise ValueError("surface query chunk size must be positive")

        points = torch.cat((global_points, regional_points), dim=0)
        context = self._prepare_query_context(
            points,
            min(self.k_neighbors, len(self.gaussians)),
        )
        size = self._query_size(points, requested, context.neighbor_indices.shape[1])
        split = global_points.shape[0]
        semantic_dim = int(getattr(self.gaussians, "semantic_dim", 1))
        experts = int(getattr(self.gaussians, "geometry_experts", 5))
        global_field = (
            self._empty(global_points, semantic_dim, experts)
            if split == 0
            else self._global_from_context(
                global_points,
                context,
                size,
                context.neighbor_indices[:split],
            )
        )
        point_regions = (
            self._empty_point_regions(
                regional_points,
                regional_ids,
                semantic_dim,
                experts,
            )
            if regional_points.shape[0] == 0
            else self._point_regions_from_context(
                regional_points,
                regional_ids,
                context,
                context.neighbor_indices[split:],
                size,
            )
        )
        return PartitionedSurfaceQueryResult(global_field, point_regions)

    forward = query


__all__ = [
    "GeometryQueryContext",
    "PartitionedSurfaceQueryResult",
    "PointRegionSurfaceQueryResult",
    "RegionOwnershipResult",
    "SemanticSurfaceField",
    "SurfaceQueryContext",
    "SurfaceQueryResult",
]
