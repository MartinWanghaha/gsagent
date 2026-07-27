"""Confidence-gated soft geometry experts for semantic Gaussian control."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .neighbor_index import GaussianNeighborIndex


EXPERT_NAMES = ("planar", "curved", "thin", "freeform", "fuzzy")


class _TensorMapping(Mapping[str, Tensor]):
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
class GeometryPolicyOutput(_TensorMapping):
    posterior: Tensor
    semantic_gate: Tensor
    density_multiplier: Tensor
    split_score: Tensor
    target_flatness: Tensor
    normal_alignment_weight: Tensor
    boundary_refinement_weight: Tensor
    sh_high_order_retention: Tensor
    prune_protection: Tensor
    surface_bandwidth: Tensor


class SoftGeometryPolicyBank(nn.Module):
    """Blend five geometry policies using learned per-Gaussian logits.

    The semantic confidence gate interpolates every directive with ordinary
    3DGS behavior.  Consequently absent or uncertain semantics cannot force a
    Gaussian into a geometric class.
    """

    expert_names = EXPERT_NAMES

    def __init__(
        self,
        num_experts: int = 5,
        confidence_floor: float = 0.05,
        learnable_profiles: bool = False,
    ) -> None:
        super().__init__()
        if num_experts != len(EXPERT_NAMES):
            raise ValueError(f"the architecture defines exactly {len(EXPERT_NAMES)} experts")
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must lie in [0,1)")
        self.num_experts = num_experts
        self.confidence_floor = float(confidence_floor)
        # Rows follow EXPERT_NAMES. Columns are density, split, flatness,
        # normal alignment, SH retention, prune protection and bandwidth.
        profiles = torch.tensor(
            [
                [0.82, 0.75, 0.08, 1.00, 0.75, 0.30, 0.65],  # planar
                [1.00, 1.00, 0.28, 0.80, 0.90, 0.35, 0.85],  # curved
                [1.65, 1.70, 0.10, 0.65, 0.85, 1.00, 0.45],  # thin
                [1.10, 1.10, 0.55, 0.35, 1.00, 0.45, 1.00],  # free-form
                [0.75, 0.45, 0.85, 0.05, 1.00, 0.15, 1.40],  # fuzzy
            ],
            dtype=torch.float32,
        )
        if learnable_profiles:
            self.profiles = nn.Parameter(profiles)
        else:
            self.register_buffer("profiles", profiles)

    @staticmethod
    def _column(value: Tensor, rows: int, reference: Tensor) -> Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(rows)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != (rows,):
            raise ValueError(f"policy evidence must have shape [{rows}] or [{rows},1]")
        return value

    def forward(
        self,
        geometry_logits: Tensor,
        semantic_confidence: Tensor,
        boundary_score: Tensor,
        geometry_error: Tensor,
    ) -> GeometryPolicyOutput:
        if geometry_logits.ndim != 2 or geometry_logits.shape[1] != self.num_experts:
            raise ValueError(
                f"geometry_logits must have shape [N,{self.num_experts}], got {tuple(geometry_logits.shape)}"
            )
        rows = geometry_logits.shape[0]
        confidence = self._column(semantic_confidence, rows, geometry_logits).clamp(0.0, 1.0)
        boundary = self._column(boundary_score, rows, geometry_logits).clamp(0.0, 1.0)
        error = self._column(geometry_error, rows, geometry_logits).clamp_min(0.0)
        gate = ((confidence - self.confidence_floor) / (1.0 - self.confidence_floor)).clamp(0.0, 1.0)
        posterior = F.softmax(geometry_logits, dim=-1)
        profiles = self.profiles.to(device=geometry_logits.device, dtype=geometry_logits.dtype)
        mixed = posterior @ profiles
        density, split, flatness, alignment, sh_retention, protection, bandwidth = mixed.unbind(-1)
        error_drive = gate * torch.tanh(error)
        boundary_drive = gate * boundary

        def blend(value: Tensor, baseline: float) -> Tensor:
            return baseline + gate * (value - baseline)

        density = blend(density, 1.0) * (1.0 + 0.60 * boundary_drive + 0.35 * error_drive)
        split = blend(split, 1.0) * (1.0 + 0.80 * boundary_drive + 0.50 * error_drive)
        flatness = blend(flatness, 1.0)
        alignment = gate * alignment
        sh_retention = blend(sh_retention, 1.0)
        protection = (gate * protection + 0.75 * boundary_drive).clamp(0.0, 1.0)
        bandwidth = blend(bandwidth, 1.0) * (1.0 - 0.25 * boundary_drive)
        return GeometryPolicyOutput(
            posterior=posterior,
            semantic_gate=gate,
            density_multiplier=density,
            split_score=split,
            target_flatness=flatness,
            normal_alignment_weight=alignment,
            boundary_refinement_weight=boundary_drive,
            sh_high_order_retention=sh_retention,
            prune_protection=protection,
            surface_bandwidth=bandwidth.clamp_min(0.1),
        )

    def from_gaussians(self, gaussians: Any) -> GeometryPolicyOutput:
        confidence = getattr(gaussians, "get_semantic_confidence", None)
        if confidence is None:
            confidence = gaussians.semantic_confidence
        return self(
            gaussians.geometry_logits,
            confidence,
            gaussians.boundary_score,
            gaussians.geometry_error,
        )

    def regularization(
        self,
        gaussians: Any,
        reference_normals: Tensor | None = None,
    ) -> dict[str, Tensor]:
        policy = self.from_gaussians(gaussians)
        scales = gaussians.get_scaling.clamp_min(1e-8)
        minimum = scales.min(dim=-1).values
        maximum = scales.max(dim=-1).values
        flatness = minimum / maximum
        gate = policy.semantic_gate
        normalizer = gate.sum().clamp_min(1.0)
        anisotropy_loss = (gate * (flatness - policy.target_flatness).square()).sum() / normalizer

        if reference_normals is None:
            normal_loss = scales.new_zeros(())
        else:
            gaussian_normals = gaussians.get_normal
            reference_normals = F.normalize(reference_normals, dim=-1, eps=1e-8)
            agreement = 1.0 - (gaussian_normals * reference_normals).sum(-1).abs()
            weight = policy.normal_alignment_weight
            normal_loss = (weight * agreement).sum() / weight.sum().clamp_min(1.0)

        features_rest = gaussians.features_rest
        if features_rest.numel() == 0:
            sh_loss = scales.new_zeros(())
        else:
            suppression = gate * (1.0 - policy.sh_high_order_retention)
            energy = features_rest.square().flatten(1).mean(-1)
            sh_loss = (suppression * energy).sum() / suppression.sum().clamp_min(1.0)
        return {
            "scale_anisotropy": anisotropy_loss,
            "normal_alignment": normal_loss,
            "sh_complexity": sh_loss,
        }


class GeometryEvidenceProjector(nn.Module):
    """Project detached local geometry evidence to five soft expert targets.

    This is deliberately not a semantic class lookup.  It combines local PCA,
    normal variation, Gaussian scale ratios, seams and reconstruction error.
    Neighbor selection is delegated to the shared Gaussian index.  It uses a
    cached cKDTree when available and a bounded exact fallback otherwise.
    """

    def __init__(
        self,
        temperature: float = 0.20,
        neighbor_index: GaussianNeighborIndex | None = None,
        *,
        propagation_enabled: bool = True,
        propagation_samples: int = 65_536,
        propagation_neighbors: int = 12,
        propagation_min_seed_confidence: float = 0.35,
        propagation_max_confidence: float = 0.85,
        propagation_momentum: float = 0.5,
        propagation_decay: float = 0.995,
        propagation_semantic_floor: float = 0.25,
        propagation_support_sigma: float = 3.0,
        propagation_boundary_barrier: float = 2.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if propagation_samples < 1 or propagation_neighbors < 1:
            raise ValueError("propagation sample and neighbor counts must be positive")
        if not 0.0 <= propagation_min_seed_confidence <= propagation_max_confidence <= 1.0:
            raise ValueError("propagation confidence thresholds must satisfy 0 <= seed <= max <= 1")
        if not 0.0 <= propagation_momentum < 1.0:
            raise ValueError("propagation_momentum must lie in [0,1)")
        if not 0.0 <= propagation_decay <= 1.0:
            raise ValueError("propagation_decay must lie in [0,1]")
        if not -1.0 < propagation_semantic_floor < 1.0:
            raise ValueError("propagation_semantic_floor must lie in (-1,1)")
        if propagation_support_sigma <= 0 or propagation_boundary_barrier < 0:
            raise ValueError("propagation support must be positive and boundary barrier non-negative")
        self.temperature = float(temperature)
        self.neighbor_index = neighbor_index
        self.propagation_enabled = bool(propagation_enabled)
        self.propagation_samples = int(propagation_samples)
        self.propagation_neighbors = int(propagation_neighbors)
        self.propagation_min_seed_confidence = float(propagation_min_seed_confidence)
        self.propagation_max_confidence = float(propagation_max_confidence)
        self.propagation_momentum = float(propagation_momentum)
        self.propagation_decay = float(propagation_decay)
        self.propagation_semantic_floor = float(propagation_semantic_floor)
        self.propagation_support_sigma = float(propagation_support_sigma)
        self.propagation_boundary_barrier = float(propagation_boundary_barrier)
        # The cursor is an optimization scheduling detail rather than learned
        # state.  Keeping it non-persistent preserves old checkpoint loading.
        self.register_buffer("_propagation_cursor", torch.zeros((), dtype=torch.long), persistent=False)
        self.last_propagation: dict[str, float | int] = {}
        self.register_buffer(
            "uncertain_prior",
            torch.tensor([0.05, 0.05, 0.05, 0.15, 0.70], dtype=torch.float32),
        )

    @staticmethod
    def _indices(selection: Tensor | None, size: int, device: torch.device) -> Tensor:
        if selection is None:
            return torch.arange(size, device=device)
        selection = torch.as_tensor(selection, device=device)
        if selection.dtype == torch.bool:
            if selection.shape != (size,):
                raise ValueError(f"selection must have shape [{size}]")
            return selection.nonzero(as_tuple=False).flatten()
        return selection.long().flatten()

    def set_neighbor_index(self, neighbor_index: GaussianNeighborIndex) -> None:
        self.neighbor_index = neighbor_index

    def _index_for(self, gaussians: Any, search_chunk: int) -> GaussianNeighborIndex:
        if self.neighbor_index is None or self.neighbor_index.gaussians is not gaussians:
            self.neighbor_index = GaussianNeighborIndex(
                gaussians,
                gaussian_chunk_size=search_chunk,
            )
        return self.neighbor_index

    @staticmethod
    def _gather(
        gaussians: Any,
        indices: Tensor,
        *,
        raw_name: str,
        getter_name: str,
        activation=None,
    ) -> Tensor:
        """Gather first, then activate, avoiding an O(N) temporary per query."""

        raw = getattr(gaussians, raw_name, None)
        if isinstance(raw, Tensor):
            value = raw.detach().index_select(0, indices)
            return value if activation is None else activation(value)
        value = getattr(gaussians, getter_name)
        value = value() if callable(value) else value
        return value.detach().index_select(0, indices)

    @classmethod
    def _scaling(cls, gaussians: Any, indices: Tensor) -> Tensor:
        return cls._gather(
            gaussians,
            indices,
            raw_name="scaling",
            getter_name="get_scaling",
            activation=torch.exp,
        )

    @classmethod
    def _rotation(cls, gaussians: Any, indices: Tensor) -> Tensor:
        return cls._gather(
            gaussians,
            indices,
            raw_name="rotation",
            getter_name="get_rotation",
            activation=lambda value: F.normalize(value, dim=-1, eps=1e-8),
        )

    @staticmethod
    def _rotation_matrix(quaternion: Tensor) -> Tensor:
        quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
        w, x, y, z = quaternion.unbind(-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(quaternion.shape[:-1] + (3, 3))

    @classmethod
    def _normal(cls, gaussians: Any, indices: Tensor) -> Tensor:
        scales = cls._scaling(gaussians, indices)
        rotation = cls._rotation_matrix(cls._rotation(gaussians, indices))
        axis = scales.argmin(dim=-1)
        gather = axis[:, None, None].expand(-1, 3, 1)
        return F.normalize(rotation.gather(2, gather).squeeze(-1), dim=-1, eps=1e-8)

    @staticmethod
    def _effective_confidence(gaussians: Any, indices: Tensor) -> Tensor:
        confidence = getattr(gaussians, "get_semantic_confidence", None)
        if confidence is None:
            confidence = gaussians.semantic_confidence
        confidence = confidence() if callable(confidence) else confidence
        return confidence.detach().index_select(0, indices).reshape(-1).clamp(0.0, 1.0)

    @torch.no_grad()
    def propagate_semantic_confidence(
        self,
        gaussians: Any,
        *,
        max_points: int | None = None,
        k: int | None = None,
        search_chunk: int = 16_384,
    ) -> dict[str, float | int]:
        """Diffuse calibrated confidence across coherent local surface support.

        Only direct camera confidence can seed propagation. Spatial support,
        semantic cosine agreement, normal agreement and boundary barriers all
        have to agree. The inferred value is stored in a separate model buffer
        with a strict ceiling, so it never becomes a synthetic observation.
        A rolling candidate window gives O(samples*k) work and eventually
        covers every Gaussian without a full-model decode.
        """

        size = len(gaussians)
        update = getattr(gaussians, "update_propagated_semantic_confidence", None)
        if not self.propagation_enabled or size <= 1 or not callable(update):
            self.last_propagation = {"visited": 0, "supported": 0, "mean": 0.0}
            return self.last_propagation
        count = min(int(max_points or self.propagation_samples), size)
        neighbors_count = min(int(k or self.propagation_neighbors), size - 1)
        device = gaussians.get_xyz.device
        start = int(self._propagation_cursor.item()) % size
        linear = torch.arange(start, start + count, device=device, dtype=torch.long)
        selected = linear.remainder(size)
        self._propagation_cursor.fill_((start + count) % size)
        neighbors = self._index_for(gaussians, search_chunk).query_indices(
            selected,
            neighbors_count,
        )
        flat = neighbors.reshape(-1)
        xyz = gaussians.get_xyz.detach()
        center = xyz.index_select(0, selected)
        neighbor_xyz = xyz.index_select(0, flat).reshape(count, neighbors_count, 3)
        distance = (neighbor_xyz - center[:, None]).norm(dim=-1)

        center_scale = self._scaling(gaussians, selected).amax(dim=-1)
        neighbor_scale = self._scaling(gaussians, flat).amax(dim=-1).reshape(
            count, neighbors_count
        )
        support = self.propagation_support_sigma * 0.5 * (
            center_scale[:, None] + neighbor_scale
        ).clamp_min(1e-7)
        spatial_affinity = torch.exp(-0.5 * (distance / support).square().clamp_max(80.0))

        embedding = gaussians.get_semantic.detach()
        center_embedding = F.normalize(
            embedding.index_select(0, selected), dim=-1, eps=1e-8
        )
        neighbor_embedding = F.normalize(
            embedding.index_select(0, flat), dim=-1, eps=1e-8
        ).reshape(count, neighbors_count, -1)
        cosine = (center_embedding[:, None] * neighbor_embedding).sum(-1)
        semantic_affinity = (
            (cosine - self.propagation_semantic_floor)
            / (1.0 - self.propagation_semantic_floor)
        ).clamp(0.0, 1.0).square()

        center_normal = self._normal(gaussians, selected)
        neighbor_normal = self._normal(gaussians, flat).reshape(count, neighbors_count, 3)
        normal_affinity = (
            center_normal[:, None] * neighbor_normal
        ).sum(-1).abs().square()
        boundary = gaussians.boundary_score.detach().reshape(-1).clamp(0.0, 1.0)
        selected_boundary = boundary.index_select(0, selected)
        barrier = torch.exp(
            -self.propagation_boundary_barrier
            * (
                selected_boundary[:, None]
                + boundary.index_select(0, flat).reshape(count, neighbors_count)
            )
        )
        affinity = spatial_affinity * semantic_affinity * normal_affinity * barrier

        # Seeds deliberately use the direct buffer, never the effective getter.
        direct = gaussians.semantic_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        seed_confidence = direct.index_select(0, flat).reshape(count, neighbors_count)
        seed = seed_confidence >= self.propagation_min_seed_confidence
        seed_weight = affinity * seed
        seed_mass = seed_weight.sum(dim=-1)
        total_mass = affinity.sum(dim=-1).clamp_min(1e-8)
        support_ratio = (seed_mass / total_mass).clamp(0.0, 1.0)
        estimate = (
            (seed_weight * seed_confidence).sum(dim=-1)
            / seed_mass.clamp_min(1e-8)
        ) * support_ratio.sqrt()
        # The target-side term otherwise cancels between the weighted mean and
        # its normalizer. Retain it explicitly so seam Gaussians do not become
        # bridges for confidence diffusion.
        estimate = estimate * torch.exp(
            -self.propagation_boundary_barrier * selected_boundary
        )
        supported = seed_mass > 1e-6
        estimate = torch.where(supported, estimate, torch.zeros_like(estimate))
        estimate = estimate.clamp(0.0, self.propagation_max_confidence)
        update(
            selected,
            estimate,
            momentum=self.propagation_momentum,
            decay=self.propagation_decay,
            maximum=self.propagation_max_confidence,
        )
        self.last_propagation = {
            "visited": int(count),
            "supported": int(supported.sum().item()),
            "mean": float(estimate[supported].mean().item()) if bool(supported.any()) else 0.0,
        }
        return self.last_propagation

    @torch.no_grad()
    def target_distribution(
        self,
        gaussians: Any,
        indices: Tensor | None = None,
        k: int = 12,
        search_chunk: int = 16_384,
    ) -> tuple[Tensor, Tensor]:
        xyz = gaussians.get_xyz.detach()
        selected = self._indices(indices, len(gaussians), xyz.device)
        if selected.numel() == 0:
            return selected, xyz.new_empty((0, len(EXPERT_NAMES)))
        if len(gaussians) <= 1:
            prior = self.uncertain_prior.to(xyz).expand(selected.numel(), -1)
            return selected, prior.clone()
        neighbors = self._index_for(gaussians, search_chunk).query_indices(selected, k)
        local = xyz[neighbors] - xyz[selected, None]
        distance2 = local.square().sum(-1)
        distance_scale = distance2.median(dim=1, keepdim=True).values.clamp_min(1e-8)
        weights = torch.exp(-distance2 / (2.0 * distance_scale))
        mean = (weights[..., None] * local).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        centered = local - mean[:, None]
        covariance = torch.einsum("mk,mki,mkj->mij", weights, centered, centered)
        covariance = covariance / weights.sum(1)[:, None, None].clamp_min(1e-8)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        l0, l1, l2 = eigenvalues.unbind(-1)
        denominator = l2.clamp_min(1e-8)
        linearity = ((l2 - l1) / denominator).clamp(0.0, 1.0)
        planarity = ((l1 - l0) / denominator).clamp(0.0, 1.0)
        scattering = (l0 / denominator).clamp(0.0, 1.0)

        center_normal = self._normal(gaussians, selected)
        neighbor_normal = self._normal(gaussians, neighbors.reshape(-1)).reshape(
            selected.shape[0], neighbors.shape[1], 3
        )
        normal_variation = (
            weights * (1.0 - (neighbor_normal * center_normal[:, None]).sum(-1).abs())
        ).sum(1) / weights.sum(1).clamp_min(1e-8)
        normal_variation = normal_variation.clamp(0.0, 1.0)

        scales = self._scaling(gaussians, selected).sort(dim=-1).values.clamp_min(1e-8)
        disk_shape = (1.0 - scales[:, 0] / scales[:, 1]).clamp(0.0, 1.0) * (
            scales[:, 1] / scales[:, 2]
        ).clamp(0.0, 1.0)
        line_shape = (1.0 - scales[:, 1] / scales[:, 2]).clamp(0.0, 1.0)
        confidence = self._effective_confidence(gaussians, selected)
        boundary = gaussians.boundary_score.detach()[selected, 0].clamp(0.0, 1.0)
        error = torch.tanh(gaussians.geometry_error.detach()[selected, 0].clamp_min(0.0))

        scores = torch.stack(
            (
                0.15 + planarity * (1.0 - normal_variation) + 0.60 * disk_shape * (1.0 - boundary),
                0.15 + planarity * normal_variation + 0.35 * (1.0 - scattering) * normal_variation,
                0.10 + linearity + 0.65 * line_shape + 0.45 * boundary,
                0.15 + scattering + 0.70 * error + 0.30 * normal_variation,
                0.10 + 0.60 * scattering + 0.55 * error,
            ),
            dim=-1,
        )
        geometric_target = F.softmax(scores / max(self.temperature, 1e-4), dim=-1)
        prior = self.uncertain_prior.to(scores).expand_as(scores)
        geometric_entropy = -(
            geometric_target.clamp_min(1e-8) * geometric_target.clamp_min(1e-8).log()
        ).sum(-1) / math.log(len(EXPERT_NAMES))
        local_reliability = (1.0 - geometric_entropy).clamp(0.0, 1.0)
        # Semantics controls how strongly geometry policies affect rendering,
        # but clear local geometry may still teach the expert classifier. This
        # avoids an 80%-fuzzy classifier before propagation has covered a scene.
        target_gate = torch.maximum(confidence, 0.5 * local_reliability)
        target = target_gate[:, None] * geometric_target + (1.0 - target_gate[:, None]) * prior
        return selected, (target / target.sum(-1, keepdim=True)).detach()

    @torch.no_grad()
    def sample_targets(
        self,
        gaussians: Any,
        max_points: int = 2048,
        k: int = 12,
        search_chunk: int = 16_384,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        size = len(gaussians)
        if max_points < 1:
            raise ValueError("max_points must be positive")
        if size <= max_points:
            indices = torch.arange(size, device=gaussians.get_xyz.device)
        else:
            # Importance sampling retains seams/high-error regions without
            # excluding ordinary surface interiors.
            weight = (
                1.0
                + gaussians.boundary_score.detach()[:, 0]
                + torch.tanh(gaussians.geometry_error.detach()[:, 0].clamp_min(0.0))
                + self._effective_confidence(
                    gaussians,
                    torch.arange(size, device=gaussians.get_xyz.device),
                )
            )
            indices = torch.multinomial(weight, max_points, replacement=False, generator=generator)
        return self.target_distribution(gaussians, indices, k, search_chunk)


__all__ = [
    "EXPERT_NAMES",
    "GeometryEvidenceProjector",
    "GeometryPolicyOutput",
    "SoftGeometryPolicyBank",
]
