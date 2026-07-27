"""Unified adaptive-density controller.

This module is the only training component allowed to change Gaussian
topology.  It preserves the familiar 3DGS gradient/opacity behaviour, while
adding confidence-gated semantic boundary, reconstruction, and geometry
signals.  With missing or zero-confidence semantic observations the score and
all topology decisions reduce to the standard photometric path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import torch

from semantic.region_membership import SparseRegionMembership
from utils.general_utils import build_rotation


@dataclass(frozen=True)
class DensityDecision:
    """Masks produced from one accumulation window."""

    clone: torch.Tensor
    split: torch.Tensor
    prune: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class TopologyBudget:
    """Optional per-step policy for a controlled topology window.

    ``max_net_growth`` bounds ``after - before`` while
    ``replacement_budget`` bounds donor churn. A zero net-growth budget is a
    true prune-and-replace step: every clone consumes one donor slot and every
    split consumes ``children - 1`` donor slots.
    """

    max_net_growth: int
    replacement_budget: int
    protect_min_confidence: float = 0.5
    protect_boundary: float = 0.25
    protect_thin_probability: float = 0.5


@dataclass(frozen=True)
class DensityReport:
    """Auditable result of a topology update."""

    before: int
    cloned: int
    split_parents: int
    split_children: int
    pruned: int
    after: int
    score_mean: float
    score_threshold: float


def _as_column(value: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    value = value.detach().to(device=device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(length)
    if value.ndim > 1:
        value = value.reshape(length, -1).mean(dim=-1)
    if value.numel() != length:
        raise ValueError(f"Expected {length} per-Gaussian values, got {value.shape}")
    return value


def _robust_unit(value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Map a positive signal to [0,1] without a scene-specific scale."""

    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    selected = value if valid is None else value[valid]
    if selected.numel() == 0:
        return torch.zeros_like(value)
    # Quantiles make thresholds stable across scenes and training phases.
    lo = torch.quantile(selected.float(), 0.10)
    hi = torch.quantile(selected.float(), 0.90)
    if (hi - lo).abs() < 1e-12:
        return (value > 0).to(value.dtype)
    return ((value - lo) / (hi - lo)).clamp(0, 1)


class DensityController:
    """Accumulate renderer evidence and atomically mutate Gaussian topology.

    Parameters are deliberately expressed in the same units as the original
    3DGS densifier.  The additional weights only affect confident semantic
    observations and therefore do not destabilize RGB-only scenes.
    """

    _ACCUMULATORS = (
        "grad_accum",
        "grad_denom",
        "rgb_accum",
        "semantic_accum",
        "boundary_accum",
        "geometry_accum",
        "geometry_denom",
        "mesh_coverage_accum",
        "mesh_coverage_denom",
        "confidence_accum",
        "observation_count",
        "max_radii",
    )

    def __init__(self, config: dict[str, Any] | Any, scene_extent: float, policy_bank=None) -> None:
        self.cfg = config if isinstance(config, dict) else vars(config)
        self.scene_extent = float(scene_extent)
        self.policy_bank = policy_bank
        self._size = 0
        self._device = torch.device("cpu")
        self._window: str | None = None
        self._clear()

    @property
    def semantic_guidance_enabled(self) -> bool:
        """Whether semantics are allowed to influence topology decisions.

        Ablation configs set all three guidance weights to zero.  Treat that as
        one architectural switch, covering eligibility, expert policies,
        pruning protection and region balancing—not only the additive score.
        """

        return any(
            float(self.cfg.get(name, default)) > 0.0
            for name, default in (
                ("semantic_weight", 0.5),
                ("boundary_weight", 0.75),
                ("geometry_weight", 0.75),
            )
        )

    def _clear(self, size: int | None = None, device: torch.device | None = None) -> None:
        self._size = self._size if size is None else int(size)
        self._device = self._device if device is None else device
        def zeros() -> torch.Tensor:
            return torch.zeros(self._size, device=self._device, dtype=torch.float32)
        self.grad_accum = zeros()
        self.grad_denom = zeros()
        self.rgb_accum = zeros()
        self.semantic_accum = zeros()
        self.boundary_accum = zeros()
        self.geometry_accum = zeros()
        self.geometry_denom = zeros()
        self.mesh_coverage_accum = zeros()
        self.mesh_coverage_denom = zeros()
        self.confidence_accum = zeros()
        self.observation_count = zeros()
        self.max_radii = zeros()

    def _ensure(self, gaussians) -> None:
        size = int(gaussians.get_xyz.shape[0])
        device = gaussians.get_xyz.device
        if size != self._size or device != self._device:
            # Topology only changes through this controller and accumulation is
            # consumed at each mutation, so resizing here is never lossy.
            self._clear(size, device)

    @torch.no_grad()
    def state_dict(self, gaussians=None) -> dict[str, Any]:
        """Serialize the in-flight density window for exact checkpoint resume."""

        if gaussians is not None:
            self._ensure(gaussians)
        return {
            "version": 3,
            "size": self._size,
            "window": self._window,
            "accumulators": {
                name: getattr(self, name).detach().clone()
                for name in self._ACCUMULATORS
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any] | None, gaussians) -> None:
        """Restore an accumulation window after Gaussian topology is restored."""

        self._ensure(gaussians)
        if not state:
            raise ValueError("density checkpoint state is required")
        version = int(state.get("version", 1))
        if version != 3:
            if version > 3:
                raise ValueError("density checkpoint was produced by a newer schema")
            raise ValueError(
                "density checkpoint does not contain the region-conditioned schema; "
                "start a fresh run"
            )
        if int(state["size"]) != self._size:
            raise ValueError(
                "density checkpoint topology does not match restored Gaussians: "
                f"{state['size']} != {self._size}"
            )
        values = state["accumulators"]
        missing = [
            name
            for name in self._ACCUMULATORS
            if name not in values
        ]
        if missing:
            raise ValueError("density checkpoint is missing: " + ", ".join(missing))
        for name in self._ACCUMULATORS:
            source = torch.as_tensor(values[name], device=self._device, dtype=torch.float32)
            if source.shape != (self._size,):
                raise ValueError(
                    f"density accumulator {name!r} has shape {tuple(source.shape)}, "
                    f"expected {(self._size,)}"
                )
            getattr(self, name).copy_(source)
        window = state["window"]
        if window is not None and not isinstance(window, str):
            raise ValueError("density checkpoint window must be a string or null")
        self._window = window

    @torch.no_grad()
    def activate_window(self, gaussians, window: str) -> None:
        """Select an accumulation window, clearing evidence on transitions."""

        if not window:
            raise ValueError("density window must be a non-empty string")
        self._ensure(gaussians)
        if self._window != window:
            self._clear(self._size, self._device)
            self._window = window

    @torch.no_grad()
    def observe(
        self,
        gaussians,
        render_pkg: dict[str, torch.Tensor],
        camera=None,
        *,
        rgb_residual: torch.Tensor | None = None,
        semantic_residual: torch.Tensor | None = None,
        geometry_error: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one view after backward has populated view-space grads."""

        self._ensure(gaussians)
        n = self._size
        if n == 0:
            return

        visible = render_pkg.get("visibility_filter")
        if visible is None:
            visible = torch.ones(n, dtype=torch.bool, device=self._device)
        else:
            visible = visible.detach().to(self._device).bool()
        indices = visible.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return

        viewspace = render_pkg.get("viewspace_points")
        if viewspace is not None and viewspace.grad is not None:
            gradient = torch.linalg.vector_norm(viewspace.grad.detach()[indices, :2], dim=-1)
            self.grad_accum[indices] += gradient
            self.grad_denom[indices] += 1

        radii = render_pkg.get("radii")
        if radii is not None:
            self.max_radii[indices] = torch.maximum(
                self.max_radii[indices], _as_column(radii, n, self._device)[indices]
            )

        dominant = render_pkg.get("dominant_index")
        if dominant is not None and dominant.numel() > 0:
            dominant = dominant.detach().to(self._device).long()
            valid_pixel = (dominant >= 0) & (dominant < n)
            if valid_pixel.any():
                ids = dominant[valid_pixel]
                ones = torch.ones_like(ids, dtype=torch.float32)
                self.observation_count.scatter_add_(0, ids, ones)

                if rgb_residual is not None:
                    residual = rgb_residual.detach().to(self._device)
                    if residual.ndim == 3:
                        residual = residual.mean(0)
                    self.rgb_accum.scatter_add_(0, ids, residual[valid_pixel].float())

                confidence = None
                if camera is not None and getattr(camera, "semantic_confidence", None) is not None:
                    confidence = camera.semantic_confidence.detach().to(self._device)[valid_pixel].float()
                    self.confidence_accum.scatter_add_(0, ids, confidence)

                if semantic_residual is not None:
                    residual = semantic_residual.detach().to(self._device)
                    if residual.ndim == 3:
                        residual = residual.mean(0)
                    contribution = residual[valid_pixel].float()
                    if confidence is not None:
                        contribution = contribution * confidence
                    self.semantic_accum.scatter_add_(0, ids, contribution)

                if geometry_error is not None and geometry_error.numel() != n:
                    residual = geometry_error.detach().to(self._device)
                    if residual.ndim == 3:
                        residual = residual.mean(0)
                    contribution = residual[valid_pixel].float()
                    self.geometry_accum.scatter_add_(0, ids, contribution)
                    self.geometry_denom.scatter_add_(0, ids, ones)

                boundary = None if camera is None else getattr(camera, "semantic_boundary", None)
                if boundary is not None:
                    contribution = boundary.detach().to(self._device)[valid_pixel].float()
                    if confidence is not None:
                        contribution = contribution * confidence
                    self.boundary_accum.scatter_add_(0, ids, contribution)

        if geometry_error is not None and geometry_error.numel() == n:
            self.geometry_accum += _as_column(geometry_error, n, self._device) * visible.float()
            self.geometry_denom += visible.float()

    @torch.no_grad()
    def observe_mesh_coverage(
        self,
        gaussians,
        indices: torch.Tensor,
        residual: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        """Accumulate sparse per-Gaussian mesh coverage residuals.

        Mesh correspondence is intentionally owned by the caller.  This
        controller only receives sparse model indices and scale-normalized
        non-negative residuals, preserving one topology evidence lifecycle for
        renderer and mesh observations alike. Duplicate indices represent
        independent observations and are averaged through scatter-add counts.
        """

        self._ensure(gaussians)
        selected = torch.as_tensor(indices, device=self._device)
        if selected.ndim != 1:
            raise ValueError("mesh coverage indices must be one-dimensional")
        values = torch.as_tensor(
            residual,
            device=self._device,
            dtype=torch.float32,
        ).detach()
        if values.ndim > 1:
            values = values.reshape(-1)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.numel() != selected.numel():
            raise ValueError("mesh coverage residual must align with indices")
        if selected.numel() == 0:
            return
        if selected.dtype == torch.bool or selected.is_floating_point() or selected.is_complex():
            raise TypeError("mesh coverage indices must have an integer dtype")
        selected = selected.long()
        if bool((selected < 0).any()) or bool((selected >= self._size).any()):
            raise IndexError("mesh coverage index lies outside Gaussian topology")

        observed = torch.isfinite(values)
        if valid is not None:
            validity = torch.as_tensor(valid, device=self._device)
            if validity.dtype != torch.bool:
                raise TypeError("mesh coverage valid mask must be boolean")
            if validity.ndim > 1:
                validity = validity.reshape(-1)
            if validity.ndim == 0:
                validity = validity.reshape(1)
            if validity.numel() != selected.numel():
                raise ValueError("mesh coverage valid mask must align with indices")
            observed &= validity
        if not bool(observed.any()):
            return

        selected = selected[observed]
        values = values[observed].clamp_min(0)
        self.mesh_coverage_accum.scatter_add_(0, selected, values)
        self.mesh_coverage_denom.scatter_add_(
            0,
            selected,
            torch.ones_like(values),
        )

    def _mean_observation(self, accumulated: torch.Tensor) -> torch.Tensor:
        return accumulated / self.observation_count.clamp_min(1)

    def _confidence_weighted_observation(self, accumulated: torch.Tensor) -> torch.Tensor:
        valid = self.confidence_accum > 0
        return torch.where(
            valid,
            accumulated / self.confidence_accum.clamp_min(1e-8),
            torch.zeros_like(accumulated),
        )

    def _semantic_confidence(self, gaussians) -> torch.Tensor:
        observed = self._mean_observation(self.confidence_accum).clamp(0, 1)
        stored = getattr(gaussians, "get_semantic_confidence", None)
        if stored is None:
            return observed
        return torch.maximum(observed, _as_column(stored, self._size, self._device).clamp(0, 1))

    def scores(self, gaussians) -> torch.Tensor:
        """Return the confidence-gated unified refinement score."""

        self._ensure(gaussians)
        valid = self.observation_count > 0
        gradient = self.grad_accum / self.grad_denom.clamp_min(1)
        rgb = self._mean_observation(self.rgb_accum)
        semantic = self._confidence_weighted_observation(self.semantic_accum)
        boundary = self._confidence_weighted_observation(self.boundary_accum)
        geometry = self.geometry_accum / self.geometry_denom.clamp_min(1)
        mesh_coverage = (
            self.mesh_coverage_accum / self.mesh_coverage_denom.clamp_min(1)
        )
        confidence = self._semantic_confidence(gaussians)

        score = float(self.cfg.get("rgb_weight", 1.0)) * _robust_unit(gradient + rgb, valid)
        if self.semantic_guidance_enabled:
            semantic_score = (
                float(self.cfg.get("semantic_weight", 0.5))
                * _robust_unit(semantic, self.confidence_accum > 0)
                + float(self.cfg.get("boundary_weight", 0.75))
                * _robust_unit(boundary, self.confidence_accum > 0)
                + float(self.cfg.get("geometry_weight", 0.75))
                * _robust_unit(geometry, self.geometry_denom > 0)
            )
            score = score + confidence * semantic_score
        mesh_observed = self.mesh_coverage_denom > 0
        mesh_weight = float(self.cfg["mesh_coverage_weight"])
        if mesh_weight > 0.0 and bool(mesh_observed.any()):
            score = score + mesh_weight * _robust_unit(
                mesh_coverage,
                mesh_observed,
            ) * mesh_observed
        if self.semantic_guidance_enabled and self.policy_bank is not None:
            policy = self.policy_bank.from_gaussians(gaussians)
            score = score * policy.density_multiplier.detach()
        return score

    def _balanced_candidates(
        self,
        score: torch.Tensor,
        candidate: torch.Tensor,
        region_membership_resolver: Callable[[torch.Tensor], SparseRegionMembership] | None,
        capacity: int,
    ) -> torch.Tensor:
        """Select globally unique candidates under soft regional budgets.

        Regional mass is the sum of ``confidence * membership_weight``.  A
        candidate may contribute to several regional quotas, but can be chosen
        only once by the global topology transaction.  Duplicate regional
        selections are removed stably and any unspent capacity is filled by the
        global score.  Missing foreground evidence (including an all-background
        membership batch) intentionally takes the same global branch as an
        absent resolver; it is not converted to a hard semantic ID.
        """

        output = torch.zeros_like(candidate)
        indices = candidate.nonzero(as_tuple=False).squeeze(-1)
        if capacity <= 0 or indices.numel() == 0:
            return output
        capacity = min(capacity, indices.numel())

        def global_fill(remaining_capacity: int) -> None:
            if remaining_capacity <= 0:
                return
            remaining = indices[~output[indices]]
            if remaining.numel() == 0:
                return
            # Candidate indices are ascending, so stable sorting also gives a
            # deterministic index tie-break without perturbing the score.
            order = torch.argsort(
                score[remaining],
                descending=True,
                stable=True,
            )
            selected = remaining[order[: min(remaining_capacity, remaining.numel())]]
            output[selected] = True

        if region_membership_resolver is None:
            global_fill(capacity)
            return output

        membership = region_membership_resolver(indices)
        if not isinstance(membership, SparseRegionMembership):
            raise TypeError(
                "region_membership_resolver must return SparseRegionMembership"
            )
        if membership.ids.shape[0] != indices.numel():
            raise ValueError(
                "region_membership_resolver must align with density candidates; "
                f"expected {indices.numel()} rows, got {membership.ids.shape[0]}"
            )
        membership = membership.to(score.device)
        ids = membership.ids.detach()
        evidence = (
            membership.weights.detach()
            * membership.confidence.detach().reshape(-1, 1)
        )
        foreground = (ids > 0) & (evidence > 0)
        if not bool(foreground.any()):
            global_fill(capacity)
            return output

        unique = ids[foreground].unique(sorted=True)
        masses = torch.stack(
            [torch.where(ids == region, evidence, 0.0).sum() for region in unique]
        )
        temperature = float(self.cfg["region_budget_temperature"])
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("density.region_budget_temperature must be finite and non-negative")
        allocation_weights = masses.pow(temperature)
        raw_budget = capacity * allocation_weights / allocation_weights.sum().clamp_min(1e-12)
        budgets = raw_budget.floor().long()
        remainder = capacity - int(budgets.sum())
        if remainder > 0:
            fractional = raw_budget - budgets.float()
            order = torch.argsort(fractional, descending=True, stable=True)
            budgets[order[: min(remainder, unique.numel())]] += 1

        regional_selections: list[int] = []
        for region, budget in zip(unique, budgets):
            row_weight = torch.where(ids == region, evidence, 0.0).sum(dim=-1)
            local_rows = (row_weight > 0).nonzero(as_tuple=False).squeeze(-1)
            count = min(int(budget), local_rows.numel())
            if count:
                priority = score[indices[local_rows]] * row_weight[local_rows]
                order = torch.argsort(priority, descending=True, stable=True)
                regional_selections.extend(
                    indices[local_rows[order[:count]]].detach().cpu().tolist()
                )

        # Preserve deterministic region-ID and within-region ranking order
        # while ensuring the topology masks contain every Gaussian at most once.
        seen: set[int] = set()
        unique_selections = []
        for index in regional_selections:
            if index not in seen:
                seen.add(index)
                unique_selections.append(index)
        if unique_selections:
            selected = torch.as_tensor(
                unique_selections[:capacity],
                device=output.device,
                dtype=torch.long,
            )
            output[selected] = True
        global_fill(capacity - int(output.sum()))
        return output

    @staticmethod
    def _limit_donors(
        donor: torch.Tensor,
        score: torch.Tensor,
        opacity: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Keep the least useful donor points within a churn budget."""

        output = torch.zeros_like(donor)
        indices = donor.nonzero(as_tuple=False).squeeze(-1)
        capacity = min(max(int(capacity), 0), indices.numel())
        if capacity == 0:
            return output
        # Opacity is the primary deletion signal; unified refinement score
        # breaks ties in favour of retaining useful geometry.
        utility = _robust_unit(score, donor)
        priority = (1.0 - opacity.clamp(0.0, 1.0)) + (1.0 - utility)
        selected = indices[priority[indices].topk(capacity).indices]
        output[selected] = True
        return output

    @staticmethod
    def _event_counts(
        score: torch.Tensor,
        clone_candidate: torch.Tensor,
        split_candidate: torch.Tensor,
        slots: int,
        split_cost: int,
    ) -> tuple[int, int]:
        """Solve the two-cost event allocation exactly for ranked scores."""

        slots = max(int(slots), 0)
        if slots == 0:
            return 0, 0
        clone_scores = score[clone_candidate]
        split_scores = score[split_candidate]
        clone_limit = min(clone_scores.numel(), slots)
        split_limit = min(split_scores.numel(), slots // split_cost)
        clone_values = (
            clone_scores.topk(clone_limit).values if clone_limit else clone_scores[:0]
        )
        split_values = (
            split_scores.topk(split_limit).values if split_limit else split_scores[:0]
        )
        clone_prefix = torch.cat((score.new_zeros(1), clone_values.cumsum(0)))
        split_prefix = torch.cat((score.new_zeros(1), split_values.cumsum(0)))
        split_counts = torch.arange(split_limit + 1, device=score.device, dtype=torch.long)
        clone_counts = (slots - split_counts * split_cost).clamp(min=0, max=clone_limit)
        objective = split_prefix + clone_prefix[clone_counts]
        # Absolute gates already established that every candidate warrants
        # refinement. If score sums tie (commonly in a uniform first window),
        # prefer the allocation that actually uses more financed slots.
        tied = objective == objective.max()
        used_slots = clone_counts + split_counts * split_cost
        best = int(torch.where(tied, used_slots, -torch.ones_like(used_slots)).argmax())
        return int(clone_counts[best]), best

    @staticmethod
    def _cached_region_membership_resolver(
        candidate: torch.Tensor,
        resolver: Callable[[torch.Tensor], SparseRegionMembership] | None,
    ) -> Callable[[torch.Tensor], SparseRegionMembership] | None:
        """Resolve a union candidate set once, then index-select sorted subsets."""

        if resolver is None:
            return None
        indices = candidate.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return None
        membership = resolver(indices)
        if not isinstance(membership, SparseRegionMembership):
            raise TypeError(
                "region_membership_resolver must return SparseRegionMembership"
            )
        if membership.ids.shape[0] != indices.numel():
            raise ValueError(
                "region_membership_resolver must align with density candidates; "
                f"expected {indices.numel()} rows, got {membership.ids.shape[0]}"
            )
        membership = membership.to(indices.device)

        def cached(query: torch.Tensor) -> SparseRegionMembership:
            positions = torch.searchsorted(indices, query)
            if positions.numel() and (
                bool((positions >= indices.numel()).any())
                or not torch.equal(indices.index_select(0, positions), query)
            ):
                raise ValueError("cached membership query is not a subset of candidates")
            return membership.index_select(positions)

        return cached

    def decide(
        self,
        gaussians,
        *,
        region_membership_resolver: Callable[[torch.Tensor], SparseRegionMembership] | None = None,
        percent_dense: float = 0.01,
        enable_size_pruning: bool = False,
        topology_budget: TopologyBudget | None = None,
    ) -> DensityDecision:
        self._ensure(gaussians)
        score = self.scores(gaussians)
        gradient = self.grad_accum / self.grad_denom.clamp_min(1)
        threshold = float(self.cfg.get("gradient_threshold", 2e-4))
        split_drive = score
        policy = None
        if self.semantic_guidance_enabled and self.policy_bank is not None:
            policy = self.policy_bank.from_gaussians(gaussians)
            split_drive = split_drive * policy.split_score.detach()
        # Absolute evidence decides whether refinement is warranted; the
        # scale-free unified score only ranks those warranted candidates.
        # This prevents a uniformly small positive residual from becoming all
        # ones after robust normalization and exponentially growing topology.
        confidence = self._semantic_confidence(gaussians)
        semantic = self._confidence_weighted_observation(self.semantic_accum)
        boundary = self._confidence_weighted_observation(self.boundary_accum)
        geometry = self.geometry_accum / self.geometry_denom.clamp_min(1)
        semantic_candidate = torch.zeros_like(gradient, dtype=torch.bool)
        if self.semantic_guidance_enabled:
            semantic_candidate = confidence >= float(
                self.cfg.get("min_semantic_confidence", 0.35)
            )
            semantic_candidate &= (
                (
                    float(self.cfg.get("semantic_weight", 0.5)) > 0.0
                    and semantic
                    >= float(self.cfg.get("semantic_residual_threshold", 0.25))
                )
                | (
                    float(self.cfg.get("boundary_weight", 0.75)) > 0.0
                    and boundary >= float(self.cfg.get("boundary_threshold", 0.10))
                )
                | (
                    float(self.cfg.get("geometry_weight", 0.75)) > 0.0
                    and geometry
                    >= float(self.cfg.get("geometry_error_threshold", 0.10))
                )
            )
        candidate = (gradient >= threshold) | semantic_candidate

        scales = gaussians.get_scaling.max(dim=-1).values
        small = scales <= float(percent_dense) * self.scene_extent
        opacity = gaussians.get_opacity.reshape(-1)
        posterior = (
            getattr(gaussians, "get_geometry_posterior", None)
            if self.semantic_guidance_enabled
            else None
        )
        if posterior is None:
            thin_probability = torch.zeros_like(opacity)
        else:
            thin_probability = posterior[:, 2].detach()

        min_opacity = float(self.cfg.get("min_opacity", 0.005))
        max_radius = float(self.cfg.get("max_screen_radius", 20.0))
        oversized = (self.max_radii > max_radius) | (scales > 0.1 * self.scene_extent)
        low_opacity = opacity < min_opacity
        # Confident thin structures and semantic seams should survive aggressive
        # screen/world-size pruning. Truly transparent points remain removable.
        protection = torch.zeros_like(confidence)
        if self.semantic_guidance_enabled:
            protection = confidence * torch.maximum(
                self._confidence_weighted_observation(self.boundary_accum),
                float(self.cfg.get("thin_protection", 0.8)) * thin_probability,
            )
        if policy is not None:
            protection = torch.maximum(protection, policy.prune_protection.detach())
        size_prune = oversized & (protection < 0.5) if enable_size_pruning else False
        prune = low_opacity | size_prune

        if topology_budget is not None:
            stored_boundary = getattr(gaussians, "get_boundary_score", None)
            if stored_boundary is not None:
                boundary = torch.maximum(
                    boundary,
                    _as_column(stored_boundary, self._size, self._device).clamp(0, 1),
                )
            protected_structure = confidence >= float(
                topology_budget.protect_min_confidence
            )
            protected_structure &= (
                boundary >= float(topology_budget.protect_boundary)
            ) | (
                thin_probability >= float(topology_budget.protect_thin_probability)
            )
            prune &= ~protected_structure
            prune = self._limit_donors(
                prune,
                split_drive,
                opacity,
                topology_budget.replacement_budget,
            )

        current = self._size
        maximum = int(self.cfg.get("max_gaussians", 2_000_000))
        replacement_enabled = bool(self.cfg["capacity_replacement_enabled"])
        near_cap_ratio = float(self.cfg["replace_near_cap_ratio"])
        near_cap = current >= int(math.floor(maximum * near_cap_ratio))
        donor_financing = replacement_enabled and (
            topology_budget is not None or near_cap
        )
        if donor_financing:
            # A financed donor cannot also be a refinement parent; otherwise
            # an event could silently consume its own slot and violate cap.
            candidate &= ~prune
        clone_candidate = candidate & small
        split_candidate = candidate & ~small

        children = int(self.cfg.get("split_children", 2))
        if children < 2:
            raise ValueError("density.split_children must be at least 2")
        growth_fraction = float(self.cfg.get("max_growth_fraction", 0.05))
        maximum_new = int(self.cfg.get("max_new_per_step", 100_000))
        if growth_fraction <= 0 or maximum_new < 1:
            raise ValueError("density growth limits must be positive")
        remaining_capacity = max(0, maximum - current)
        global_growth = min(
            remaining_capacity,
            maximum_new,
            max(1, int(math.ceil(current * growth_fraction))),
        )
        donor_count = int(prune.sum())
        if topology_budget is None:
            replacement_limit = int(self.cfg["max_replacements_per_step"])
            replacement_slots = (
                min(donor_count, replacement_limit)
                if near_cap and replacement_enabled
                else 0
            )
            net_growth = global_growth
        else:
            replacement_slots = donor_count if replacement_enabled else 0
            net_growth = min(global_growth, max(int(topology_budget.max_net_growth), 0))
        event_slots = net_growth + replacement_slots
        clone_count, split_count = self._event_counts(
            split_drive,
            clone_candidate,
            split_candidate,
            event_slots,
            children - 1,
        )
        cached_resolver = self._cached_region_membership_resolver(
            candidate if clone_count + split_count else torch.zeros_like(candidate),
            region_membership_resolver if self.semantic_guidance_enabled else None,
        )
        clone = self._balanced_candidates(
            split_drive,
            clone_candidate,
            cached_resolver,
            clone_count,
        )
        split = self._balanced_candidates(
            split_drive,
            split_candidate,
            cached_resolver,
            split_count,
        )
        prune &= ~(clone | split)
        donor_count = int(prune.sum())
        expected_after = current - donor_count + clone_count + split_count * (children - 1)
        if expected_after > maximum:
            raise RuntimeError(
                "density decision exceeds max_gaussians: "
                f"{current} - {donor_count} + {clone_count} + "
                f"{split_count}*{children - 1} = {expected_after} > {maximum}"
            )
        return DensityDecision(clone=clone, split=split, prune=prune, score=split_drive)

    def _policy_offsets(self, gaussians, split: torch.Tensor, children: int) -> torch.Tensor:
        """Sample split offsets from a posterior mixture of geometry experts."""

        indices = split.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return gaussians.get_xyz.new_empty((0, children, 3))
        scales = gaussians.get_scaling[indices]
        rotation = build_rotation(gaussians.get_rotation[indices])
        posterior = (
            getattr(gaussians, "get_geometry_posterior", None)
            if self.semantic_guidance_enabled
            else None
        )
        if posterior is None:
            multipliers = torch.ones_like(scales)
        else:
            probability = posterior[indices].detach()
            # Axis order follows Gaussian local scales. Policies are continuous:
            # planar suppresses the shortest-axis offset, thin favours the long
            # axis, fuzzy/freeform remain close to standard covariance sampling.
            order = scales.argsort(dim=-1)
            inverse = order.argsort(dim=-1)
            expert_sorted = scales.new_tensor(
                [
                    [0.12, 1.0, 1.0],  # planar
                    [0.55, 0.85, 1.0],  # curved
                    [0.25, 0.45, 1.0],  # thin
                    [1.0, 1.0, 1.0],   # freeform
                    [0.8, 0.8, 0.8],   # fuzzy / conservative
                ]
            )
            sorted_multiplier = probability @ expert_sorted
            multipliers = sorted_multiplier.gather(1, inverse)
        local = torch.randn(indices.numel(), children, 3, device=scales.device, dtype=scales.dtype)
        local = local * (scales * multipliers)[:, None, :]
        # Centre children around the parent so a split does not drift geometry.
        local = local - local.mean(dim=1, keepdim=True)
        return torch.einsum("mij,mcj->mci", rotation, local)

    @torch.no_grad()
    def update_evidence(self, gaussians) -> None:
        update = getattr(gaussians, "update_evidence", None)
        if update is None or self._size == 0:
            return
        observed = self.observation_count > 0
        indices = observed.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return
        update(
            indices=indices,
            semantic_confidence=self._mean_observation(self.confidence_accum)[
                indices
            ].unsqueeze(-1),
            boundary_score=self._confidence_weighted_observation(
                self.boundary_accum
            )[indices].unsqueeze(-1),
            geometry_error=(
                self.geometry_accum / self.geometry_denom.clamp_min(1)
            )[indices].unsqueeze(-1),
            # GaussianModel owns a cumulative cross-window count. Passing this
            # window's pixel count would overwrite history and make unseen
            # Gaussians lose confidence at every topology step.
            momentum=0.9,
        )

    @torch.no_grad()
    def step(
        self,
        gaussians,
        *,
        region_membership_resolver: Callable[[torch.Tensor], SparseRegionMembership] | None = None,
        percent_dense: float = 0.01,
        enable_size_pruning: bool = False,
        topology_budget: TopologyBudget | None = None,
    ) -> DensityReport:
        """Consume accumulated evidence and perform one atomic topology step."""

        self._ensure(gaussians)
        self.update_evidence(gaussians)
        decision = self.decide(
            gaussians,
            region_membership_resolver=region_membership_resolver,
            percent_dense=percent_dense,
            enable_size_pruning=enable_size_pruning,
            topology_budget=topology_budget,
        )
        before = self._size
        threshold = float(self.cfg.get("gradient_threshold", 2e-4))
        clone_count = int(decision.clone.sum())
        split_count = int(decision.split.sum())
        prune_count = int(decision.prune.sum())
        children = int(self.cfg.get("split_children", 2))

        # All masks refer to the same source topology. The registry constructs
        # survivors, clones, and split children in one optimizer-safe transaction.
        changed = clone_count + split_count + prune_count > 0
        if changed:
            offsets = self._policy_offsets(gaussians, decision.split, children)
            mutate = getattr(gaussians, "mutate_topology", None)
            if mutate is None:
                raise AttributeError("GaussianModel must implement atomic mutate_topology()")
            mutate(
                decision.clone,
                decision.split,
                decision.prune,
                children=children,
                offsets=offsets,
                scale_factor=0.8,
            )

        after = int(gaussians.get_xyz.shape[0])
        maximum = int(self.cfg.get("max_gaussians", 2_000_000))
        if after > maximum:
            raise RuntimeError(
                f"atomic topology mutation produced {after} Gaussians, cap is {maximum}"
            )
        self._clear(after, gaussians.get_xyz.device)
        return DensityReport(
            before=before,
            cloned=clone_count,
            split_parents=split_count,
            split_children=split_count * children,
            pruned=prune_count,
            after=after,
            score_mean=float(decision.score.mean()) if decision.score.numel() else 0.0,
            score_threshold=threshold,
        )

    @torch.no_grad()
    def reset_opacity(self, gaussians, maximum: float = 0.01) -> None:
        """Delegate periodic opacity reset without bypassing the registry."""

        reset = getattr(gaussians, "reset_opacity", None)
        if reset is None:
            raise AttributeError("GaussianModel must implement reset_opacity()")
        try:
            reset(maximum=maximum)
        except TypeError:
            reset()
