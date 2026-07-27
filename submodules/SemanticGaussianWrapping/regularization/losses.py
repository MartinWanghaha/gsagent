"""Joint photometric, semantic, manifold, and geometry-policy losses."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from semantic.neighbor_index import GaussianNeighborIndex
from utils.general_utils import build_rotation
from utils.graphics_utils import DEFAULT_NORMAL_ALPHA_THRESHOLD, depth_normal_residual
from utils.loss_utils import (
    l1_loss,
    soft_dice_loss,
    ssim,
    symmetric_kl,
)


class PixelSemanticDecoder(nn.Module):
    """Apply the scene-owned point decoder to a semantic feature image."""

    def __init__(self, semantic_dim: int, num_classes: int, decoder: nn.Module) -> None:
        super().__init__()
        if decoder is None:
            raise ValueError("PixelSemanticDecoder requires the scene semantic decoder")
        self.semantic_dim = int(semantic_dim)
        self.num_classes = int(num_classes)
        self.decoder = decoder

    def decode_points(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode tensors whose final axis is the semantic feature dimension."""

        if embedding.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"semantic embedding must have {self.semantic_dim} channels"
            )
        logits = self.decoder(embedding)
        if logits.shape != (*embedding.shape[:-1], self.num_classes):
            raise ValueError("scene semantic decoder returned an invalid logits shape")
        return logits

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 4:
            raise ValueError("semantic feature image must have shape [B,D,H,W]")
        if embedding.shape[1] != self.semantic_dim:
            raise ValueError(
                f"semantic feature image must have {self.semantic_dim} channels"
            )
        pixels = embedding.permute(0, 2, 3, 1)
        logits = self.decode_points(pixels)
        return logits.permute(0, 3, 1, 2)


@dataclass
class LossBundle:
    photometric: torch.Tensor
    auxiliary: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return self.photometric + self.auxiliary


class SemanticLossSystem(nn.Module):
    """Converts rendered semantic embeddings into all semantic supervision.

    The decoder is deliberately scene-specific, matching Gaga's open-world
    grouping while keeping the per-Gaussian embedding dimension fixed.
    """

    def __init__(
        self,
        semantic_dim: int,
        num_classes: int,
        weights: dict[str, float],
        semantic_decoder: nn.Module,
        policy_bank=None,
        evidence_projector=None,
        evidence_interval: int = 100,
        evidence_samples: int = 2048,
        evidence_weight: float = 1.0,
        evidence_entropy_weight: float = 0.05,
        evidence_balance_weight: float = 0.10,
        neighbor_index: GaussianNeighborIndex | None = None,
        normal_alpha_threshold: float = DEFAULT_NORMAL_ALPHA_THRESHOLD,
        region_top_k: int = 3,
        region_decode_chunk_size: int = 32_768,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        self.weights = weights
        self.policy_bank = policy_bank
        self.evidence_projector = evidence_projector
        self.neighbor_index = neighbor_index
        if evidence_projector is not None and neighbor_index is not None:
            setter = getattr(evidence_projector, "set_neighbor_index", None)
            if callable(setter):
                setter(neighbor_index)
        self.evidence_interval = max(int(evidence_interval), 1)
        self.evidence_samples = max(int(evidence_samples), 1)
        self.evidence_weight = float(evidence_weight)
        self.evidence_entropy_weight = float(evidence_entropy_weight)
        self.evidence_balance_weight = float(evidence_balance_weight)
        if self.evidence_entropy_weight < 0 or self.evidence_balance_weight < 0:
            raise ValueError("geometry evidence regularization weights must be non-negative")
        self.normal_alpha_threshold = float(normal_alpha_threshold)
        if not 0.0 <= self.normal_alpha_threshold <= 1.0:
            raise ValueError("normal_alpha_threshold must be in [0,1]")
        if isinstance(region_top_k, bool) or int(region_top_k) < 1:
            raise ValueError("region_top_k must be positive")
        if isinstance(region_decode_chunk_size, bool) or int(region_decode_chunk_size) < 1:
            raise ValueError("region_decode_chunk_size must be positive")
        self.region_top_k = int(region_top_k)
        self.region_decode_chunk_size = int(region_decode_chunk_size)
        self._evidence_iteration = -1
        self._evidence_indices: torch.Tensor | None = None
        self._evidence_targets: torch.Tensor | None = None
        self.classifier = PixelSemanticDecoder(semantic_dim, num_classes, semantic_decoder)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(semantic_dim, max(semantic_dim // 2, 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(semantic_dim // 2, 4), 1, 1),
        )

    def photometric_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = l1_loss(rendered, target)
        structural = 1.0 - ssim(rendered, target)
        dssim = self.weights.get("lambda_dssim", 0.2)
        loss = (1.0 - dssim) * l1 + dssim * structural
        return loss, {"l1": l1, "dssim": structural}

    def _region_balanced_mean(
        self,
        value: torch.Tensor,
        camera,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Average a pixel residual inside regions, then across regions.

        Background remains governed by the global objective.  Foreground
        regions receive a tempered area weight so a small, consistently
        observed object is not erased by the surrounding scene.
        """

        ids = getattr(camera, "semantic_ids", None)
        confidence = getattr(camera, "semantic_confidence", None)
        if ids is None or confidence is None:
            return value.new_zeros(())
        ids = ids.to(device=value.device, dtype=torch.long)
        confidence = confidence.to(device=value.device, dtype=value.dtype)
        if value.shape != ids.shape or confidence.shape != ids.shape:
            raise ValueError("region-balanced residual and semantic maps must share [H,W]")
        mask = (ids > 0) & (confidence > 0) & torch.isfinite(value)
        if valid is not None:
            if valid.shape != value.shape:
                raise ValueError("region-balanced valid mask must have shape [H,W]")
            mask &= valid.to(device=value.device, dtype=torch.bool)
        if not bool(mask.any()):
            return value.new_zeros(())
        selected_ids = ids[mask]
        unique, inverse = torch.unique(selected_ids, sorted=True, return_inverse=True)
        selected_weight = confidence[mask]
        mass = value.new_zeros(unique.numel())
        total = value.new_zeros(unique.numel())
        mass.scatter_add_(0, inverse, selected_weight)
        total.scatter_add_(0, inverse, value[mask] * selected_weight)
        region_mean = total / mass.clamp_min(1e-8)
        temperature = float(self.weights["region_area_temperature"])
        region_weight = mass.detach().pow(temperature)
        region_weight = region_weight / region_weight.sum().clamp_min(1e-8)
        return (region_weight * region_mean).sum()

    def region_photometric_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        camera,
    ) -> torch.Tensor:
        residual = (rendered - target).abs().mean(dim=0)
        return self._region_balanced_mean(residual, camera)

    def semantic_image_losses(self, rendered_embedding: torch.Tensor, camera) -> dict[str, torch.Tensor]:
        zero = rendered_embedding.new_zeros(())
        if getattr(camera, "semantic_ids", None) is None:
            return {"semantic": zero, "boundary": zero}
        ids = camera.semantic_ids.to(rendered_embedding.device)
        confidence = camera.semantic_confidence.to(rendered_embedding.device)
        if not bool((confidence > 0).any()):
            return {"semantic": zero, "boundary": zero}
        if ids.shape != rendered_embedding.shape[1:] or confidence.shape != ids.shape:
            raise ValueError("semantic supervision must match rendered image resolution")
        pixels = rendered_embedding.permute(1, 2, 0).reshape(-1, self.semantic_dim)
        targets = ids.reshape(-1)
        weights = confidence.reshape(-1)
        ignore_index = int(getattr(camera, "ignore_label", -1))
        valid_weight = weights * (targets != ignore_index)

        def chunk_numerator(
            features: torch.Tensor,
            labels: torch.Tensor,
            chunk_weight: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.classifier.decode_points(features)
            per_pixel = F.cross_entropy(
                logits,
                labels,
                ignore_index=ignore_index,
                reduction="none",
            )
            return (per_pixel * chunk_weight).sum()

        numerators = []
        for start in range(0, pixels.shape[0], self.region_decode_chunk_size):
            end = start + self.region_decode_chunk_size
            features = pixels[start:end]
            labels = targets[start:end]
            chunk_weight = valid_weight[start:end]
            numerators.append(
                checkpoint(
                    chunk_numerator,
                    features,
                    labels,
                    chunk_weight,
                    use_reentrant=False,
                )
            )
        semantic_loss = torch.stack(numerators).sum() / valid_weight.sum().clamp_min(1.0)
        if getattr(camera, "semantic_boundary", None) is None:
            boundary_loss = zero
        else:
            boundary_logits = self.boundary_head(rendered_embedding[None])[0, 0]
            boundary = camera.semantic_boundary.to(rendered_embedding.device)
            bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary, reduction="none")
            boundary_loss = 0.5 * (bce * confidence).sum() / confidence.sum().clamp_min(1.0)
            boundary_loss = boundary_loss + 0.5 * soft_dice_loss(
                boundary_logits,
                boundary,
                confidence,
            )
        return {"semantic": semantic_loss, "boundary": boundary_loss}

    @torch.no_grad()
    def semantic_residual(
        self,
        rendered_embedding: torch.Tensor,
        camera,
    ) -> torch.Tensor | None:
        """Decode a dense residual map without retaining a scene-sized logit tensor."""

        if getattr(camera, "semantic_ids", None) is None:
            return None
        ids = camera.semantic_ids.to(rendered_embedding.device)
        if ids.shape != rendered_embedding.shape[1:]:
            raise ValueError("semantic supervision must match rendered image resolution")
        pixels = rendered_embedding.permute(1, 2, 0).reshape(-1, self.semantic_dim)
        targets = ids.reshape(-1)
        residual = rendered_embedding.new_empty(targets.shape)
        ignore_index = int(getattr(camera, "ignore_label", -1))
        for start in range(0, pixels.shape[0], self.region_decode_chunk_size):
            end = start + self.region_decode_chunk_size
            logits = self.classifier.decode_points(pixels[start:end])
            residual[start:end] = F.cross_entropy(
                logits,
                targets[start:end],
                ignore_index=ignore_index,
                reduction="none",
            )
        return residual.reshape_as(ids)

    def set_neighbor_index(self, neighbor_index: GaussianNeighborIndex) -> None:
        self.neighbor_index = neighbor_index
        setter = getattr(self.evidence_projector, "set_neighbor_index", None)
        if callable(setter):
            setter(neighbor_index)

    def _index_for(self, gaussians) -> GaussianNeighborIndex:
        if self.neighbor_index is None or self.neighbor_index.gaussians is not gaussians:
            self.set_neighbor_index(GaussianNeighborIndex(gaussians))
        return self.neighbor_index

    @torch.no_grad()
    def _sample_knn(
        self,
        gaussians,
        xyz: torch.Tensor,
        max_points: int = 512,
        k: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample centers and query their shared detached neighbor index."""

        if xyz.shape[0] <= 1:
            empty = torch.empty(0, dtype=torch.long, device=xyz.device)
            return empty, empty.view(0, 0)
        count = min(max_points, xyz.shape[0])
        query_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:count]
        return query_idx, self._index_for(gaussians).query_indices(query_idx, k)

    def manifold_loss(self, gaussians) -> torch.Tensor:
        xyz = gaussians.get_xyz
        query, neighbors = self._sample_knn(gaussians, xyz)
        if query.numel() == 0 or neighbors.numel() == 0:
            return xyz.new_zeros(())
        decoder = getattr(gaussians, "point_region_memberships", None)
        if not callable(decoder):
            raise TypeError("region manifold loss requires point_region_memberships")
        query_membership = decoder(
            query,
            top_k=self.region_top_k,
            chunk_size=self.region_decode_chunk_size,
        )
        flat_neighbors = neighbors.reshape(-1)
        neighbor_membership = decoder(
            flat_neighbors,
            top_k=self.region_top_k,
            chunk_size=self.region_decode_chunk_size,
        )
        neighbor_ids = neighbor_membership.ids.reshape(
            neighbors.shape[0],
            neighbors.shape[1],
            -1,
        )
        neighbor_weights = neighbor_membership.weights.reshape_as(neighbor_ids)
        matches = query_membership.ids[:, None, :, None] == neighbor_ids[:, :, None, :]
        agreement = (
            query_membership.weights[:, None, :, None]
            * neighbor_weights[:, :, None, :]
            * matches
        ).sum(dim=(-1, -2))
        query_confidence = query_membership.confidence[:, 0]
        neighbor_confidence = neighbor_membership.confidence.reshape(
            neighbors.shape[0],
            neighbors.shape[1],
        )
        same_weight = (
            agreement
            * torch.sqrt(
                query_confidence[:, None] * neighbor_confidence
            )
        ).detach().clamp(0, 1)
        geometry_logits = gaussians.get_geometry_logits
        consistency = symmetric_kl(
            geometry_logits[query, None].expand(-1, neighbors.shape[1], -1),
            geometry_logits[neighbors],
        )
        normalizer = same_weight.sum().clamp_min(1.0)
        policy_consistency = (same_weight * consistency).sum() / normalizer

        # Semantically coherent neighbors provide geometry evidence without a
        # category-to-shape lookup table. Planar experts align local normals and
        # centers to a tangent sheet; thin experts align their long axes.
        scales = gaussians.get_scaling
        query_scales = scales[query]
        neighbor_scales = scales[neighbors]
        query_rotation = build_rotation(gaussians.get_rotation[query])
        neighbor_rotation = build_rotation(gaussians.get_rotation[neighbors].reshape(-1, 4)).reshape(
            query.shape[0], neighbors.shape[1], 3, 3
        )
        query_batch = torch.arange(query.shape[0], device=scales.device)
        neighbor_batch = torch.arange(neighbors.numel(), device=scales.device)
        query_minimum = query_scales.argmin(-1)
        query_maximum = query_scales.argmax(-1)
        neighbor_minimum = neighbor_scales.reshape(-1, 3).argmin(-1)
        neighbor_maximum = neighbor_scales.reshape(-1, 3).argmax(-1)
        query_normals = query_rotation[query_batch, :, query_minimum]
        query_long_axes = query_rotation[query_batch, :, query_maximum]
        flat_neighbor_rotation = neighbor_rotation.reshape(-1, 3, 3)
        neighbor_normals = flat_neighbor_rotation[neighbor_batch, :, neighbor_minimum].reshape(
            query.shape[0], neighbors.shape[1], 3
        )
        neighbor_long_axes = flat_neighbor_rotation[neighbor_batch, :, neighbor_maximum].reshape(
            query.shape[0], neighbors.shape[1], 3
        )
        delta = xyz[neighbors] - xyz[query, None]
        distance = delta.norm(dim=-1).clamp_min(1e-8)
        tangent_error = (delta * query_normals[:, None]).sum(-1).abs() / distance
        normal_error = 1.0 - (query_normals[:, None] * neighbor_normals).sum(-1).abs()
        long_axis_error = 1.0 - (query_long_axes[:, None] * neighbor_long_axes).sum(-1).abs()
        posterior = F.softmax(geometry_logits, dim=-1)
        planar_weight = same_weight * posterior[query, 0, None]
        thin_weight = same_weight * posterior[query, 2, None]
        planar_loss = (planar_weight * (tangent_error + 0.5 * normal_error)).sum() / planar_weight.sum().clamp_min(1.0)
        thin_loss = (thin_weight * long_axis_error).sum() / thin_weight.sum().clamp_min(1.0)
        return policy_consistency + 0.25 * planar_loss + 0.15 * thin_loss

    def invalidate_geometry_evidence(self) -> None:
        self._evidence_iteration = -1
        self._evidence_indices = None
        self._evidence_targets = None

    def evidence_state_dict(self) -> dict[str, torch.Tensor | int | None]:
        propagation_cursor = getattr(
            self.evidence_projector,
            "_propagation_cursor",
            None,
        )
        return {
            "version": 1,
            "iteration": int(self._evidence_iteration),
            "propagation_cursor": (
                None
                if not isinstance(propagation_cursor, torch.Tensor)
                else int(propagation_cursor.item())
            ),
            "indices": (
                None
                if self._evidence_indices is None
                else self._evidence_indices.detach().clone()
            ),
            "targets": (
                None
                if self._evidence_targets is None
                else self._evidence_targets.detach().clone()
            ),
        }

    def load_evidence_state_dict(
        self, state: dict[str, torch.Tensor | int | None] | None
    ) -> None:
        if not isinstance(state, dict):
            raise ValueError("geometry-evidence checkpoint state is required")
        if type(state.get("version")) is not int or state["version"] != 1:
            raise ValueError("geometry-evidence checkpoint schema must be version 1")
        required = {"iteration", "propagation_cursor", "indices", "targets"}
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "geometry-evidence checkpoint is missing: "
                + ", ".join(sorted(missing))
            )
        iteration = state["iteration"]
        if type(iteration) is not int or iteration < -1:
            raise ValueError("geometry-evidence iteration must be an integer >= -1")
        self._evidence_iteration = iteration
        indices = state["indices"]
        targets = state["targets"]
        if (indices is None) != (targets is None):
            raise ValueError(
                "geometry-evidence indices and targets must both be present or absent"
            )
        reference = next(self.parameters())
        self._evidence_indices = (
            None
            if indices is None
            else torch.as_tensor(indices, device=reference.device).long()
        )
        self._evidence_targets = (
            None
            if targets is None
            else torch.as_tensor(
                targets,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        propagation_cursor = getattr(
            self.evidence_projector,
            "_propagation_cursor",
            None,
        )
        saved_cursor = state["propagation_cursor"]
        if saved_cursor is not None and (
            type(saved_cursor) is not int or saved_cursor < 0
        ):
            raise ValueError(
                "geometry-evidence propagation cursor must be a non-negative integer"
            )
        if isinstance(propagation_cursor, torch.Tensor) and saved_cursor is not None:
            propagation_cursor.fill_(int(saved_cursor))

    def _expert_evidence_loss(self, gaussians, iteration: int | None) -> torch.Tensor:
        if self.evidence_projector is None or len(gaussians) == 0:
            return gaussians.get_xyz.new_zeros(())
        refresh = self._evidence_indices is None
        refresh |= self._evidence_indices is not None and (
            self._evidence_indices.numel() == 0
            or int(self._evidence_indices.max()) >= len(gaussians)
        )
        if iteration is not None:
            refresh |= self._evidence_iteration < 0 or iteration - self._evidence_iteration >= self.evidence_interval
        if refresh:
            propagate = getattr(
                self.evidence_projector,
                "propagate_semantic_confidence",
                None,
            )
            if callable(propagate):
                propagate(gaussians)
            indices, targets = self.evidence_projector.sample_targets(
                gaussians,
                max_points=self.evidence_samples,
            )
            self._evidence_indices = indices.detach()
            self._evidence_targets = targets.detach()
            self._evidence_iteration = -1 if iteration is None else int(iteration)
        if self._evidence_indices is None or self._evidence_indices.numel() == 0:
            return gaussians.get_xyz.new_zeros(())
        logits = gaussians.get_geometry_logits[self._evidence_indices]
        targets = self._evidence_targets.to(logits)
        log_posterior = F.log_softmax(logits, dim=-1)
        posterior = log_posterior.exp()
        evidence_kl = F.kl_div(log_posterior, targets, reduction="batchmean")

        # Confident, unambiguous geometry should select an expert decisively.
        # Ambiguous targets are exempt, avoiding forced certainty at semantic
        # seams. A batch target-distribution term counters single-expert
        # collapse while retaining the scene's naturally imbalanced geometry.
        confidence_source = getattr(gaussians, "get_semantic_confidence", None)
        if confidence_source is None:
            confidence_source = gaussians.semantic_confidence
        confidence_source = (
            confidence_source() if callable(confidence_source) else confidence_source
        )
        confidence = confidence_source[self._evidence_indices].reshape(-1).detach().clamp(0, 1)
        normalizer = math.log(max(logits.shape[-1], 2))
        target_entropy = -(
            targets.clamp_min(1e-8) * targets.clamp_min(1e-8).log()
        ).sum(-1) / normalizer
        safe_posterior = posterior.clamp_min(1e-8)
        posterior_entropy = -(
            safe_posterior * safe_posterior.log()
        ).sum(-1) / normalizer
        certainty_weight = (0.25 + 0.75 * confidence) * (1.0 - target_entropy).clamp(0, 1)
        certainty_loss = (
            certainty_weight * posterior_entropy
        ).sum() / certainty_weight.sum().clamp_min(1.0)
        mean_target = targets.mean(0).clamp_min(1e-8)
        mean_target = mean_target / mean_target.sum()
        mean_posterior = posterior.mean(0).clamp_min(1e-8)
        balance_loss = F.kl_div(mean_posterior.log(), mean_target, reduction="sum")
        return (
            evidence_kl
            + self.evidence_entropy_weight * certainty_loss
            + self.evidence_balance_weight * balance_loss
        )

    def geometry_policy_loss(
        self,
        gaussians,
        iteration: int | None = None,
        policy_terms: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Soft expert constraints on Gaussian covariance without class hard-coding."""

        if self.policy_bank is not None:
            terms = (
                self.policy_bank.regularization(gaussians)
                if policy_terms is None
                else policy_terms
            )
            return (
                terms["scale_anisotropy"]
                + terms["normal_alignment"]
                + self.evidence_weight * self._expert_evidence_loss(gaussians, iteration)
            )

        posterior = F.softmax(gaussians.get_geometry_logits, dim=-1)
        scales = gaussians.get_scaling
        ordered, _ = scales.sort(dim=-1)
        thinness = ordered[:, 0] / ordered[:, 2].clamp_min(1e-8)
        planarity = ordered[:, 0] / ordered[:, 1].clamp_min(1e-8)
        planar = posterior[:, 0]
        thin = posterior[:, 2]
        fuzzy = posterior[:, 4]
        # Planar Gaussians are surfel-like; thin Gaussians retain one long axis;
        # fuzzy regions are intentionally exempt from flattening.
        return ((1.0 - fuzzy) * (planar * planarity + thin * thinness)).mean()

    def normal_consistency_loss(self, render_pkg: dict[str, torch.Tensor], camera) -> torch.Tensor:
        if "expected_depth" not in render_pkg or "normal" not in render_pkg:
            return render_pkg["render"].new_zeros(())
        residual, valid = depth_normal_residual(
            camera,
            render_pkg["expected_depth"],
            render_pkg["normal"],
            render_pkg.get("alpha"),
            alpha_threshold=self.normal_alpha_threshold,
        )
        # ``where`` inside depth_normal_residual keeps an autograd connection
        # even when no pixels are valid, with exactly zero invalid gradients.
        global_mean = residual.sum() / valid.sum().clamp_min(1)
        ids = getattr(camera, "semantic_ids", None)
        confidence = getattr(camera, "semantic_confidence", None)
        if ids is None or confidence is None:
            return global_mean
        region_valid = (
            ids.to(valid.device) > 0
        ) & (confidence.to(valid.device) > 0) & valid
        if not bool(region_valid.any()):
            return global_mean
        region_mean = self._region_balanced_mean(residual, camera, valid)
        mix = float(self.weights["region_geometry_mix"])
        return (1.0 - mix) * global_mean + mix * region_mean

    def sh_prior_loss(
        self,
        gaussians,
        policy_terms: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.policy_bank is not None:
            terms = (
                self.policy_bank.regularization(gaussians)
                if policy_terms is None
                else policy_terms
            )
            return terms["sh_complexity"]
        rest = gaussians.get_features[:, 1:, :]
        if rest.numel() == 0:
            return gaussians.get_xyz.new_zeros(())
        posterior = F.softmax(gaussians.get_geometry_logits, dim=-1)
        confidence = gaussians.get_semantic_confidence.squeeze(-1)
        stable = confidence * (posterior[:, 0] + 0.5 * posterior[:, 1])
        return (stable[:, None, None] * rest.square()).mean()

    def forward(
        self,
        render_pkg: dict[str, torch.Tensor],
        camera,
        gaussians,
        curriculum_weights: dict[str, float],
        surface_loss: torch.Tensor | None = None,
        mesh_loss: torch.Tensor | None = None,
        iteration: int | None = None,
        target_image: torch.Tensor | None = None,
    ) -> LossBundle:
        target = camera.original_image if target_image is None else target_image
        photo, terms = self.photometric_loss(
            render_pkg["render"], target.to(render_pkg["render"].device)
        )
        region_rgb = self.region_photometric_loss(
            render_pkg["render"],
            target.to(render_pkg["render"].device),
            camera,
        )
        terms["region_rgb"] = region_rgb
        zero = render_pkg["render"].new_zeros(())

        def active(name: str) -> bool:
            return (
                self.weights.get(f"lambda_{name}", 0.0) != 0.0
                and curriculum_weights.get(name, 0.0) != 0.0
            )

        if active("semantic") or active("boundary"):
            terms.update(self.semantic_image_losses(render_pkg["semantic"], camera))
        else:
            terms.update(semantic=zero, boundary=zero)
        terms["manifold"] = self.manifold_loss(gaussians) if active("manifold") else zero
        shared_policy_terms = None
        if self.policy_bank is not None and (active("geometry") or active("sh")):
            # Scale/rotation/SH activations span every Gaussian. Compute them
            # once per optimization step and share them across both losses.
            shared_policy_terms = self.policy_bank.regularization(gaussians)
        terms["geometry"] = (
            self.geometry_policy_loss(
                gaussians,
                iteration,
                shared_policy_terms,
            )
            + self.normal_consistency_loss(render_pkg, camera)
            if active("geometry")
            else zero
        )
        terms["sh"] = (
            self.sh_prior_loss(gaussians, shared_policy_terms)
            if active("sh")
            else zero
        )
        terms["surface"] = zero if surface_loss is None or not active("surface") else surface_loss
        terms["mesh"] = zero if mesh_loss is None or not active("mesh") else mesh_loss

        auxiliary = photo.new_zeros(())
        for name in (
            "region_rgb",
            "semantic",
            "boundary",
            "manifold",
            "geometry",
            "sh",
            "surface",
            "mesh",
        ):
            base_weight = self.weights.get(f"lambda_{name}", 0.0)
            auxiliary = auxiliary + base_weight * curriculum_weights.get(name, 0.0) * terms[name]
        return LossBundle(photo, auxiliary, terms)
