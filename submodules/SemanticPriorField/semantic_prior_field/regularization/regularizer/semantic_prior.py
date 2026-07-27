"""Semantic Prior Field regularization.

Train-loop entry points for the SPF consumers that act through losses:

  * orientation prior  — align the learned per-Gaussian normal with the
    per-instance proxy normal (plane / quadric), confidence-weighted;
  * selective flattening — the PGSR min-scale loss restricted to Gaussians
    of planar instances (thin instances are explicitly exempt);
  * SH region consistency — same-instance neighbours should have similar
    view-dependent appearance (chunked kNN, never N x N);
  * SH outlier decay — inside a coherent instance, Gaussians with
    anomalously high high-order SH energy are decayed. This targets the
    "geometry error baked into view-dependent color" failure mode.

Follows the same initialize / compute / reset contract as the normal-field,
multiview and mesh-in-the-loop regularizers.
"""

from typing import Any, Dict, Optional

import torch

from semantic.prior_field import (
    PRIOR_PLANAR,
    PRIOR_QUADRIC,
    BoundaryWeightCache,
    SemanticPriorField,
)


def initialize_semantic_prior(
    scene,
    config: Dict[str, Any],
    observations,
) -> Dict[str, Any]:
    state = {
        "prior_field": SemanticPriorField(config),
        "boundary_cache": BoundaryWeightCache(observations, config),
    }
    return state


def reset_semantic_prior_state_at_next_iteration(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Invalidate cached per-Gaussian quantities after a topology change."""
    state["prior_field"].invalidate()
    return state


def get_boundary_weight_map(
    state: Dict[str, Any],
    viewpoint_cam,
    device,
) -> torch.Tensor:
    """(H, W) map in [boundary_weight, 1], cached per view."""
    return state["boundary_cache"].get(
        viewpoint_cam.image_name,
        viewpoint_cam.image_height,
        viewpoint_cam.image_width,
        device,
    )


def maybe_refresh_prior_field(
    iteration: int,
    gaussians,
    semantic_head,
    config: Dict[str, Any],
    state: Dict[str, Any],
    verbose: bool = True,
) -> SemanticPriorField:
    prior_field: SemanticPriorField = state["prior_field"]
    if iteration >= int(config["start_iter"]) and prior_field.needs_refresh(iteration):
        prior_field.refresh(gaussians, semantic_head, iteration)
        if verbose:
            print(f"[INFO] Refreshed Semantic Prior Field at iteration {iteration}.")
            print(f"        > {prior_field.summary()}")
    return prior_field


def compute_semantic_prior_regularization(
    iteration: int,
    gaussians,
    semantic_head,
    config: Dict[str, Any],
    state: Dict[str, Any],
    args,
    visibility_filter: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    device = gaussians.get_xyz.device
    zero = torch.zeros(size=(), device=device)

    prior_field = maybe_refresh_prior_field(
        iteration, gaussians, semantic_head, config, state
    )
    if not prior_field.valid:
        return {
            "semantic_prior_loss": zero,
            "orientation_prior_loss": zero,
            "selective_flatten_loss": zero,
            "sh_consistency_loss": zero,
            "sh_decay_loss": zero,
        }

    # --- Orientation prior -------------------------------------------------
    orientation_prior_loss = zero
    orient_active = (
        getattr(args, "sp_orient", True)
        and gaussians.use_gaussian_features
        and iteration >= int(config.get("orient_start_iter", 20_001))
    )
    if orient_active:
        prior_mask = prior_field.prior_weight > 0
        if visibility_filter is not None:
            prior_mask = prior_mask & visibility_filter
        if prior_mask.any():
            normals = gaussians.convert_features_to_normals(normalize=True)[prior_mask]
            proxy_normals = prior_field.prior_normals[prior_mask]
            weights = prior_field.prior_weight[prior_mask]
            # Sign-agnostic alignment: the sign channel is supervised by the
            # normal-field losses; the proxy constrains the axis only.
            alignment_error = 1.0 - (normals * proxy_normals).sum(dim=-1).abs()
            orientation_prior_loss = (
                float(config.get("orient_weight", 0.05))
                * (weights * alignment_error).sum()
                / weights.sum().clamp_min(1e-8)
            )

    # --- Selective flattening ---------------------------------------------
    selective_flatten_loss = zero
    if getattr(args, "sp_flatten", True):
        flatten_mask = (
            (prior_field.prior_type == PRIOR_PLANAR)
            | (prior_field.prior_type == PRIOR_QUADRIC)
        ) & (prior_field.prior_weight > 0)
        if visibility_filter is not None:
            flatten_mask = flatten_mask & visibility_filter
        if flatten_mask.any():
            min_scaling = gaussians.get_scaling_with_3D_filter[flatten_mask].min(dim=-1).values
            weights = prior_field.prior_weight[flatten_mask]
            selective_flatten_loss = (
                float(config.get("flatten_weight", 10.0))
                * (weights * min_scaling / gaussians.spatial_lr_scale).sum()
                / weights.sum().clamp_min(1e-8)
            )

    # --- SH region consistency ---------------------------------------------
    sh_consistency_loss = zero
    if getattr(args, "sp_sh", True) and float(config.get("sh_consistency_weight", 0.0)) > 0:
        sh_consistency_loss = float(config.get("sh_consistency_weight", 0.01)) * (
            _sh_region_consistency(
                features_rest=gaussians.get_features_rest,
                positions=gaussians.get_xyz.detach(),
                labels=prior_field.labels,
                sample_size=int(config.get("sh_consistency_samples", 4096)),
                neighbors=int(config.get("sh_consistency_neighbors", 5)),
            )
        )

    # --- SH outlier decay ---------------------------------------------------
    sh_decay_loss = zero
    if getattr(args, "sp_sh", True) and float(config.get("sh_decay_weight", 0.0)) > 0:
        sh_decay_loss = float(config.get("sh_decay_weight", 0.01)) * _sh_outlier_decay(
            features_rest=gaussians.get_features_rest,
            labels=prior_field.labels,
            outlier_factor=float(config.get("sh_outlier_factor", 4.0)),
        )

    semantic_prior_loss = (
        orientation_prior_loss
        + selective_flatten_loss
        + sh_consistency_loss
        + sh_decay_loss
    )
    return {
        "semantic_prior_loss": semantic_prior_loss,
        "orientation_prior_loss": (
            orientation_prior_loss.detach()
            if torch.is_tensor(orientation_prior_loss)
            else orientation_prior_loss
        ),
        "selective_flatten_loss": (
            selective_flatten_loss.detach()
            if torch.is_tensor(selective_flatten_loss)
            else selective_flatten_loss
        ),
        "sh_consistency_loss": (
            sh_consistency_loss.detach()
            if torch.is_tensor(sh_consistency_loss)
            else sh_consistency_loss
        ),
        "sh_decay_loss": (
            sh_decay_loss.detach() if torch.is_tensor(sh_decay_loss) else sh_decay_loss
        ),
    }


def _sh_region_consistency(
    features_rest: torch.Tensor,
    positions: torch.Tensor,
    labels: torch.Tensor,
    sample_size: int = 4096,
    neighbors: int = 5,
) -> torch.Tensor:
    """Same-instance kNN smoothness on high-order SH coefficients.

    Sampled anchors are compared against sampled candidates only; pairs with
    different labels are masked out of the neighbourhood, so appearance is
    never smoothed across instance boundaries.
    """
    labelled = (labels >= 0).nonzero(as_tuple=True)[0]
    if labelled.numel() < 2:
        return features_rest.sum() * 0.0
    sample_size = min(int(sample_size), labelled.numel())
    sampled = labelled[torch.randperm(labelled.numel(), device=labels.device)[:sample_size]]
    sample_positions = positions[sampled]
    sample_labels = labels[sampled]
    sample_features = features_rest[sampled].reshape(sample_size, -1)

    k = min(int(neighbors) + 1, sample_size)
    chunk_size = min(2048, sample_size)
    losses = []
    for start in range(0, sample_size, chunk_size):
        end = min(start + chunk_size, sample_size)
        distances = torch.cdist(sample_positions[start:end], sample_positions)
        same_label = sample_labels[start:end, None] == sample_labels[None, :]
        distances = torch.where(
            same_label, distances, torch.full_like(distances, float("inf"))
        )
        knn = distances.topk(k=k, largest=False)
        neighbour_indices = knn.indices[:, 1:]
        neighbour_valid = knn.values[:, 1:].isfinite()
        if not neighbour_valid.any():
            continue
        neighbour_features = sample_features[neighbour_indices]  # (C, k-1, D)
        anchor_features = sample_features[start:end].unsqueeze(1)
        difference = ((anchor_features - neighbour_features) ** 2).mean(dim=-1)
        losses.append(difference[neighbour_valid].mean())
    if not losses:
        return features_rest.sum() * 0.0
    return torch.stack(losses).mean()


def _sh_outlier_decay(
    features_rest: torch.Tensor,
    labels: torch.Tensor,
    outlier_factor: float = 4.0,
) -> torch.Tensor:
    """Decay anomalously view-dependent Gaussians inside coherent instances.

    Within one instance the high-order SH energy should be spatially
    coherent; isolated Gaussians whose energy exceeds ``outlier_factor``
    times the instance median are typically compensating a geometry error
    and are pulled back toward the instance statistics. Instances are never
    decayed as a whole, so genuinely specular regions keep their capacity.
    """
    flat = features_rest.reshape(features_rest.shape[0], -1)
    energy = flat.norm(dim=-1)
    labelled_mask = labels >= 0
    if not labelled_mask.any():
        return features_rest.sum() * 0.0
    labelled_labels = labels[labelled_mask]
    labelled_energy = energy[labelled_mask]

    unique_labels, inverse = torch.unique(labelled_labels, return_inverse=True)
    # Per-instance median energy without materializing per-instance slices.
    medians = torch.zeros(unique_labels.numel(), device=energy.device)
    for i in range(unique_labels.numel()):
        medians[i] = labelled_energy[inverse == i].detach().median()
    thresholds = outlier_factor * medians[inverse]
    excess = (labelled_energy - thresholds).clamp_min(0.0)
    if excess.numel() == 0:
        return features_rest.sum() * 0.0
    return excess.mean()
