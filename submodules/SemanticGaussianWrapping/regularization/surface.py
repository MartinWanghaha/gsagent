"""Differentiable consistency between Gaussians and the shared surface field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation


@dataclass(frozen=True)
class GaussianSurfaceBatch:
    """One sampled consistency batch whose field query may be shared."""

    indices: torch.Tensor
    gaussian_normals: torch.Tensor
    query_points: torch.Tensor
    region_ids: torch.Tensor
    region_weights: torch.Tensor


def _bounded_random_indices(
    count: int,
    samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample unique rows without allocating a permutation of a 2M model."""

    target = min(max(int(samples), 0), int(count))
    if target == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if target == count or target > count // 4:
        return torch.randperm(count, device=device)[:target]
    selected = torch.empty(0, device=device, dtype=torch.long)
    while selected.numel() < target:
        missing = target - selected.numel()
        draw = torch.randint(
            count,
            (max(2 * missing, 32),),
            device=device,
        )
        selected = torch.unique(torch.cat((selected, draw)), sorted=False)
    return selected[:target]


def _selected_geometry(gaussians, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    registry = getattr(gaussians, "registry", None)
    if registry is not None and "scaling" in registry and "rotation" in registry:
        scaling = registry["scaling"].index_select(0, indices).exp()
        rotation = F.normalize(
            registry["rotation"].index_select(0, indices),
            dim=-1,
            eps=1e-8,
        )
        return scaling, rotation
    return (
        gaussians.get_scaling.index_select(0, indices),
        gaussians.get_rotation.index_select(0, indices),
    )


def _selected_confidence(gaussians, indices: torch.Tensor) -> torch.Tensor:
    registry = getattr(gaussians, "registry", None)
    if registry is not None and "semantic_confidence" in registry:
        direct = registry["semantic_confidence"].index_select(0, indices)
        if "propagated_semantic_confidence" in registry:
            propagated = registry["propagated_semantic_confidence"].index_select(
                0,
                indices,
            )
            direct = torch.maximum(direct, propagated)
        return direct
    return gaussians.get_semantic_confidence.index_select(0, indices)


def prepare_gaussian_surface_consistency(
    gaussians,
    sample_points: int = 8192,
    pivot_factor: float = 1.5,
    *,
    region_top_k: int = 3,
    region_decode_chunk_size: int = 32_768,
    minimum_region_weight: float = 0.05,
) -> GaussianSurfaceBatch | None:
    count = gaussians.get_xyz.shape[0]
    if count == 0 or int(sample_points) <= 0:
        return None
    indices = _bounded_random_indices(
        count,
        sample_points,
        gaussians.get_xyz.device,
    )
    membership_decoder = getattr(gaussians, "point_region_memberships", None)
    if not callable(membership_decoder):
        raise TypeError(
            "Gaussian surface consistency requires point_region_memberships"
        )
    membership = membership_decoder(
        indices,
        top_k=int(region_top_k),
        chunk_size=int(region_decode_chunk_size),
    )
    if membership.ids.shape[1] == 0:
        return None
    # Keep the sparse posterior intact: one spatial sample is evaluated against
    # all sufficiently supported Gaga regions in the same field reduction.
    evidence_weight = membership.weights * membership.confidence
    valid_membership = evidence_weight >= float(minimum_region_weight)
    keep = valid_membership.any(dim=1)
    if not bool(keep.any()):
        return None
    indices = indices[keep]
    region_ids = membership.ids[keep]
    # Confidence is applied once by the consistency objective; keep this term
    # as the unrenormalized semantic membership probability.
    region_weights = torch.where(
        valid_membership[keep],
        membership.weights[keep],
        torch.zeros_like(membership.weights[keep]),
    ).detach()
    centers = gaussians.get_xyz[indices]
    scales, quaternions = _selected_geometry(gaussians, indices)
    rotations = build_rotation(quaternions)
    smallest = scales.argmin(-1)
    batch = torch.arange(indices.shape[0], device=indices.device)
    gaussian_normals = rotations[batch, :, smallest]
    offset = pivot_factor * scales[batch, smallest, None] * gaussian_normals
    return GaussianSurfaceBatch(
        indices=indices,
        gaussian_normals=gaussian_normals,
        query_points=torch.cat((centers, centers + offset, centers - offset), dim=0),
        region_ids=region_ids,
        region_weights=region_weights,
    )


def gaussian_surface_consistency(
    gaussians,
    surface_field,
    sample_points: int = 8192,
    pivot_factor: float = 1.5,
    *,
    prepared: GaussianSurfaceBatch | None = None,
    query_result=None,
    region_top_k: int = 3,
    region_decode_chunk_size: int = 32_768,
    minimum_region_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prepared is None:
        prepared = prepare_gaussian_surface_consistency(
            gaussians,
            sample_points,
            pivot_factor,
            region_top_k=region_top_k,
            region_decode_chunk_size=region_decode_chunk_size,
            minimum_region_weight=minimum_region_weight,
        )
    if prepared is None:
        zero = gaussians.get_xyz.new_zeros(())
        return zero, {"center": zero, "crossing": zero, "normal": zero}

    # One field call shares discrete routing, compact activated Gaussian
    # attributes and policy outputs across all three consistency probes.
    # Concatenating does not couple their differentiable field evaluations;
    # the result is exactly the same three point sets split back apart.
    point_regions = prepared.region_ids.repeat((3, 1))
    query = (
        surface_field.query_point_regions(
            prepared.query_points,
            point_regions,
        )
        if query_result is None
        else query_result
    )
    expected_shape = (prepared.query_points.shape[0], prepared.region_ids.shape[1])
    if query.sdf.shape != expected_shape:
        raise ValueError("surface consistency query result has the wrong shape")
    samples = prepared.indices.shape[0]
    valid = query.valid
    center_sdf, positive_sdf, negative_sdf = torch.where(
        valid,
        query.sdf,
        torch.zeros_like(query.sdf),
    ).split(samples)
    center_valid, positive_valid, negative_valid = valid.split(samples)
    center_normal = query.normal[:samples]
    confidence = _selected_confidence(gaussians, prepared.indices).squeeze(-1)
    confidence = confidence.detach().clamp(0, 1)
    weight = (0.25 + 0.75 * confidence[:, None]) * prepared.region_weights

    center_weight = weight * center_valid
    center_loss = (center_weight * center_sdf.abs()).sum() / center_weight.sum().clamp_min(1.0)
    # An oriented surface must cross one of the two normal pivots.  The normal
    # sign itself is irrelevant, so only the product is constrained.
    crossing_weight = weight * positive_valid * negative_valid
    crossing_loss = (
        crossing_weight * F.relu(positive_sdf * negative_sdf)
    ).sum() / crossing_weight.sum().clamp_min(1.0)
    normal_alignment = 1.0 - (
        F.normalize(center_normal, dim=-1)
        * prepared.gaussian_normals[:, None, :]
    ).sum(-1).abs()
    normal_loss = (center_weight * normal_alignment).sum() / center_weight.sum().clamp_min(1.0)
    total = center_loss + crossing_loss + 0.25 * normal_loss
    return total, {"center": center_loss, "crossing": crossing_loss, "normal": normal_loss}


__all__ = [
    "GaussianSurfaceBatch",
    "gaussian_surface_consistency",
    "prepare_gaussian_surface_consistency",
]
