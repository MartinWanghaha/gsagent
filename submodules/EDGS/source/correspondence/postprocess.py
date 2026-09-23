"""Paper-faithful MV-RoMa pair filtering and multi-view track sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import (
    DenseMultiViewCorrespondence,
    DirectedPair,
    ImageSamplingGeometry,
    MultiViewTracks,
    PairField,
    PairQuality,
)
from .pair_store import PairFieldStore
from .planning import CameraGroup


@dataclass(frozen=True)
class GroupTrackCandidates:
    group_index: int
    group: CameraGroup
    flat_source_indices: torch.Tensor
    tracks: MultiViewTracks

    def __post_init__(self) -> None:
        if self.flat_source_indices.ndim != 1:
            raise ValueError("flat_source_indices must be one-dimensional")
        if self.flat_source_indices.shape[0] != self.tracks.coordinates.shape[0]:
            raise ValueError("candidate indices and tracks differ in length")


def merge_dense_group(
    store: PairFieldStore,
    group: CameraGroup,
    dense: DenseMultiViewCorrespondence,
) -> None:
    """Merge one dense group into ordered-pair fields using paper Eq.11."""

    if dense.target_count != len(group.target_indices):
        raise ValueError("dense target count differs from planned camera group")
    height, width = dense.spatial_shape
    geometry = dense.source_geometry or ImageSamplingGeometry.identity(height, width)
    probability = dense.certainty_logits[:, 0].sigmoid()
    for local_target, target_index in enumerate(group.target_indices):
        warp = dense.target_coordinates[local_target]
        confidence = probability[local_target]
        valid = (
            dense.source_valid
            & dense.target_valid[local_target]
            & torch.isfinite(warp).all(dim=0)
            & torch.isfinite(confidence)
            & (warp.abs() <= 1.0).all(dim=0)
        )
        store.merge(
            DirectedPair(group.source_index, target_index),
            source_coordinates=dense.source_coordinates,
            target_coordinates=warp,
            confidence=confidence,
            valid=valid,
            source_geometry=geometry,
        )


def reciprocal_cycle_mask(
    forward: PairField,
    reverse: PairField,
    *,
    max_error_px: float,
    device: torch.device | str,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Evaluate paper Eq.12 for one direction in source-image pixels."""

    if max_error_px <= 0 or row_chunk <= 0:
        raise ValueError("cycle threshold and row_chunk must be positive")
    target_device = torch.device(device)
    reverse_coordinates = reverse.target_coordinates.to(target_device)
    reverse_valid = reverse.valid.to(target_device)
    forward_target = forward.target_coordinates.to(target_device)
    source_coordinates = forward.source_coordinates.to(target_device)
    height, width = forward.valid.shape
    output = torch.zeros((height, width), dtype=torch.bool, device=target_device)

    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        semantic_grid = forward_target[:, start:stop].permute(1, 2, 0)
        storage_grid = reverse.source_geometry.semantic_to_storage_grid(
            semantic_grid
        ).unsqueeze(0)
        sampled_reverse = F.grid_sample(
            reverse_coordinates.unsqueeze(0),
            storage_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0].permute(1, 2, 0)
        sampled_valid = F.grid_sample(
            reverse_valid.float()[None, None],
            storage_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0]
        source = source_coordinates[:, start:stop].permute(1, 2, 0)
        delta_px = forward.source_geometry.normalized_delta_to_storage_pixels(
            sampled_reverse - source
        )
        output[start:stop] = (
            forward.valid[start:stop].to(target_device)
            & (sampled_valid >= 0.999)
            & torch.isfinite(sampled_reverse).all(dim=-1)
            & (torch.linalg.vector_norm(delta_px, dim=-1) <= max_error_px)
        )
    return output.cpu()


def apply_reciprocal_cycle_filter(
    store: PairFieldStore,
    *,
    max_error_px: float,
    device: torch.device | str,
) -> dict[DirectedPair, PairQuality]:
    """Filter all fields together; missing reverse predictions become invalid."""

    keys = sorted(store.keys())
    qualities: dict[DirectedPair, PairQuality] = {}
    visited: set[frozenset[int]] = set()
    for key in keys:
        undirected = frozenset((key.source_index, key.target_index))
        if undirected in visited:
            continue
        visited.add(undirected)
        reverse_key = key.reverse()
        forward = store.read(key)
        forward_before = int(forward.valid.sum().item())
        if reverse_key not in store:
            forward_mask = torch.zeros_like(forward.valid)
            store.apply_valid_mask(key, forward_mask)
            qualities[key] = PairQuality(0.0, 0.0, 0)
            continue

        reverse = store.read(reverse_key)
        reverse_before = int(reverse.valid.sum().item())
        # Compute both masks from the unmodified merged fields.
        forward_mask = reciprocal_cycle_mask(
            forward,
            reverse,
            max_error_px=max_error_px,
            device=device,
        )
        reverse_mask = reciprocal_cycle_mask(
            reverse,
            forward,
            max_error_px=max_error_px,
            device=device,
        )
        store.apply_valid_mask(key, forward_mask)
        store.apply_valid_mask(reverse_key, reverse_mask)

        for pair_key, before in ((key, forward_before), (reverse_key, reverse_before)):
            field = store.read(pair_key)
            count = int(field.valid.sum().item())
            mean_confidence = (
                float(field.confidence[field.valid].mean().item()) if count else 0.0
            )
            qualities[pair_key] = PairQuality(
                cycle_inlier_ratio=count / max(1, before),
                mean_confidence=mean_confidence,
                valid_pixels=count,
            )
    return qualities


def spatial_nms(
    score: torch.Tensor,
    *,
    radius_px: int,
    max_points: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic square-radius NMS with score/index tie breaking."""

    if score.ndim != 2:
        raise ValueError("score must have shape [H,W]")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive or null")
    score_cpu = score.detach().to(device="cpu", dtype=torch.float32)
    height, width = score_cpu.shape
    flat = torch.nonzero(score_cpu.reshape(-1) > 0, as_tuple=False).squeeze(1)
    if flat.numel() == 0:
        return flat, score_cpu.new_empty((0,))
    values = score_cpu.reshape(-1)[flat]
    flat_np = flat.numpy()
    values_np = values.numpy()
    order = np.lexsort((flat_np, -values_np))
    suppressed = np.zeros((height, width), dtype=np.bool_)
    selected: list[int] = []
    for position in order.tolist():
        index = int(flat_np[position])
        y, x = divmod(index, width)
        if suppressed[y, x]:
            continue
        selected.append(index)
        y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
        x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
        suppressed[y0:y1, x0:x1] = True
        if max_points is not None and len(selected) >= max_points:
            break
    indices = torch.tensor(selected, dtype=torch.long)
    return indices, score_cpu.reshape(-1)[indices]


def grid_balanced_spatial_nms(
    score: torch.Tensor,
    *,
    radius_px: int,
    grid_size_px: int,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave NMS candidates across occupied image tiles.

    NMS still chooses the strongest representative in each local neighborhood.
    Round-robin tile allocation then prevents high-visibility regions from
    consuming the entire per-source budget before valid boundary regions are
    considered.
    """

    if grid_size_px <= 0 or max_points <= 0:
        raise ValueError("grid_size_px and max_points must be positive")
    candidates, candidate_scores = spatial_nms(
        score,
        radius_px=radius_px,
        max_points=None,
    )
    if candidates.numel() <= max_points:
        return candidates, candidate_scores

    width = int(score.shape[1])
    columns = (width + grid_size_px - 1) // grid_size_px
    buckets: dict[int, list[int]] = {}
    for position, flat_index in enumerate(candidates.tolist()):
        y, x = divmod(flat_index, width)
        cell = (y // grid_size_px) * columns + x // grid_size_px
        buckets.setdefault(cell, []).append(position)

    selected_positions: list[int] = []
    depth = 0
    cells = sorted(buckets)
    while len(selected_positions) < max_points:
        added = False
        for cell in cells:
            bucket = buckets[cell]
            if depth < len(bucket):
                selected_positions.append(bucket[depth])
                added = True
                if len(selected_positions) == max_points:
                    break
        if not added:
            break
        depth += 1
    positions = torch.tensor(selected_positions, dtype=torch.long)
    return candidates[positions], candidate_scores[positions]


def build_group_track_candidates(
    store: PairFieldStore,
    group: CameraGroup,
    *,
    group_index: int,
    confidence_threshold: float,
    min_target_views: int,
    nms_radius_px: int,
    sampling_strategy: str,
    sampling_grid_size_px: int,
    visibility_score_weight: float,
    max_points: int,
) -> GroupTrackCandidates | None:
    """Construct paper Eq.13 tracks from already cycle-filtered pair fields."""

    available = [
        target
        for target in group.target_indices
        if DirectedPair(group.source_index, target) in store
    ]
    if len(available) < min_target_views:
        return None
    first = store.read(DirectedPair(group.source_index, available[0]))
    height, width = first.valid.shape
    target_coordinates = []
    target_confidence = []
    target_valid = []
    for target in group.target_indices:
        key = DirectedPair(group.source_index, target)
        if key in store:
            field = store.read(key)
            if field.valid.shape != (height, width):
                raise ValueError("a source camera produced inconsistent field sizes")
            target_coordinates.append(field.target_coordinates)
            target_confidence.append(field.confidence)
            target_valid.append(
                field.valid & (field.confidence > confidence_threshold)
            )
        else:
            target_coordinates.append(torch.zeros((2, height, width)))
            target_confidence.append(torch.zeros((height, width)))
            target_valid.append(torch.zeros((height, width), dtype=torch.bool))

    coordinates = torch.stack(target_coordinates)
    confidence = torch.stack(target_confidence)
    valid = torch.stack(target_valid)
    length = valid.sum(dim=0)
    mean_confidence = (confidence * valid).sum(dim=0) / length.clamp_min(1)
    eligible = length >= min_target_views
    score = (
        visibility_score_weight * length.float() + mean_confidence
    ).masked_fill(~eligible, 0.0)
    if sampling_strategy == "grid_balanced":
        selected, selected_score = grid_balanced_spatial_nms(
            score,
            radius_px=nms_radius_px,
            grid_size_px=sampling_grid_size_px,
            max_points=max_points,
        )
    elif sampling_strategy == "score":
        selected, selected_score = spatial_nms(
            score,
            radius_px=nms_radius_px,
            max_points=max_points,
        )
    else:
        raise ValueError(f"unsupported sampling strategy: {sampling_strategy!r}")
    if selected.numel() == 0:
        return None

    source_xy = first.source_coordinates.permute(1, 2, 0).reshape(-1, 2)[selected]
    target_xy = coordinates.permute(2, 3, 0, 1).reshape(
        -1, len(group.target_indices), 2
    )[selected]
    target_probability = confidence.permute(1, 2, 0).reshape(
        -1, len(group.target_indices)
    )[selected]
    selected_valid = valid.permute(1, 2, 0).reshape(
        -1, len(group.target_indices)
    )[selected]
    sample_count = selected.shape[0]
    tracks = MultiViewTracks(
        coordinates=torch.cat((source_xy[:, None], target_xy), dim=1),
        confidence=torch.cat(
            (torch.ones((sample_count, 1)), target_probability), dim=1
        ),
        valid=torch.cat(
            (torch.ones((sample_count, 1), dtype=torch.bool), selected_valid),
            dim=1,
        ),
        sampling_score=selected_score,
    )
    return GroupTrackCandidates(
        group_index=group_index,
        group=group,
        flat_source_indices=selected,
        tracks=tracks,
    )
