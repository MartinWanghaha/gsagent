"""Multi-view consensus and conservative raw-mask correction."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import slic

from .config import RobustAssociationConfig
from .observations import reliable_core_map
from .types import MaskObservation, RefinedView, ViewProjection


def build_gaussian_consensus(
    num_gaussians: int,
    observations: list[MaskObservation],
    *,
    margin_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate all confirmed observations into one probabilistic 3D label field."""
    active = [
        observation
        for observation in observations
        if observation.track_id > 0 and observation.gaussian_ids.size
    ]
    labels = np.zeros(num_gaussians, dtype=np.uint16)
    confidence = np.zeros(num_gaussians, dtype=np.float32)
    if not active:
        return labels, confidence
    maximum_label = max(observation.track_id for observation in active)
    ids = np.concatenate([observation.gaussian_ids for observation in active])
    instance_ids = np.concatenate(
        [
            np.full(
                observation.gaussian_ids.size,
                observation.track_id,
                dtype=np.int64,
            )
            for observation in active
        ]
    )
    votes = np.concatenate(
        [
            observation.gaussian_weights
            * observation.quality
            * max(observation.assignment_score, 0.25)
            for observation in active
        ]
    )
    composite = ids * (maximum_label + 1) + instance_ids
    order = np.argsort(composite, kind="stable")
    composite = composite[order]
    votes = votes[order]
    unique_composite, first = np.unique(composite, return_index=True)
    summed_votes = np.add.reduceat(votes, first)
    gaussian_ids = unique_composite // (maximum_label + 1)
    track_ids = unique_composite % (maximum_label + 1)

    ranking = np.lexsort((-summed_votes, gaussian_ids))
    gaussian_ids = gaussian_ids[ranking]
    track_ids = track_ids[ranking]
    summed_votes = summed_votes[ranking]
    group_start = np.concatenate(([0], np.flatnonzero(np.diff(gaussian_ids)) + 1))
    group_end = np.concatenate((group_start[1:], [gaussian_ids.size]))
    for start, end in zip(group_start, group_end):
        gaussian_id = int(gaussian_ids[start])
        best = float(summed_votes[start])
        second = float(summed_votes[start + 1]) if end - start > 1 else 0.0
        margin = (best - second) / max(best, 1e-8)
        if margin >= margin_threshold:
            labels[gaussian_id] = int(track_ids[start])
            confidence[gaussian_id] = float(np.clip(margin, 0.0, 1.0))
    return labels, confidence


def _visible_seeds(
    projection: ViewProjection,
    gaussian_labels: np.ndarray,
    gaussian_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = gaussian_labels[projection.gaussian_ids]
    confidence = gaussian_confidence[projection.gaussian_ids]
    valid = labels > 0
    seed_labels = np.zeros((projection.height, projection.width), dtype=np.uint16)
    seed_confidence = np.zeros(
        (projection.height, projection.width),
        dtype=np.float32,
    )
    if not valid.any():
        return seed_labels, seed_confidence
    flat = projection.pixel_y[valid].astype(
        np.int64
    ) * projection.width + projection.pixel_x[valid].astype(np.int64)
    depth = projection.depth[valid]
    order = np.lexsort((depth, flat))
    flat = flat[order]
    first = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1))
    chosen = order[first]
    chosen_flat = flat[first]
    seed_labels.reshape(-1)[chosen_flat] = labels[valid][chosen]
    seed_confidence.reshape(-1)[chosen_flat] = confidence[valid][chosen]
    return seed_labels, seed_confidence


def _build_superpixels(
    image_bgr: np.ndarray,
    config: RobustAssociationConfig,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, config.superpixel_max_edge / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        working = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    else:
        working = image_bgr
    requested = max(1, round(height * width / config.superpixel_size**2))
    requested = min(requested, max(1, working.shape[0] * working.shape[1] // 4))
    segments = slic(
        cv2.cvtColor(working, cv2.COLOR_BGR2RGB),
        n_segments=requested,
        compactness=config.superpixel_compactness,
        sigma=1.0,
        start_label=0,
        enforce_connectivity=True,
        channel_axis=-1,
    ).astype(np.int32, copy=False)
    if segments.shape != (height, width):
        segments = cv2.resize(
            segments,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return segments


def _split_mask_with_superpixels(
    raw_region: np.ndarray,
    superpixels: np.ndarray,
    seed_labels: np.ndarray,
    seed_confidence: np.ndarray,
    owner_label: int,
    allowed_labels: np.ndarray,
    config: RobustAssociationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(raw_region.shape, dtype=np.uint16)
    result[raw_region] = owner_label
    confidence = np.zeros(raw_region.shape, dtype=np.float32)
    valid = raw_region & np.isin(seed_labels, allowed_labels)
    if not valid.any():
        return result, confidence

    segment_count = int(superpixels.max()) + 1
    vote_mass = np.zeros((allowed_labels.size, segment_count), dtype=np.float32)
    vote_count = np.zeros((allowed_labels.size, segment_count), dtype=np.int32)
    for index, label in enumerate(allowed_labels):
        label_seeds = valid & (seed_labels == label)
        if not label_seeds.any():
            continue
        segment_ids = superpixels[label_seeds]
        vote_mass[index] = np.bincount(
            segment_ids,
            weights=np.maximum(seed_confidence[label_seeds], 1e-3),
            minlength=segment_count,
        )
        vote_count[index] = np.bincount(
            segment_ids,
            minlength=segment_count,
        )
    total_mass = vote_mass.sum(axis=0)
    winner_index = vote_mass.argmax(axis=0)
    segment_ids = np.arange(segment_count)
    winner_mass = vote_mass[winner_index, segment_ids]
    winner_count = vote_count[winner_index, segment_ids]
    winner_labels = allowed_labels[winner_index]
    purity = winner_mass / np.maximum(total_mass, 1e-8)
    reassign = (
        (winner_labels != owner_label)
        & (winner_count >= config.split_min_seed_points)
        & (purity >= config.split_seed_purity)
    )
    pixel_reassign = raw_region & reassign[superpixels]
    result[pixel_reassign] = winner_labels[superpixels[pixel_reassign]]
    confidence[pixel_reassign] = purity[superpixels[pixel_reassign]]

    minimum_area = max(
        config.split_min_area_pixels,
        int(np.ceil(raw_region.sum() * config.split_min_area_fraction)),
    )
    for label in allowed_labels:
        if label == owner_label:
            continue
        candidate = (result == label).astype(np.uint8)
        component_count, components, statistics, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
        for component in range(1, component_count):
            if statistics[component, cv2.CC_STAT_AREA] < minimum_area:
                rejected = components == component
                result[rejected] = owner_label
                confidence[rejected] = 0.0
    return result, confidence


def refine_view(
    raw_labels: np.ndarray,
    image_bgr: np.ndarray,
    projection: ViewProjection,
    observations: list[MaskObservation],
    gaussian_labels: np.ndarray,
    gaussian_confidence: np.ndarray,
    *,
    config: RobustAssociationConfig,
) -> RefinedView:
    if raw_labels.shape != (projection.height, projection.width):
        raise ValueError("Raw mask and projection resolution differ")
    if image_bgr.shape[:2] != raw_labels.shape:
        raise ValueError("RGB image and raw mask resolution differ")
    output = np.zeros(raw_labels.shape, dtype=np.uint16)
    confidence = np.full(raw_labels.shape, 0.5, dtype=np.float32)
    core = reliable_core_map(raw_labels.copy(), config.core_radius)
    seed_labels, seed_confidence = _visible_seeds(
        projection,
        gaussian_labels,
        gaussian_confidence,
    )
    observation_by_local = {
        observation.local_id: observation for observation in observations
    }
    projected_raw = raw_labels[projection.pixel_y, projection.pixel_x]
    projected_global = gaussian_labels[projection.gaussian_ids]
    projected_confidence = gaussian_confidence[projection.gaussian_ids]
    split_count = 0
    uncertain_count = 0
    superpixels = None

    for local_id in np.unique(raw_labels):
        local_id = int(local_id)
        if local_id == 0:
            continue
        region = raw_labels == local_id
        observation = observation_by_local.get(local_id)
        if observation is None or observation.track_id <= 0:
            uncertain_count += 1
            output[region] = config.ignore_label
            confidence[region] = 0.0
            continue

        projected = projected_raw == local_id
        candidate_labels = projected_global[projected]
        candidate_confidence = projected_confidence[projected]
        valid = candidate_labels > 0
        significant = np.empty(0, dtype=np.uint16)
        if valid.any():
            mass = np.bincount(
                candidate_labels[valid].astype(np.int64),
                weights=np.maximum(candidate_confidence[valid], 1e-3),
            )
            total = float(mass.sum())
            significant = np.flatnonzero(
                mass >= config.split_fraction * max(total, 1e-8)
            ).astype(np.uint16)
            significant = significant[significant > 0]

        can_split = (
            significant.size >= 2
            and observation.track_id in significant
            and np.any(significant != observation.track_id)
        )
        if can_split:
            if superpixels is None:
                superpixels = _build_superpixels(image_bgr, config)
            x0, y0, x1, y1 = observation.bbox
            crop_region = region[y0:y1, x0:x1]
            split_labels, split_confidence = _split_mask_with_superpixels(
                crop_region,
                superpixels[y0:y1, x0:x1],
                seed_labels[y0:y1, x0:x1],
                seed_confidence[y0:y1, x0:x1],
                observation.track_id,
                significant,
                config,
            )
            output_crop = output[y0:y1, x0:x1]
            confidence_crop = confidence[y0:y1, x0:x1]
            base = observation.quality * max(observation.assignment_score, 0.25)
            output_crop[crop_region] = split_labels[crop_region]
            confidence_crop[crop_region] = np.where(
                split_confidence[crop_region] > 0,
                split_confidence[crop_region],
                base
                * np.where(
                    core[y0:y1, x0:x1][crop_region],
                    1.0,
                    config.boundary_weight,
                ),
            )
            if np.any(split_labels[crop_region] != observation.track_id):
                split_count += 1
        else:
            output[region] = observation.track_id
            base = observation.quality * max(observation.assignment_score, 0.25)
            confidence[region] = base * np.where(
                core[region],
                1.0,
                config.boundary_weight,
            )

    confidence = np.clip(confidence, 0.0, 1.0)
    valid = output != config.ignore_label
    return RefinedView(
        image_name=projection.image_name,
        labels=output,
        confidence=confidence,
        valid=valid,
        diagnostics={
            "raw_instances": int(raw_labels.max()),
            "split_masks": split_count,
            "uncertain_masks": uncertain_count,
            "ignore_pixels": int((~valid).sum()),
        },
    )
