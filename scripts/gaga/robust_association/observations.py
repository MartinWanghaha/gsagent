"""Build noise-aware mask observations and sparse Gaussian evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import RobustAssociationConfig
from .types import MaskObservation, ViewProjection


def reliable_core_map(labels: np.ndarray, radius: int) -> np.ndarray:
    """Return pixels whose local neighborhood keeps the same non-zero label."""
    core = labels > 0
    for _ in range(radius):
        padded = np.pad(labels, 1, mode="edge")
        same = (
            (labels == padded[:-2, 1:-1])
            & (labels == padded[2:, 1:-1])
            & (labels == padded[1:-1, :-2])
            & (labels == padded[1:-1, 2:])
            & (labels == padded[:-2, :-2])
            & (labels == padded[:-2, 2:])
            & (labels == padded[2:, :-2])
            & (labels == padded[2:, 2:])
        )
        core &= same
        labels = np.where(core, labels, 0)
    return core


def _label_statistics(
    labels: np.ndarray,
    core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, int, int]]]:
    maximum = int(labels.max()) if labels.size else 0
    flat = labels.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(flat, minlength=maximum + 1)
    core_counts = np.bincount(
        flat[core.reshape(-1)],
        minlength=maximum + 1,
    )
    height, width = labels.shape
    ys, xs = np.nonzero(labels)
    values = labels[ys, xs].astype(np.int64, copy=False)
    min_x = np.full(maximum + 1, width, dtype=np.int64)
    min_y = np.full(maximum + 1, height, dtype=np.int64)
    max_x = np.full(maximum + 1, -1, dtype=np.int64)
    max_y = np.full(maximum + 1, -1, dtype=np.int64)
    if values.size:
        np.minimum.at(min_x, values, xs)
        np.minimum.at(min_y, values, ys)
        np.maximum.at(max_x, values, xs)
        np.maximum.at(max_y, values, ys)
    boxes = {
        local_id: (
            int(min_x[local_id]),
            int(min_y[local_id]),
            int(max_x[local_id]) + 1,
            int(max_y[local_id]) + 1,
        )
        for local_id in range(1, maximum + 1)
        if counts[local_id] > 0
    }
    return counts, core_counts, boxes


def appearance_descriptors(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    *,
    max_edge: int = 512,
    hue_bins: int = 12,
) -> dict[int, np.ndarray]:
    """Compute compact color descriptors for all instances in one pass."""
    height, width = labels.shape
    scale = min(1.0, max_edge / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(
            labels.astype(np.int32),
            size,
            interpolation=cv2.INTER_NEAREST,
        )
    maximum = int(labels.max()) if labels.size else 0
    flat_labels = labels.reshape(-1).astype(np.int64, copy=False)
    valid = flat_labels > 0
    if not valid.any():
        return {}
    counts = np.bincount(flat_labels[valid], minlength=maximum + 1).astype(np.float64)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    channels = []
    for channel in range(3):
        values = lab[valid, channel]
        sums = np.bincount(
            flat_labels[valid],
            weights=values,
            minlength=maximum + 1,
        )
        squares = np.bincount(
            flat_labels[valid],
            weights=values * values,
            minlength=maximum + 1,
        )
        mean = sums / np.maximum(counts, 1.0)
        std = np.sqrt(np.maximum(squares / np.maximum(counts, 1.0) - mean * mean, 0))
        channels.extend((mean / 255.0, std / 128.0))

    hue = np.minimum(
        (hsv[valid, 0].astype(np.int64) * hue_bins) // 180,
        hue_bins - 1,
    )
    combined = flat_labels[valid] * hue_bins + hue
    histogram = np.bincount(
        combined,
        minlength=(maximum + 1) * hue_bins,
    ).reshape(maximum + 1, hue_bins)
    histogram = histogram / np.maximum(counts[:, None], 1.0)

    descriptors = {}
    base = np.stack(channels, axis=1)
    for local_id in range(1, maximum + 1):
        if counts[local_id] == 0:
            continue
        descriptor = np.concatenate((base[local_id], histogram[local_id])).astype(
            np.float32
        )
        norm = float(np.linalg.norm(descriptor))
        descriptors[local_id] = descriptor / max(norm, 1e-8)
    return descriptors


def resolve_image(scene_dir: Path, images: str, image_name: str) -> Path:
    directory = scene_dir / images
    candidates = [directory / image_name]
    candidates.extend(
        directory / f"{Path(image_name).stem}{suffix}"
        for suffix in (
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve RGB image for {image_name!r}")


def project_view(projector, viewpoint, view_index: int) -> ViewProjection:
    projected = projector.project_gaussian(viewpoint)
    flattened = projected["p_proj_flatten"].reshape(-1).detach().cpu().numpy()
    gaussian_ids = projected["p_proj_inside_indices"].reshape(-1).detach().cpu().numpy()
    depth_all = projected["p_hom_z"].detach().cpu().numpy()
    pixel_x = flattened // projector.image_height
    pixel_y = flattened % projector.image_height
    return ViewProjection(
        view_index=view_index,
        image_name=viewpoint.image_name,
        height=projector.image_height,
        width=projector.image_width,
        gaussian_ids=gaussian_ids.astype(np.int64, copy=False),
        pixel_x=pixel_x.astype(np.int32, copy=False),
        pixel_y=pixel_y.astype(np.int32, copy=False),
        depth=depth_all[gaussian_ids].astype(np.float32, copy=False),
    )


def _aggregate_selected(
    projection: ViewProjection,
    labels: np.ndarray,
    core: np.ndarray,
    config: RobustAssociationConfig,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    local_labels = labels[projection.pixel_y, projection.pixel_x].astype(
        np.int64,
        copy=False,
    )
    valid = local_labels > 0
    if not valid.any():
        return {}
    pixel_x = projection.pixel_x[valid]
    pixel_y = projection.pixel_y[valid]
    local_labels = local_labels[valid]
    gaussian_ids = projection.gaussian_ids[valid]
    depth = projection.depth[valid]
    patch_width = math.ceil(projection.width / config.num_patches)
    patch_height = math.ceil(projection.height / config.num_patches)
    patch_x = np.minimum(pixel_x // patch_width, config.num_patches - 1)
    patch_y = np.minimum(pixel_y // patch_height, config.num_patches - 1)
    patch_ids = patch_y * config.num_patches + patch_x
    group_keys = local_labels * (config.num_patches**2) + patch_ids
    order = np.lexsort((depth, group_keys))
    ordered_keys = group_keys[order]
    boundaries = np.flatnonzero(np.diff(ordered_keys)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [order.size]))
    selected_ids: dict[int, list[np.ndarray]] = defaultdict(list)
    selected_weights: dict[int, list[np.ndarray]] = defaultdict(list)
    for start, end in zip(starts, ends):
        group_order = order[start:end]
        count = max(int(config.front_percentage * group_order.size), 1)
        chosen = group_order[:count]
        local_id = int(local_labels[chosen[0]])
        rank = np.arange(count, dtype=np.float32)
        rank /= max(count - 1, 1)
        depth_weight = np.exp(-config.depth_decay * rank)
        pixel_weight = np.where(
            core[pixel_y[chosen], pixel_x[chosen]],
            1.0,
            config.boundary_weight,
        ).astype(np.float32)
        selected_ids[local_id].append(gaussian_ids[chosen])
        selected_weights[local_id].append(depth_weight * pixel_weight)

    result = {}
    for local_id, id_parts in selected_ids.items():
        ids = np.concatenate(id_parts).astype(np.int64, copy=False)
        weights = np.concatenate(selected_weights[local_id]).astype(
            np.float32,
            copy=False,
        )
        sort_order = np.argsort(ids, kind="stable")
        ids = ids[sort_order]
        weights = weights[sort_order]
        unique_ids, first = np.unique(ids, return_index=True)
        maximum_weights = np.maximum.reduceat(weights, first)
        result[local_id] = (unique_ids, maximum_weights)
    return result


def build_view_observations(
    projector,
    viewpoint,
    *,
    view_index: int,
    node_offset: int,
    scene_dir: Path,
    images: str,
    raw_mask_dir: Path,
    gaussian_xyz: np.ndarray,
    config: RobustAssociationConfig,
) -> tuple[list[MaskObservation], ViewProjection]:
    mask_path = raw_mask_dir / f"{viewpoint.image_name}.png"
    labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.ndim != 2:
        raise RuntimeError(f"Could not decode label mask: {mask_path}")
    labels = labels.astype(np.int32, copy=False)
    if labels.shape != (projector.image_height, projector.image_width):
        raise ValueError(
            f"Mask {mask_path} has shape {labels.shape}, expected "
            f"{(projector.image_height, projector.image_width)}"
        )
    core = reliable_core_map(labels.copy(), config.core_radius)
    counts, core_counts, boxes = _label_statistics(labels, core)
    image_path = resolve_image(scene_dir, images, viewpoint.image_name)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode RGB image: {image_path}")
    appearances = appearance_descriptors(image, labels)
    projection = project_view(projector, viewpoint, view_index)
    evidence = _aggregate_selected(projection, labels, core, config)

    observations = []
    for local_id in range(1, len(counts)):
        area = int(counts[local_id])
        if area == 0:
            continue
        ids, weights = evidence.get(
            local_id,
            (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            ),
        )
        core_ratio = float(core_counts[local_id] / max(area, 1))
        area_score = 1.0 - math.exp(-area / 512.0)
        support_score = 1.0 - math.exp(-ids.size / 32.0)
        quality = float(
            np.clip(
                0.45 * core_ratio + 0.20 * area_score + 0.35 * support_score,
                0.05,
                1.0,
            )
        )
        if ids.size:
            centroid = np.average(
                gaussian_xyz[ids],
                axis=0,
                weights=np.maximum(weights, 1e-6),
            ).astype(np.float32)
        else:
            centroid = np.full(3, np.nan, dtype=np.float32)
        appearance = appearances.get(
            local_id,
            np.zeros(18, dtype=np.float32),
        )
        observations.append(
            MaskObservation(
                node_id=node_offset + len(observations),
                view_index=view_index,
                image_name=viewpoint.image_name,
                local_id=local_id,
                area=area,
                core_ratio=core_ratio,
                quality=quality,
                bbox=boxes[local_id],
                image_shape=labels.shape,
                gaussian_ids=ids,
                gaussian_weights=weights,
                appearance=appearance,
                centroid_3d=centroid,
            )
        )
    return observations, projection
