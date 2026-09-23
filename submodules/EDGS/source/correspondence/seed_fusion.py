"""Budgeted triangulation and deterministic fusion of MV-RoMa seed points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .config import DedupSettings, SeedBudgetSettings
from .contracts import MultiViewTracks, normalized_to_pixel
from .geometry import weighted_multiview_dlt
from .postprocess import GroupTrackCandidates


@dataclass(frozen=True)
class GaussianSeedBatch:
    xyz: torch.Tensor
    rgb: torch.Tensor
    distance_to_source: torch.Tensor
    reprojection_error: torch.Tensor
    triangulation_angle_deg: torch.Tensor
    sampling_score: torch.Tensor
    source_index: torch.Tensor
    group_index: torch.Tensor
    source_pixel_index: torch.Tensor

    def __post_init__(self) -> None:
        count = self.xyz.shape[0]
        if self.xyz.shape != (count, 3) or self.rgb.shape != (count, 3):
            raise ValueError("seed xyz/rgb must have shape [N,3]")
        one_dimensional = (
            self.distance_to_source,
            self.reprojection_error,
            self.triangulation_angle_deg,
            self.sampling_score,
            self.source_index,
            self.group_index,
            self.source_pixel_index,
        )
        if any(value.shape != (count,) for value in one_dimensional):
            raise ValueError("seed metadata must have shape [N]")

    def take(self, indices: torch.Tensor | np.ndarray | Sequence[int]) -> "GaussianSeedBatch":
        index = torch.as_tensor(indices, dtype=torch.long)
        return GaussianSeedBatch(
            xyz=self.xyz[index],
            rgb=self.rgb[index],
            distance_to_source=self.distance_to_source[index],
            reprojection_error=self.reprojection_error[index],
            triangulation_angle_deg=self.triangulation_angle_deg[index],
            sampling_score=self.sampling_score[index],
            source_index=self.source_index[index],
            group_index=self.group_index[index],
            source_pixel_index=self.source_pixel_index[index],
        )


def _concatenate(batches: Sequence[GaussianSeedBatch]) -> GaussianSeedBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty seed sequence")
    fields = GaussianSeedBatch.__dataclass_fields__
    return GaussianSeedBatch(
        **{
            name: torch.cat([getattr(batch, name) for batch in batches])
            for name in fields
        }
    )


def _sample_source_colors(camera: Any, normalized_xy: torch.Tensor) -> torch.Tensor:
    image = getattr(camera, "original_image", None)
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("camera.original_image must have shape [C,H,W]")
    height, width = int(image.shape[1]), int(image.shape[2])
    pixels = normalized_to_pixel(normalized_xy, height, width).round().long()
    x = pixels[:, 0].clamp(0, width - 1)
    y = pixels[:, 1].clamp(0, height - 1)
    return image[:3, y, x].transpose(0, 1).to(
        device=normalized_xy.device, dtype=torch.float32
    )


def _slice_tracks(tracks: MultiViewTracks, indices: torch.Tensor) -> MultiViewTracks:
    return MultiViewTracks(
        coordinates=tracks.coordinates[indices],
        confidence=tracks.confidence[indices],
        valid=tracks.valid[indices],
        sampling_score=tracks.sampling_score[indices],
    )


def _triangulate_group_subset(
    candidates: GroupTrackCandidates,
    local_indices: torch.Tensor,
    cameras: Sequence[Any],
    *,
    device: torch.device,
    min_target_views: int,
    max_reprojection_error: float,
    min_triangulation_angle_deg: float,
    reject_outliers: bool,
) -> GaussianSeedBatch | None:
    tracks_cpu = _slice_tracks(candidates.tracks, local_indices)
    tracks = MultiViewTracks(
        coordinates=tracks_cpu.coordinates.to(device),
        confidence=tracks_cpu.confidence.to(device),
        valid=tracks_cpu.valid.to(device),
        sampling_score=tracks_cpu.sampling_score.to(device),
    )
    camera_indices = candidates.group.all_indices
    projections = torch.stack(
        [cameras[index].full_proj_transform for index in camera_indices]
    ).to(device=device, dtype=tracks.coordinates.dtype)
    centers = torch.stack(
        [cameras[index].camera_center for index in camera_indices]
    ).to(device=device, dtype=tracks.coordinates.dtype)
    triangulated = weighted_multiview_dlt(
        projections,
        tracks,
        camera_centers=centers,
        min_views=min_target_views + 1,
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        reject_outliers=reject_outliers,
    )
    accepted = triangulated.accepted
    if not bool(accepted.any().item()):
        return None
    xyz = triangulated.points[accepted]
    source_xy = tracks.coordinates[accepted, 0]
    rgb = _sample_source_colors(cameras[candidates.group.source_index], source_xy)
    distance = torch.linalg.vector_norm(xyz - centers[0], dim=1)
    maximum_error = triangulated.reprojection_error.masked_fill(
        ~triangulated.valid_observations, float("-inf")
    ).amax(dim=1)[accepted]
    accepted_local = local_indices[accepted.cpu()]
    count = int(accepted.sum().item())
    return GaussianSeedBatch(
        xyz=xyz.detach().cpu(),
        rgb=rgb.detach().cpu(),
        distance_to_source=distance.detach().cpu(),
        reprojection_error=maximum_error.detach().cpu(),
        triangulation_angle_deg=triangulated.triangulation_angle_deg[
            accepted
        ].detach().cpu(),
        sampling_score=tracks.sampling_score[accepted].detach().cpu(),
        source_index=torch.full(
            (count,), candidates.group.source_index, dtype=torch.int64
        ),
        group_index=torch.full((count,), candidates.group_index, dtype=torch.int64),
        source_pixel_index=candidates.flat_source_indices[accepted_local].to(
            dtype=torch.int64
        ),
    )


def _best_per_source_pixel(
    batch: GaussianSeedBatch,
    *,
    limit: int,
) -> GaussianSeedBatch:
    score = batch.sampling_score.numpy()
    error = batch.reprojection_error.numpy()
    angle = batch.triangulation_angle_deg.numpy()
    group = batch.group_index.numpy()
    pixel = batch.source_pixel_index.numpy()
    # Pixel is primary; the remaining keys choose its best successful group.
    order = np.lexsort((group, -angle, error, -score, pixel))
    ordered_pixel = pixel[order]
    first = np.ones(len(order), dtype=np.bool_)
    first[1:] = ordered_pixel[1:] != ordered_pixel[:-1]
    selected = order[first]
    # The source budget is quality ordered after per-pixel fallback resolution.
    quality_order = np.lexsort(
        (
            batch.group_index[selected].numpy(),
            batch.source_pixel_index[selected].numpy(),
            -batch.triangulation_angle_deg[selected].numpy(),
            batch.reprojection_error[selected].numpy(),
            -batch.sampling_score[selected].numpy(),
        )
    )
    return batch.take(selected[quality_order[:limit]])


def triangulate_source_candidates(
    candidate_groups: Sequence[GroupTrackCandidates],
    cameras: Sequence[Any],
    settings: SeedBudgetSettings,
    *,
    device: torch.device | str,
    min_target_views: int,
    max_reprojection_error: float,
    min_triangulation_angle_deg: float,
    reject_outliers: bool,
    triangulation_batch_size: int,
) -> GaussianSeedBatch | None:
    """Triangulate score-ordered proposals until the source budget is filled."""

    if not candidate_groups:
        return None
    proposal_group: list[int] = []
    proposal_local: list[int] = []
    proposal_score: list[float] = []
    proposal_pixel: list[int] = []
    for group_position, candidates in enumerate(candidate_groups):
        count = candidates.tracks.coordinates.shape[0]
        proposal_group.extend([group_position] * count)
        proposal_local.extend(range(count))
        proposal_score.extend(candidates.tracks.sampling_score.tolist())
        proposal_pixel.extend(candidates.flat_source_indices.tolist())
    score = np.asarray(proposal_score, dtype=np.float32)
    pixel = np.asarray(proposal_pixel, dtype=np.int64)
    group = np.asarray(proposal_group, dtype=np.int64)
    local = np.asarray(proposal_local, dtype=np.int64)
    order = np.lexsort((local, group, pixel, -score))

    accepted_batches: list[GaussianSeedBatch] = []
    accepted_pixels: set[int] = set()
    cursor = 0
    torch_device = torch.device(device)
    while cursor < len(order) and len(accepted_pixels) < settings.per_source:
        remaining = settings.per_source - len(accepted_pixels)
        take = max(settings.proposal_batch_size, 4 * remaining)
        proposal_indices = order[cursor : cursor + take]
        cursor += len(proposal_indices)
        for group_position in np.unique(group[proposal_indices]):
            mask = group[proposal_indices] == group_position
            local_indices = torch.from_numpy(local[proposal_indices][mask]).long()
            for local_batch in local_indices.split(triangulation_batch_size):
                batch = _triangulate_group_subset(
                    candidate_groups[int(group_position)],
                    local_batch,
                    cameras,
                    device=torch_device,
                    min_target_views=min_target_views,
                    max_reprojection_error=max_reprojection_error,
                    min_triangulation_angle_deg=min_triangulation_angle_deg,
                    reject_outliers=reject_outliers,
                )
                if batch is not None:
                    accepted_batches.append(batch)
                    accepted_pixels.update(batch.source_pixel_index.tolist())
    if not accepted_batches:
        return None
    return _best_per_source_pixel(
        _concatenate(accepted_batches), limit=settings.per_source
    )


def deduplicate_seed_batches(
    batches: Sequence[GaussianSeedBatch],
    settings: DedupSettings,
    *,
    scaling_factor: float,
    total_limit: int | None,
) -> tuple[GaussianSeedBatch, float | None]:
    """Fuse duplicate 3D seeds while preserving the best geometric sample."""

    combined = _concatenate(batches)
    if not settings.enabled:
        result = combined
        voxel_size = None
    else:
        voxel_size = settings.voxel_size
        if voxel_size is None:
            initial_scale = combined.distance_to_source * float(scaling_factor)
            voxel_size = float(initial_scale.median().item()) * settings.voxel_scale
        voxel_size = max(voxel_size, np.finfo(np.float32).eps)
        voxel = torch.floor(combined.xyz / voxel_size).to(torch.int64).numpy()
        score = combined.sampling_score.numpy()
        error = combined.reprojection_error.numpy()
        angle = combined.triangulation_angle_deg.numpy()
        source = combined.source_index.numpy()
        group = combined.group_index.numpy()
        pixel = combined.source_pixel_index.numpy()
        order = np.lexsort(
            (pixel, group, source, -angle, error, -score, voxel[:, 2], voxel[:, 1], voxel[:, 0])
        )
        ordered_voxel = voxel[order]
        first = np.ones(len(order), dtype=np.bool_)
        first[1:] = np.any(ordered_voxel[1:] != ordered_voxel[:-1], axis=1)
        result = combined.take(order[first])
    if total_limit is not None and result.xyz.shape[0] > total_limit:
        order = np.lexsort(
            (
                result.group_index.numpy(),
                result.source_pixel_index.numpy(),
                result.reprojection_error.numpy(),
                -result.sampling_score.numpy(),
            )
        )
        result = result.take(order[:total_limit])
    return result, voxel_size
