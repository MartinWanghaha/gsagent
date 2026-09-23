"""Optional MV-RoMa-aware PGSR neighbor ranking.

The existing pose graph remains the candidate gate.  This module only reranks
geometrically safe candidates with overlap and correspondence quality, so the
PGSR loss interface and legacy ``build_neighbor_map`` stay untouched.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import MVRoMaTrainingSettings
from .contracts import DirectedPair, PairQuality


def build_hybrid_neighbor_map(
    cameras: Sequence[Any],
    pose_neighbor_map: Mapping[str, Sequence[Any]],
    overlap_matrix: np.ndarray,
    pair_quality: Mapping[DirectedPair, PairQuality],
    settings: MVRoMaTrainingSettings,
) -> dict[str, list[Any]]:
    """Rerank pose-valid neighbors without admitting unsafe camera pairs."""

    cameras = list(cameras)
    overlap = np.asarray(overlap_matrix, dtype=np.float64)
    if overlap.shape != (len(cameras), len(cameras)):
        raise ValueError("overlap matrix does not match the training cameras")
    names = [str(getattr(camera, "image_name", index)) for index, camera in enumerate(cameras)]
    if len(set(names)) != len(names):
        raise ValueError("camera image_name values must be unique")
    index_by_name = {name: index for index, name in enumerate(names)}
    result: dict[str, list[Any]] = {}

    for source_index, source_name in enumerate(names):
        pose_candidates = list(pose_neighbor_map.get(source_name, ()))
        ranked = []
        for pose_rank, target_camera in enumerate(pose_candidates):
            target_name = str(getattr(target_camera, "image_name", ""))
            if target_name not in index_by_name:
                continue
            target_index = index_by_name[target_name]
            mutual_overlap = float(
                min(overlap[source_index, target_index], overlap[target_index, source_index])
            )
            forward = pair_quality.get(DirectedPair(source_index, target_index))
            reverse = pair_quality.get(DirectedPair(target_index, source_index))
            qualities = [quality.score for quality in (forward, reverse) if quality is not None]
            quality_score = float(sum(qualities) / len(qualities)) if qualities else 0.0
            if (
                mutual_overlap < settings.min_overlap
                or quality_score < settings.min_pair_quality
            ):
                continue
            pose_score = 1.0 - pose_rank / max(1, len(pose_candidates))
            score = (
                settings.overlap_weight * mutual_overlap
                + settings.pair_quality_weight * quality_score
                + settings.pose_rank_weight * pose_score
            )
            ranked.append((score, pose_rank, target_name, target_camera))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = [item[3] for item in ranked[: settings.max_neighbors]]
        if not selected and settings.fallback_to_pose:
            selected = pose_candidates[: settings.max_neighbors]
        result[source_name] = selected
    return result
