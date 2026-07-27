"""Typed records shared by robust association stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaskObservation:
    node_id: int
    view_index: int
    image_name: str
    local_id: int
    area: int
    core_ratio: float
    quality: float
    bbox: tuple[int, int, int, int]
    image_shape: tuple[int, int]
    gaussian_ids: np.ndarray
    gaussian_weights: np.ndarray
    appearance: np.ndarray
    centroid_3d: np.ndarray
    assignment_score: float = 0.0
    track_id: int = -1
    status: str = "unassigned"

    @property
    def evidence_mass(self) -> float:
        return float(self.gaussian_weights.sum())

    def summary(self) -> dict:
        return {
            "node_id": self.node_id,
            "view_index": self.view_index,
            "image_name": self.image_name,
            "local_id": self.local_id,
            "area": self.area,
            "core_ratio": self.core_ratio,
            "quality": self.quality,
            "bbox": list(self.bbox),
            "gaussian_count": int(self.gaussian_ids.size),
            "evidence_mass": self.evidence_mass,
            "centroid_3d": self.centroid_3d.tolist(),
            "assignment_score": self.assignment_score,
            "track_id": self.track_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class AssociationEdge:
    source: int
    target: int
    source_view: int
    target_view: int
    score: float
    weighted_jaccard: float
    bidirectional_coverage: float
    appearance: float
    spatial: float
    quality: float
    selected: bool = False


@dataclass
class InstanceTrack:
    internal_id: int
    node_ids: list[int]
    view_ids: list[int]
    quality: float
    mean_edge_score: float
    status: str
    global_id: int = 0
    gaussian_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    gaussian_weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    def summary(self) -> dict:
        return {
            "internal_id": self.internal_id,
            "global_id": self.global_id,
            "status": self.status,
            "node_ids": self.node_ids,
            "view_ids": self.view_ids,
            "support_views": len(set(self.view_ids)),
            "quality": self.quality,
            "mean_edge_score": self.mean_edge_score,
            "gaussian_count": int(self.gaussian_ids.size),
        }


@dataclass
class ViewProjection:
    view_index: int
    image_name: str
    height: int
    width: int
    gaussian_ids: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    depth: np.ndarray


@dataclass
class RefinedView:
    image_name: str
    labels: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    diagnostics: dict
