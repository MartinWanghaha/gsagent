"""Core data containers for semantic surface sampling and triangle meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def _as_float_array(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    return np.ascontiguousarray(array)


@dataclass
class SurfaceSamples:
    """A NumPy representation of ``SemanticSurfaceField.query`` results."""

    points: np.ndarray
    occupancy: np.ndarray
    sdf: np.ndarray
    normal: np.ndarray
    semantic: np.ndarray
    geometry_posterior: np.ndarray
    uncertainty: np.ndarray

    def __post_init__(self) -> None:
        self.points = _as_float_array(self.points, 2, "points")
        self.occupancy = _as_float_array(self.occupancy, 1, "occupancy")
        self.sdf = _as_float_array(self.sdf, 1, "sdf")
        self.normal = _as_float_array(self.normal, 2, "normal")
        self.semantic = _as_float_array(self.semantic, 2, "semantic")
        self.geometry_posterior = _as_float_array(
            self.geometry_posterior, 2, "geometry_posterior"
        )
        self.uncertainty = _as_float_array(self.uncertainty, 1, "uncertainty")

        count = len(self.points)
        for name in (
            "occupancy",
            "sdf",
            "normal",
            "semantic",
            "geometry_posterior",
            "uncertainty",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} length does not match points")
        if self.points.shape[1] != 3 or self.normal.shape[1] != 3:
            raise ValueError("points and normal must have shape [N, 3]")
        if self.geometry_posterior.shape[1] != 5:
            raise ValueError("geometry_posterior must have shape [N, 5]")
        if self.semantic.shape[1] < 1:
            raise ValueError("semantic must have at least one channel")

    def __len__(self) -> int:
        return len(self.points)

    def take(self, indices: np.ndarray) -> "SurfaceSamples":
        indices = np.asarray(indices)
        return SurfaceSamples(
            points=self.points[indices],
            occupancy=self.occupancy[indices],
            sdf=self.sdf[indices],
            normal=self.normal[indices],
            semantic=self.semantic[indices],
            geometry_posterior=self.geometry_posterior[indices],
            uncertainty=self.uncertainty[indices],
        )

    @classmethod
    def concatenate(cls, samples: list["SurfaceSamples"]) -> "SurfaceSamples":
        if not samples:
            raise ValueError("at least one SurfaceSamples object is required")
        return cls(
            **{
                name: np.concatenate([getattr(item, name) for item in samples], axis=0)
                for name in (
                    "points",
                    "occupancy",
                    "sdf",
                    "normal",
                    "semantic",
                    "geometry_posterior",
                    "uncertainty",
                )
            }
        )


@dataclass
class RegionOwnershipSamples:
    """Memory-bounded soft region ownership at a set of mesh vertices."""

    requested_region_ids: np.ndarray
    region_id: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        requested = np.asarray(self.requested_region_ids)
        if requested.ndim != 1 or requested.dtype.kind not in "iu":
            raise ValueError("requested_region_ids must be a one-dimensional integer array")
        self.requested_region_ids = np.ascontiguousarray(requested, dtype=np.int64)
        if not len(self.requested_region_ids) or np.any(self.requested_region_ids <= 0):
            raise ValueError("requested_region_ids must contain foreground IDs")
        if np.any(np.diff(self.requested_region_ids) <= 0):
            raise ValueError("requested_region_ids must be sorted and unique")

        region_id = np.asarray(self.region_id)
        if region_id.ndim != 1 or region_id.dtype.kind not in "iu":
            raise ValueError("region_id must be a one-dimensional integer array")
        self.region_id = np.ascontiguousarray(region_id, dtype=np.int64)
        self.confidence = np.ascontiguousarray(
            np.asarray(self.confidence, dtype=np.float32)
        )
        self.valid = np.ascontiguousarray(np.asarray(self.valid, dtype=bool))
        if self.confidence.shape != self.region_id.shape or self.valid.shape != self.region_id.shape:
            raise ValueError("region ownership arrays must share shape [P]")
        if not np.all(np.isfinite(self.confidence)):
            raise ValueError("region ownership confidence must be finite")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("region ownership confidence must lie in [0,1]")
        if np.any(self.region_id[~self.valid] != -1) or np.any(
            self.confidence[~self.valid] != 0.0
        ):
            raise ValueError("invalid ownership rows must use ID -1 and zero confidence")
        if np.any(self.region_id[self.valid] <= 0) or np.any(
            ~np.isin(self.region_id[self.valid], self.requested_region_ids)
        ):
            raise ValueError("valid ownership IDs must belong to requested_region_ids")

    def __len__(self) -> int:
        return len(self.region_id)


@dataclass
class TriangleMesh:
    """Triangle mesh with per-vertex attributes and optional face ownership.

    ``face_region_id`` is the topology-level ownership contract used by the
    region-conditioned exporter.  Non-negative values identify decoder
    regions, ``-1`` identifies the coverage-conserving residual region, and
    ``-2`` marks a face whose duplicate chart owners disagree.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None
    semantic_id: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    face_region_id: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = _as_float_array(self.vertices, 2, "vertices")
        self.faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64))
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape [F, 3]")
        if self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [V, 3]")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must be finite")
        if self.faces.size and (
            self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("face index is outside the vertex array")

        count = len(self.vertices)
        if self.normals is not None:
            self.normals = _as_float_array(self.normals, 2, "normals")
            if self.normals.shape != self.vertices.shape:
                raise ValueError("normals must have shape [V, 3]")
            if not np.all(np.isfinite(self.normals)):
                raise ValueError("normals must be finite")
        if self.semantic is not None:
            self.semantic = _as_float_array(self.semantic, 2, "semantic")
            if len(self.semantic) != count:
                raise ValueError("semantic must have shape [V, D]")
            if not np.all(np.isfinite(self.semantic)):
                raise ValueError("semantic must be finite")
        if self.semantic_id is not None:
            self.semantic_id = np.ascontiguousarray(
                np.asarray(self.semantic_id, dtype=np.int32).reshape(-1)
            )
            if len(self.semantic_id) != count:
                raise ValueError("semantic_id must have shape [V]")
        if self.uncertainty is not None:
            self.uncertainty = np.ascontiguousarray(
                np.asarray(self.uncertainty, dtype=np.float32).reshape(-1)
            )
            if len(self.uncertainty) != count:
                raise ValueError("uncertainty must have shape [V]")
            if not np.all(np.isfinite(self.uncertainty)):
                raise ValueError("uncertainty must be finite")
        if self.face_region_id is not None:
            raw_face_region_id = np.asarray(self.face_region_id)
            if raw_face_region_id.dtype.kind not in "iu":
                raise ValueError("face_region_id must be an integer array")
            face_region_id = np.asarray(
                raw_face_region_id, dtype=np.int64
            ).reshape(-1)
            if len(face_region_id) != len(self.faces):
                raise ValueError("face_region_id must have shape [F]")
            if np.any(face_region_id < -2) or np.any(
                face_region_id > np.iinfo(np.int32).max
            ):
                raise ValueError(
                    "face_region_id values must be -2, -1, or non-negative int32"
                )
            self.face_region_id = np.ascontiguousarray(
                face_region_id, dtype=np.int32
            )

    @classmethod
    def empty(cls, semantic_dim: int = 1) -> "TriangleMesh":
        return cls(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            normals=np.empty((0, 3), dtype=np.float32),
            semantic=np.empty((0, semantic_dim), dtype=np.float32),
            semantic_id=None,
            uncertainty=np.empty((0,), dtype=np.float32),
            face_region_id=None,
        )

    def copy(self) -> "TriangleMesh":
        return TriangleMesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=None if self.normals is None else self.normals.copy(),
            semantic=None if self.semantic is None else self.semantic.copy(),
            semantic_id=None if self.semantic_id is None else self.semantic_id.copy(),
            uncertainty=None if self.uncertainty is None else self.uncertainty.copy(),
            face_region_id=None
            if self.face_region_id is None
            else self.face_region_id.copy(),
            metadata=dict(self.metadata),
        )

    def compact(self) -> "TriangleMesh":
        """Drop unreferenced vertices while retaining vertex and face attributes."""
        used, inverse = np.unique(self.faces.reshape(-1), return_inverse=True)
        return TriangleMesh(
            vertices=self.vertices[used],
            faces=inverse.reshape(-1, 3),
            normals=None if self.normals is None else self.normals[used],
            semantic=None if self.semantic is None else self.semantic[used],
            semantic_id=None if self.semantic_id is None else self.semantic_id[used],
            uncertainty=None if self.uncertainty is None else self.uncertainty[used],
            face_region_id=None
            if self.face_region_id is None
            else self.face_region_id.copy(),
            metadata=dict(self.metadata),
        )


@dataclass
class RegionAwareMesh:
    """One global topology with explicit per-vertex and per-face ownership."""

    global_mesh: TriangleMesh
    region_ids: np.ndarray
    vertex_region_id: np.ndarray
    vertex_region_confidence: np.ndarray
    face_region_id: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.global_mesh, TriangleMesh):
            raise TypeError("global_mesh must be a TriangleMesh")
        raw_region_ids = np.asarray(self.region_ids)
        if raw_region_ids.ndim != 1 or raw_region_ids.dtype.kind not in "iu":
            raise ValueError("region_ids must be a one-dimensional integer array")
        self.region_ids = np.ascontiguousarray(raw_region_ids, dtype=np.int64)
        if not len(self.region_ids) or np.any(self.region_ids <= 0):
            raise ValueError("region_ids must contain foreground IDs greater than zero")
        if np.any(np.diff(self.region_ids) <= 0):
            raise ValueError("region_ids must be sorted and unique")

        self.vertex_region_id = np.ascontiguousarray(
            np.asarray(self.vertex_region_id, dtype=np.int64).reshape(-1)
        )
        self.face_region_id = np.ascontiguousarray(
            np.asarray(self.face_region_id, dtype=np.int64).reshape(-1)
        )
        self.vertex_region_confidence = np.ascontiguousarray(
            np.asarray(self.vertex_region_confidence, dtype=np.float32).reshape(-1)
        )
        if len(self.vertex_region_id) != len(self.global_mesh.vertices):
            raise ValueError("vertex_region_id must have shape [V]")
        if len(self.vertex_region_confidence) != len(self.global_mesh.vertices):
            raise ValueError("vertex_region_confidence must have shape [V]")
        if len(self.face_region_id) != len(self.global_mesh.faces):
            raise ValueError("face_region_id must have shape [F]")
        allowed = np.concatenate((np.array([-1], dtype=np.int64), self.region_ids))
        if not np.all(np.isin(self.vertex_region_id, allowed)):
            raise ValueError("vertex_region_id contains an unrequested region ID")
        if not np.all(np.isin(self.face_region_id, allowed)):
            raise ValueError("face_region_id contains an unrequested region ID")
        if not np.all(np.isfinite(self.vertex_region_confidence)) or np.any(
            (self.vertex_region_confidence < 0.0)
            | (self.vertex_region_confidence > 1.0)
        ):
            raise ValueError("vertex_region_confidence must be finite and lie in [0, 1]")

        if len(self.global_mesh.faces):
            owners = self.vertex_region_id[self.global_mesh.faces]
            unanimous = (owners[:, 0] > 0) & np.all(owners == owners[:, :1], axis=1)
            expected = np.where(unanimous, owners[:, 0], -1)
            if not np.array_equal(self.face_region_id, expected):
                raise ValueError("face ownership must match unanimous vertex ownership")

    def region_view(self, region_id: int) -> TriangleMesh:
        """Return a compact face subset without rebuilding or modifying topology."""
        if isinstance(region_id, (bool, np.bool_)) or not isinstance(
            region_id, (int, np.integer)
        ):
            raise TypeError("region_id must be an integer")
        region_id = int(region_id)
        if region_id not in self.region_ids:
            raise ValueError("region_id was not requested for this mesh")
        selected = self.face_region_id == region_id
        view = TriangleMesh(
            vertices=self.global_mesh.vertices.copy(),
            faces=self.global_mesh.faces[selected].copy(),
            normals=None if self.global_mesh.normals is None else self.global_mesh.normals.copy(),
            semantic=None if self.global_mesh.semantic is None else self.global_mesh.semantic.copy(),
            semantic_id=None
            if self.global_mesh.semantic_id is None
            else self.global_mesh.semantic_id.copy(),
            uncertainty=None
            if self.global_mesh.uncertainty is None
            else self.global_mesh.uncertainty.copy(),
            face_region_id=self.face_region_id[selected].copy(),
            metadata={**self.global_mesh.metadata, "region_id": region_id},
        ).compact()
        view.metadata["region_id"] = region_id
        return view
