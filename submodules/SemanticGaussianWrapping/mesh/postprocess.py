"""Seam-aware simplification and semantic instance preserving cleanup."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .topology import connected_face_components, seam_vertices
from .types import TriangleMesh


def recompute_vertex_normals(mesh: TriangleMesh) -> TriangleMesh:
    result = mesh.copy()
    normals = np.zeros_like(result.vertices)
    if len(result.faces):
        face_normals = np.cross(
            result.vertices[result.faces[:, 1]] - result.vertices[result.faces[:, 0]],
            result.vertices[result.faces[:, 2]] - result.vertices[result.faces[:, 0]],
        )
        for corner in range(3):
            np.add.at(normals, result.faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    result.normals = normals
    return result


def _clean_faces_with_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_region_id: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Remove invalid/duplicate faces while reducing their ownership safely."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        regions = (
            None
            if face_region_id is None
            else np.empty((0,), dtype=np.int32)
        )
        return faces, regions
    valid = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    source_indices = np.flatnonzero(valid)
    faces = faces[valid]
    if not len(faces):
        regions = (
            None
            if face_region_id is None
            else np.empty((0,), dtype=np.int32)
        )
        return faces, regions
    area_twice = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    non_degenerate = area_twice > 1e-12
    source_indices = source_indices[non_degenerate]
    faces = faces[non_degenerate]
    if not len(faces):
        regions = (
            None
            if face_region_id is None
            else np.empty((0,), dtype=np.int32)
        )
        return faces, regions

    canonical = np.sort(faces, axis=1)
    _, first, inverse = np.unique(
        canonical,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    group_order = np.argsort(first)
    cleaned_faces = faces[first[group_order]]
    if face_region_id is None:
        return cleaned_faces, None

    source_regions = np.asarray(face_region_id, dtype=np.int32)[source_indices]
    group_count = len(first)
    minimum = np.full(group_count, np.iinfo(np.int32).max, dtype=np.int32)
    maximum = np.full(group_count, np.iinfo(np.int32).min, dtype=np.int32)
    np.minimum.at(minimum, inverse, source_regions)
    np.maximum.at(maximum, inverse, source_regions)
    reduced = np.where(minimum == maximum, minimum, -2).astype(np.int32)
    return cleaned_faces, reduced[group_order]


def _clean_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compatibility helper for callers that do not carry face ownership."""
    return _clean_faces_with_regions(vertices, faces, None)[0]


def remove_small_components(
    mesh: TriangleMesh,
    *,
    min_faces: int = 20,
    preserve_semantic_instances: bool = True,
    min_instance_vertices: int = 3,
    background_id: Optional[int] = None,
    preserve_max_uncertainty: Optional[float] = None,
) -> TriangleMesh:
    """Remove noise without deleting a trustworthy unique semantic instance.

    ``preserve_max_uncertainty`` is useful for training feedback: a tiny
    component survives only when it is both a label's sole instance and its
    mean field uncertainty is below the threshold. Offline cleanup retains
    the historical behavior when the threshold is omitted.
    """
    if preserve_max_uncertainty is not None and not 0.0 <= preserve_max_uncertainty <= 1.0:
        raise ValueError("preserve_max_uncertainty must lie in [0,1]")
    if min_faces <= 1 or not len(mesh.faces):
        return mesh.copy()
    components = connected_face_components(mesh.faces)
    if not components:
        return mesh.copy()

    majority: list[Optional[int]] = []
    label_components: dict[int, int] = {}
    for component in components:
        vertices = np.unique(mesh.faces[component])
        if mesh.semantic_id is None or not len(vertices):
            majority.append(None)
            continue
        labels, counts = np.unique(mesh.semantic_id[vertices], return_counts=True)
        label = int(labels[np.argmax(counts)])
        if label < 0:
            majority.append(None)
            continue
        majority.append(label)
        label_components[label] = label_components.get(label, 0) + 1

    keep: list[np.ndarray] = []
    preserved = 0
    for component, label in zip(components, majority):
        if len(component) >= min_faces:
            keep.append(component)
            continue
        vertices = np.unique(mesh.faces[component])
        unique_instance = (
            preserve_semantic_instances
            and label is not None
            and label != background_id
            and label_components.get(label, 0) == 1
            and len(vertices) >= min_instance_vertices
        )
        if unique_instance and preserve_max_uncertainty is not None:
            # Missing uncertainty is not evidence of a trustworthy instance.
            # The historical preserve-all behavior remains available by
            # leaving the threshold unset for offline cleanup.
            unique_instance = mesh.uncertainty is not None and bool(
                np.mean(mesh.uncertainty[vertices]) <= preserve_max_uncertainty
            )
        if unique_instance:
            keep.append(component)
            preserved += 1

    selected = (
        np.concatenate(keep)
        if keep
        else np.empty((0,), dtype=np.int64)
    )
    result = TriangleMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[selected].copy(),
        normals=None if mesh.normals is None else mesh.normals.copy(),
        semantic=None if mesh.semantic is None else mesh.semantic.copy(),
        semantic_id=None if mesh.semantic_id is None else mesh.semantic_id.copy(),
        uncertainty=None if mesh.uncertainty is None else mesh.uncertainty.copy(),
        face_region_id=None
        if mesh.face_region_id is None
        else mesh.face_region_id[selected].copy(),
        metadata=dict(mesh.metadata),
    )
    result.metadata["small_instances_preserved"] = preserved
    result.metadata["components_removed"] = len(components) - len(keep)
    return result.compact()


def seam_aware_vertex_clustering(
    mesh: TriangleMesh,
    voxel_size: float,
    *,
    protect_uncertainty_threshold: Optional[float] = None,
) -> TriangleMesh:
    """Cluster interior vertices but never merge semantic seam/detail vertices."""
    if voxel_size <= 0 or not len(mesh.faces):
        return mesh.copy()
    protected = seam_vertices(mesh)
    if protect_uncertainty_threshold is not None and mesh.uncertainty is not None:
        protected |= mesh.uncertainty >= float(protect_uncertainty_threshold)

    spatial = np.floor(
        (mesh.vertices - mesh.vertices.min(axis=0, keepdims=True)) / float(voxel_size)
    ).astype(np.int64)
    labels = (
        np.zeros((len(mesh.vertices), 1), dtype=np.int64)
        if mesh.semantic_id is None
        else mesh.semantic_id[:, None].astype(np.int64)
    )
    identity = np.zeros((len(mesh.vertices), 1), dtype=np.int64)
    identity[protected, 0] = np.nonzero(protected)[0] + 1
    keys = np.concatenate([spatial, labels, identity], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    weights = np.bincount(inverse, minlength=count).astype(np.float64)

    def average(values: np.ndarray) -> np.ndarray:
        output = np.zeros((count, values.shape[1]), dtype=np.float64)
        np.add.at(output, inverse, values)
        return (output / weights[:, None]).astype(np.float32)

    def aggregate_semantic_ids() -> Optional[np.ndarray]:
        """Keep decoded IDs; Gaga embedding dimensions are not class IDs.

        Labels are part of the clustering key today, so the common case has a
        single label per cluster.  A confidence-weighted vote still makes the
        reduction correct if the key policy is relaxed later.  Lower field
        uncertainty means a stronger vote; exact ties prefer lower mean
        uncertainty and then the smallest ID for deterministic output.
        """

        if mesh.semantic_id is None:
            return None
        labels = np.asarray(mesh.semantic_id, dtype=np.int32)
        order = np.argsort(inverse, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(weights.astype(np.int64))))
        output = np.full(count, -1, dtype=np.int32)
        for cluster in range(count):
            members = order[offsets[cluster] : offsets[cluster + 1]]
            known = members[labels[members] >= 0]
            if not len(known):
                continue
            values, local = np.unique(labels[known], return_inverse=True)
            if mesh.uncertainty is None:
                vote_weight = np.ones(len(known), dtype=np.float64)
                uncertainty_value = np.zeros(len(known), dtype=np.float64)
            else:
                uncertainty_value = np.clip(
                    np.asarray(mesh.uncertainty[known], dtype=np.float64), 0.0, 1.0
                )
                vote_weight = 1.0 - uncertainty_value
                if not np.any(vote_weight > 0):
                    vote_weight = np.ones(len(known), dtype=np.float64)
            votes = np.bincount(local, weights=vote_weight, minlength=len(values))
            candidates = np.flatnonzero(np.isclose(votes, votes.max()))
            if len(candidates) > 1:
                mean_uncertainty = np.asarray(
                    [uncertainty_value[local == index].mean() for index in candidates]
                )
                candidates = candidates[
                    np.isclose(mean_uncertainty, mean_uncertainty.min())
                ]
            output[cluster] = int(values[int(candidates[0])])
        return output

    vertices = average(mesh.vertices)
    semantic = None if mesh.semantic is None else average(mesh.semantic)
    semantic_id = aggregate_semantic_ids()
    normals = None if mesh.normals is None else average(mesh.normals)
    if normals is not None:
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    uncertainty = None
    if mesh.uncertainty is not None:
        total = np.bincount(inverse, weights=mesh.uncertainty, minlength=count)
        uncertainty = (total / weights).astype(np.float32)
    faces, face_region_id = _clean_faces_with_regions(
        vertices,
        inverse[mesh.faces],
        mesh.face_region_id,
    )
    result = TriangleMesh(
        vertices=vertices,
        faces=faces,
        normals=normals,
        semantic=semantic,
        semantic_id=semantic_id,
        uncertainty=uncertainty,
        face_region_id=face_region_id,
        metadata=dict(mesh.metadata),
    ).compact()
    result.metadata["seam_vertices_protected"] = int(protected.sum())
    return result


def simplify_to_face_budget(
    mesh: TriangleMesh,
    target_faces: int,
    *,
    initial_voxel_size: Optional[float] = None,
    protect_uncertainty_threshold: Optional[float] = None,
    iterations: int = 10,
) -> TriangleMesh:
    """Dependency-free face-budget approximation using protected clustering."""
    if target_faces < 1:
        raise ValueError("target_faces must be positive")
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    extent = np.ptp(mesh.vertices, axis=0)
    diagonal = max(float(np.linalg.norm(extent)), 1e-8)
    voxel = initial_voxel_size or diagonal / 1024.0
    best = mesh.copy()
    for _ in range(iterations):
        candidate = seam_aware_vertex_clustering(
            mesh,
            voxel,
            protect_uncertainty_threshold=protect_uncertainty_threshold,
        )
        if len(candidate.faces) <= target_faces:
            return candidate
        if len(candidate.faces) < len(best.faces):
            best = candidate
        voxel *= 1.7
    return best


def postprocess_mesh(
    mesh: TriangleMesh,
    *,
    min_component_faces: int = 20,
    preserve_semantic_instances: bool = True,
    min_instance_vertices: int = 3,
    background_id: Optional[int] = None,
    simplify_voxel_size: Optional[float] = None,
    target_faces: Optional[int] = None,
    protect_uncertainty_threshold: Optional[float] = None,
) -> TriangleMesh:
    result = remove_small_components(
        mesh,
        min_faces=min_component_faces,
        preserve_semantic_instances=preserve_semantic_instances,
        min_instance_vertices=min_instance_vertices,
        background_id=background_id,
    )
    if simplify_voxel_size is not None and simplify_voxel_size > 0:
        result = seam_aware_vertex_clustering(
            result,
            simplify_voxel_size,
            protect_uncertainty_threshold=protect_uncertainty_threshold,
        )
    if target_faces is not None:
        result = simplify_to_face_budget(
            result,
            target_faces,
            initial_voxel_size=simplify_voxel_size,
            protect_uncertainty_threshold=protect_uncertainty_threshold,
        )
    return recompute_vertex_normals(result)
