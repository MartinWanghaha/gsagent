"""Semantic compatibility and contact-aware mesh topology constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .types import TriangleMesh


@dataclass
class ContactGraph:
    """Undirected semantic contact graph with optional learned edge scores."""

    scores: dict[tuple[int, int], float] = field(default_factory=dict)
    threshold: float = 0.5

    @staticmethod
    def _edge(first: int, second: int) -> tuple[int, int]:
        first, second = int(first), int(second)
        return (first, second) if first <= second else (second, first)

    def add(self, first: int, second: int, score: float = 1.0) -> None:
        self.scores[self._edge(first, second)] = float(score)

    def allows(self, first: int, second: int) -> bool:
        if int(first) == int(second):
            return True
        return self.scores.get(self._edge(first, second), 0.0) >= self.threshold

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Sequence[float | int]],
        *,
        threshold: float = 0.5,
    ) -> "ContactGraph":
        graph = cls(threshold=threshold)
        for edge in edges:
            if len(edge) < 2:
                raise ValueError("contact graph edges need at least two labels")
            score = 1.0 if len(edge) < 3 else float(edge[2])
            graph.add(int(edge[0]), int(edge[1]), score)
        return graph

    @classmethod
    def from_json(cls, path: str | Path, *, threshold: float = 0.5) -> "ContactGraph":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return cls.from_edges(data, threshold=threshold)
        if not isinstance(data, Mapping):
            raise ValueError("contact graph JSON must be a list or object")
        graph = cls(threshold=float(data.get("threshold", threshold)))
        edges = data.get("edges")
        if edges is not None:
            for edge in edges:
                if isinstance(edge, Mapping):
                    graph.add(
                        int(edge.get("source", edge.get("a"))),
                        int(edge.get("target", edge.get("b"))),
                        float(edge.get("score", 1.0)),
                    )
                else:
                    graph.add(
                        int(edge[0]), int(edge[1]), 1.0 if len(edge) < 3 else float(edge[2])
                    )
            return graph
        for source, targets in data.items():
            if source == "threshold":
                continue
            if isinstance(targets, Mapping):
                for target, score in targets.items():
                    graph.add(int(source), int(target), float(score))
            else:
                for target in targets:
                    graph.add(int(source), int(target), 1.0)
        return graph

    @classmethod
    def from_gaussians(
        cls,
        gaussians,
        *,
        decoder=None,
        threshold: float = 0.5,
        neighbors: int = 8,
        distance_factor: float = 2.5,
        min_support: float = 4.0,
        confidence_floor: float = 0.2,
        background_id: int | None = 0,
        max_points: int = 500_000,
        seed: int = 0,
    ) -> "ContactGraph":
        """Estimate physical instance contacts from the learned Gaussian state.

        Semantic labels alone must not disconnect two objects that physically
        touch. Cross-label Gaussian pairs vote for an edge only when their
        centers overlap at the scale of their covariance and both carry
        semantic evidence. Scores combine overlap strength and repeated local
        support, so a single projected contour coincidence is insufficient.
        """

        graph = cls(threshold=threshold)
        if neighbors < 1 or max_points < 2:
            raise ValueError("neighbors and max_points must be positive")
        xyz_value = getattr(gaussians, "get_xyz")
        xyz_value = xyz_value() if callable(xyz_value) else xyz_value
        if len(xyz_value) < 2:
            return graph
        try:
            from scipy.spatial import cKDTree
        except ImportError as error:
            raise RuntimeError("automatic contact-graph estimation requires scipy") from error

        def numpy(value):
            value = value() if callable(value) else value
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return np.asarray(value)

        xyz = numpy(xyz_value).astype(np.float32, copy=False)
        scales = numpy(getattr(gaussians, "get_scaling")).astype(np.float32, copy=False)
        confidence_source = getattr(gaussians, "get_semantic_confidence", None)
        if confidence_source is None:
            confidence_source = getattr(gaussians, "semantic_confidence")
        confidence = numpy(confidence_source).reshape(-1).astype(np.float32, copy=False)
        semantic_source = getattr(gaussians, "get_semantic", None)
        if semantic_source is None:
            semantic_source = getattr(gaussians, "semantic_embedding")
        semantic = semantic_source() if callable(semantic_source) else semantic_source
        selected_decoder = decoder or getattr(gaussians, "semantic_decoder", None)
        if selected_decoder is None:
            return graph

        # Decoder stays on its owning device and runs in bounded chunks.
        labels_chunks = []
        try:
            import torch

            with torch.no_grad():
                for start in range(0, len(xyz), 65_536):
                    logits = selected_decoder(semantic[start : start + 65_536])
                    if logits is None:
                        return graph
                    labels_chunks.append(logits.argmax(-1).detach().cpu().numpy())
        except (ImportError, TypeError, AttributeError):
            decoded = selected_decoder(numpy(semantic))
            if decoded is None:
                return graph
            logits = numpy(decoded)
            labels_chunks = [np.argmax(logits, axis=-1)]
        labels = np.concatenate(labels_chunks).astype(np.int64, copy=False)

        valid = confidence >= float(confidence_floor)
        if background_id is not None:
            valid &= labels != int(background_id)
        candidates = np.flatnonzero(valid)
        if len(candidates) < 2:
            return graph
        if len(candidates) > max_points:
            generator = np.random.default_rng(seed)
            probability = confidence[candidates].astype(np.float64)
            probability /= probability.sum()
            candidates = np.sort(
                generator.choice(candidates, size=max_points, replace=False, p=probability)
            )

        points = xyz[candidates]
        k = min(int(neighbors) + 1, len(points))
        distance, local_neighbor = cKDTree(points).query(points, k=k, workers=-1)
        if k == 1:
            return graph
        distance = np.asarray(distance)[:, 1:].reshape(-1)
        local_neighbor = np.asarray(local_neighbor)[:, 1:].reshape(-1)
        source_local = np.repeat(np.arange(len(points)), k - 1)
        # Count each undirected Gaussian pair once.
        unique_pair = source_local < local_neighbor
        source = candidates[source_local[unique_pair]]
        target = candidates[local_neighbor[unique_pair]]
        distance = distance[unique_pair]
        first_label, second_label = labels[source], labels[target]
        cross_label = first_label != second_label
        source, target, distance = source[cross_label], target[cross_label], distance[cross_label]
        first_label, second_label = first_label[cross_label], second_label[cross_label]
        if not len(source):
            return graph

        support_radius = float(distance_factor) * (
            scales[source].max(axis=1) + scales[target].max(axis=1)
        )
        overlap = distance <= support_radius
        source, target, distance = source[overlap], target[overlap], distance[overlap]
        first_label, second_label = first_label[overlap], second_label[overlap]
        support_radius = support_radius[overlap]
        if not len(source):
            return graph
        strength = np.exp(-distance / np.maximum(support_radius, 1e-8))
        strength *= np.sqrt(confidence[source] * confidence[target])
        label_a = np.minimum(first_label, second_label)
        label_b = np.maximum(first_label, second_label)
        pairs = np.stack((label_a, label_b), axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float32)
        sums = np.bincount(inverse, weights=strength).astype(np.float32)
        mean_strength = sums / np.maximum(counts, 1.0)
        repeated_support = 1.0 - np.exp(-counts / max(float(min_support), 1e-6))
        scores = mean_strength * repeated_support
        for (first, second), score in zip(unique, scores):
            graph.add(int(first), int(second), float(score))
        return graph


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return numerator / np.maximum(denominator, 1e-8)


def compatible_pairs(
    first_semantic: np.ndarray,
    second_semantic: np.ndarray,
    first_label: np.ndarray,
    second_label: np.ndarray,
    *,
    cosine_threshold: float = 0.85,
    contact_graph: Optional[ContactGraph] = None,
) -> np.ndarray:
    """Compatibility is soft-semantic agreement or an explicit physical contact."""
    first_label = np.asarray(first_label, dtype=np.int64)
    second_label = np.asarray(second_label, dtype=np.int64)
    same = (first_label >= 0) & (second_label >= 0) & (first_label == second_label)
    similar = cosine_similarity(first_semantic, second_semantic) >= cosine_threshold
    allowed = same | similar
    if contact_graph is not None:
        flat_first = first_label.reshape(-1)
        flat_second = second_label.reshape(-1)
        graph_allowed = np.fromiter(
            (
                a >= 0 and b >= 0 and contact_graph.allows(a, b)
                for a, b in zip(flat_first, flat_second)
            ),
            dtype=bool,
            count=len(flat_first),
        ).reshape(first_label.shape)
        allowed |= graph_allowed
    return allowed


def semantic_edge_compatibility(
    semantic: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    *,
    cosine_threshold: float = 0.85,
    contact_graph: Optional[ContactGraph] = None,
) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    return compatible_pairs(
        semantic[edges[:, 0]],
        semantic[edges[:, 1]],
        labels[edges[:, 0]],
        labels[edges[:, 1]],
        cosine_threshold=cosine_threshold,
        contact_graph=contact_graph,
    )


def face_compatibility_mask(
    mesh: TriangleMesh,
    *,
    cosine_threshold: float = 0.85,
    contact_graph: Optional[ContactGraph] = None,
    max_edge_length: Optional[float] = None,
) -> np.ndarray:
    if not len(mesh.faces):
        return np.empty((0,), dtype=bool)
    if mesh.semantic is None or mesh.semantic_id is None:
        semantic_ok = np.ones(len(mesh.faces), dtype=bool)
    else:
        edges = mesh.faces[:, ((0, 1), (1, 2), (2, 0))]
        flat_edges = edges.reshape(-1, 2)
        edge_ok = semantic_edge_compatibility(
            mesh.semantic,
            mesh.semantic_id,
            flat_edges,
            cosine_threshold=cosine_threshold,
            contact_graph=contact_graph,
        )
        semantic_ok = edge_ok.reshape(-1, 3).all(axis=1)
    if max_edge_length is None:
        return semantic_ok
    edges = mesh.faces[:, ((0, 1), (1, 2), (2, 0))]
    length = np.linalg.norm(
        mesh.vertices[edges[:, :, 0]] - mesh.vertices[edges[:, :, 1]], axis=-1
    )
    return semantic_ok & (length <= float(max_edge_length)).all(axis=1)


def filter_semantic_topology(
    mesh: TriangleMesh,
    *,
    cosine_threshold: float = 0.85,
    contact_graph: Optional[ContactGraph] = None,
    max_edge_length: Optional[float] = None,
) -> TriangleMesh:
    mask = face_compatibility_mask(
        mesh,
        cosine_threshold=cosine_threshold,
        contact_graph=contact_graph,
        max_edge_length=max_edge_length,
    )
    result = mesh.copy()
    result.faces = result.faces[mask]
    result.metadata["semantic_faces_removed"] = int((~mask).sum())
    return result.compact()


def seam_vertices(mesh: TriangleMesh, *, cosine_threshold: float = 0.85) -> np.ndarray:
    """Return vertices adjacent to a semantic label seam."""
    seam = np.zeros(len(mesh.vertices), dtype=bool)
    if not len(mesh.faces):
        return seam
    mixed = np.zeros(len(mesh.faces), dtype=bool)
    if mesh.semantic_id is not None:
        labels = mesh.semantic_id[mesh.faces]
        known = labels >= 0
        mixed |= (
            known[:, :, None]
            & known[:, None, :]
            & (labels[:, :, None] != labels[:, None, :])
        ).any(axis=(1, 2))
    if mesh.semantic is not None:
        edges = mesh.faces[:, ((0, 1), (1, 2), (2, 0))]
        similarity = cosine_similarity(
            mesh.semantic[edges[:, :, 0]], mesh.semantic[edges[:, :, 1]]
        )
        mixed |= (similarity < cosine_threshold).any(axis=1)
    seam[mesh.faces[mixed].reshape(-1)] = True
    return seam


def connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    """Connected face components using shared vertices as adjacency."""
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return []
    vertex_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), []).append(face_index)
    visited = np.zeros(len(faces), dtype=bool)
    components: list[np.ndarray] = []
    for seed in range(len(faces)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component = []
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for vertex in faces[face_index]:
                for neighbour in vertex_faces[int(vertex)]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        stack.append(neighbour)
        components.append(np.asarray(component, dtype=np.int64))
    return components
