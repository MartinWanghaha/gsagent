"""Sparse cross-view graph construction and high-confidence matching."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import RobustAssociationConfig
from .types import AssociationEdge, MaskObservation


def camera_neighbor_pairs(
    camera_centers: np.ndarray,
    neighbors: int,
) -> list[tuple[int, int]]:
    count = camera_centers.shape[0]
    if count < 2:
        return []
    distances = np.linalg.norm(
        camera_centers[:, None, :] - camera_centers[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)
    pairs = set()
    for source in range(count):
        nearest = np.argsort(distances[source])[: min(neighbors, count - 1)]
        pairs.update(tuple(sorted((source, int(target)))) for target in nearest)
    return sorted(pairs)


def _view_evidence(
    observations: list[MaskObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = []
    nodes = []
    weights = []
    for observation in observations:
        ids.append(observation.gaussian_ids)
        nodes.append(
            np.full(
                observation.gaussian_ids.size,
                observation.node_id,
                dtype=np.int64,
            )
        )
        weights.append(observation.gaussian_weights)
    if not ids:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    ids_array = np.concatenate(ids)
    order = np.argsort(ids_array, kind="stable")
    return (
        ids_array[order],
        np.concatenate(nodes)[order],
        np.concatenate(weights)[order],
    )


def _appearance_similarity(source: MaskObservation, target: MaskObservation) -> float:
    if not source.appearance.any() or not target.appearance.any():
        return 0.0
    cosine = float(np.dot(source.appearance, target.appearance))
    # The descriptor is non-negative and L2-normalized, so cosine already lies
    # in [0, 1]. Re-centering it would give unrelated colors a misleading 0.5.
    return float(np.clip(cosine, 0.0, 1.0))


def _spatial_similarity(
    source: MaskObservation,
    target: MaskObservation,
    scene_scale: float,
) -> float:
    if (
        not np.isfinite(source.centroid_3d).all()
        or not np.isfinite(target.centroid_3d).all()
    ):
        return 0.0
    distance = float(np.linalg.norm(source.centroid_3d - target.centroid_3d))
    return float(np.exp(-distance / max(scene_scale * 0.15, 1e-6)))


def pair_view_edges(
    source_observations: list[MaskObservation],
    target_observations: list[MaskObservation],
    *,
    config: RobustAssociationConfig,
    scene_scale: float,
) -> list[AssociationEdge]:
    if not source_observations or not target_observations:
        return []
    source_ids, source_nodes, source_weights = _view_evidence(source_observations)
    target_ids, target_nodes, target_weights = _view_evidence(target_observations)
    common, source_indices, target_indices = np.intersect1d(
        source_ids,
        target_ids,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        return []
    del common
    source_node = source_nodes[source_indices]
    target_node = target_nodes[target_indices]
    pair_keys = np.stack((source_node, target_node), axis=1)
    unique_pairs, inverse = np.unique(pair_keys, axis=0, return_inverse=True)
    intersection = np.bincount(
        inverse,
        weights=np.minimum(
            source_weights[source_indices],
            target_weights[target_indices],
        ),
        minlength=unique_pairs.shape[0],
    )
    lookup = {
        observation.node_id: observation
        for observation in (*source_observations, *target_observations)
    }
    cue_sum = (
        config.gaussian_weight
        + config.coverage_weight
        + config.appearance_weight
        + config.spatial_weight
    )
    edges = []
    for pair, shared_mass in zip(unique_pairs, intersection):
        source = lookup[int(pair[0])]
        target = lookup[int(pair[1])]
        source_mass = max(source.evidence_mass, 1e-8)
        target_mass = max(target.evidence_mass, 1e-8)
        union = source_mass + target_mass - shared_mass
        jaccard = float(shared_mass / max(union, 1e-8))
        coverage_source = float(shared_mass / source_mass)
        coverage_target = float(shared_mass / target_mass)
        coverage = (
            2.0
            * coverage_source
            * coverage_target
            / max(coverage_source + coverage_target, 1e-8)
        )
        appearance = _appearance_similarity(source, target)
        spatial = _spatial_similarity(source, target, scene_scale)
        quality = float(np.sqrt(source.quality * target.quality))
        score = (
            quality
            * (
                config.gaussian_weight * jaccard
                + config.coverage_weight * coverage
                + config.appearance_weight * appearance
                + config.spatial_weight * spatial
            )
            / cue_sum
        )
        if score < config.candidate_threshold:
            continue
        edges.append(
            AssociationEdge(
                source=source.node_id,
                target=target.node_id,
                source_view=source.view_index,
                target_view=target.view_index,
                score=float(score),
                weighted_jaccard=jaccard,
                bidirectional_coverage=coverage,
                appearance=appearance,
                spatial=spatial,
                quality=quality,
            )
        )
    return edges


def select_hungarian_edges(
    edges: list[AssociationEdge],
    *,
    config: RobustAssociationConfig,
) -> list[AssociationEdge]:
    grouped: dict[tuple[int, int], list[AssociationEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.source_view, edge.target_view), []).append(edge)
    selected = []
    for pair_edges in grouped.values():
        source_nodes = sorted({edge.source for edge in pair_edges})
        target_nodes = sorted({edge.target for edge in pair_edges})
        source_index = {node: index for index, node in enumerate(source_nodes)}
        target_index = {node: index for index, node in enumerate(target_nodes)}
        scores = np.zeros((len(source_nodes), len(target_nodes)), dtype=np.float32)
        edge_lookup = {}
        for edge in pair_edges:
            row = source_index[edge.source]
            column = target_index[edge.target]
            if edge.score > scores[row, column]:
                scores[row, column] = edge.score
                edge_lookup[(row, column)] = edge
        rows, columns = linear_sum_assignment(-scores)
        for row, column in zip(rows, columns):
            score = float(scores[row, column])
            if score < config.match_threshold:
                continue
            row_alternatives = np.delete(scores[row], column)
            column_alternatives = np.delete(scores[:, column], row)
            alternative = max(
                float(row_alternatives.max()) if row_alternatives.size else 0.0,
                float(column_alternatives.max()) if column_alternatives.size else 0.0,
            )
            if score - alternative < config.match_margin:
                continue
            selected.append(replace(edge_lookup[(row, column)], selected=True))
    return selected


def build_association_graph(
    observations_by_view: list[list[MaskObservation]],
    camera_centers: np.ndarray,
    *,
    config: RobustAssociationConfig,
    scene_scale: float,
) -> tuple[list[AssociationEdge], list[AssociationEdge], list[tuple[int, int]]]:
    view_pairs = camera_neighbor_pairs(camera_centers, config.view_neighbors)
    candidates = []
    for source_view, target_view in view_pairs:
        candidates.extend(
            pair_view_edges(
                observations_by_view[source_view],
                observations_by_view[target_view],
                config=config,
                scene_scale=scene_scale,
            )
        )
    selected = select_hungarian_edges(candidates, config=config)
    return candidates, selected, view_pairs
