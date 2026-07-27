"""Order-independent constrained clustering for mask observations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import RobustAssociationConfig
from .types import AssociationEdge, InstanceTrack, MaskObservation


class ConstrainedDisjointSet:
    def __init__(self, observations: list[MaskObservation]) -> None:
        self.parent = np.arange(len(observations), dtype=np.int64)
        self.rank = np.zeros(len(observations), dtype=np.int8)
        self.views = {
            observation.node_id: {observation.view_index}
            for observation in observations
        }

    def find(self, node: int) -> int:
        parent = int(self.parent[node])
        if parent != node:
            self.parent[node] = self.find(parent)
        return int(self.parent[node])

    def conflicts(self, source: int, target: int) -> set[int]:
        source_root = self.find(source)
        target_root = self.find(target)
        if source_root == target_root:
            return set()
        return self.views[source_root] & self.views[target_root]

    def union(self, source: int, target: int, *, allow_conflicts: bool = False) -> bool:
        source_root = self.find(source)
        target_root = self.find(target)
        if source_root == target_root:
            return False
        if not allow_conflicts and self.views[source_root] & self.views[target_root]:
            return False
        if self.rank[source_root] < self.rank[target_root]:
            source_root, target_root = target_root, source_root
        self.parent[target_root] = source_root
        self.views[source_root] |= self.views[target_root]
        del self.views[target_root]
        if self.rank[source_root] == self.rank[target_root]:
            self.rank[source_root] += 1
        return True


def _appearance_similarity(source: MaskObservation, target: MaskObservation) -> float:
    if not source.appearance.any() or not target.appearance.any():
        return 0.0
    return float(np.clip(np.dot(source.appearance, target.appearance), 0.0, 1.0))


def _bbox_gap(source: MaskObservation, target: MaskObservation) -> float:
    source_x0, source_y0, source_x1, source_y1 = source.bbox
    target_x0, target_y0, target_x1, target_y1 = target.bbox
    dx = max(source_x0 - target_x1, target_x0 - source_x1, 0)
    dy = max(source_y0 - target_y1, target_y0 - source_y1, 0)
    height, width = source.image_shape
    return float(np.hypot(dx, dy) / max(np.hypot(width, height), 1.0))


def _fragment_conflicts_are_compatible(
    source_root: int,
    target_root: int,
    dsu: ConstrainedDisjointSet,
    observations: list[MaskObservation],
    config: RobustAssociationConfig,
) -> bool:
    conflicts = dsu.views[source_root] & dsu.views[target_root]
    if len(conflicts) != 1:
        return False
    conflict_view = next(iter(conflicts))
    source_nodes = [
        observation
        for observation in observations
        if dsu.find(observation.node_id) == source_root
        and observation.view_index == conflict_view
    ]
    target_nodes = [
        observation
        for observation in observations
        if dsu.find(observation.node_id) == target_root
        and observation.view_index == conflict_view
    ]
    if len(source_nodes) != 1 or len(target_nodes) != 1:
        return False
    source = source_nodes[0]
    target = target_nodes[0]
    return (
        _appearance_similarity(source, target) >= config.fragment_appearance_threshold
        and _bbox_gap(source, target) <= 0.03
    )


def _track_prototype(
    nodes: list[MaskObservation],
) -> tuple[np.ndarray, np.ndarray]:
    id_parts = []
    weight_parts = []
    for observation in nodes:
        if observation.gaussian_ids.size == 0:
            continue
        id_parts.append(observation.gaussian_ids)
        weight_parts.append(observation.gaussian_weights * observation.quality)
    if not id_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    ids = np.concatenate(id_parts)
    weights = np.concatenate(weight_parts)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    weights = weights[order]
    unique_ids, first = np.unique(ids, return_index=True)
    summed = np.add.reduceat(weights, first)
    summed /= max(float(summed.max()), 1e-8)
    return unique_ids, summed.astype(np.float32, copy=False)


def _track_area_fraction(
    track: InstanceTrack,
    observations: list[MaskObservation],
) -> float:
    nodes = [observations[node_id] for node_id in track.node_ids]
    pixels = sum(
        height * width for height, width in (node.image_shape for node in nodes)
    )
    return sum(node.area for node in nodes) / max(pixels, 1)


def _propagation_target(
    track: InstanceTrack,
    observations: list[MaskObservation],
    tracks_by_node: dict[int, InstanceTrack],
    candidate_edges: list[AssociationEdge],
    config: RobustAssociationConfig,
) -> tuple[int, float] | None:
    node_ids = set(track.node_ids)
    evidence: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for edge in candidate_edges:
        if edge.source in node_ids:
            neighbor_id = edge.target
        elif edge.target in node_ids:
            neighbor_id = edge.source
        else:
            continue
        neighbor_track = tracks_by_node[neighbor_id]
        if neighbor_track.status != "confirmed":
            continue
        neighbor_view = observations[neighbor_id].view_index
        evidence[neighbor_track.global_id].append((edge.score, neighbor_view))

    ranked = []
    for global_id, edges in evidence.items():
        strongest = sorted(edges, reverse=True)[: config.tentative_max_neighbor_edges]
        neighbor_views = {view for _, view in strongest}
        if len(neighbor_views) < config.tentative_min_neighbor_views:
            continue
        ranked.append((float(np.mean([score for score, _ in strongest])), global_id))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    best_score, best_id = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if (
        best_score < config.tentative_propagation_threshold
        or best_score - runner_up < config.tentative_propagation_margin
    ):
        return None
    return best_id, best_score


def cluster_observations(
    observations: list[MaskObservation],
    candidate_edges: list[AssociationEdge],
    selected_edges: list[AssociationEdge],
    *,
    config: RobustAssociationConfig,
) -> list[InstanceTrack]:
    if not observations:
        return []
    if [observation.node_id for observation in observations] != list(
        range(len(observations))
    ):
        raise ValueError("Observation node IDs must be contiguous")
    dsu = ConstrainedDisjointSet(observations)
    for edge in sorted(selected_edges, key=lambda item: item.score, reverse=True):
        dsu.union(edge.source, edge.target)

    # A conservative second pass can merge same-view fragments only when their
    # appearance is strong and their bounding boxes touch.
    for edge in sorted(candidate_edges, key=lambda item: item.score, reverse=True):
        if edge.score < config.fragment_merge_threshold:
            break
        source_root = dsu.find(edge.source)
        target_root = dsu.find(edge.target)
        if source_root == target_root:
            continue
        conflicts = dsu.conflicts(source_root, target_root)
        if not conflicts:
            dsu.union(source_root, target_root)
        elif _fragment_conflicts_are_compatible(
            source_root,
            target_root,
            dsu,
            observations,
            config,
        ):
            dsu.union(source_root, target_root, allow_conflicts=True)

    grouped: dict[int, list[MaskObservation]] = defaultdict(list)
    for observation in observations:
        grouped[dsu.find(observation.node_id)].append(observation)
    edge_scores: dict[int, list[float]] = defaultdict(list)
    for edge in candidate_edges:
        root = dsu.find(edge.source)
        if root == dsu.find(edge.target):
            edge_scores[root].append(edge.score)

    tracks = []
    for internal_id, (root, nodes) in enumerate(
        sorted(grouped.items(), key=lambda item: min(node.node_id for node in item[1]))
    ):
        views = [node.view_index for node in nodes]
        quality = float(np.mean([node.quality for node in nodes]))
        mean_edge = float(np.mean(edge_scores[root])) if edge_scores[root] else 0.0
        support_views = len(set(views))
        evidence_count = sum(node.gaussian_ids.size for node in nodes)
        if quality <= config.min_track_quality * 0.5 or evidence_count == 0:
            status = "rejected"
        elif (
            support_views >= config.min_track_views
            and quality >= config.min_track_quality
        ):
            status = "confirmed"
        else:
            status = "tentative"
        gaussian_ids, gaussian_weights = _track_prototype(nodes)
        tracks.append(
            InstanceTrack(
                internal_id=internal_id,
                node_ids=[node.node_id for node in nodes],
                view_ids=views,
                quality=quality,
                mean_edge_score=mean_edge,
                status=status,
                gaussian_ids=gaussian_ids,
                gaussian_weights=gaussian_weights,
            )
        )

    confirmed = sorted(
        (track for track in tracks if track.status == "confirmed"),
        key=lambda item: (-len(set(item.view_ids)), -item.quality, item.internal_id),
    )
    next_global_id = 1
    for track in confirmed:
        track.global_id = next_global_id
        next_global_id += 1

    tracks_by_node = {node_id: track for track in tracks for node_id in track.node_ids}
    assignment_scores: dict[int, float] = {}
    tentative = sorted(
        (track for track in tracks if track.status == "tentative"),
        key=lambda item: (-item.quality, item.internal_id),
    )
    for track in tentative:
        propagated = _propagation_target(
            track,
            observations,
            tracks_by_node,
            candidate_edges,
            config,
        )
        if propagated is not None:
            track.global_id, assignment_scores[track.internal_id] = propagated
            track.status = "propagated"
            continue
        if (
            _track_area_fraction(track, observations)
            >= config.tentative_min_area_fraction
            and track.quality >= config.tentative_min_quality
            and track.gaussian_ids.size >= config.tentative_min_gaussians
        ):
            track.global_id = next_global_id
            next_global_id += 1
            track.status = "promoted"
            assignment_scores[track.internal_id] = track.quality

    for track in tracks:
        default_score = max(track.mean_edge_score, track.quality * 0.25)
        score = assignment_scores.get(track.internal_id, default_score)
        for node_id in track.node_ids:
            observation = observations[node_id]
            observation.track_id = track.global_id
            observation.status = track.status
            observation.assignment_score = score if track.global_id > 0 else 0.0
    return tracks
