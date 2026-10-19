"""Certified view graph and deterministic, bounded multistart beam search."""
from __future__ import annotations

import numpy as np
from view_planning_geometry import world_to_camera, certify_segment


def build_graph(scene, free, positions, cameras, metrics, roi, config, safe_radius, margin):
    from scipy.spatial import cKDTree
    selection = config["selection"]
    maximum = roi["radius"]*selection["max_neighbor_translation_over_target_radius"]
    graph = {i: {} for i in range(len(positions))}
    tree = cKDTree(positions)
    rotations = [world_to_camera(c)[:3, :3] for c in cameras]
    rotations = [r/np.linalg.norm(r[:, 0]) for r in rotations]
    tested = set()
    for i in range(len(positions)):
        neighbors = tree.query(positions[i], k=min(len(positions), selection["max_neighbors"]+1), distance_upper_bound=maximum)[1]
        for j in np.atleast_1d(neighbors):
            j = int(j)
            pair = tuple(sorted((i, j)))
            if i == j or j >= len(positions) or pair in tested:
                continue
            tested.add(pair)
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rotations[i]@rotations[j].T)-1)/2, -1, 1))))
            common = int((metrics[i]["visible"] & metrics[j]["visible"]).sum())
            if angle > selection["max_neighbor_rotation_deg"] or common < selection["minimum_common_target_samples"]:
                continue
            edge = certify_segment(scene, free, positions[i], positions[j], safe_radius,
                margin=margin, min_step=roi["radius"]*config["collision"]["edge_min_step_over_target_radius"],
                max_depth=config["collision"]["edge_max_subdivisions"])
            if edge:
                edge.update(rotation_deg=angle, common_target_samples=common,
                            length=float(np.linalg.norm(positions[i]-positions[j])))
                graph[i][j] = edge
                graph[j][i] = edge
    return graph


def select_sequence(graph, metrics, positions, anchors, count, config, *, banned=frozenset()):
    """All bends/transitions consume a frame. No duplicate or shortcut padding."""
    active = set(graph)-set(banned)
    # Only components connected to an actually validated observation anchor.
    reachable = set()
    for anchor in anchors:
        if anchor not in active:
            continue
        queue = [anchor]
        while queue:
            node = queue.pop()
            if node in reachable:
                continue
            reachable.add(node)
            queue.extend(j for j in graph[node] if j in active and j not in reachable)
    if len(reachable) < count:
        raise ValueError(f"no anchored component can supply {count} safe distinct frames (reachable={len(reachable)})")
    starts = sorted(reachable, key=lambda i: (-metrics[i]["score"], i))[:config["selection"]["start_count"]]
    beam = [((i,), metrics[i]["visible"].copy(), metrics[i]["score"]) for i in starts]
    for _ in range(1, count):
        expanded = []
        for route, covered, score in beam:
            for nxt, edge in sorted(graph[route[-1]].items()):
                if nxt not in reachable or nxt in route:
                    continue
                added = int((metrics[nxt]["visible"] & ~covered).sum())/len(covered)
                # Saturating known-surface coverage plus quality and motion cost.
                gain = 4*added + .1*metrics[nxt]["score"] - .02*edge["rotation_deg"]/180
                expanded.append((route+(nxt,), covered | metrics[nxt]["visible"], score+gain))
        if not expanded:
            raise ValueError("N-frame route infeasible within bounded search; increase candidates/beam budget or N and use a new RUN_NAME")
        expanded.sort(key=lambda state: (-state[2], state[0]))
        beam = expanded[:config["selection"]["beam_width"]]
    for route, coverage, score in beam:
        if coverage.mean() >= config["view_quality"]["min_known_surface_coverage"]:
            return list(route), float(coverage.mean()), float(score)
    raise ValueError("known target surface coverage below the configured minimum")
