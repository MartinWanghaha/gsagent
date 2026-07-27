"""CPU-only tests for noise-robust multi-view association."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from robust_association.clustering import cluster_observations
from robust_association.config import RobustAssociationConfig
from robust_association.exporter import export_association
from robust_association.engine import RobustAssociationEngine
from robust_association.graph import build_association_graph
from robust_association.observations import reliable_core_map
from robust_association.refinement import (
    build_gaussian_consensus,
    refine_view,
)
from robust_association.types import (
    AssociationEdge,
    InstanceTrack,
    MaskObservation,
    RefinedView,
    ViewProjection,
)


def observation(
    node_id: int,
    view: int,
    local_id: int,
    gaussian_ids,
    *,
    appearance=None,
    quality: float = 1.0,
) -> MaskObservation:
    ids = np.asarray(gaussian_ids, dtype=np.int64)
    return MaskObservation(
        node_id=node_id,
        view_index=view,
        image_name=f"view_{view}",
        local_id=local_id,
        area=100,
        core_ratio=0.8,
        quality=quality,
        bbox=(0, 0, 10, 10),
        image_shape=(10, 10),
        gaussian_ids=ids,
        gaussian_weights=np.ones(ids.size, dtype=np.float32),
        appearance=np.asarray(
            appearance if appearance is not None else [1.0, 0.0],
            dtype=np.float32,
        ),
        centroid_3d=np.array([local_id, 0, 0], dtype=np.float32),
    )


def test_reliable_core_downweights_boundaries_and_keeps_instances_separate():
    labels = np.zeros((9, 12), dtype=np.int32)
    labels[1:8, 1:6] = 1
    labels[1:8, 6:11] = 2
    core = reliable_core_map(labels, 1)
    assert core[4, 3]
    assert core[4, 8]
    assert not core[1, 3]
    assert not core[4, 5]
    assert not core[4, 6]


def test_sparse_graph_prefers_consistent_gaussian_identity():
    observations = [
        observation(0, 0, 1, [1, 2, 3, 4]),
        observation(1, 0, 2, [20, 21, 22]),
        observation(2, 1, 1, [1, 2, 3, 5]),
        observation(3, 1, 2, [20, 21, 23]),
    ]
    config = RobustAssociationConfig(
        view_neighbors=1,
        candidate_threshold=0.01,
        match_threshold=0.05,
        match_margin=0.0,
        min_track_views=2,
    )
    candidates, selected, _ = build_association_graph(
        [observations[:2], observations[2:]],
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        config=config,
        scene_scale=10.0,
    )
    selected_pairs = {(edge.source, edge.target) for edge in selected}
    assert selected_pairs == {(0, 2), (1, 3)}
    tracks = cluster_observations(
        observations,
        candidates,
        selected,
        config=config,
    )
    confirmed = [track for track in tracks if track.status == "confirmed"]
    assert len(confirmed) == 2
    assert {tuple(track.node_ids) for track in confirmed} == {(0, 2), (1, 3)}


def test_single_view_false_positive_is_not_confirmed():
    observations = [
        observation(0, 0, 1, [1, 2]),
        observation(1, 1, 1, [1, 2]),
        observation(2, 1, 2, [99], quality=0.1),
    ]
    selected = [AssociationEdge(0, 1, 0, 1, 0.9, 0.8, 0.8, 0.8, 0.8, 1.0, True)]
    cluster_observations(
        observations,
        selected,
        selected,
        config=RobustAssociationConfig(min_track_views=2),
    )
    assert observations[0].track_id == observations[1].track_id > 0
    assert observations[2].track_id == 0
    assert observations[2].status == "rejected"


def test_tentative_track_uses_neighbor_propagation():
    observations = [
        observation(0, 0, 1, [0, 1, 2]),
        observation(1, 1, 1, [0, 1, 3]),
        observation(2, 2, 1, [0, 2], quality=0.7),
    ]
    selected = [AssociationEdge(0, 1, 0, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, True)]
    candidates = selected + [
        AssociationEdge(2, 0, 2, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7),
        AssociationEdge(2, 1, 2, 1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7),
    ]
    cluster_observations(
        observations,
        candidates,
        selected,
        config=RobustAssociationConfig(
            tentative_propagation_threshold=0.3,
            tentative_min_neighbor_views=2,
        ),
    )
    assert observations[2].track_id == observations[0].track_id > 0
    assert observations[2].status == "propagated"


def test_large_high_quality_tentative_track_is_promoted():
    item = observation(0, 0, 1, np.arange(64), quality=0.9)
    tracks = cluster_observations(
        [item],
        [],
        [],
        config=RobustAssociationConfig(),
    )
    assert tracks[0].status == "promoted"
    assert item.track_id == 1


def test_undersegmented_raw_mask_is_split_on_rgb_superpixels():
    height, width = 40, 80
    grid_y, grid_x = np.mgrid[2:height:4, 2:width:4]
    pixel_x = grid_x.reshape(-1).astype(np.int32)
    pixel_y = grid_y.reshape(-1).astype(np.int32)
    gaussian_ids = np.arange(pixel_x.size, dtype=np.int64)
    left_ids = gaussian_ids[pixel_x < width // 2]
    right_ids = gaussian_ids[pixel_x >= width // 2]
    first = observation(0, 0, 1, left_ids)
    second = observation(1, 1, 1, right_ids)
    first.track_id = 1
    second.track_id = 2
    first.assignment_score = second.assignment_score = 1.0
    labels, confidence = build_gaussian_consensus(
        gaussian_ids.size,
        [first, second],
        margin_threshold=0.1,
    )

    raw = np.ones((height, width), dtype=np.int32)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (20, 30, 220)
    image[:, width // 2 :] = (220, 180, 20)
    projection = ViewProjection(
        view_index=0,
        image_name="frame",
        height=height,
        width=width,
        gaussian_ids=gaussian_ids,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        depth=np.ones(gaussian_ids.size, dtype=np.float32),
    )
    merged_observation = observation(0, 0, 1, gaussian_ids)
    merged_observation.bbox = (0, 0, width, height)
    merged_observation.image_shape = (height, width)
    merged_observation.area = height * width
    merged_observation.track_id = 1
    merged_observation.assignment_score = 1.0
    refined = refine_view(
        raw,
        image,
        projection,
        [merged_observation],
        labels,
        confidence,
        config=RobustAssociationConfig(
            core_radius=0,
            split_fraction=0.2,
            split_min_seed_points=1,
            split_min_area_pixels=32,
            split_min_area_fraction=0.0,
            superpixel_size=8,
        ),
    )
    assert refined.diagnostics["split_masks"] == 1
    assert np.mean(refined.labels[:, : width // 2] == 1) > 0.95
    assert np.mean(refined.labels[:, width // 2 :] == 2) > 0.95


def test_sparse_competing_seeds_do_not_split_a_raw_mask():
    raw = np.ones((20, 20), dtype=np.int32)
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    projection = ViewProjection(
        view_index=0,
        image_name="frame",
        height=20,
        width=20,
        gaussian_ids=np.array([0, 1], dtype=np.int64),
        pixel_x=np.array([5, 15], dtype=np.int32),
        pixel_y=np.array([10, 10], dtype=np.int32),
        depth=np.ones(2, dtype=np.float32),
    )
    item = observation(0, 0, 1, [0, 1])
    item.area = 400
    item.bbox = (0, 0, 20, 20)
    item.image_shape = (20, 20)
    item.track_id = 1
    item.assignment_score = 1.0
    refined = refine_view(
        raw,
        image,
        projection,
        [item],
        np.array([1, 2], dtype=np.uint16),
        np.ones(2, dtype=np.float32),
        config=RobustAssociationConfig(
            core_radius=0,
            split_min_seed_points=8,
            split_min_area_pixels=1,
        ),
    )
    assert refined.diagnostics["split_masks"] == 0
    assert np.all(refined.labels == 1)


def test_export_is_uint16_atomic_and_preserves_previous_output(tmp_path):
    raw_dir = tmp_path / "raw_entityseg_mask"
    raw_dir.mkdir()
    cv2.imwrite(
        str(raw_dir / "frame.png"),
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8),
    )
    point_cloud = tmp_path / "point_cloud.ply"
    point_cloud.write_bytes(b"fixture")
    refined = RefinedView(
        image_name="frame",
        labels=np.array([[0, 300, 65_535], [1, 2, 3]], dtype=np.uint16),
        confidence=np.ones((2, 3), dtype=np.float32),
        valid=np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
        diagnostics={
            "split_masks": 0,
            "uncertain_masks": 1,
            "ignore_pixels": 1,
        },
    )
    tracks = [
        InstanceTrack(
            internal_id=index,
            node_ids=[],
            view_ids=[0, 1],
            quality=1.0,
            mean_edge_score=1.0,
            status="confirmed",
            global_id=global_id,
        )
        for index, global_id in enumerate((1, 2, 3, 300))
    ]
    config = RobustAssociationConfig(qa_max_ignore_fraction=0.25)
    output, backup = export_association(
        scene_dir=tmp_path,
        output_name="entityseg_mask",
        raw_mask_dir=raw_dir,
        point_cloud=point_cloud,
        refined_views=[refined],
        observations=[],
        tracks=tracks,
        candidate_edges=[],
        selected_edges=[],
        config=config,
        visualize=True,
        force=False,
        extra_diagnostics={},
    )
    assert backup is None
    mask = cv2.imread(str(output / "frame.png"), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint16
    assert int(mask.max()) == 65_535
    manifest = json.loads((output / "association" / "manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["num_mask"] == 4
    assert mask[0, 1] == 4
    qa = json.loads((output / "association" / "qa.json").read_text())
    assert qa["passed"]
    assert qa["empty_classes_after_compaction"] == 0

    second_output, backup = export_association(
        scene_dir=tmp_path,
        output_name="entityseg_mask",
        raw_mask_dir=raw_dir,
        point_cloud=point_cloud,
        refined_views=[refined],
        observations=[],
        tracks=tracks,
        candidate_edges=[],
        selected_edges=[],
        config=config,
        visualize=False,
        force=True,
        extra_diagnostics={},
    )
    assert second_output == output
    assert backup is not None and backup.is_dir()


def test_export_rejects_fragmented_labels_at_qa_gate(tmp_path):
    raw_dir = tmp_path / "raw_entityseg_mask"
    raw_dir.mkdir()
    raw = np.ones((10, 10), dtype=np.uint8)
    cv2.imwrite(str(raw_dir / "frame.png"), raw)
    point_cloud = tmp_path / "point_cloud.ply"
    point_cloud.write_bytes(b"fixture")
    yy, xx = np.indices(raw.shape)
    labels = ((xx + yy) % 2 + 1).astype(np.uint16)
    refined = RefinedView(
        image_name="frame",
        labels=labels,
        confidence=np.ones(raw.shape, dtype=np.float32),
        valid=np.ones(raw.shape, dtype=bool),
        diagnostics={"split_masks": 1, "uncertain_masks": 0, "ignore_pixels": 0},
    )
    tracks = [
        InstanceTrack(index, [], [0], 1.0, 1.0, "confirmed", global_id=index + 1)
        for index in range(2)
    ]
    with pytest.raises(RuntimeError, match="QA gate failed"):
        export_association(
            scene_dir=tmp_path,
            output_name="entityseg_mask",
            raw_mask_dir=raw_dir,
            point_cloud=point_cloud,
            refined_views=[refined],
            observations=[],
            tracks=tracks,
            candidate_edges=[],
            selected_edges=[],
            config=RobustAssociationConfig(),
            visualize=False,
            force=False,
            extra_diagnostics={},
        )
    assert not (tmp_path / "entityseg_mask").exists()


class _FakeView:
    def __init__(self, image_name: str, center) -> None:
        self.image_name = image_name
        self.camera_center = torch.tensor(center, dtype=torch.float32)

    def to(self, _device):
        return self


class _FakeProjector:
    device = torch.device("cpu")
    image_height = 4
    image_width = 8

    def __init__(self) -> None:
        self.gaussians_xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )
        self.viewpoint_camera = [
            _FakeView("frame_0", [0.0, 0.0, 0.0]),
            _FakeView("frame_1", [1.0, 0.0, 0.0]),
        ]

    def project_gaussian(self, _viewpoint):
        x = torch.tensor([1, 2, 5, 6], dtype=torch.long)
        y = torch.tensor([1, 2, 1, 2], dtype=torch.long)
        return {
            "p_proj_flatten": x * self.image_height + y,
            "p_proj_inside_indices": torch.arange(4, dtype=torch.long),
            "p_hom_z": torch.ones(4, dtype=torch.float32),
        }


def test_engine_streams_a_complete_two_view_association(tmp_path):
    images = tmp_path / "images"
    raw_masks = tmp_path / "raw_entityseg_mask"
    images.mkdir()
    raw_masks.mkdir()
    for index in range(2):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :] = (30, 80, 160)
        labels = np.ones((4, 8), dtype=np.uint8)
        assert cv2.imwrite(str(images / f"frame_{index}.png"), image)
        assert cv2.imwrite(str(raw_masks / f"frame_{index}.png"), labels)
    point_cloud = tmp_path / "point_cloud.ply"
    point_cloud.write_bytes(b"fixture")

    engine = RobustAssociationEngine(
        _FakeProjector(),
        scene_dir=tmp_path,
        images="images",
        raw_mask_dir=raw_masks,
        point_cloud=point_cloud,
        output_name="entityseg_mask",
        config=RobustAssociationConfig(
            front_percentage=1.0,
            num_patches=2,
            core_radius=0,
            view_neighbors=1,
            candidate_threshold=0.01,
            match_threshold=0.05,
            match_margin=0.0,
            min_track_views=2,
        ),
        visualize=True,
        force=False,
    )
    output, backup = engine.run()

    assert backup is None
    assert (
        cv2.imread(
            str(output / "frame_0.png"),
            cv2.IMREAD_UNCHANGED,
        ).dtype
        == np.uint16
    )
    assert np.all(cv2.imread(str(output / "frame_1.png"), cv2.IMREAD_UNCHANGED) == 1)
    manifest = json.loads((output / "association" / "manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["diagnostics"]["views"] == 2
    assert manifest["diagnostics"]["global_instances"] == 1
    graph = np.load(output / "association" / "graph_edges.npz")
    assert graph["selected"].tolist() == [True]
