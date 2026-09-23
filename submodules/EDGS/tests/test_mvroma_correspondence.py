from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from source.correspondence.contracts import (  # noqa: E402
    DenseMultiViewCorrespondence,
    MultiViewTracks,
    normalized_image_grid,
    normalized_to_pixel,
    pixel_to_normalized,
)
from source.correspondence.geometry import (  # noqa: E402
    project_row_vector,
    weighted_multiview_dlt,
)
from source.correspondence.mvroma_runtime import (  # noqa: E402
    InferenceGroup,
    MVRoMaRuntime,
    MVRoMaSettings,
)
from source.correspondence.local_dinov2 import LocalDINOv2  # noqa: E402
from source.correspondence.overlap import (  # noqa: E402
    build_colmap_visibility_overlap,
)
from source.correspondence.planning import (  # noqa: E402
    CameraGroup,
    OverlapPlannerSettings,
    plan_overlap_aware_groups,
)


def test_colmap_visibility_overlap_is_directed_and_name_aligned(tmp_path: Path):
    cameras = [
        SimpleNamespace(image_name="a"),
        SimpleNamespace(image_name="b.JPG"),
        SimpleNamespace(image_name="nested/c.JPG"),
    ]
    images = {
        7: SimpleNamespace(name="nested/c.JPG", point3D_ids=[3, 4, 5, 6]),
        3: SimpleNamespace(name="a.JPG", point3D_ids=[1, 2, 3, 4, -1]),
        5: SimpleNamespace(name="b.JPG", point3D_ids=[2, 3]),
    }

    estimate = build_colmap_visibility_overlap(
        cameras,
        images,
        sparse_model_path=tmp_path,
    )

    expected = np.array(
        (
            (0.0, 0.5, 0.5),
            (1.0, 0.0, 0.5),
            (0.5, 0.25, 0.0),
        )
    )
    assert np.allclose(estimate.matrix, expected)
    assert estimate.camera_names == ("a.JPG", "b.JPG", "nested/c.JPG")
    assert estimate.visible_point_counts == (4, 2, 4)


def test_overlap_planner_uses_target_coherence_instead_of_matrix_knn():
    overlap = np.full((5, 5), 0.05, dtype=np.float64)
    np.fill_diagonal(overlap, 0.0)
    overlap[0, 1] = 0.90
    overlap[0, 2] = 0.80
    overlap[0, 3] = 0.70
    overlap[1, 2] = 0.00
    overlap[1, 3] = 0.95

    plan = plan_overlap_aware_groups(
        overlap,
        OverlapPlannerSettings(
            targets_per_group=2,
            primary_group_budget=5,
            augment_reciprocity=False,
        ),
    )

    source_zero = next(group for group in plan.groups if group.source_index == 0)
    assert source_zero.target_indices == (1, 3)
    assert plan.primary_group_count == 5
    assert plan.neighbor_table().shape == (5, 2)


def test_overlap_planner_allocates_quotas_and_closes_directed_pairs():
    overlap = np.array(
        (
            (0.0, 0.9, 0.7, 0.2, 0.1),
            (0.8, 0.0, 0.6, 0.3, 0.2),
            (0.7, 0.6, 0.0, 0.8, 0.4),
            (0.2, 0.3, 0.9, 0.0, 0.8),
            (0.1, 0.2, 0.4, 0.9, 0.0),
        )
    )
    settings = OverlapPlannerSettings(
        targets_per_group=1,
        primary_group_budget=7,
        overlap_threshold=0.5,
        augment_reciprocity=True,
    )

    first = plan_overlap_aware_groups(overlap, settings)
    second = plan_overlap_aware_groups(overlap, settings)

    assert first.groups == second.groups
    assert first.primary_group_count == 7
    assert sum(first.source_quotas) == 7
    assert min(first.source_quotas) == 1
    assert first.reciprocal_group_count > 0
    assert first.reciprocity_coverage == 1.0
    selected = first.pair_counts > 0
    assert np.array_equal(selected, selected.T)


def test_overlap_planner_rejects_budget_that_omits_source_views():
    overlap = np.ones((4, 4), dtype=np.float64) - np.eye(4)
    with pytest.raises(ValueError, match="at least the camera count"):
        plan_overlap_aware_groups(
            overlap,
            OverlapPlannerSettings(
                targets_per_group=2,
                primary_group_budget=3,
            ),
        )


def test_align_corners_false_coordinate_roundtrip_and_grid():
    grid = normalized_image_grid(2, 3, device="cpu")
    expected_x = torch.tensor((-2.0 / 3.0, 0.0, 2.0 / 3.0))
    expected_y = torch.tensor((-0.5, 0.5))

    assert torch.allclose(grid[0, 0], expected_x)
    assert torch.allclose(grid[1, :, 0], expected_y)

    coordinates = torch.tensor(((0.0, 0.0), (2.0, 1.0), (1.25, 0.75)))
    normalized = pixel_to_normalized(coordinates, height=2, width=3)
    assert torch.allclose(
        normalized_to_pixel(normalized, height=2, width=3), coordinates, atol=1e-6
    )


def test_dense_correspondence_samples_shared_source_tracks():
    source = normalized_image_grid(2, 2, device="cpu")
    targets = source.unsqueeze(0).repeat(2, 1, 1, 1)
    dense = DenseMultiViewCorrespondence(
        source_coordinates=source,
        target_coordinates=targets,
        certainty_logits=torch.full((2, 1, 2, 2), 10.0),
        source_valid=torch.ones(2, 2, dtype=torch.bool),
        target_valid=torch.ones(2, 2, 2, dtype=torch.bool),
    )

    tracks = dense.sample_tracks(
        4,
        confidence_threshold=0.5,
        min_target_views=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert tracks.coordinates.shape == (4, 3, 2)
    assert tracks.valid.all()
    assert torch.allclose(tracks.coordinates[:, :1], tracks.coordinates[:, 1:])
    assert torch.all(tracks.confidence[:, 0] == 1)


def _projection(center_x: float) -> torch.Tensor:
    world_view = torch.eye(4, dtype=torch.float64)
    world_view[3, 0] = -center_x
    projection = torch.zeros(4, 4, dtype=torch.float64)
    projection[0, 0] = 1.0
    projection[1, 1] = 1.0
    projection[2, 2] = 1.0
    projection[2, 3] = 1.0
    return world_view @ projection


def test_weighted_dlt_uses_row_vector_homogeneous_column():
    projections = torch.stack([_projection(-1.0), _projection(0.0), _projection(1.0)])
    expected = torch.tensor(
        ((0.2, -0.1, 4.0, 1.0), (-0.4, 0.3, 6.0, 1.0)), dtype=torch.float64
    )
    coordinates, depth = project_row_vector(expected, projections)
    tracks = MultiViewTracks(
        coordinates=coordinates,
        confidence=torch.ones(2, 3, dtype=torch.float64),
        valid=torch.ones(2, 3, dtype=torch.bool),
        sampling_score=torch.ones(2, dtype=torch.float64),
    )
    centers = torch.tensor(
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        dtype=torch.float64,
    )

    result = weighted_multiview_dlt(
        projections,
        tracks,
        camera_centers=centers,
        min_views=3,
        max_reprojection_error=1e-7,
        min_triangulation_angle_deg=0.1,
    )

    assert torch.all(depth > 0)
    assert result.accepted.tolist() == [True, True]
    assert torch.allclose(result.points_h, expected, atol=1e-7, rtol=1e-7)


def test_weighted_dlt_drops_one_bad_target_when_enough_views_remain():
    projections = torch.stack(
        [_projection(-1.5), _projection(-0.5), _projection(0.5), _projection(1.5)]
    )
    point = torch.tensor(((0.1, 0.2, 5.0, 1.0),), dtype=torch.float64)
    coordinates, _ = project_row_vector(point, projections)
    coordinates[:, 3, 0] += 0.5
    tracks = MultiViewTracks(
        coordinates=coordinates,
        confidence=torch.ones(1, 4, dtype=torch.float64),
        valid=torch.ones(1, 4, dtype=torch.bool),
        sampling_score=torch.ones(1, dtype=torch.float64),
    )
    centers = torch.tensor(
        ((-1.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (1.5, 0.0, 0.0)),
        dtype=torch.float64,
    )

    result = weighted_multiview_dlt(
        projections,
        tracks,
        camera_centers=centers,
        min_views=3,
        max_reprojection_error=1e-6,
        reject_outliers=True,
    )

    assert result.accepted.item()
    assert not result.valid_observations[0, 3]
    assert torch.allclose(result.points_h, point, atol=1e-7, rtol=1e-7)


def test_runtime_releases_prematcher_before_building_dense_model(tmp_path: Path):
    root = tmp_path / "MV-RoMa"
    (root / "src").mkdir(parents=True)
    (root / "src" / "build_model.py").write_text("# test checkout\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test")
    dinov2_root = tmp_path / "DINOv2"
    (dinov2_root / "dinov2" / "hub").mkdir(parents=True)
    (dinov2_root / "hubconf.py").write_text("# test hub\n", encoding="utf-8")
    (dinov2_root / "dinov2" / "hub" / "backbones.py").write_text(
        "# test backbones\n", encoding="utf-8"
    )
    dinov2_checkpoint = tmp_path / "dinov2.pth"
    dinov2_checkpoint.write_bytes(b"test")
    settings = MVRoMaSettings(
        root=root,
        checkpoint=checkpoint,
        dinov2_root=dinov2_root,
        dinov2_checkpoint=dinov2_checkpoint,
    )
    events: list[str] = []

    def prematcher_factory(_name, _device):
        events.append("prematcher")
        return object()

    def track_provider(_prematcher, _group, _settings, _device):
        events.append("tracks")
        return torch.zeros(1, 2, 4)

    def model_factory(_settings, _device):
        events.append("model")
        return torch.nn.Identity()

    def dense_runner(_model, _prepared, _settings, _device):
        events.append("dense")
        source = normalized_image_grid(1, 1, device="cpu")
        return DenseMultiViewCorrespondence(
            source_coordinates=source,
            target_coordinates=source.unsqueeze(0),
            certainty_logits=torch.ones(1, 1, 1, 1),
            source_valid=torch.ones(1, 1, dtype=torch.bool),
            target_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        )

    group = InferenceGroup(
        cameras=CameraGroup(0, (1,)),
        source_path=tmp_path / "a.png",
        target_paths=(tmp_path / "b.png",),
    )
    with MVRoMaRuntime(
        settings,
        "cpu",
        prematcher_factory=prematcher_factory,
        track_provider=track_provider,
        model_factory=model_factory,
        dense_runner=dense_runner,
    ) as runtime:
        prepared = runtime.prepare_tracks([group])
        assert events == ["prematcher", "tracks"]
        prediction = runtime.predict_dense(prepared[0])
        assert prediction.target_count == 1

    assert events == ["prematcher", "tracks", "model", "dense"]


def test_uniception_dinov2_request_is_redirected_to_strict_local_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "DINOv2"
    source_file = root / "dinov2" / "models" / "vision_transformer.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# test source\n", encoding="utf-8")
    (root / "dinov2" / "hub").mkdir(parents=True)
    (root / "hubconf.py").write_text("# test hub\n", encoding="utf-8")
    (root / "dinov2" / "hub" / "backbones.py").write_text(
        "# test backbones\n", encoding="utf-8"
    )
    expected = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "dinov2_vitl14_pretrain.pth"
    torch.save(expected.state_dict(), checkpoint)
    loader = LocalDINOv2(root=root, checkpoint=checkpoint)
    loader.validate()

    calls = []

    def fake_hub_load(repo_or_dir, model_name, *args, **kwargs):
        calls.append((str(repo_or_dir), model_name, args, kwargs))
        assert str(repo_or_dir) == str(root.resolve())
        assert model_name == "dinov2_vitl14"
        assert kwargs["source"] == "local"
        assert kwargs["pretrained"] is False
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    monkeypatch.setattr(
        "source.correspondence.local_dinov2.inspect.getfile",
        lambda _class: str(source_file),
    )

    with loader.redirect_uniception_hub_call():
        model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            force_reload=True,
        )

    assert torch.hub.load is fake_hub_load
    assert len(calls) == 1
    for name, value in expected.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)
