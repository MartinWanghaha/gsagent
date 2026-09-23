from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from source.correspondence.config import (  # noqa: E402
    MVRoMaTrainingSettings,
    PairStoreSettings,
    resolve_group_budget,
)
from source.correspondence.contracts import (  # noqa: E402
    DenseMultiViewCorrespondence,
    DirectedPair,
    ImageSamplingGeometry,
    PairQuality,
    normalized_image_grid,
)
from source.correspondence.pair_store import PairFieldStore  # noqa: E402
from source.correspondence.pgsr_graph import build_hybrid_neighbor_map  # noqa: E402
from source.correspondence.manifest import write_mvroma_artifacts  # noqa: E402
from source.correspondence.planning import CameraGroup, CameraGroupPlan  # noqa: E402
from source.correspondence.postprocess import (  # noqa: E402
    apply_reciprocal_cycle_filter,
    grid_balanced_spatial_nms,
    merge_dense_group,
    spatial_nms,
)
from source.correspondence.seed_fusion import GaussianSeedBatch  # noqa: E402


def _dense(warp: torch.Tensor, probability: torch.Tensor):
    height, width = probability.shape
    logits = torch.logit(probability.clamp(1e-5, 1 - 1e-5))[None, None]
    return DenseMultiViewCorrespondence(
        source_coordinates=normalized_image_grid(height, width, device="cpu"),
        target_coordinates=warp[None],
        certainty_logits=logits,
        source_valid=torch.ones(height, width, dtype=torch.bool),
        target_valid=torch.ones(1, height, width, dtype=torch.bool),
        source_geometry=ImageSamplingGeometry.identity(height, width),
        target_geometries=(ImageSamplingGeometry.identity(height, width),),
    )


@pytest.mark.parametrize("mode", ["memory", "mmap"])
def test_pair_merge_keeps_pixelwise_highest_confidence(tmp_path, mode):
    source = normalized_image_grid(2, 2, device="cpu")
    first_warp = source.clone()
    second_warp = source.clone()
    second_warp[0, 0, 0] += 0.25
    first_probability = torch.full((2, 2), 0.8)
    second_probability = torch.full((2, 2), 0.2)
    second_probability[0, 0] = 0.9
    settings = PairStoreSettings(
        mode=mode,
        max_ram_gb=1,
        temp_dir=tmp_path,
    )

    with PairFieldStore(settings, expected_pair_count=1) as store:
        merge_dense_group(store, CameraGroup(0, (1,)), _dense(first_warp, first_probability))
        merge_dense_group(store, CameraGroup(0, (1,)), _dense(second_warp, second_probability))
        field = store.read(DirectedPair(0, 1))
        assert field.target_coordinates[0, 0, 0] == second_warp[0, 0, 0]
        assert torch.allclose(
            field.target_coordinates[:, 1, 1], first_warp[:, 1, 1]
        )
        assert field.confidence[0, 0] == pytest.approx(0.9, abs=1e-5)


def test_reciprocal_cycle_uses_pixel_threshold():
    height = width = 9
    grid = normalized_image_grid(height, width, device="cpu")
    geometry = ImageSamplingGeometry.identity(height, width)
    settings = PairStoreSettings(mode="memory", max_ram_gb=1)
    store = PairFieldStore(settings, expected_pair_count=2)
    valid = torch.ones(height, width, dtype=torch.bool)
    confidence = torch.full((height, width), 0.9)
    store.merge(
        DirectedPair(0, 1),
        source_coordinates=grid,
        target_coordinates=grid,
        confidence=confidence,
        valid=valid,
        source_geometry=geometry,
    )
    shifted_reverse = grid.clone()
    shifted_reverse[0] += 4.0 * 2.0 / width
    store.merge(
        DirectedPair(1, 0),
        source_coordinates=grid,
        target_coordinates=shifted_reverse,
        confidence=confidence,
        valid=valid,
        source_geometry=geometry,
    )

    qualities = apply_reciprocal_cycle_filter(
        store,
        max_error_px=3.0,
        device="cpu",
    )

    assert not store.read(DirectedPair(0, 1)).valid.any()
    assert qualities[DirectedPair(0, 1)].valid_pixels == 0
    store.close()


def test_cycle_pixel_scale_tracks_mvroma_output_not_original_resolution():
    geometry = ImageSamplingGeometry(
        original_height=1080,
        original_width=1920,
        storage_height=560,
        storage_width=840,
    )
    one_output_pixel = torch.tensor((2.0 / 840, 2.0 / 560))

    converted = geometry.normalized_delta_to_storage_pixels(one_output_pixel)

    assert torch.allclose(converted, torch.ones(2))


def test_spatial_nms_is_deterministic_on_plateaus():
    score = torch.ones(1, 6)
    first_indices, first_scores = spatial_nms(score, radius_px=1)
    second_indices, second_scores = spatial_nms(score, radius_px=1)

    assert first_indices.tolist() == [0, 2, 4]
    assert torch.equal(first_indices, second_indices)
    assert torch.equal(first_scores, second_scores)


def test_grid_balanced_nms_reserves_budget_for_each_occupied_tile():
    score = torch.tensor(
        (
            (10.0, 9.0, 8.0, 7.0, 2.0, 1.9, 1.8, 1.7),
            (6.0, 5.0, 4.0, 3.0, 1.6, 1.5, 1.4, 1.3),
        )
    )

    selected, _ = grid_balanced_spatial_nms(
        score,
        radius_px=0,
        grid_size_px=4,
        max_points=2,
    )

    assert sorted((selected % score.shape[1] // 4).tolist()) == [0, 1]


def test_vectorized_mvroma_medoid_snap_is_deterministic():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "submodules"
        / "MV-RoMa"
        / "src"
        / "track_cluster.py"
    )
    spec = importlib.util.spec_from_file_location("paintmesh_track_cluster", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = torch.tensor(((0.0,), (1.0,), (10.0,), (11.0,)))
    centers = torch.tensor(((0.5,), (10.5,), (50.0,)))

    snapped = module._snap_centroids_to_assigned_samples(samples, centers)

    assert snapped[:, 0].tolist() == [0.0, 10.0, 50.0]


def test_group_budget_policies_are_explicit():
    assert resolve_group_budget("one_per_source", 16) == 16
    assert resolve_group_budget("paper_half", 16) == 32
    assert resolve_group_budget("paper_full", 16) == 64
    assert resolve_group_budget("fixed", 16, 20) == 20


def test_hybrid_pgsr_graph_only_reranks_pose_valid_candidates():
    cameras = [SimpleNamespace(image_name=name) for name in ("a", "b", "c")]
    pose = {"a": [cameras[1], cameras[2]], "b": [], "c": []}
    overlap = np.array(
        ((0.0, 0.4, 0.9), (0.4, 0.0, 0.0), (0.9, 0.0, 0.0))
    )
    quality = {
        DirectedPair(0, 1): PairQuality(0.8, 0.8, 10),
        DirectedPair(1, 0): PairQuality(0.8, 0.8, 10),
        DirectedPair(0, 2): PairQuality(0.9, 0.9, 10),
        DirectedPair(2, 0): PairQuality(0.9, 0.9, 10),
    }
    settings = MVRoMaTrainingSettings(
        pgsr_neighbor_strategy="hybrid",
        min_overlap=0.1,
        max_neighbors=2,
        pose_rank_weight=0.0,
    )

    result = build_hybrid_neighbor_map(cameras, pose, overlap, quality, settings)

    assert [camera.image_name for camera in result["a"]] == ["c", "b"]
    assert result["b"] == []
    assert result["c"] == []


def test_manifest_validator_rejects_modified_sidecars(tmp_path):
    cameras = [
        SimpleNamespace(
            image_name=f"{index}.png",
            image_width=3,
            image_height=3,
            full_proj_transform=torch.eye(4),
        )
        for index in range(2)
    ]
    groups = (CameraGroup(0, (1,)), CameraGroup(1, (0,)))
    plan = CameraGroupPlan(
        groups=groups,
        primary_group_count=2,
        source_quotas=(1, 1),
        pair_counts=np.asarray(((0, 1), (1, 0)), dtype=np.int32),
        mean_primary_source_overlap=0.8,
        minimum_primary_source_overlap=0.8,
    )
    mvroma_config = {
        "schema_version": 2,
        "algorithm_version": "paper-v1",
        "group_planner": {"budget_policy": "paper_half"},
    }
    write_mvroma_artifacts(
        tmp_path,
        mvroma_config=mvroma_config,
        cameras=cameras,
        overlap_matrix=np.asarray(((0.0, 0.8), (0.8, 0.0))),
        group_plan=plan,
        pair_quality={},
        diagnostics={},
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "init_wC": {
                    "use": True,
                    "backend": "mvroma",
                    "mvroma": mvroma_config,
                }
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "tools"
            / "validate_correspondence_contract.py"
        ),
        "--config",
        str(config),
        "--backend",
        "mvroma",
    ]
    subprocess.run(arguments, check=True, capture_output=True, text=True)

    groups_path = tmp_path / "correspondence_init" / "groups.json"
    groups_path.write_text("{}\n", encoding="utf-8")
    failed = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "SHA-256 mismatch" in failed.stderr


def test_manifest_serializes_omegaconf_lists_as_json_arrays(tmp_path):
    cameras = [
        SimpleNamespace(
            image_name=f"{index}.png",
            image_width=3,
            image_height=3,
            full_proj_transform=torch.eye(4),
        )
        for index in range(2)
    ]
    plan = CameraGroupPlan(
        groups=(CameraGroup(0, (1,)), CameraGroup(1, (0,))),
        primary_group_count=2,
        source_quotas=(1, 1),
        pair_counts=np.asarray(((0, 1), (1, 0)), dtype=np.int32),
        mean_primary_source_overlap=1.0,
        minimum_primary_source_overlap=1.0,
    )
    config = OmegaConf.create(
        {
            "schema_version": 2,
            "algorithm_version": "paper-v1",
            "coarse_resolution": [560, 560],
            "group_planner": {"budget_policy": "one_per_source"},
        }
    )

    write_mvroma_artifacts(
        tmp_path,
        mvroma_config=config,
        cameras=cameras,
        overlap_matrix=np.asarray(((0.0, 1.0), (1.0, 0.0))),
        group_plan=plan,
        pair_quality={},
        diagnostics={},
    )

    payload = json.loads(
        (tmp_path / "correspondence_init" / "manifest.json").read_text()
    )
    assert payload["identity"]["config"]["coarse_resolution"] == [560, 560]


def test_manifest_validator_accepts_legacy_stringified_lists(tmp_path):
    cameras = [
        SimpleNamespace(
            image_name=f"{index}.png",
            image_width=3,
            image_height=3,
            full_proj_transform=torch.eye(4),
        )
        for index in range(2)
    ]
    plan = CameraGroupPlan(
        groups=(CameraGroup(0, (1,)), CameraGroup(1, (0,))),
        primary_group_count=2,
        source_quotas=(1, 1),
        pair_counts=np.asarray(((0, 1), (1, 0)), dtype=np.int32),
        mean_primary_source_overlap=1.0,
        minimum_primary_source_overlap=1.0,
    )
    mvroma_config = {
        "schema_version": 2,
        "algorithm_version": "paper-v1",
        "coarse_resolution": [560, 560],
        "group_planner": {"budget_policy": "one_per_source"},
    }
    write_mvroma_artifacts(
        tmp_path,
        mvroma_config=mvroma_config,
        cameras=cameras,
        overlap_matrix=np.asarray(((0.0, 1.0), (1.0, 0.0))),
        group_plan=plan,
        pair_quality={},
        diagnostics={},
    )

    manifest_path = tmp_path / "correspondence_init" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["config"]["coarse_resolution"] = "[560, 560]"
    identity_bytes = json.dumps(
        manifest["identity"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    manifest["artifact_id"] = hashlib.sha256(identity_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "init_wC": {
                    "use": True,
                    "backend": "mvroma",
                    "mvroma": mvroma_config,
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "validate_correspondence_contract.py"
            ),
            "--config",
            str(config_path),
            "--backend",
            "mvroma",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_paper_backend_orchestrates_merge_cycle_and_shared_source_budget(
    tmp_path, monkeypatch
):
    import source.correspondence.mvroma_backend as backend
    from source.correspondence.contracts import MVRoMaInitializationResult
    from source.correspondence.overlap import OverlapEstimate
    from source.correspondence.planning import CameraGroupPlan

    cameras = [
        SimpleNamespace(
            image_name=f"{index}.png",
            original_image=torch.full((3, 3, 3), 0.5),
            image_width=3,
            image_height=3,
            full_proj_transform=torch.eye(4),
            camera_center=torch.tensor((float(index), 0.0, 0.0)),
        )
        for index in range(3)
    ]
    groups = (
        CameraGroup(0, (1, 2)),
        CameraGroup(1, (0, 2)),
        CameraGroup(2, (0, 1)),
    )
    pair_counts = np.ones((3, 3), dtype=np.int32) - np.eye(3, dtype=np.int32)
    plan = CameraGroupPlan(
        groups=groups,
        primary_group_count=3,
        source_quotas=(1, 1, 1),
        pair_counts=pair_counts,
        mean_primary_source_overlap=0.8,
        minimum_primary_source_overlap=0.8,
    )
    overlap = OverlapEstimate(
        matrix=pair_counts.astype(np.float64) * 0.8,
        camera_names=tuple(camera.image_name for camera in cameras),
        visible_point_counts=(10, 10, 10),
        sparse_model_path=tmp_path,
    )

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prepare_tracks(self, inference_groups):
            return [
                SimpleNamespace(
                    group=group,
                    paired_tracks=torch.zeros(len(group.target_paths), 1, 4),
                )
                for group in inference_groups
            ]

        def predict_dense(self, prepared):
            grid = normalized_image_grid(3, 3, device="cpu")
            count = len(prepared.group.cameras.target_indices)
            return DenseMultiViewCorrespondence(
                source_coordinates=grid,
                target_coordinates=grid[None].repeat(count, 1, 1, 1),
                certainty_logits=torch.full((count, 1, 3, 3), 10.0),
                source_valid=torch.ones(3, 3, dtype=torch.bool),
                target_valid=torch.ones(count, 3, 3, dtype=torch.bool),
                source_geometry=ImageSamplingGeometry.identity(3, 3),
                target_geometries=tuple(
                    ImageSamplingGeometry.identity(3, 3) for _ in range(count)
                ),
            )

    monkeypatch.setattr(backend, "MVRoMaRuntime", FakeRuntime)
    monkeypatch.setattr(
        backend,
        "_build_camera_group_plan",
        lambda *args, **kwargs: (overlap, plan),
    )

    def fake_triangulate(candidate_groups, _cameras, _settings, **kwargs):
        if not candidate_groups:
            return None
        source = candidate_groups[0].group.source_index
        return GaussianSeedBatch(
            xyz=torch.tensor(((float(source), 0.0, 1.0),)),
            rgb=torch.full((1, 3), 0.5),
            distance_to_source=torch.ones(1),
            reprojection_error=torch.zeros(1),
            triangulation_angle_deg=torch.ones(1),
            sampling_score=torch.ones(1),
            source_index=torch.tensor((source,)),
            group_index=torch.tensor((source,)),
            source_pixel_index=torch.tensor((0,)),
        )

    monkeypatch.setattr(backend, "triangulate_source_candidates", fake_triangulate)
    monkeypatch.setattr(
        backend,
        "_append_gaussian_seeds",
        lambda _gaussians, batches, **kwargs: sum(batch.xyz.shape[0] for batch in batches),
    )
    config = {
        "matches_per_ref": 4,
        "num_refs": 3,
        "scaling_factor": 0.001,
        "proj_err_tolerance": 0.01,
        "mvroma": {
            "root": str(tmp_path / "mvroma"),
            "checkpoint": str(tmp_path / "mvroma.pth"),
            "dinov2_root": str(tmp_path / "dinov2"),
            "dinov2_checkpoint": str(tmp_path / "dinov2.pth"),
            "postprocess": {
                "mode": "paper",
                "confidence_threshold": 0.3,
                "cycle_threshold_px": 3.0,
                "nms_radius_px": 1,
                "min_target_views": 2,
                "pair_store": {"mode": "memory"},
            },
            "seed_budget": {"per_source": 4},
            "group_planner": {"overlap_provider": "colmap_tracks"},
            "training": {"pgsr_neighbor_strategy": "pose"},
        },
    }
    scene = SimpleNamespace(getTrainCameras=lambda: cameras)

    result = backend.init_gaussians_with_mvroma(object(), scene, config, "cpu")

    assert isinstance(result, MVRoMaInitializationResult)
    assert result.diagnostics["accepted_points"] == 3
    assert result.diagnostics["cycle_valid_pairs"] == 6
