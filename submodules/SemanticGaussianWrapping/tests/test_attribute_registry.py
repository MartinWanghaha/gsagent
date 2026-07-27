from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.dataset_readers import BasicPointCloud
from scene.gaussian_attributes import GaussianAttributeRegistry
from scene.gaussian_model import GaussianModel, _exponential_lr
from scene import Scene


def test_registry_clone_prune_updates_every_attribute_and_adam_state():
    registry = GaussianAttributeRegistry()
    registry.register("xyz", torch.arange(9, dtype=torch.float32).reshape(3, 3), optimizer_group="xyz")
    registry.register("dynamic_feature", torch.arange(6, dtype=torch.float32).reshape(3, 2), optimizer_group="dynamic")
    registry.register("evidence", torch.arange(3, dtype=torch.float32)[:, None], trainable=False)
    optimizer = torch.optim.Adam(
        [
            {"params": [registry["xyz"]], "lr": 1e-2, "name": "xyz"},
            {"params": [registry["dynamic_feature"]], "lr": 1e-2, "name": "dynamic"},
        ]
    )
    registry.bind_optimizer(optimizer)
    (registry["xyz"].sum() + registry["dynamic_feature"].sum()).backward()
    optimizer.step()

    registry.clone(torch.tensor([False, True, False]))
    assert len(registry) == 4
    assert registry["evidence"].shape == (4, 1)
    assert torch.equal(registry["evidence"][-1], registry["evidence"][1])
    for name in ("xyz", "dynamic_feature"):
        parameter = registry[name]
        assert any(parameter is item for group in optimizer.param_groups for item in group["params"])
        assert optimizer.state[parameter]["exp_avg"].shape[0] == 4
        assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"][-1]) == 0

    registry.prune(torch.tensor([1]))
    assert len(registry) == 3
    assert all(value.shape[0] == 3 for _, value in registry.named_attributes())


def test_position_schedule_has_no_implicit_full_training_warmup():
    initial, final = 1.6e-4, 1.6e-6
    assert _exponential_lr(0, initial, final, 30_000, 0.01, 0) == pytest.approx(initial)
    assert _exponential_lr(30_000, initial, final, 30_000, 0.01, 0) == pytest.approx(final)
    assert _exponential_lr(0, initial, final, 30_000, 0.01, 1_000) == pytest.approx(initial * 0.01)
    assert _exponential_lr(1_000, initial, final, 30_000, 0.01, 1_000) > initial * 0.5


def test_feature_dc_and_rest_learning_rates_are_independent() -> None:
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    args = SimpleNamespace(
        feature_dc_lr=0.003,
        feature_rest_lr=0.00007,
    )
    model.training_setup(args)
    rates = {group["name"]: group["lr"] for group in model.optimizer.param_groups}
    assert rates["f_dc"] == pytest.approx(0.003)
    assert rates["f_rest"] == pytest.approx(0.00007)


def test_single_attribute_replace_preserves_sibling_parameters_and_resets_moments():
    registry = GaussianAttributeRegistry()
    registry.register("xyz", torch.randn(3, 3), optimizer_group="xyz")
    registry.register("opacity", torch.randn(3, 1), optimizer_group="opacity")
    optimizer = torch.optim.Adam(
        [
            {"params": [registry["xyz"]], "lr": 1e-2, "name": "xyz"},
            {"params": [registry["opacity"]], "lr": 1e-2, "name": "opacity"},
        ]
    )
    registry.bind_optimizer(optimizer)
    (registry["xyz"].sum() + registry["opacity"].sum()).backward()
    optimizer.step()
    old_xyz = registry["xyz"]
    old_opacity = registry["opacity"]

    replacement = registry.replace(
        "opacity",
        torch.zeros_like(old_opacity),
        reset_optimizer_moments=True,
    )

    assert registry["xyz"] is old_xyz
    assert replacement is registry["opacity"] and replacement is old_opacity
    assert any(replacement is item for group in optimizer.param_groups for item in group["params"])
    assert torch.count_nonzero(optimizer.state[replacement]["exp_avg"]) == 0
    assert torch.count_nonzero(optimizer.state[replacement]["exp_avg_sq"]) == 0


def test_semantic_lift_keeps_rgb_and_semantic_parameters_trainable():
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    cloud = BasicPointCloud(
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        colors=np.full((2, 3), 0.5, dtype=np.float32),
        normals=np.zeros((2, 3), dtype=np.float32),
    )
    model.create_from_pcd(cloud, 1.0)
    model.set_training_stage("semantic_lift")
    assert all(parameter.requires_grad for parameter in model.render_parameters())
    assert model.semantic_embedding.requires_grad
    assert not model.geometry_logits.requires_grad


def test_registry_ply_round_trip_preserves_dynamic_schema(tmp_path):
    registry = GaussianAttributeRegistry()
    registry.register("xyz", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    registry.register("custom_code", torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), trainable=False)
    registry.register("semantic_embedding", torch.randn(2, 7), ply_prefix="semantic")
    path = tmp_path / "point_cloud.ply"
    registry.save_ply(path)

    restored = GaussianAttributeRegistry()
    restored.load_ply(path, device="cpu")
    assert restored.names == registry.names
    assert restored.specs["custom_code"].trailing_shape == (2,)
    for name in registry.names:
        assert torch.allclose(restored[name].float(), registry[name].detach().float())


def test_gaussian_model_combined_topology_is_based_on_old_indices():
    cloud = BasicPointCloud(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        colors=np.full((4, 3), 0.5, dtype=np.float32),
        normals=np.zeros((4, 3), dtype=np.float32),
    )
    model = GaussianModel(2, semantic_dim=8, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    args = SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=1.6e-4,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=100,
        feature_lr=2.5e-3,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        semantic_lr=0.0025,
        geometry_lr=0.001,
    )
    model.training_setup(args)
    with torch.no_grad():
        model.registry["semantic_confidence"].copy_(
            torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        )
        model.registry["propagated_semantic_confidence"].copy_(
            torch.tensor([[0.5], [0.6], [0.7], [0.8]])
        )
    old_semantic = model.semantic_embedding.detach().clone()
    old_confidence = model.semantic_confidence.detach().clone()
    old_propagated = model.propagated_semantic_confidence.detach().clone()
    result = model.mutate_topology(
        clone_mask=torch.tensor([True, False, False, False]),
        split_mask=torch.tensor([False, True, False, False]),
        prune_mask=torch.tensor([False, False, True, False]),
        children=2,
        offsets=torch.zeros(1, 2, 3),
    )
    assert result == {
        "old_size": 4,
        "new_size": 5,
        "survived": 2,
        "cloned": 1,
        "split_parents": 1,
        "split_children": 2,
        "pruned": 2,
    }
    assert all(value.shape[0] == 5 for _, value in model.registry.named_attributes())
    # survivor old 0/3, clone old 0, then two children of old 1
    assert torch.allclose(model.semantic_embedding, old_semantic[[0, 3, 0, 1, 1]])
    assert torch.allclose(model.semantic_embedding[2], old_semantic[0])
    assert torch.allclose(model.semantic_embedding[3], old_semantic[1])
    assert torch.equal(model.semantic_confidence, old_confidence[[0, 3, 0, 1, 1]])
    assert torch.equal(
        model.propagated_semantic_confidence,
        old_propagated[[0, 3, 0, 1, 1]],
    )
    assert model.topology_generation == 1
    # Three rows were created (one clone and two split children), while two
    # source rows were removed. Churn counts changed slots without double
    # counting both sides of the transaction.
    assert model.cumulative_topology_churn == 3
    assert model.topology_stamp.gaussian_count == 5


def test_zero_net_growth_prune_and_replace_advances_topology_stamp() -> None:
    cloud = BasicPointCloud(
        points=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        colors=np.full((4, 3), 0.5, dtype=np.float32),
        normals=np.zeros((4, 3), dtype=np.float32),
    )
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)

    result = model.mutate_topology(
        clone_mask=torch.tensor([True, False, False, False]),
        split_mask=None,
        prune_mask=torch.tensor([False, True, False, False]),
    )

    assert result["old_size"] == result["new_size"] == 4
    assert model.topology_generation == 1
    assert model.cumulative_topology_churn == 1
    assert model.topology_stamp.gaussian_count == 4

    # An identity transaction does not manufacture a new topology version.
    model.mutate_topology(None, None, None)
    assert model.topology_generation == 1
    assert model.cumulative_topology_churn == 1


def test_topology_stamp_round_trips_and_legacy_snapshots_default_to_zero() -> None:
    cloud = BasicPointCloud(
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        colors=np.full((2, 3), 0.5, dtype=np.float32),
        normals=np.zeros((2, 3), dtype=np.float32),
    )
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.mutate_topology(
        clone_mask=torch.tensor([True, False]),
        split_mask=None,
        prune_mask=None,
    )

    for snapshot in (model.capture(), model.capture_inference("cpu")):
        assert snapshot["format_version"] == 3
        restored = GaussianModel(1, semantic_dim=4, device="cpu")
        restored.restore(snapshot)
        assert restored.topology_stamp == model.topology_stamp

    legacy = model.capture()
    legacy["format_version"] = 2
    legacy.pop("topology_generation")
    legacy.pop("cumulative_topology_churn")
    restored_legacy = GaussianModel(1, semantic_dim=4, device="cpu")
    restored_legacy.restore(legacy)
    assert restored_legacy.topology_generation == 0
    assert restored_legacy.cumulative_topology_churn == 0
    assert restored_legacy.topology_stamp.gaussian_count == 3


def test_point_cloud_and_ply_import_define_fresh_topology_baselines(tmp_path) -> None:
    cloud = BasicPointCloud(
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        colors=np.full((2, 3), 0.5, dtype=np.float32),
        normals=np.zeros((2, 3), dtype=np.float32),
    )
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    model.clone(torch.tensor([True, False]))
    assert model.topology_generation == 1

    model.create_from_pcd(cloud, 1.0)
    assert model.topology_generation == 0
    assert model.cumulative_topology_churn == 0

    model.clone(torch.tensor([True, False]))
    path = tmp_path / "model.ply"
    model.save_ply(path)
    restored = GaussianModel(1, semantic_dim=4, device="cpu")
    restored.load_ply(path)
    assert len(restored) == 3
    assert restored.topology_generation == 0
    assert restored.cumulative_topology_churn == 0


def test_colmap_scene_loads_aligned_gaga_observations(tmp_path):
    source = tmp_path / "source"
    sparse = source / "sparse" / "0"
    images = source / "images"
    masks = source / "sam_mask"
    sparse.mkdir(parents=True)
    images.mkdir()
    masks.mkdir()
    (sparse / "cameras.txt").write_text(
        "# camera\n1 PINHOLE 4 3 4 4 1.5 1\n", encoding="utf8"
    )
    (sparse / "images.txt").write_text(
        "# image\n1 1 0 0 0 0 0 0 1 frame.png\n\n", encoding="utf8"
    )
    (sparse / "points3D.txt").write_text(
        "1 0 0 1 255 0 0 0.1\n2 0.1 0 1 0 255 0 0.1\n", encoding="utf8"
    )
    Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8)).save(images / "frame.png")
    labels = np.zeros((3, 4), dtype=np.uint8)
    labels[:, 2:] = 1
    Image.fromarray(labels).save(masks / "frame.png")
    (masks / "info.json").write_text(json.dumps({"num_mask": 1}), encoding="utf8")
    args = SimpleNamespace(
        source_path=str(source),
        model_path=str(tmp_path / "output"),
        images="images",
        resolution=1,
        white_background=False,
        data_device="cpu",
        eval=False,
        llffhold=8,
        semantic_path="sam_mask",
        semantic_confidence_path="",
        semantic_boundary_path="",
        semantic_ignore_label=-1,
        semantic_background_label=0,
        semantic_temperature=0.1,
        boundary_width=1,
    )
    model = GaussianModel(1, semantic_dim=4, device="cpu")
    scene = Scene(args, model, shuffle=False)
    camera = scene.getTrainCameras()[0]
    assert camera.original_image.shape == (3, 3, 4)
    assert camera.semantic_ids.shape == (3, 4)
    assert camera.semantic_confidence[:, :2].sum() == 0  # background is not geometry evidence
    assert camera.semantic_confidence[:, 2:].min() == 1
    assert scene.num_semantic_classes == 2
    assert scene.extent > 0
