from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gaussian_renderer import render
from model_io import (
    build_surface_field,
    dataset_namespace,
    load_trained_scene,
    optimization_namespace,
    pipeline_namespace,
)
from scene import GaussianModel, Scene
from semantic import GeometryEvidenceProjector
from training import SemanticGaussianTrainer
from utils.config_utils import load_config, save_config


ROOT = Path(__file__).resolve().parents[1]


def _write_blender_scene(root: Path) -> None:
    root.mkdir()
    (root / "sam_mask").mkdir()
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[..., :3] = np.array([90, 150, 210], dtype=np.uint8)
    image[..., 3] = 255
    Image.fromarray(image).save(root / "view.png")
    mask = np.ones((16, 16), dtype=np.uint8)
    mask[:, 8:] = 2
    Image.fromarray(mask).save(root / "sam_mask" / "view.png")
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = 4.0
    metadata = {
        "camera_angle_x": 0.7,
        "frames": [{"file_path": "view.png", "transform_matrix": transform.tolist()}],
    }
    (root / "transforms_train.json").write_text(json.dumps(metadata), encoding="utf8")


def test_train_checkpoint_reload_and_render(tmp_path: Path) -> None:
    source, output = tmp_path / "scene", tmp_path / "output"
    _write_blender_scene(source)
    config = load_config(ROOT / "configs" / "default.yaml")
    config["device"] = "cpu"
    config["data"].update(data_device="cpu", random_points=4, semantic_path="sam_mask")
    config["renderer"]["backend"] = "reference"
    config["optimization"]["iterations"] = 4
    config["phases"].update(semantic_from=1, joint_from=2, surface_from=3, ramp_iterations=1)
    config["density"].update(from_iter=100, until_iter=100)
    config["surface"]["enabled"] = False
    config["logging"].update(
        log_interval=1,
        profile_interval=2,
        save_iterations=[3],
    )
    config["runtime"] = {"source_path": str(source), "model_path": str(output)}
    output.mkdir()
    save_config(config, output / "config.yaml")

    dataset = dataset_namespace(config, source, output, "cpu")
    gaussians = GaussianModel(3, 16, 5, "cpu")
    scene = Scene(dataset, gaussians, shuffle=False)
    optimization = optimization_namespace(config)
    gaussians.training_setup(optimization)
    field = build_surface_field(gaussians, config)
    trainer = SemanticGaussianTrainer(
        scene,
        gaussians,
        pipeline_namespace(config),
        config,
        surface_field=field,
        policy_bank=gaussians.policy_bank,
        evidence_projector=GeometryEvidenceProjector(),
        output_path=output,
    )
    assert trainer.neighbor_index is field.neighbor_index
    assert trainer.loss_system.neighbor_index is field.neighbor_index
    assert trainer.loss_system.evidence_projector.neighbor_index is field.neighbor_index

    def save(iteration, state):
        scene.save(iteration)
        torch.save(state, output / f"chkpnt{iteration}.pth")

    result = trainer.train(save_iterations=[3], save_callback=save)
    assert result.iteration == 4
    records = {
        record["iteration"]: record
        for record in (
            json.loads(line)
            for line in (output / "training.jsonl").read_text(
                encoding="utf8"
            ).splitlines()
        )
    }
    profile_keys = {
        "time_render_ms",
        "time_surface_ms",
        "time_backward_ms",
        "time_topology_ms",
        "time_step_ms",
    }
    assert profile_keys.isdisjoint(records[1])
    assert profile_keys <= records[2].keys()
    assert profile_keys.isdisjoint(records[3])
    assert profile_keys <= records[4].keys()
    assert all(records[2][key] >= 0.0 for key in profile_keys)
    assert (output / "point_cloud" / "iteration_3" / "point_cloud.ply").is_file()
    assert (output / "chkpnt3.pth").is_file()

    loaded = load_trained_scene(output, iteration=3, device="cpu")
    camera = loaded["scene"].getTrainCameras()[0]
    package = render(
        camera,
        loaded["gaussians"],
        loaded["pipeline"],
        torch.zeros(3),
        backend="reference",
    )
    assert package["render"].shape == (3, 16, 16)
    assert package["semantic"].shape == (16, 16, 16)
    assert torch.isfinite(package["render"]).all()
    assert loaded["surface_field"].query(loaded["gaussians"].get_xyz[:2]).sdf.shape == (2,)
    assert loaded["surface_field"].neighbor_index is loaded["neighbor_index"]

    # Checkpoints are self-contained; retaining a matching PLY is optional.
    (output / "point_cloud" / "iteration_3" / "point_cloud.ply").unlink()
    checkpoint_only = load_trained_scene(output, iteration=3, device="cpu")
    assert checkpoint_only["gaussians"].get_xyz.shape[0] == result.gaussian_count

    # Reproduce the real CLI construction order and resume across a decoder.
    resumed_gaussians = GaussianModel(3, 16, 5, "cpu")
    resumed_scene = Scene(dataset, resumed_gaussians, shuffle=False)
    resumed_gaussians.training_setup(optimization)
    resumed_field = build_surface_field(resumed_gaussians, config)
    resumed_trainer = SemanticGaussianTrainer(
        resumed_scene,
        resumed_gaussians,
        pipeline_namespace(config),
        config,
        surface_field=resumed_field,
        policy_bank=resumed_gaussians.policy_bank,
        evidence_projector=GeometryEvidenceProjector(),
    )
    checkpoint = torch.load(output / "chkpnt3.pth", map_location="cpu", weights_only=True)
    start = resumed_trainer.load_state_dict(checkpoint, optimization)
    assert start == 3
    resumed = resumed_trainer.train(start_iteration=start)
    assert resumed.iteration == 4
