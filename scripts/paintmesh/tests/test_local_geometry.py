from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from local_geometry_io import (check_preserved, gate_membership, load_config, prepare_rgb,
    read_json, record, save_rgb_receipt, validate_local, validate_rgb, verify_targets, write_json)
from source.paintmesh_local_data import LocalGaussians
from source.paintmesh_local_losses import geometry_ramp, interior_depth_mask, local_depth_normal, local_losses
from source.pgsr_geometry import depth_to_normal

CONFIG = REPO / "scripts/paintmesh/configs/local_geometry.yaml"


def gaussian_ply(path, size=3, tilt=0.):
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    fields += [f"obj_dc_{i}" for i in range(16)]
    points = np.zeros(size * size, dtype=[(name, "f4") for name in fields])
    xx, yy = np.meshgrid(np.linspace(-.8, .8, size), np.linspace(-.8, .8, size))
    points["x"], points["y"], points["z"] = xx.ravel(), yy.ravel(), 2
    points["scale_0"] = points["scale_1"] = np.log(.35)
    points["scale_2"] = np.log(.02)
    points["rot_0"], points["opacity"] = 1, 2
    points["rot_1"] = tilt
    for i in range(3):
        points[f"f_dc_{i}"] = -.2 / .28209479177387814
    for i in range(16):
        points[f"obj_dc_{i}"] = i
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(points, "vertex")]).write(str(path))
    return points


def loss_inputs():
    h, w = 7, 9
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[1:-1, 1:-1] = True
    target_normal = torch.zeros(3, h, w)
    target_normal[2] = -1
    target = dict(rgb=torch.ones(3, h, w) * .4, depth=torch.ones(h, w) * 2,
                  normal=target_normal, normal_valid=torch.ones_like(mask), mask=mask)
    camera = SimpleNamespace(image_width=w, image_height=h, FoVx=1., FoVy=1.,
                             world_view_transform=torch.eye(4))
    normal = target_normal.clone()
    normal[0] = .2
    normal.requires_grad_()
    depth = torch.full((1, h, w), 2.2, requires_grad=True)
    package = dict(render=torch.full((3, h, w), .3, requires_grad=True),
                   rendered_alpha=torch.ones(1, h, w, requires_grad=True),
                   rendered_normal=normal, plane_depth=depth,
                   depth_normal=depth_to_normal(camera, depth))
    baseline = dict(rgb=torch.full((3, h, w), .3), alpha=torch.ones(h, w))
    return package, target, baseline


def test_local_schedule_and_config():
    cfg = load_config(CONFIG)
    assert [geometry_ramp(t, cfg) for t in (99, 100, 101, 300, 500, 999)] == [0, 0, .0025, .5, 1, 1]
    for updates in (dict(iterations=10), dict(geometry_ramp_iters=0),
                    dict(geometry_from_iter=-1), dict(iterations=True)):
        with pytest.raises(ValueError):
            load_config(CONFIG, **updates)


def test_lama_normal_is_separate_and_differentiable():
    package, target, baseline = loss_inputs()
    loss, stats = local_losses(package, target, baseline, 500, load_config(CONFIG))
    assert stats["lama_normal"] > 0 and stats["depth"] > 0
    assert stats["depth_normal_consistency"] > 0
    loss.backward()
    for key in ("plane_depth", "rendered_normal"):
        assert torch.isfinite(package[key].grad).all()
        assert package[key].grad.abs().sum() > 0


def test_geometry_disabled_before_ramp_and_empty_masks_safe():
    package, target, baseline = loss_inputs()
    loss, stats = local_losses(package, target, baseline, 100, load_config(CONFIG))
    assert all(stats[key] == 0 for key in ("depth", "lama_normal", "depth_normal_consistency"))
    target["mask"].fill_(False)
    loss, stats = local_losses(package, target, baseline, 500, load_config(CONFIG))
    assert torch.isfinite(loss) and stats["normal_pixels"] == 0
    loss.backward()


def test_nan_invalid_normal_and_depth_do_not_poison_gradients():
    package, target, baseline = loss_inputs()
    with torch.no_grad():
        package["plane_depth"][0, 3, 3] = float("nan")
        package["rendered_normal"][:, 2, 2] = float("nan")
        target["normal"][:, 4, 4] = 0
    loss, stats = local_losses(package, target, baseline, 500, load_config(CONFIG))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(package["plane_depth"].grad).all()
    assert torch.isfinite(package["rendered_normal"].grad).all()


def test_low_alpha_keeps_coverage_penalty():
    package, target, baseline = loss_inputs()
    with torch.no_grad():
        package["rendered_alpha"].fill_(0)
    loss, stats = local_losses(package, target, baseline, 500, load_config(CONFIG))
    assert stats["normal_pixels"] == 0 and stats["coverage"] == 0
    assert stats["alpha_preserve"] > 0
    loss.backward()
    assert package["rendered_alpha"].grad.sum() < 0


def test_depth_edges_and_hole_border_excluded():
    depth = torch.ones(7, 7)
    depth[:, 4:] = 2
    valid = torch.ones_like(depth, dtype=torch.bool)
    interior = interior_depth_mask(depth, valid, .05)
    assert not interior[0].any() and not interior[:, 3:5].any()
    assert interior[3, 2]


def test_nonfinite_depth_is_sanitized_before_normal_differentiation():
    camera = SimpleNamespace(image_width=7, image_height=7, FoVx=1., FoVy=1.,
                             world_view_transform=torch.eye(4))
    depth = torch.full((1, 7, 7), 2., requires_grad=True)
    with torch.no_grad():
        depth[0, 3, 3] = float("nan")
    normal = local_depth_normal(camera, depth)
    normal.sum().backward()
    assert torch.isfinite(normal).all() and torch.isfinite(depth.grad).all()


def test_frozen_rows_semantics_and_adam_are_preserved(tmp_path):
    ply = tmp_path / "input.ply"
    points = gaussian_ply(ply)
    editable = np.zeros(len(points), bool)
    editable[4] = True
    model = LocalGaussians(ply, editable, device="cpu")
    optimizer = model.optimizer(load_config(CONFIG)["optimizer"])
    before = model.local_state()
    for _ in range(3):
        loss = model.get_xyz.sum() + model.get_scaling.sum() + model.get_rotation.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert all(parameter.shape[0] == 1 for parameter in model.parameters())
    assert not torch.equal(before["xyz"], model.xyz)
    output = tmp_path / "result.ply"
    model.save(output)
    check_preserved(ply, output, editable)
    np.testing.assert_array_equal(PlyData.read(str(ply))["vertex"].data, points)
    state = model.local_state()
    model.restore_rows(torch.tensor([True]))
    torch.testing.assert_close(model.xyz, model.xyz_base[model.indices])
    model.restore_local(state)
    torch.testing.assert_close(model.xyz, state["xyz"])


def test_gate_rejects_far_and_offscreen_points(tmp_path):
    support = tmp_path / "support.ply"
    gaussian_ply(support)
    gate = tmp_path / "gate.npz"
    np.savez(gate, projection=np.eye(4, dtype=np.float32), mask=np.ones((9, 9), bool),
             distance_threshold=np.array(.3))
    inside = gate_membership(np.array([[0., 0, 2], [10., 0, 2], [0., 0, 50]], np.float32), gate, support)
    assert inside.tolist() == [True, False, False]


def test_receipt_rejects_wrong_row_count_and_frozen_changes(tmp_path):
    source, output = tmp_path / "source.ply", tmp_path / "output.ply"
    points = gaussian_ply(source)
    gaussian_ply(output)
    editable = np.zeros(len(points), bool)
    editable[0] = True
    data = PlyData.read(str(output), mmap=False)
    data["vertex"].data["obj_dc_0"][0] = 10
    data.write(str(output))
    with pytest.raises(ValueError, match="frozen"):
        check_preserved(source, output, editable)
    with pytest.raises(ValueError, match="row count"):
        check_preserved(source, output, editable[:-1])


def test_opposite_normal_is_penalized_not_absolute_dot():
    package, target, baseline = loss_inputs()
    package["rendered_normal"] = -target["normal"].clone().requires_grad_()
    _, stats = local_losses(package, target, baseline, 500, load_config(CONFIG))
    torch.testing.assert_close(stats["lama_normal"], torch.tensor(2.))


def test_unknown_weights_and_nonfinite_config_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    for value in ("loss:\n  normal: 1\n", "optimizer:\n  rotation_lr: .nan\n",
                  "validity:\n  coverage_alpha_floor: 0\n"):
        path.write_text(value)
        with pytest.raises(ValueError):
            load_config(path)


def test_local_debug_default_schedule_disabled_and_images(tmp_path):
    from PIL import Image
    from source.paintmesh_local_debug import LocalGeometryDebug
    package, target, baseline = loss_inputs()
    cfg = load_config(CONFIG)
    assert cfg["debug"]["enabled"] is True
    debug = LocalGeometryDebug(tmp_path, cfg["debug"], target, .01)
    assert [debug.due(step) for step in (0, 1, 99, 100)] == [True, False, False, True]
    _, stats = local_losses(package, target, baseline, 500, cfg)
    debug.save(package, target, stats, 500)
    path = tmp_path / "debug/step_000500_view_00004.jpg"
    with Image.open(path) as image:
        assert image.size == (1120, 375)
    receipt = read_json(path.with_suffix(".json"))
    assert receipt["completed_steps"] == 500 and receipt["metrics"]["ramp"] == 1
    assert debug.depth_range == (2., 2.)
    disabled = dict(cfg["debug"], enabled=False)
    writer = LocalGeometryDebug(tmp_path / "disabled", disabled, target, .01)
    assert not writer.due(0)
    writer.save(package, target, stats, 0)
    assert not (tmp_path / "disabled").exists()


@pytest.mark.parametrize("setting", ["enabled: yesplease", "interval: 0", "from_step: -1",
                                      "view_index: 30", "jpeg_quality: 101"])
def test_invalid_debug_settings_rejected(tmp_path, setting):
    path = tmp_path / "debug.yaml"
    path.write_text("debug:\n  " + setting + "\n")
    with pytest.raises(ValueError, match="debug"):
        load_config(path)


@pytest.mark.skipif(os.environ.get("PAINTMESH_LOCAL_GPU_TEST") != "1", reason="opt-in CUDA integration")
def test_gpu_worker_30_frames_resume_and_reuse(tmp_path):
    # Uses the real existing LaMa artifact fixture, NOT the LaMa network.
    sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
    from tools.tests.test_paintmesh_normal import NormalFixture
    fixture = NormalFixture(tmp_path, frames=30, shape=(16, 16))
    fixture.prepare()
    fixture.make_valid_outputs()
    fixture.make_normal_outputs()
    fixture.validate()
    verify_targets(fixture.completion_manifest, fixture.camera_path)
    root = tmp_path / "local_geometry"
    source = tmp_path / "source.ply"
    points = gaussian_ply(source, size=5, tilt=.07)
    rgb_ply = tmp_path / "rgb.ply"
    context = root / "rgb_context.json"
    rgb_manifest = tmp_path / "manifests/rgb_finetune_manifest.json"
    classifier = tmp_path / "classifier.pth"
    classifier.write_bytes(b"frozen classifier")
    fusion, inpaint_config = tmp_path / "fusion.json", tmp_path / "inpaint.json"
    write_json(fusion, {})
    write_json(inpaint_config, {"finetune_iteration": 5})
    args = SimpleNamespace(source_ply=source, classifier=classifier, inpaint_config=inpaint_config,
        camera=fixture.camera_path, lama=fixture.completion_manifest, fusion=fusion, support=source,
        rgb_ply=rgb_ply, context=context, manifest=rgb_manifest, rgb_iterations=5, seed_frame=4)
    prepare_rgb(args)
    gaussian_ply(rgb_ply, size=5, tilt=.07)
    editable = np.zeros(len(points), bool)
    editable[6:19] = True
    save_rgb_receipt(context, rgb_ply, editable, dict(projection=np.eye(4, dtype=np.float32),
        mask=np.ones((16, 16), bool), distance_threshold=np.array(2.)))
    validate_rgb(rgb_manifest)
    edgs_config = tmp_path / "edgs.yaml"
    edgs_config.write_text("gs:\n  renderer:\n    backend: pgsr\n  dataset:\n    white_background: false\n")
    local_manifest = tmp_path / "manifests/local_geometry_manifest.json"
    local_config = tmp_path / "local.yaml"
    local_config.write_text("checkpoint_interval: 1\ndebug:\n  interval: 1\n")
    command = [sys.executable, str(REPO / "submodules/EDGS/tools/finetune_pgsr_geometry.py"),
        "--rgb-manifest", str(rgb_manifest), "--lama", str(fixture.completion_manifest),
        "--camera", str(fixture.camera_path), "--edgs-config", str(edgs_config),
        "--config", str(local_config), "--output-root", str(root), "--manifest", str(local_manifest),
        "--iterations", "3", "--geometry-from-iter", "0", "--geometry-ramp-iters", "1"]
    interrupted_launcher = """
import pathlib, runpy, sys
worker = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(worker.parents[3] / 'scripts/paintmesh'))
import local_geometry_io
original = local_geometry_io.atomic_write
def interrupted(path, write):
    original(path, write)
    if pathlib.Path(path).name == 'latest.pth':
        raise RuntimeError('simulated interruption after checkpoint')
local_geometry_io.atomic_write = interrupted
sys.argv = sys.argv[1:]
runpy.run_path(str(worker), run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", interrupted_launcher, *command[1:]],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "simulated interruption" in result.stderr, result.stdout + result.stderr
    assert not local_manifest.exists()
    checkpoint_state = torch.load(root / "checkpoints/latest.pth", map_location="cpu", weights_only=False)
    assert checkpoint_state["next_step"] == 1
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "Resuming local geometry at step 1" in result.stdout, result.stdout + result.stderr
    receipt = validate_local(local_manifest)
    assert receipt["parameters"]["local_geometry_iterations"] == 3
    assert receipt["outputs"]["ply"]["sha256"] != record(rgb_ply)["sha256"]
    for stem in ("step_000000", "step_000001", "step_000002", "step_000003", "final"):
        assert (root / "debug" / f"{stem}_view_00004.jpg").is_file()
    assert read_json(root / "debug/final_view_00004.json")["state"] == "final_gated"
    result = subprocess.run(command + ["--validate-only"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "Reused" in result.stdout, result.stdout + result.stderr
    # Simulate interruption after last checkpoint, before receipt commit.
    local_manifest.rename(local_manifest.with_suffix(".previous.json"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "Resuming" in result.stdout, result.stdout + result.stderr
    resumed = validate_local(local_manifest)
    assert resumed["outputs"]["ply"]["sha256"] == receipt["outputs"]["ply"]["sha256"]
    result = subprocess.run(command + ["--iterations", "4"], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    # Do not accept stale targets, even if a completed PLY already exists.
    missing = fixture.normal_output / "00000.npy"
    missing.rename(missing.with_suffix(".backup"))
    result = subprocess.run(command + ["--validate-only"], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
