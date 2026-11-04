"""Protect the two-snapshot experiment and automatic hole annotation contracts."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from track_holes_headless import hole_reference, correct_prediction, review_pages
from test_joint_density import fixture, step
from source.paintmesh_joint_data import JointGaussians
from run_inpaint360_full import stage_command


def test_same_run_serialization_does_not_change_optimization_or_rng(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    _, _, source_a, new, trainable = fixture(a)
    _, _, source_b, _, _ = fixture(b)
    first = JointGaussians(source_a, new, device="cpu", trainable_mask=trainable)
    control = JointGaussians(source_b, new, device="cpu", trainable_mask=trainable)
    from edgs_inpaint_io import load_config, check_preserved
    opt = first.optimizer(load_config()["optimizer"])
    ctrl = control.optimizer(load_config()["optimizer"])
    for _ in range(2): step(first, opt); step(control, ctrl)
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with torch.no_grad(): first.save(tmp_path / "at_2.ply")
    assert random.getstate() == py_rng
    np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for _ in range(3): step(first, opt); step(control, ctrl)
    for name in first.parameter_names:
        torch.testing.assert_close(getattr(first, name), getattr(control, name), rtol=0, atol=0)
        torch.testing.assert_close(opt.state[getattr(first,name)]["exp_avg"],
                                   ctrl.state[getattr(control,name)]["exp_avg"], rtol=0, atol=0)
    with torch.no_grad(): first.save(tmp_path / "at_5.ply")
    check_preserved(source_a, tmp_path / "at_5.ply", new,
                    retained_rows=first.retained_rows.numpy(), trainable_mask=trainable)


def test_projected_hole_annotation_and_drift_correction():
    original = np.zeros((100,120,3),dtype=np.float32)
    removed = original.copy(); removed[30:60,40:80] = .6
    reference = hole_reference(original, removed)
    assert reference.shape == (100,120) and reference.sum() == 1200
    drift = np.zeros_like(reference); drift[:20,:20] = True
    corrected, stats = correct_prediction(drift, reference)
    assert stats["reset_reference"]
    np.testing.assert_array_equal(corrected, reference)
    corrected, stats = correct_prediction(reference, reference)
    assert not stats["reset_reference"] and stats["reference_coverage"] == 1
    with pytest.raises(ValueError, match="unusable"):
        hole_reference(original, original)


def test_stage_resume_uses_positional_argument_instead_of_ignored_environment():
    for script in ("run_remove.sh", "run_inpaint.sh"):
        args = ["bash", "scripts/paintmesh/" + script, "Inpaint360GS/inpaint360", "bag", "4", "36", "none", "1"]
        resumed = stage_command(args, 5)
        assert resumed[-1] == "5" and args[-1] == "1"
        assert resumed[2:7] == args[2:7]


def test_review_pages_are_written_for_every_frame(tmp_path):
    frames = [np.zeros((100,120,3),dtype=np.uint8) for _ in range(13)]
    masks = [np.zeros((100,120),dtype=np.uint8) for _ in frames]
    for mask in masks: mask[30:60,40:80] = 1
    review_pages(frames, masks, tmp_path)
    assert len(list(tmp_path.glob('frames_*.jpg'))) == 2


def test_scene_specific_camera_config_reaches_resumed_shell(tmp_path, monkeypatch):
    import run_inpaint360_full as batch
    override = tmp_path / "scene-camera.json"
    override.write_text('{}\n')
    runner = object.__new__(batch.FullJob)
    runner.env = dict(GPU="1", VIRTUAL_ADAPTIVE_CONFIG="original.json")
    runner.job = tmp_path
    runner.req = dict(run_name="test")
    runner.status = dict(scenes={"cone_red": {}})
    runner.save = lambda: None
    captured = {}
    class Process:
        pid = 123
        def wait(self): return 0
    def spawn(args, **kwargs):
        captured.update(args=args, **kwargs)
        return Process()
    monkeypatch.setattr(batch.subprocess, "Popen", spawn)
    scene = dict(name="cone_red", removal_root=str(tmp_path / "removal"),
                 inpaint_root=str(tmp_path / "inpaint"), mask_dilation=30,
                 adaptive_config=str(override))
    runner.command(scene, "removal", ["bash", "scripts/paintmesh/run_remove.sh",
        "Inpaint360GS/inpaint360", "cone_red", "4", "24", "none", "1"], start=4, end=4)
    assert captured["args"][-1] == "4"
    assert captured["env"]["VIRTUAL_ADAPTIVE_CONFIG"] == str(override)
    assert captured["env"]["GPU"] == "1"
    assert runner.env["VIRTUAL_ADAPTIVE_CONFIG"] == "original.json"


def test_new_job_materializes_recorded_scene_camera_overrides(tmp_path, monkeypatch):
    import run_inpaint360_full as batch
    from run_geometry_ablation import read, write
    monkeypatch.setattr(batch, "REPO", tmp_path)
    (tmp_path / "logs").mkdir()
    config = tmp_path / "scripts/paintmesh/configs"
    base = dict(inclination_range_deg=[30.,50.], target_roi_fraction=.08, min_context_support=.4)
    write(config / "inpaint360_adaptive180.json", base)
    write(config / "inpaint360_full_scenes.json", dict(
        bag=dict(target_ids=[36], mask_dilation=30),
        garden_toys=dict(target_ids=[14], mask_dilation=20,
                         adaptive_options=dict(inclination_range_deg=[10.,30.], min_context_support=.3))))
    job = batch.create_job()
    scenes = read(job / "request.json")["scenes"]
    assert "adaptive_config" not in scenes[0]
    actual = read(scenes[1]["adaptive_config"])
    assert actual == dict(inclination_range_deg=[10.,30.], target_roi_fraction=.08, min_context_support=.3)
    assert read(config / "inpaint360_adaptive180.json") == base
