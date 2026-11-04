"""Objective gradients and controlled experiment isolation for muti_plan."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))

from edgs_inpaint_io import load_config
from run_muti_plan import changes, make_variants, validate_evaluation
from source.paintmesh_joint_losses import joint_losses


def rgb_example():
    rgb = torch.tensor([.2, .8, .9]).repeat(3, 1).reshape(3, 1, 3).requires_grad_()
    target = dict(rgb=torch.zeros_like(rgb), mask=torch.tensor([[True, True, False]]),
                  rgb_weight=torch.tensor([[1., .1, .7]], requires_grad=True))
    package = dict(render=rgb, rendered_alpha=torch.ones(1, 1, 3),
                   plane_depth=torch.ones(1, 1, 3), rendered_normal=torch.zeros(3, 1, 3))
    baseline = dict(rgb=rgb.detach().clone(), alpha=torch.ones(1, 3))
    cfg = load_config()
    cfg["loss"] = {key: 0. for key in cfg["loss"]}
    cfg["loss"]["rgb_hole"] = 1.
    return rgb, target, package, baseline, cfg


def test_uniform_rgb_recovers_high_frequency_pixel_gradient_without_touching_outside():
    rgb, target, package, baseline, cfg = rgb_example()
    original, _ = joint_losses(package, target, baseline, 0, cfg)
    original.backward()
    expected = torch.tensor((.2 + .8 * .1) / 1.1)
    torch.testing.assert_close(original, expected)
    assert target["rgb_weight"].grad is None
    assert torch.allclose(rgb.grad[:, :, 0] / rgb.grad[:, :, 1], torch.full((3, 1), 10.))
    rgb.grad = None
    cfg["appearance"]["rgb_weight_mode"] = "uniform"
    uniform, _ = joint_losses(package, target, baseline, 0, cfg)
    uniform.backward()
    torch.testing.assert_close(uniform, torch.tensor(.5))
    torch.testing.assert_close(rgb.grad[:, :, :2], torch.full((3, 1, 2), 1 / 6))
    assert (rgb.grad[:, :, 2] == 0).all()
    assert target["rgb_weight"].grad is None


def test_legacy_missing_mode_retains_exact_weighted_objective():
    _, target, package, baseline, cfg = rgb_example()
    value, _ = joint_losses(package, target, baseline, 0, cfg)
    del cfg["appearance"]["rgb_weight_mode"]
    legacy, _ = joint_losses(package, target, baseline, 0, cfg)
    assert torch.equal(value, legacy)


def test_uniform_empty_mask_has_finite_zero_gradient():
    rgb, target, package, baseline, cfg = rgb_example()
    target["mask"][:] = False
    cfg["appearance"]["rgb_weight_mode"] = "uniform"
    value, _ = joint_losses(package, target, baseline, 0, cfg)
    value.backward()
    assert value == 0 and torch.isfinite(rgb.grad).all() and (rgb.grad == 0).all()


def test_rgb_mode_invalid_is_rejected(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"appearance": {"rgb_weight_mode": "bad"}}))
    with pytest.raises(ValueError, match="rgb_weight_mode"):
        load_config(config)


def test_matrix_only_changes_declared_objective_fields():
    cfg = load_config(train_depth=True, train_normal=True)
    original = copy.deepcopy(cfg)
    variants = make_variants(cfg)
    assert original == cfg
    control = variants[0]["config"]
    expected = [set(), {"loss.lama_normal"}, {"appearance.rgb_weight_mode"},
                {"loss.lama_normal", "appearance.rgb_weight_mode"}]
    for variant, keys in zip(variants, expected):
        v = variant["config"]
        assert set(changes(control, v)) == keys
        assert v["iterations"] == 10000 and v["pruning"]["enabled"]
        assert v["init"] == cfg["init"] and v["optimizer"] == cfg["optimizer"]
        assert v["local_background"] == cfg["local_background"]
        assert v["supervision"] == cfg["supervision"]
        assert v["loss"]["depth_normal_consistency"] > 0 and v["loss"]["depth"] > 0
        assert not v["adaptive_density"]["enabled"]
    variants[1]["config"]["loss"]["depth"] = 99
    assert variants[0]["config"]["loss"]["depth"] == cfg["loss"]["depth"]


def test_eval_rejects_other_test_frames_even_with_better_metrics():
    from run_muti_plan import METRICS
    source = dict(complete=True, protocol={"resolution": 4}, frame_count=2,
        inputs=dict(frames=["test_a", "test_b"], source_path="scene", resolution=4, renderer="pgsr",
                    reference_images={}, official_masks={}, colmap_images={}, colmap_cameras={}),
        metrics={key: 1. for key in METRICS})
    modified = copy.deepcopy(source)
    validate_evaluation(modified, source)
    modified["inputs"]["frames"] = modified["inputs"]["frames"][:-1]
    with pytest.raises(ValueError, match="frames"):
        validate_evaluation(modified, source)
