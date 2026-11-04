"""Scale regularization contract, official appearance, and paired experiment plan."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from edgs_inpaint_io import load_config
from source.paintmesh_joint_losses import joint_losses, scale_flatten_loss
from source.pgsr_losses import PGSRLossComposer

spec = importlib.util.spec_from_file_location("appearance_scale_ablation", REPO / "scripts/paintmesh/run_appearance_scale_ablation.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def original_config():
    cfg = load_config(init_depth=True, train_depth=True, train_normal=True)
    cfg["init"].update(use_depth=True, use_normal=False)
    cfg["local_background"]["enabled"] = True
    cfg["supervision"].update(use_depth=True, use_normal=True)
    cfg["loss"].update(ssim=.2, lpips=.005)
    return cfg


def test_visible_scale_gradient_matches_pgsr_and_keeps_frozen_rows():
    frozen = torch.tensor([[.1, .5, .9]])
    parameter = torch.nn.Parameter(torch.tensor([[.03, .2, .4], [.7, .06, .5]]).log())
    model = SimpleNamespace(get_scaling=torch.cat((frozen, parameter.exp())))
    pkg = dict(visibility_filter=torch.tensor([True, True, False]))
    loss = scale_flatten_loss(pkg, model)
    reference = PGSRLossComposer(dict(enabled=True), lambda_dssim=.2)._scale_term(pkg, model, loss * 0)
    torch.testing.assert_close(loss, torch.tensor(.065))
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    (100 * loss).backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([[1.5, 0., 0.], [0., 0., 0.]]))
    assert not frozen.requires_grad and frozen.grad is None


def test_empty_visibility_is_finite_zero_with_zero_gradient():
    scales = torch.ones(3, 3, requires_grad=True)
    loss = scale_flatten_loss(dict(visibility_filter=torch.zeros(3, dtype=torch.bool)), SimpleNamespace(get_scaling=scales))
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert scales.grad.count_nonzero() == 0


def test_isotropic_seed_flattens_one_axis_instead_of_shrinking_all_axes():
    scales = torch.full((2, 3), .02, requires_grad=True)
    loss = scale_flatten_loss(dict(visibility_filter=torch.ones(2, dtype=torch.bool)), SimpleNamespace(get_scaling=scales))
    (100 * loss).backward()
    assert torch.equal((scales.grad != 0).sum(dim=1), torch.ones(2, dtype=torch.int64))
    torch.testing.assert_close(scales.grad.sum(dim=1), torch.full((2,), 50.))


def test_joint_scale_loss_is_active_before_geometry_ramp_and_requires_model():
    cfg = original_config()
    cfg["loss"] = {key: 0. for key in cfg["loss"]}
    cfg["loss"]["scale_flatten"] = 100.
    rgb = torch.ones(3, 24, 24, requires_grad=True)
    pkg = dict(render=rgb, plane_depth=torch.ones(1, 24, 24), rendered_alpha=torch.ones(1, 24, 24),
               rendered_normal=torch.zeros(3, 24, 24), visibility_filter=torch.tensor([True]))
    target = dict(rgb=rgb.detach(), mask=torch.ones(24, 24, dtype=torch.bool))
    base = dict(rgb=rgb.detach(), alpha=torch.ones(24, 24))
    scales = torch.tensor([[.02, .04, .06]], requires_grad=True)
    with pytest.raises(ValueError, match="without Gaussian"):
        joint_losses(pkg, target, base, 0, cfg)
    loss, metrics = joint_losses(pkg, target, base, 0, cfg, gaussians=SimpleNamespace(get_scaling=scales))
    assert metrics["ramp"] == 0 and metrics["weight_scale_flatten"] == 100
    torch.testing.assert_close(loss, torch.tensor(2.))
    loss.backward()
    torch.testing.assert_close(scales.grad, torch.tensor([[100., 0., 0.]]))
    cfg["loss"]["scale_flatten"] = 0.
    zero, _ = joint_losses(pkg, target, base, 0, cfg)  # Legacy callers need no Gaussian object.
    assert zero == 0


def test_default_zero_and_bad_weight_rejected(tmp_path):
    assert load_config()["loss"]["scale_flatten"] == 0
    for value in (-1, float("nan"), True):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(dict(loss=dict(scale_flatten=value))))
        with pytest.raises(ValueError, match="scale_flatten"):
            load_config(path)


def test_pair_changes_only_scale_and_preserves_all_other_supervision():
    cfg = original_config(); original = copy.deepcopy(cfg)
    variants = batch.make_variants(cfg)
    assert cfg == original and len(variants) == 2
    off, on = [copy.deepcopy(v["config"]) for v in variants]
    assert off["loss"].pop("scale_flatten") == 0
    assert on["loss"].pop("scale_flatten") == 100
    assert off == on
    assert off["appearance"]["mode"] == "inpaint360gs"
    assert off["loss"]["rgb_hole"] == 0 and off["loss"]["rgb_known_preserve"] == pytest.approx(.2)
    assert off["loss"]["ssim"] == .8 and off["loss"]["lpips"] == .0005
    for key in batch.PRESERVED_LOSSES:
        assert off["loss"][key] == cfg["loss"][key]
    for key in ("init", "supervision", "local_background", "pruning", "adaptive_density", "optimizer", "iterations", "seed"):
        assert off[key] == cfg[key]


def test_report_preserves_previous_results_and_displays_on_off_delta(tmp_path):
    runner = batch.ScaleBatch.__new__(batch.ScaleBatch)
    runner.job = tmp_path / "logs"; runner.job.mkdir()
    runner.output = tmp_path / "suite"; runner.output.mkdir()
    runner.report = tmp_path / "comparison.md"; runner.report.write_text("Previous experiments.\n")
    metrics = {key: 1. for key in batch.METRICS}
    result = tmp_path / "reference.json"; result.write_text(json.dumps(dict(metrics=metrics)))
    variants = batch.make_variants(original_config())
    runner.req = dict(variants=variants, baseline_results=str(result), reference_results=str(result), paper=metrics, frame_count=58)
    runner.status = dict(state="running", phase="00: training", variants=[])
    runner.report_progress()
    for index, variant in enumerate(variants):
        runner.status["variants"].append(dict(name=variant["name"], label=variant["label"], state="complete",
            metrics={key: index + 2. for key in metrics}))
    runner.status["state"] = "complete"
    runner.report_progress()
    report = runner.report.read_text()
    assert report.startswith("Previous experiments.") and report.count("<!-- APPEARANCE_SCALE_ABLATION:logs:BEGIN -->") == 1
    assert "| masked_PSNR ↑ | 1.000000 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | +1.000000 |" in report
    assert json.loads((runner.output / "summary.json").read_text())["complete"]
    assert "100.0" in (runner.output / "summary.csv").read_text()
