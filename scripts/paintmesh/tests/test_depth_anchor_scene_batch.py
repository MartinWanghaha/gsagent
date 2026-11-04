"""The scene queue must preserve the bag recipe and keep failed scenes out of comparison."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_depth_anchor_scene_batch as batch
from edgs_inpaint_io import load_config


def config():
    cfg = load_config()
    cfg.update(iterations=5000, freeze_background=True)
    cfg["init"].update(use_depth=True, use_normal=False)
    cfg["surface_init"]["mode"] = "depth_anchor"
    cfg["appearance"]["mode"] = "masked"
    cfg["pruning"]["enabled"] = True
    cfg["local_background"]["enabled"] = True
    cfg["adaptive_density"]["enabled"] = False
    cfg["background_writeback"]["enabled"] = False
    cfg["loss"].update(rgb_hole=1., rgb_known_preserve=1., ssim=.2, lpips=.005,
        alpha_preserve=.1, depth=.1, lama_normal=.05, depth_normal_consistency=.01, multiview=.01)
    return cfg


def test_reference_preserved_and_official_recipe_rejected():
    reference = config()
    result = batch.experiment_config(reference)
    assert result == reference and result is not reference
    result["loss"]["depth"] = 9
    assert reference["loss"]["depth"] == .1
    reference["appearance"]["mode"] = "inpaint360gs"
    with pytest.raises(ValueError):
        batch.experiment_config(reference)


def test_subset_aggregation_never_uses_unavailable_scene_or_paper_total():
    paper = {n: {k: v for k in batch.METRICS} for n, v in (("car", 10), ("truck", 20), ("toys", 999))}
    entries = [dict(name="car", frame_count=1, metrics={k: 12 for k in batch.METRICS}),
               dict(name="truck", frame_count=3, metrics={k: 22 for k in batch.METRICS})]
    summary = batch.matched_comparison(entries, paper)
    assert summary["paper_same_subset_macro"]["PSNR"] == 15
    assert summary["ours_scene_macro"]["PSNR"] == 17
    assert summary["ours_official_code_average"]["PSNR"] == 19.5
    assert summary["ours_official_code_average"]["FID"] == 17
    assert summary["delta"]["PSNR"] == 2
    assert batch.matched_comparison([], paper) is None
    with pytest.raises(ValueError, match="Duplicate"):
        batch.matched_comparison(entries + entries, paper)


def test_initializer_failure_does_not_start_selection_or_training(tmp_path, monkeypatch):
    import edgs_inpaint_io as io
    monkeypatch.setattr(io, "verify_record", lambda value: None)
    runner = batch.SceneBatch.__new__(batch.SceneBatch)
    runner.req = dict(python="python", records=[], target_plys=[], initial={"inputs": {
        key: dict(path=f"/car/{key}") for key in ("removed_model", "inpaint_config", "lama", "camera")}})
    commands = runner.commands(tmp_path)
    phase, command, _ = next(commands)
    assert phase == "initialization"
    assert "--proposal-initialization" not in command
    assert command[command.index("--removed-model") + 1] == "/car/removed_model"
    # The caller aborts the generator on a failed subprocess; no post-init side effects may run.
    commands.close()
    assert not list(tmp_path.iterdir())


def test_scene_failure_continues_queue_without_old_model_fallback(tmp_path, monkeypatch):
    import edgs_inpaint_io as io
    monkeypatch.setattr(io, "verify_record", lambda value: None)
    monkeypatch.setattr(batch.signal, "signal", lambda *args: None)
    cfg = config()
    scenes = []
    for name in ("car", "truck"):
        root = tmp_path / name
        batch.write(root / "config.yaml", cfg)
        baseline = root / "old_results.json"
        batch.write(baseline, {})
        scenes.append(dict(name=name, output=str(root), frame_count=3,
            baseline_results=str(baseline), baseline_results_sha256=batch.sha(baseline)))
    plan = dict(output=str(tmp_path / "report"), report=str(tmp_path / "report/comparison.md"), gpu=1,
        code={}, reference_results={}, appearance_weights={}, python="python", scenes=scenes, reference_config=cfg)
    job = tmp_path / "logs"
    batch.write(job / "request.json", plan)
    batch.write(job / "status.json", {"state": "prepared", "scenes": []})
    runner = batch.SceneBatch(job)
    calls = []
    monkeypatch.setattr(runner, "report_progress", lambda: None)
    monkeypatch.setattr(runner, "commands", lambda root: iter([
        ("initialization", [], root), ("training", [], root), ("evaluation", [], root)]))
    def stage(name, *args):
        calls.append(name)
        if name == "car: initialization":
            raise RuntimeError("insufficient confident anchors")
    monkeypatch.setattr(runner, "stage", stage)
    monkeypatch.setattr(runner, "verify_result", lambda *args: dict(metrics={k: 1. for k in batch.METRICS}))
    runner.run()
    assert "car: training" not in calls
    assert "truck: training" in calls and "truck: evaluation" in calls
    assert runner.status["state"] == "complete_with_failures"
    assert runner.status["failed_scenes"] == ["car"]


def test_report_labels_missing_scenes_and_preserves_per_scene_directions(tmp_path):
    runner = batch.SceneBatch.__new__(batch.SceneBatch)
    runner.job = tmp_path / "logs"
    runner.output = tmp_path / "out"
    runner.report = runner.output / "comparison.md"
    metrics = {k: 2. for k in batch.METRICS}
    runner.plan = dict(scenes=[], reference_metrics=metrics, reference_frame_count=5,
        paper={"bag": {k: 1. for k in batch.METRICS}}, paper_source="https://example.org/paper",
        missing=[dict(name="toys", reason="缺少前置输入")], python="python")
    runner.status = dict(state="running", scenes=[])
    runner.report_progress()
    text = runner.report.read_text()
    assert "toys" in text and "缺少前置输入" in text
    assert "bag（已有参考）" in text
    assert "| PSNR ↑ | 1.000000 | 2.000000 | +1.000000 | 更好 |" in text
    assert "| LPIPS ↓ | 1.000000 | 2.000000 | +1.000000 | 更差 |" in text
    assert batch.read(runner.output / "summary.json")["comparisons"]["new_scenes"] is None
