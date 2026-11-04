"""A surface initialization experiment must keep supervision and old-row eligibility fixed."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_surface_initialization_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("surface_initialization_ablation", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def reference():
    from edgs_inpaint_io import load_config
    cfg = load_config()
    cfg.update(iterations=5000, freeze_background=True)
    cfg["init"].update(use_depth=True, use_normal=False)
    cfg["appearance"]["mode"] = "masked"
    cfg["surface_init"]["mode"] = "none"
    cfg["local_background"]["enabled"] = True
    cfg["background_writeback"]["enabled"] = False
    cfg["adaptive_density"]["enabled"] = False
    cfg["pruning"]["enabled"] = True
    cfg["loss"].update(rgb_hole=1., rgb_known_preserve=1., ssim=.2, lpips=.005,
        alpha_preserve=.1, depth=.1, lama_normal=.05, depth_normal_consistency=.01, multiview=.01)
    return cfg


def test_only_initialization_mode_changes():
    cfg = reference(); before = copy.deepcopy(cfg)
    variant = batch.make_variant(cfg)
    assert cfg == before
    after = copy.deepcopy(variant["config"])
    assert after["surface_init"]["mode"] == "depth_anchor"
    after["surface_init"]["mode"] = "none"
    assert after == cfg
    for section in ("loss", "optimizer", "pruning", "local_background", "init", "supervision"):
        assert variant["config"][section] == cfg[section]


@pytest.mark.parametrize("section,key,value", [
    ("appearance", "mode", "inpaint360gs"), ("loss", "depth", 0),
    ("adaptive_density", "enabled", True), ("background_writeback", "enabled", True),
    ("local_background", "enabled", False), ("pruning", "enabled", False)])
def test_reject_different_supervision_or_density_policy(section, key, value):
    cfg = reference(); cfg[section][key] = value
    with pytest.raises(ValueError):
        batch.make_variant(cfg)


def test_new_population_changes_without_changing_any_old_eligibility():
    old = np.array([False, False, False, False, True, True, True])
    trainable = np.array([False, True, True, False, True, True, True])
    new = np.array([False, False, False, False, True, True])
    old_votes = np.array([0, 20, 3, 1, 0, 0, 0], dtype=np.uint16)
    before = [array.copy() for array in (old, trainable, new, old_votes)]
    mask, votes, stats = batch.rebind_arrays(old, trainable, new, old_votes)
    np.testing.assert_array_equal(mask, [False, True, True, False, True, True])
    np.testing.assert_array_equal(votes, old_votes[:6])
    assert stats["initial_old"] == 4 and stats["initial_new"] == 2
    assert stats["trainable_old"] == 2 and stats["frozen_old"] == 2
    assert stats["old_mask_bitwise_equal"]
    for current, expected in zip((old, trainable, new, old_votes), before):
        np.testing.assert_array_equal(current, expected)
    with pytest.raises(ValueError, match="ordering/count"):
        batch.rebind_arrays(old, trainable, np.array([False, False, True, True]), old_votes)
    with pytest.raises(ValueError, match="ordering/count"):
        batch.rebind_arrays(old, trainable, np.array([False, True, False, False, False, True]), old_votes)


def test_train_publish_evaluate_use_new_initialization_and_same_iteration(tmp_path):
    runner = batch.SurfaceInitializationBatch.__new__(batch.SurfaceInitializationBatch)
    inputs = {name: dict(path=f"/new/{name}.json") for name in
              ("initialization", "lama", "camera", "edgs_config", "local_selection")}
    pub = {name: dict(path=f"/shared/{name}") for name in ("classifier", "edgs_config", "cfg_args",
        "removed_model_manifest", "removal_manifest", "workspace_manifest", "tracking_session",
        "lama_manifest", "inpaint_config")}
    runner.req = dict(python="/env/python", reference_config=reference(), training_inputs=inputs,
        publish_inputs=pub, source="/data/bag", publish_parameters=dict(source_iteration=2000,
        target_ids=[36], surrounding_ids=[]))
    train, publish, evaluate = [command for _, command, _ in runner.commands(tmp_path)]
    for flag, key in (("initialization", "initialization"), ("local-selection", "local_selection")):
        assert train[train.index("--" + flag)+1] == inputs[key]["path"]
    assert not any("checkpoint" in value for value in train)
    assert "iteration_5000/point_cloud.ply" in publish[publish.index("--inpainted-ply")+1]
    assert evaluate[evaluate.index("--iteration")+1] == "5000"


def test_report_preserves_historical_results_and_includes_pending_scores(tmp_path):
    runner = batch.SurfaceInitializationBatch.__new__(batch.SurfaceInitializationBatch)
    runner.job = tmp_path / "logs"; runner.job.mkdir()
    runner.output = tmp_path / "suite"; runner.output.mkdir()
    runner.report = tmp_path / "comparison.md"; runner.report.write_text("Keep previous results.\n")
    metrics = {key: 1. for key in batch.METRICS}
    path = tmp_path / "results.json"; path.write_text(json.dumps(dict(metrics=metrics)))
    runner.req = dict(reference_config=reference(), baseline_results=str(path), paper=metrics,
        python="/env/python", frame_count=58, local_selection_check=dict(initial_old=4,
        initial_new=2, trainable_old=2, frozen_old=2, baseline_summary=dict(initial_new=3)))
    runner.status = dict(state="running", phase="depth_anchor: training", variants=[])
    runner.report_progress()
    runner.status["variants"].append(dict(name="depth_anchor", state="complete", metrics=metrics))
    runner.status["state"] = "complete"
    runner.report_progress()
    report = runner.report.read_text()
    assert report.startswith("Keep previous results.")
    assert report.count("<!-- DEPTH_ANCHOR_INITIALIZATION_ABLATION:logs:BEGIN -->") == 1
    assert "| masked_PSNR ↑ | 1.000000 | 1.000000 | 1.000000 | +0.000000 |" in report
    assert json.loads((runner.output / "summary.json").read_text())["complete"]


def test_fresh_initializer_has_same_inputs_and_does_not_reuse_old_cache(tmp_path):
    initial = dict(inputs={key: dict(path=f"/baseline/{key}.json") for key in
        ("removed_model", "inpaint_config", "lama", "camera")})
    command = batch.initialization_command("/env/python", initial, tmp_path)
    assert "--proposal-initialization" not in command
    for flag, key in (("removed-model", "removed_model"), ("inpaint-config", "inpaint_config"),
                      ("lama", "lama"), ("camera", "camera")):
        assert command[command.index("--"+flag)+1] == initial["inputs"][key]["path"]
    assert command[command.index("--manifest")+1] == str(tmp_path/"manifests/edgs_init_manifest.json")


def test_initialization_precedes_selection_and_training(tmp_path, monkeypatch):
    import edgs_inpaint_io
    runner = batch.SurfaceInitializationBatch.__new__(batch.SurfaceInitializationBatch)
    runner.job = tmp_path; runner.status = {}
    runner.save = lambda: None; runner.report_progress = lambda: None
    runner.req = dict(initialization=dict(mode="fresh_matching", command=["new-init"], manifest="/new/init.json"),
        variants=[dict(config=reference())], baseline_training_inputs={"lama": dict(path="/lama.json")})
    events = []
    def bind(*args):
        events.append("bind")
        return dict(path="/new/selection.json"), dict(old_mask_bitwise_equal=True), dict(outputs=dict(surface_model=dict(path="/new/model.json")))
    monkeypatch.setattr(batch, "rebind_selection", bind)
    monkeypatch.setattr(edgs_inpaint_io, "record", lambda path: dict(path=str(path)))
    monkeypatch.setattr(batch, "read", lambda path: dict(mode="depth_anchor"))
    monkeypatch.setattr(batch.Batch, "commands", lambda *args, **kwargs: (("training", ["train"], tmp_path),))
    commands = runner.commands(tmp_path)
    assert next(commands)[:2] == ("initialization", ["new-init"])
    assert not events  # Initializer runs while generator is suspended.
    assert next(commands)[:2] == ("training", ["train"])
    assert events == ["bind"]
    assert runner.req["training_inputs"]["initialization"]["path"] == "/new/init.json"
    assert (tmp_path/"resolved_initialization.json").is_file()


def test_report_supports_pending_initialization(tmp_path):
    runner = batch.SurfaceInitializationBatch.__new__(batch.SurfaceInitializationBatch)
    runner.job = tmp_path/"job"; runner.job.mkdir()
    runner.output = tmp_path/"output"; runner.output.mkdir()
    runner.report = tmp_path/"report.md"; runner.report.write_text("Existing report.\n")
    metrics = {key: 1. for key in batch.METRICS}
    results = tmp_path/"result.json"; results.write_text(json.dumps(dict(metrics=metrics)))
    runner.req = dict(reference_config=reference(), local_selection_check=None, baseline_results=str(results),
        paper=metrics, frame_count=58, python="/env/python")
    runner.status = dict(state="running", phase="depth_anchor: initialization", variants=[])
    runner.report_progress()
    assert "初始化尚未完成" in runner.report.read_text()
    assert not json.loads((runner.output/"summary.json").read_text())["complete"]
