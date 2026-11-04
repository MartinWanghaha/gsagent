"""Protect cumulative configuration and fresh-start commands for long GPU jobs."""
import copy
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_geometry_ablation.py"
spec = importlib.util.spec_from_file_location("geometry_ablation_batch", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def reference_config():
    return dict(appearance=dict(mode="inpaint360gs"), init=dict(use_depth=True, use_normal=False),
        supervision=dict(use_depth=True, use_normal=True), iterations=5000, seed=0,
        loss=dict(rgb_hole=0., rgb_known_preserve=.2, ssim=.8, lpips=.0005, alpha_preserve=.1,
                  depth=.1, lama_normal=.05, depth_normal_consistency=.01, multiview=.01))


def test_five_cumulative_objectives_share_inputs_and_disable_coverage():
    cfg = reference_config()
    unchanged = copy.deepcopy(cfg)
    variants = batch.make_variants(cfg)
    assert cfg == unchanged and len(variants) == 5
    for index, variant in enumerate(variants):
        actual = variant["config"]
        assert actual["appearance"] == cfg["appearance"]
        assert actual["init"] == cfg["init"]
        assert actual["supervision"] == cfg["supervision"]
        assert actual["loss"]["alpha_preserve"] == 0
        for key in ("rgb_hole", "rgb_known_preserve", "ssim", "lpips"):
            assert actual["loss"][key] == cfg["loss"][key]
        for position, term in enumerate(batch.GEOMETRY):
            assert actual["loss"][term] == (cfg["loss"][term] if position < index else 0)
    variants[-1]["config"]["init"]["use_depth"] = False
    assert variants[0]["config"]["init"]["use_depth"] and cfg == unchanged


def test_all_five_groups_enable_same_density_without_changing_losses():
    cfg = reference_config()
    density = dict(enabled=True, from_iter=500, until_iter=5000, interval=100,
                   grad_threshold=.0002, percent_dense=.01, max_new_points=500000)
    previous = batch.make_variants(cfg)
    current = batch.make_variants(cfg, density)
    for old, new in zip(previous, current):
        actual = new["config"].copy()
        assert actual.pop("adaptive_density") == density
        assert actual == old["config"]
    current[0]["config"]["adaptive_density"]["enabled"] = False
    assert current[1]["config"]["adaptive_density"]["enabled"] and density["enabled"]


def test_every_group_starts_from_shared_initialization_not_previous_checkpoint(tmp_path):
    runner = batch.Batch.__new__(batch.Batch)
    inputs = {name: dict(path=f"/shared/{name}.json") for name in
              ("initialization", "lama", "camera", "edgs_config", "local_selection")}
    pub = {name: dict(path=f"/shared/{name}") for name in
           ("classifier", "edgs_config", "cfg_args", "removed_model_manifest", "removal_manifest",
            "workspace_manifest", "tracking_session", "lama_manifest", "inpaint_config")}
    runner.req = dict(python="/env/python", reference_config=reference_config(), training_inputs=inputs,
        publish_inputs=pub, source="/data/bag", publish_parameters=dict(source_iteration=2000,
        target_ids=[36], surrounding_ids=[]))
    for name in batch.NAMES:
        root = tmp_path / name
        train = runner.commands(root)[0][1]
        assert train[train.index("--initialization") + 1] == "/shared/initialization.json"
        assert train[train.index("--local-selection") + 1] == "/shared/local_selection.json"
        assert train[train.index("--output-root") + 1] == str(root / "edgs_joint")
        assert not any("checkpoint" in value or "start-checkpoint" in value for value in train)
        publish = runner.commands(root)[1][1]
        assert publish[:3] == ["/env/python", "-m", "tools.publish_inpainted_edgs_model"]


def test_report_preserves_previous_comparisons_and_updates_partial_metrics(tmp_path):
    runner = batch.Batch.__new__(batch.Batch)
    runner.job = tmp_path / "logs" / "test_suite"
    runner.job.mkdir(parents=True)
    runner.output = tmp_path / "outputs"
    runner.output.mkdir()
    runner.report = tmp_path / "comparison.md"
    before = "# Existing comparison\n\nOld results must remain.\n"
    runner.report.write_text(before)
    metrics = {key: 1. for key in batch.METRICS}
    result = tmp_path / "existing.json"
    result.write_text(json.dumps(dict(metrics=metrics)))
    runner.req = dict(variants=batch.make_variants(reference_config()), baseline_results=str(result),
        reference_results=str(result), paper=metrics, frame_count=58)
    runner.status = dict(state="running", phase="00: evaluation", variants=[])
    runner.report_progress()
    runner.status["variants"] = [dict(name=batch.NAMES[0], label=batch.LABELS[0], state="complete", metrics=metrics)]
    runner.report_progress()
    text = runner.report.read_text()
    assert text.startswith(before.rstrip())
    assert text.count("<!-- GEOMETRY_ABLATION:test_suite:BEGIN -->") == 1
    assert "masked_PSNR" in text and "58 个" in text
    summary = json.loads((runner.output / "summary.json").read_text())
    assert not summary["complete"] and len(summary["variants"]) == 1
    assert (runner.output / "summary.csv").is_file()


def test_density_report_compares_matching_pruning_group(tmp_path):
    runner = batch.Batch.__new__(batch.Batch)
    runner.job = tmp_path / "logs"; runner.job.mkdir()
    runner.output = tmp_path / "suite"; runner.output.mkdir()
    runner.report = tmp_path / "comparison.md"; runner.report.write_text("Existing report.\n")
    metrics = {key: 1. for key in batch.METRICS}
    result = tmp_path / "previous.json"; result.write_text(json.dumps(dict(metrics=metrics)))
    density = dict(enabled=True, from_iter=500, until_iter=5000, interval=100,
                   grad_threshold=.0002, percent_dense=.01, max_new_points=500000)
    root = runner.output / batch.NAMES[0]
    diag = root / "edgs_joint/diagnostics/final.json"; diag.parent.mkdir(parents=True)
    diag.write_text(json.dumps(dict(adaptive_density=dict(counters=dict(cloned=12, split_parents=3,
        split_children=6, pruned_new=2)), pruning=dict(remaining_new=113))))
    runner.req = dict(variants=batch.make_variants(reference_config(), density), baseline_results=str(result),
        reference_results=str(result), paper=metrics, frame_count=58, adaptive_density=True,
        pruning_reference={batch.NAMES[0]: dict(path=str(result))})
    runner.status = dict(state="running", phase="01: training", variants=[dict(name=batch.NAMES[0],
        label=batch.LABELS[0], state="complete", metrics={key: 2. for key in batch.METRICS}, output=str(root))])
    runner.report_progress()
    text = runner.report.read_text()
    assert text.startswith("Existing report.") and "累计克隆" in text
    assert "| 官方外观 | 12 | 3 | 6 | 2 | 113 |" in text
    assert "| 官方外观 | masked_PSNR | 1.000000 | 2.000000 | +1.000000 |" in text
