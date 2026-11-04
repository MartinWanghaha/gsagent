"""Duration ablations must preserve supervision and evaluate their own models."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_iteration_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("iteration_ablation", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def reference():
    return dict(iterations=5000, appearance=dict(mode="masked"), seed=0,
        init=dict(use_depth=True, use_normal=False),
        geometry_from_iter=100, geometry_ramp_iters=2000,
        adaptive_density=dict(enabled=False), pruning=dict(enabled=True, min_opacity=.005, until_iter=15000),
        loss=dict(rgb_hole=1., rgb_known_preserve=1., ssim=.2, lpips=.005, alpha_preserve=.1,
                  depth=.1, lama_normal=.05, depth_normal_consistency=.01, multiview=.01))


def test_only_duration_changes_and_all_original_losses_remain_enabled():
    cfg = reference(); before = copy.deepcopy(cfg)
    variants = batch.make_variants(cfg)
    assert cfg == before
    assert [v["config"]["iterations"] for v in variants] == [8000, 10000, 15000, 20000]
    for variant in variants:
        actual = copy.deepcopy(variant["config"])
        actual["iterations"] = 5000
        assert actual == cfg
        assert variant["config"]["pruning"]["until_iter"] == 15000
    variants[0]["config"]["loss"]["rgb_hole"] = 0
    assert cfg == before and variants[1]["config"]["loss"]["rgb_hole"] == 1


def test_reject_official_supervision_and_disabled_geometry():
    cfg = reference(); cfg["appearance"]["mode"] = "inpaint360gs"
    with pytest.raises(ValueError, match="original masked"):
        batch.make_variants(cfg)
    cfg = reference(); cfg["loss"]["depth"] = 0
    with pytest.raises(ValueError, match="remain enabled"):
        batch.make_variants(cfg)


def test_each_duration_starts_fresh_and_publishes_and_evaluates_correct_iteration(tmp_path):
    runner = batch.IterationBatch.__new__(batch.IterationBatch)
    inputs = {name: dict(path=f"/shared/{name}.json") for name in
              ("initialization", "lama", "camera", "edgs_config", "local_selection")}
    pub = {name: dict(path=f"/shared/{name}") for name in
           ("classifier", "edgs_config", "cfg_args", "removed_model_manifest", "removal_manifest",
            "workspace_manifest", "tracking_session", "lama_manifest", "inpaint_config")}
    runner.req = dict(python="/env/python", reference_config=reference(), training_inputs=inputs,
        publish_inputs=pub, source="/data/bag", publish_parameters=dict(source_iteration=2000,
        target_ids=[36], surrounding_ids=[]))
    for variant in batch.make_variants(reference()):
        root = tmp_path / variant["name"]; root.mkdir()
        (root / "config.yaml").write_text(json.dumps(variant["config"]))
        target = str(variant["config"]["iterations"])
        commands = runner.commands(root)
        train, publish, evaluate = [command for _, command, _ in commands]
        assert train[train.index("--initialization")+1] == "/shared/initialization.json"
        assert train[train.index("--config")+1] == str(root / "config.yaml")
        assert not any("checkpoint" in argument for argument in train)
        assert publish[publish.index("--source-iteration")+1] == "2000"
        assert publish[publish.index("--output-iteration")+1] == target
        assert f"iteration_{target}/point_cloud.ply" in publish[publish.index("--inpainted-ply")+1]
        assert evaluate[evaluate.index("--iteration")+1] == target


def test_evaluation_must_match_target_iteration_and_model(tmp_path):
    runner = batch.IterationBatch.__new__(batch.IterationBatch)
    root = tmp_path / "experiment"
    model = root / "inpainted_3dgs"
    (model / "evaluation/inpaint360_r4").mkdir(parents=True)
    (root / "manifests").mkdir()
    ply = model / "point_cloud.ply"; ply.write_bytes(b"trained 8000-step model")
    result_path = model / "evaluation/inpaint360_r4/results.json"
    inputs = {key: "same" for key in ("source_path", "scene", "resolution", "renderer", "frames", "training_resolution",
            "training_eval", "training_exclude_test_images", "reference_images", "official_masks", "colmap_images", "colmap_cameras")}
    inputs.update(iteration=5000, model=dict(sha256="old-model"))
    baseline = dict(complete=True, frame_count=58, protocol="official", inputs=inputs,
                    metrics={key: 1. for key in batch.METRICS})
    base_path = tmp_path / "baseline.json"; base_path.write_text(json.dumps(baseline))
    variant = batch.make_variants(reference())[0]
    runner.req = dict(baseline_results=str(base_path), training_inputs=dict(shared=True), implementation={}, appearance_weights={})
    joint = dict(request=dict(config=variant["config"], implementation={}, appearance_weights={}), inputs=dict(shared=True))
    (root / "manifests/edgs_joint_manifest.json").write_text(json.dumps(joint))
    (model / "model_manifest.json").write_text(json.dumps(dict(complete=True, model=dict(point_cloud="point_cloud.ply",
         point_cloud_storage=dict(sha256=batch.sha(ply))))))
    result = copy.deepcopy(baseline)
    result["inputs"].update(iteration=8000, model=dict(sha256=batch.sha(ply)))
    result_path.write_text(json.dumps(result))
    assert runner.verify_result(root, variant)["inputs"]["iteration"] == 8000
    result["inputs"]["iteration"] = 5000; result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="iteration"):
        runner.verify_result(root, variant)
    result["inputs"].update(iteration=8000, model=dict(sha256="old-model")); result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="different model"):
        runner.verify_result(root, variant)


def test_report_keeps_previous_results_and_includes_pending_duration_groups(tmp_path):
    runner = batch.IterationBatch.__new__(batch.IterationBatch)
    runner.job = tmp_path / "logs"; runner.job.mkdir()
    runner.output = tmp_path / "suite"; runner.output.mkdir()
    runner.report = tmp_path / "comparison.md"; runner.report.write_text("Keep previous experiments.\n")
    metrics = {key: 1. for key in batch.METRICS}
    result = tmp_path / "baseline.json"; result.write_text(json.dumps(dict(metrics=metrics)))
    runner.req = dict(variants=batch.make_variants(reference()), reference_config=reference(),
        baseline_results=str(result), baseline_iteration=5000, paper=metrics, frame_count=58)
    runner.status = dict(state="running", phase="iter_08000: training", variants=[])
    runner.report_progress()
    runner.status["variants"].append(dict(name="iter_08000", label="8000 步", state="complete", metrics=metrics))
    runner.report_progress()
    report = runner.report.read_text()
    assert report.startswith("Keep previous experiments.")
    assert report.count("<!-- ITERATION_ABLATION:logs:BEGIN -->") == 1
    assert "| 指标 | 论文 bag | 原始 5000 步 | 8000 步 | 10000 步 | 15000 步 | 20000 步 |" in report
    assert "| masked_PSNR ↑ | 1.000000 | 1.000000 | 1.000000 | — | — | — |" in report
    assert not json.loads((runner.output / "summary.json").read_text())["complete"]
    assert "8000" in (runner.output / "summary.csv").read_text()


def test_seven_depth_anchor_durations_share_all_other_configuration():
    cfg = reference()
    cfg['surface_init'] = dict(mode='depth_anchor', grid_pixels=32, anchor_radius_pixels=64)
    cfg['freeze_background'] = True
    cfg['local_background'] = dict(enabled=True)
    before = copy.deepcopy(cfg)
    durations = [2000, 4000, 7000, 9000, 10000, 15000, 20000]
    variants = batch.make_variants(cfg, durations)
    assert [v['name'] for v in variants] == ['iter_02000', 'iter_04000', 'iter_07000', 'iter_09000', 'iter_10000', 'iter_15000', 'iter_20000']
    assert cfg == before
    for v in variants:
        actual = copy.deepcopy(v['config']); actual['iterations'] = 5000
        assert actual == before


def test_depth_anchor_report_labels_baseline_and_all_seven_groups(tmp_path):
    runner = batch.IterationBatch.__new__(batch.IterationBatch)
    runner.job = tmp_path/'logs'; runner.job.mkdir()
    runner.output = tmp_path/'suite'; runner.output.mkdir()
    runner.report = tmp_path/'comparison.md'; runner.report.write_text('Previous results.\n')
    metrics = {key: 1. for key in batch.METRICS}
    result = tmp_path/'baseline.json'; result.write_text(json.dumps(dict(metrics=metrics)))
    cfg = reference(); cfg['surface_init'] = dict(mode='depth_anchor')
    variants = batch.make_variants(cfg, [2000,4000,7000,9000,10000,15000,20000])
    runner.req = dict(variants=variants, reference_config=cfg, baseline_results=str(result),
        baseline_iteration=5000, paper=metrics, frame_count=58)
    runner.status = dict(state='running', phase='iter_02000: training', variants=[])
    runner.report_progress()
    text = runner.report.read_text()
    assert 'depth_anchor 基线 5000 步' in text
    assert '7 组仅改变' in text
    assert '四组' not in text
    assert all(v['label'] in text for v in variants)
