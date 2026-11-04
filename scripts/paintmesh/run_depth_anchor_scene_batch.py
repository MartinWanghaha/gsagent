#!/usr/bin/env python3
"""Serial fresh depth-anchor initialization, original-supervision 5k training and evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from run_geometry_ablation import Batch, LOWER, METRICS, REPO, atomic, now, read, sha, write
from run_surface_initialization_ablation import initialization_command

PREFIX = "original_supervision_depth_anchor_5000"
SCENE_ROOT = REPO / "output/paintmesh/Inpaint360GS/inpaint360"
REFERENCE = SCENE_ROOT / ("bag/removal/table1_adaptive180_sweep/inpaint/"
    "original_supervision_depth_anchor_init_20261022_215217/depth_anchor")
PAPER = Path(__file__).parent / "configs/inpaint360_paper_table7.json"
SOURCE_SUFFIX = "removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP"


def experiment_config(reference):
    """Use the completed bag recipe verbatim, never migrate an old scene's losses."""
    cfg = copy.deepcopy(reference)
    if cfg["iterations"] != 5000 or cfg["surface_init"]["mode"] != "depth_anchor":
        raise ValueError("Reference must be the completed 5000-step depth_anchor experiment")
    if (cfg["appearance"]["mode"] != "masked" or not cfg["init"]["use_depth"]
            or cfg["init"]["use_normal"] or not cfg["freeze_background"]
            or not cfg["local_background"]["enabled"] or not cfg["pruning"]["enabled"]
            or cfg["adaptive_density"]["enabled"] or cfg["background_writeback"]["enabled"]):
        raise ValueError("Reference must use original appearance, local old-row updates and prune-only")
    for key in ("rgb_hole", "rgb_known_preserve", "ssim", "lpips", "alpha_preserve",
                "depth", "lama_normal", "depth_normal_consistency", "multiview"):
        if cfg["loss"][key] <= 0:
            raise ValueError("Original supervision must stay enabled: " + key)
    return cfg


def matched_comparison(entries, paper):
    """Macro means over exactly the successful scenes; missing scenes never become zeros."""
    if not entries:
        return None
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate scene in comparison")
    ours = {k: sum(e["metrics"][k] for e in entries) / len(entries) for k in METRICS}
    published = {k: sum(paper[n][k] for n in names) / len(names) for k in METRICS}
    count = sum(e["frame_count"] for e in entries)
    weighted = {k: sum(e["metrics"][k] * e["frame_count"] for e in entries) / count
                for k in METRICS if k != "FID"}
    weighted["FID"] = ours["FID"]
    return dict(scenes=names, frame_count=count, ours_scene_macro=ours,
        paper_same_subset_macro=published, delta={k: ours[k] - published[k] for k in METRICS},
        ours_official_code_average=weighted)


def prepare(args):
    from edgs_inpaint_io import appearance_weights, implementation, load_config, record
    ref = args.reference_run.resolve(strict=True)
    cfg = experiment_config(load_config(ref / "edgs_joint/config.resolved.json"))
    reference_joint = read(ref / "manifests/edgs_joint_manifest.json")
    reference_result = ref / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
    ref_result = read(reference_result)
    if not reference_joint.get("complete") or not ref_result.get("complete"):
        raise ValueError("Reference training and evaluation must be complete")
    if reference_joint["request"]["config"] != cfg or ref_result["inputs"]["iteration"] != 5000:
        raise ValueError("Reference manifest/config/evaluation mismatch")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = REPO / "logs" / f"{PREFIX}_scenes_gpu{args.gpu}_{stamp}"
    output = REPO / "output/evaluation" / f"{PREFIX}_scenes_{stamp}"
    paper_data = read(PAPER)
    paper = {n: dict(zip(paper_data["metrics"], values)) for n, values in paper_data["scenes"].items()}
    scenes, missing = [], []
    for directory in sorted(args.scene_root.resolve(strict=True).iterdir()):
        name = directory.name
        if not directory.is_dir() or name == "bag" or name not in paper:
            continue
        source = directory / SOURCE_SUFFIX
        needed = [source / name for name in ("manifests/edgs_init_manifest.json",
            "manifests/edgs_joint_manifest.json", "inpainted_3dgs/model_manifest.json",
            "inpainted_3dgs/evaluation/inpaint360_r4/results.json")]
        absent = [str(p) for p in needed if not p.is_file()]
        if absent:
            missing.append(dict(name=name, reason="缺少 removal/LaMa/inpaint 前置产物", missing=absent))
            continue
        initial, joint, published, result = map(read, needed)
        if not all(item.get("complete") for item in (initial, joint, published, result)):
            raise ValueError("Incomplete input artifacts: " + name)
        for key in ("camera", "lama"):
            if initial["inputs"][key] != joint["inputs"][key]:
                raise ValueError(f"{name}: initialization/training {key} mismatch")
        if result["protocol"] != ref_result["protocol"]:
            raise ValueError("Evaluation protocol differs from bag: " + name)
        for key in ("resolution", "training_resolution", "training_eval", "training_exclude_test_images"):
            if result["inputs"][key] != ref_result["inputs"][key]:
                raise ValueError(f"{name}: evaluation policy differs: {key}")
        background = Path(initial["inputs"]["background"]["path"])
        targets = [record(background.with_name(f"point_cloud_{i}.ply"))
                   for i in published["parameters"]["target_ids"]]
        camera = read(initial["inputs"]["camera"]["path"])
        root = source.parent / f"{PREFIX}_{stamp}"
        scenes.append(dict(name=name, output=str(root), old_run=str(source), initial=initial,
            baseline_results=str(needed[-1]), baseline_results_sha256=sha(needed[-1]),
            publish_inputs=published["inputs"], publish_parameters=published["parameters"],
            edgs_config=joint["inputs"]["edgs_config"], target_plys=targets,
            records=[record(p) for p in needed], source=result["inputs"]["source_path"],
            frame_count=result["frame_count"], virtual_views=len(camera["cameras"])))
    if not scenes:
        raise ValueError("No complete other-scene inputs found")
    assets = appearance_weights(cfg)
    if assets != reference_joint["request"]["appearance_weights"]:
        raise ValueError("LPIPS assets differ from the bag recipe")
    code = implementation()
    for name in ("scripts/paintmesh/run_depth_anchor_scene_batch.py",
                 "scripts/paintmesh/run_geometry_ablation.py", "scripts/paintmesh/run_surface_initialization_ablation.py",
                 "scripts/paintmesh/eval_inpaint360.py", "submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py",
                 "submodules/Inpaint360GS/utils/image_utils.py", "submodules/Inpaint360GS/utils/loss_utils.py"):
        code[name] = sha(REPO / name)
    req = dict(created_at=now(), gpu=args.gpu, python=str(Path(args.python).resolve(strict=True)),
        output=str(output), report=str(output / "comparison.md"), reference=str(ref),
        reference_results=record(reference_result), reference_config=cfg, implementation=implementation(),
        code=code, appearance_weights=assets, paper=paper, paper_source=paper_data["source"],
        scenes=scenes, missing=missing, reference_metrics=ref_result["metrics"],
        reference_frame_count=ref_result["frame_count"], reference_protocol=ref_result["protocol"],
        implementation_differences_from_bag=[k for k, v in implementation().items()
            if reference_joint["request"]["implementation"].get(k) != v])
    job.mkdir(parents=True)
    output.mkdir(parents=True)
    for scene in scenes:
        root = Path(scene["output"])
        (root / "manifests").mkdir(parents=True)
        write(root / "config.yaml", cfg)
    write(job / "request.json", req)
    write(job / "status.json", dict(state="prepared", scenes=[]))
    write(output / "experiment_plan.json", req)
    return job


class SceneBatch(Batch):
    def __init__(self, job):
        super().__init__(job)
        self.plan = copy.deepcopy(self.req)
        previous = read(self.job / "status.json")
        self.status["scenes"] = previous.get("scenes", [])
        self.status.pop("variants", None)

    def bind(self, scene):
        self.req = {**self.plan, **scene, "training_inputs": None}

    def commands(self, root, *, iterations=None):
        from edgs_inpaint_io import record, validate_init, validate_selection, verify_record
        req, py = self.req, self.req["python"]
        for value in req["records"] + req["target_plys"]:
            verify_record(value)
        yield "initialization", initialization_command(py, req["initial"], root), REPO
        initial_path = root / "manifests/edgs_init_manifest.json"
        selection_path = root / "manifests/local_selection_manifest.json"
        select = [py, str(REPO / "submodules/EDGS/tools/select_paintmesh_local_background.py"),
            "--initialization", str(initial_path), "--lama", req["initial"]["inputs"]["lama"]["path"],
            "--camera", req["initial"]["inputs"]["camera"]["path"], "--config", str(root / "config.yaml"),
            "--output-root", str(root / "local_selection"), "--manifest", str(selection_path)]
        for value in req["target_plys"]:
            select += ["--target-ply", value["path"]]
        yield "local_selection", select, REPO
        initial = validate_init(initial_path)
        selection = validate_selection(selection_path, initial=initial, config=req["reference_config"]["local_background"])
        req["training_inputs"] = dict(initialization=record(initial_path), local_selection=record(selection_path),
            camera=req["initial"]["inputs"]["camera"], lama=req["initial"]["inputs"]["lama"], edgs_config=req["edgs_config"])
        write(root / "resolved_initialization.json", dict(training_inputs=req["training_inputs"],
            surface_model=read(initial["outputs"]["surface_model"]["path"]),
            selection=read(selection["outputs"]["summary"]["path"])))
        yield from super().commands(root, iterations=5000)

    def report_progress(self):
        plan, status = self.plan, self.status
        completed = [e for e in status["scenes"] if e["state"] == "complete"]
        reference = dict(name="bag", metrics=plan["reference_metrics"], frame_count=plan["reference_frame_count"])
        comparisons = dict(new_scenes=matched_comparison(completed, plan["paper"]),
            including_existing_bag=matched_comparison([reference, *completed], plan["paper"]))
        lines = ["# 新版 depth_anchor 初始化：其他场景 5000 步", "",
            f"状态：{status['state']}；阶段：{status.get('phase', '准备')}；完成 {len(completed)}/{len(plan['scenes'])}。", "",
            "采用现有 bag 5000 步实验的完整配置：原本 masked 外观监督＋全部几何监督；"
            "depth_anchor（RGB 锚点校正深度，无局部平面拟合）；局部旧高斯更新、远处冻结；只剪枝，不克隆/分裂。",
            "每场景重新匹配并初始化，依次训练、发布和评估；复用各自已有 removal、mask 和 LaMa RGB/depth/normal。"
            "新输出目录保留历史结果。初始化无法满足置信度条件时，该场景记为失败，不回退到旧模式。", "",
            "本批次其他场景为 table1_hemi90 的 90 帧；bag 参考为 adaptive180 的 180 帧。"
            "这是将 bag 配方应用到其他场景，不是相对其他场景旧实验只改变单一变量的消融。",
            "评估：resolution=4、真实 test_* 相机、官方 unseen_mask；masked 指标为 mask 外置零后按整幅图计算。"
            "训练沿用既有 eval=false / exclude_test_images=true；与论文完整训练协议仍有差异。"
            "这里只训练和评估 3DGS，未执行 mesh 的 TSDF 后处理。", "",
            f"论文值来源：[Inpaint360GS Table 7]({plan['paper_source']})，为论文公布的四舍五入数据。", "",
            "| 场景 | 目标 ID | 虚拟视图 | test 帧数 | 状态 | 输出 |", "|---|---|---:|---:|---|---|"]
        states = {e["name"]: e for e in status["scenes"]}
        for scene in plan["scenes"]:
            entry = states.get(scene["name"], {})
            state = entry.get("state", "pending")
            if entry.get("error"):
                state += ": " + entry["error"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {scene['name']} | {scene['publish_parameters']['target_ids']} | {scene['virtual_views']} | "
                         f"{scene['frame_count']} | {state} | [目录]({scene['output']}) |")
        for entry in plan["missing"]:
            lines.append(f"| {entry['name']} | — | — | — | {entry['reason']} | — |")
        lines += ["", "bag 使用已有同配置 5000 步结果作为单独参考，不在本批次重新训练。", ""]
        csv_rows = []
        for entry in [reference, *completed]:
            label = entry["name"] + ("（已有参考）" if entry["name"] == "bag" else "")
            lines += [f"## {label}", "", "| 指标 | 论文 | 本方法 | 差值（本方法−论文） | 比较 |",
                      "|---|---:|---:|---:|---|"]
            for key in METRICS:
                paper, value = plan["paper"][entry["name"]][key], entry["metrics"][key]
                delta = value - paper
                better = delta < 0 if key in LOWER else delta > 0
                label = "相同" if delta == 0 else "更好" if better else "更差"
                lines.append(f"| {key} {'↓' if key in LOWER else '↑'} | {paper:.6f} | {value:.6f} | {delta:+.6f} | {label} |")
                csv_rows.append(dict(scene=entry["name"], metric=key, paper=paper, ours=value, delta=delta))
            lines.append("")
        for name, summary in comparisons.items():
            if summary is None:
                continue
            lines += [f"## {'仅本批新训练场景' if name == 'new_scenes' else '包含已有 bag 参考'}：相同场景子集均值", "",
                "场景：" + ", ".join(summary["scenes"]) + "。", "",
                "| 指标 | 论文同子集场景平均 | 本方法场景平均 | 差值 | 本方法官方代码聚合 |",
                "|---|---:|---:|---:|---:|"]
            for k in METRICS:
                lines.append(f"| {k} | {summary['paper_same_subset_macro'][k]:.6f} | {summary['ours_scene_macro'][k]:.6f} | "
                    f"{summary['delta'][k]:+.6f} | {summary['ours_official_code_average'][k]:.6f} |")
            lines += ["", "论文子集均值由 Table 7 对应行计算。官方代码聚合：前六项按 test 帧数加权、FID 按场景平均；"
                "不把本子集与论文 11 场景 Table 1 总均值直接比较。失败/未完成场景不计入均值。", ""]
        if status.get("error"):
            lines += ["批次错误：" + status["error"], ""]
        lines += [f"总日志：`{self.job / 'run.log'}`", f"状态：`{self.job / 'status.json'}`", "",
            f"恢复命令：`{plan['python']} {Path(__file__).resolve()} --resume {self.job} --detach`", ""]
        atomic(self.report, "\n".join(lines))
        write(self.output / "summary.json", dict(complete=status["state"] == "complete", status=status,
            comparisons=comparisons, missing=plan["missing"], request=str(self.job / "request.json")))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["scene", "metric", "paper", "ours", "delta"])
        writer.writeheader(); writer.writerows(csv_rows)
        atomic(self.output / "comparison.csv", buf.getvalue())

    def run(self):
        from edgs_inpaint_io import record, verify_record
        with (self.job / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
            try:
                for name, expected in self.plan["code"].items():
                    if sha(REPO / name) != expected:
                        raise ValueError("Recorded implementation changed: " + name)
                verify_record(self.plan["reference_results"])
                for value in self.plan["appearance_weights"].values():
                    verify_record(value)
                self.stage("CUDA preflight", [self.plan["python"], "-c",
                    "import os,torch; assert torch.cuda.is_available(); "
                    "assert torch.isfinite(torch.ones(1,device='cuda')).all(); "
                    "print('CUDA preflight OK:',os.environ['CUDA_VISIBLE_DEVICES'],torch.cuda.get_device_name(0),flush=True)"],
                    REPO, self.job / "preflight.log")
                for index, scene in enumerate(self.plan["scenes"]):
                    self.bind(scene)
                    root = Path(scene["output"])
                    if read(root / "config.yaml") != self.plan["reference_config"]:
                        raise ValueError("Experiment config changed: " + scene["name"])
                    entry = next((e for e in self.status["scenes"] if e["name"] == scene["name"]), None)
                    if entry is None:
                        entry = dict(name=scene["name"], output=str(root))
                        self.status["scenes"].append(entry)
                    entry.update(state="running", frame_count=scene["frame_count"])
                    entry.pop("error", None)
                    self.status.update(current_index=index, current_scene=scene["name"], training_progress="")
                    result_path = root / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
                    try:
                        if not result_path.exists():
                            for phase, command, cwd in self.commands(root):
                                self.stage(scene["name"] + ": " + phase, command, cwd, self.job / (scene["name"] + ".log"))
                        else:
                            self.req["training_inputs"] = read(root / "resolved_initialization.json")["training_inputs"]
                            for value in self.req["training_inputs"].values():
                                verify_record(value)
                        if sha(scene["baseline_results"]) != scene["baseline_results_sha256"]:
                            raise ValueError("Evaluation reference changed")
                        result = self.verify_result(root, dict(config=self.plan["reference_config"]))
                        entry.update(state="complete", metrics=result["metrics"], results=str(result_path), finished_at=now())
                    except Exception as error:
                        if isinstance(error, InterruptedError):
                            raise
                        entry.update(state="failed", error=str(error), finished_at=now())
                        traceback.print_exc()
                        print("SCENE FAILED; proceeding to next scene:", scene["name"], flush=True)
                    self.save(); self.report_progress()
                failed = [e["name"] for e in self.status["scenes"] if e["state"] != "complete"]
                self.status.update(state="complete_with_failures" if failed else "complete",
                    phase="Finished queued scenes", finished_at=now(), failed_scenes=failed)
                self.status.pop("child_pid", None)
                self.save(); self.report_progress()
            except BaseException as error:
                self.status.update(state="stopped" if isinstance(error, InterruptedError) else "failed",
                    error=str(error), finished_at=now())
                for entry in self.status["scenes"]:
                    if entry["state"] == "running":
                        entry.update(state=self.status["state"], error=str(error))
                self.save(); self.report_progress()
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    parser.add_argument("--scene-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", default="/home/member/.conda/envs/paintmesh/bin/python")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("GPU must be nonnegative")
    job = args.resume.resolve(strict=True) if args.resume else prepare(args)
    if args.detach:
        with (job / "run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("This batch is already running")
        runner = SceneBatch(job)
        runner.status["state"] = "queued"
        runner.save(); runner.report_progress()
        with (job / "run.log").open("a") as log:
            child = subprocess.Popen([args.python, str(Path(__file__).resolve()), "--resume", str(job)],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(job / "run.pid", str(child.pid) + "\n")
        info = dict(pid=child.pid, job=str(job), log=str(job / "run.log"), status=str(job / "status.json"),
            report=str(runner.report), output=str(runner.output), scenes=[s["name"] for s in runner.plan["scenes"]])
        write(REPO / "logs" / f"{PREFIX}_scenes_gpu{runner.plan['gpu']}.latest.json", info)
        print(json.dumps(info, ensure_ascii=False))
        return 0
    try:
        SceneBatch(job).run()
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
