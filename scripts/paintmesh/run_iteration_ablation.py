#!/usr/bin/env python3
"""Original PaintMesh appearance + geometry; vary only the training duration."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_geometry_ablation import Batch, METRICS, LOWER, REPO, atomic, now, read, sha, write

DEFAULT_BASELINE = REPO / ("output/paintmesh/Inpaint360GS/inpaint360/bag/removal/"
    "table1_adaptive180_sweep/inpaint/edgs_depth_local_prune_perceptual")
ITERATIONS = (8000, 10000, 15000, 20000)


def make_variants(reference, iterations=ITERATIONS):
    if reference["appearance"]["mode"] != "masked":
        raise ValueError("This experiment requires the original masked appearance supervision")
    if any(reference["loss"][k] <= 0 for k in ("rgb_hole", "rgb_known_preserve", "ssim", "lpips",
                                               "alpha_preserve", "depth", "lama_normal",
                                               "depth_normal_consistency", "multiview")):
        raise ValueError("Original appearance and all geometry/coverage losses must remain enabled")
    if not iterations or len(set(iterations)) != len(iterations) or any(type(n) is not int or n <= 0 for n in iterations):
        raise ValueError("Training iterations must be distinct positive integers")
    result = []
    for count in iterations:
        cfg = copy.deepcopy(reference)
        cfg["iterations"] = count
        result.append(dict(name=f"iter_{count:05d}", label=f"{count} 步", config=cfg))
    return result


def prepare(args):
    from edgs_inpaint_io import (appearance_weights, implementation, load_config, verify_record,
        validate_init, validate_selection, check_initialization_config)
    baseline = args.baseline_run.expanduser().resolve(strict=True)
    cfg = load_config(baseline / "edgs_joint/config.resolved.json")
    variants = make_variants(cfg, args.iterations)
    joint = read(baseline / "manifests/edgs_joint_manifest.json")
    published = read(baseline / "inpainted_3dgs/model_manifest.json")
    result_path = baseline / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
    results = read(result_path)
    if not all(obj.get("complete") for obj in (joint, published, results)):
        raise ValueError("Baseline training, publication and evaluation must be complete")
    if cfg["iterations"] != results["inputs"]["iteration"]:
        raise ValueError("Baseline evaluation iteration differs from its training")
    current = implementation()
    allowed = {"scripts/paintmesh/configs/edgs_inpaint.yaml", "scripts/paintmesh/edgs_inpaint_io.py",
               "submodules/EDGS/source/paintmesh_joint_losses.py", "submodules/EDGS/tools/train_paintmesh_pgsr.py"}
    changed = {}
    for name, expected in joint["request"]["implementation"].items():
        if current.get(name) != expected:
            if name not in allowed:
                raise ValueError("Unaudited training implementation difference: " + name)
            changed[name] = dict(baseline=expected, current=current.get(name))
    assets = appearance_weights(cfg)
    if assets != joint["request"]["appearance_weights"]:
        raise ValueError("Use the baseline paintmesh Python environment; LPIPS assets differ")
    for record in (*joint["inputs"].values(), *assets.values()):
        verify_record(record)
    initial = validate_init(joint["inputs"]["initialization"]["path"])
    check_initialization_config(initial, cfg)
    validate_selection(joint["inputs"]["local_selection"]["path"], initial=initial, config=cfg["local_background"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "original_supervision_iteration_ablation"
    job = REPO / "logs" / f"{prefix}_gpu{args.gpu}_{stamp}"
    output = args.output_root.expanduser().resolve() if args.output_root else baseline.parent / f"{prefix}_{stamp}"
    report = args.report.expanduser().resolve(strict=True) if args.report else baseline / "comparison_paper_bag/comparison.md"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output contains files; use --resume JOB_DIR or another --output-root")
    if not Path(args.python).is_file() or not report.is_file():
        raise ValueError("Python executable or comparison report is missing")
    paper_path = (args.paper_comparison.expanduser().resolve(strict=True) if args.paper_comparison else
                  baseline / "comparison_paper_bag/comparison.json")
    rows = read(paper_path)["rows"]
    metric_files = ("scripts/paintmesh/eval_inpaint360.py", "submodules/Inpaint360GS/utils/image_utils.py",
                    "submodules/Inpaint360GS/utils/loss_utils.py")
    orchestration = ("scripts/paintmesh/run_iteration_ablation.py", "scripts/paintmesh/run_geometry_ablation.py")
    request = dict(created_at=now(), repo=str(REPO), baseline=str(baseline), reference=str(baseline),
        baseline_iteration=cfg["iterations"], output=str(output), report=str(report), gpu=args.gpu,
        python=str(Path(args.python).resolve()), baseline_results=str(result_path), reference_results=str(result_path),
        baseline_results_sha256=sha(result_path), reference_results_sha256=sha(result_path),
        paper={row["metric"]: row["paper_bag"] for row in rows},
        paper_comparison=dict(path=str(paper_path), sha256=sha(paper_path)),
        paper_source="https://arxiv.org/html/2511.06457v1#A4", variants=variants,
        training_inputs=joint["inputs"], implementation=current, implementation_update=changed,
        implementation_added=sorted(set(current) - set(joint["request"]["implementation"])),
        appearance_weights=assets, publish_inputs=published["inputs"], publish_parameters=published["parameters"],
        metric_implementation={name: sha(REPO / name) for name in metric_files},
        orchestration_implementation={name: sha(REPO / name) for name in orchestration},
        source=results["inputs"]["source_path"], frame_count=results["frame_count"], reference_config=cfg,
        initialization_mode=cfg.get("surface_init", {}).get("mode", "none"),
        varying_fields=["iterations"], policy="independent training from identical initial PLY; original appearance and geometry")
    job.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        root = output / variant["name"]
        (root / "manifests").mkdir(parents=True)
        write(root / "config.yaml", variant["config"])
    write(job / "request.json", request)
    write(job / "status.json", dict(state="prepared", created_at=now(), variants=[]))
    atomic(job / "comparison.before.md", report.read_text())
    write(job / "configuration_diff.json", {v["name"]: dict(iterations=dict(before=cfg["iterations"],
                                                                           after=v["config"]["iterations"])) for v in variants})
    return job


class IterationBatch(Batch):
    def commands(self, root):
        return super().commands(root, iterations=read(root / "config.yaml")["iterations"])

    def run(self):
        for name, expected in self.req["orchestration_implementation"].items():
            if sha(REPO / name) != expected:
                raise ValueError("Batch implementation changed; preserve this job: " + name)
        super().run()

    def report_progress(self):
        req, status = self.req, self.status
        cfg, loss = req["reference_config"], req["reference_config"]["loss"]
        states = {v["name"]: v for v in status["variants"]}
        completed = [v for v in status["variants"] if v.get("state") == "complete"]
        baseline = read(req["baseline_results"])["metrics"]
        lines = ["## 原始外观＋几何监督：训练步数消融", "",
            f"状态：{status['state']}；当前阶段：{status.get('phase', '准备')}。", "",
            f"{len(req['variants'])} 组仅改变 `iterations`；每组从同一初始化点云重新训练，不从已有 {req['baseline_iteration']} 步或上一组 checkpoint 续训。", "",
            f"初始化模式：`{cfg.get('surface_init', {}).get('mode', 'none')}`。复用同一初始化 manifest/PLY 和旧点更新名单，不重新执行匹配或初始化，也不清理或替换其中的旧高斯。", "",
            "原始监督：洞内 RGB=" + str(loss["rgb_hole"]) + "、背景保持=" + str(loss["rgb_known_preserve"]) +
            "、SSIM=" + str(loss["ssim"]) + "、masked LPIPS=" + str(loss["lpips"]) + "、覆盖=" + str(loss["alpha_preserve"]) +
            "、深度=" + str(loss["depth"]) + "、法线=" + str(loss["lama_normal"]) +
            "、深度—法线一致性=" + str(loss["depth_normal_consistency"]) + "、多视图=" + str(loss["multiview"]) + "。", "",
            f"固定 seed={cfg['seed']}、初始化 depth={cfg['init']['use_depth']}/normal={cfg['init']['use_normal']}、同一 LaMa 与 180 帧相机、相同学习率和局部旧高斯选择；远处背景冻结。几何从 {cfg['geometry_from_iter']} 步开始，渐增 {cfg['geometry_ramp_iters']} 步，不随总步数拉伸。", "",
            f"密度配置保持基线：adaptive_density.enabled={cfg['adaptive_density']['enabled']}；pruning={json.dumps(cfg['pruning'])}。不延长剪枝窗口；20000 步组在达到既定截止点后继续优化但不再剪枝。", "",
            "| 实验 | 进度 |", "|---|---|"]
        for variant in req["variants"]:
            lines.append(f"| {variant['label']} | {states.get(variant['name'], {}).get('state', '等待')} |")
        lines += ["", f"相同 {req['frame_count']} 个 test_* 相机、resolution=4、官方 unseen_mask；每组用自己的最终迭代 PLY 评估。", "",
            f"| 指标 | 论文 bag | {'depth_anchor 基线' if cfg.get('surface_init', {}).get('mode') == 'depth_anchor' else '原始'} {req['baseline_iteration']} 步 | " + " | ".join(v["label"] for v in req["variants"]) + " |",
            "|---|---:|---:|" + "---:|" * len(req["variants"])]
        for key in METRICS:
            values = [f"{req['paper'][key]:.6f}", f"{baseline[key]:.6f}"]
            for variant in req["variants"]:
                value = states.get(variant["name"], {})
                values.append(f"{value['metrics'][key]:.6f}" if value.get("state") == "complete" else "—")
            lines.append(f"| {key} {'↓' if key in LOWER else '↑'} | " + " | ".join(values) + " |")
        lines += ["", f"{len(req['variants'])} 组训练配置逐字段相同，仅 iterations 不同；配置差异见日志目录的 configuration_diff.json。基线 {req['baseline_iteration']} 步为同初始化的已完成结果；代码哈希变化记录在 request.json/implementation_update 中。论文列为已发表值。", ""]
        if status.get("error"):
            lines += ["错误：`" + status["error"] + "`。", ""]
        lines += [f"结果目录：`{self.output}`", f"总日志：`{self.job / 'run.log'}`", f"状态：`{self.job / 'status.json'}`", "",
            f"恢复命令：`{sys.executable} {Path(__file__).resolve()} --resume {self.job} --detach`", "",
            "本批次训练、发布并评估 3DGS，不额外执行 TSDF mesh 后处理。"]
        body = "\n".join(lines) + "\n"
        begin, end = [f"<!-- ITERATION_ABLATION:{self.job.name}:{tag} -->" for tag in ("BEGIN", "END")]
        with (self.report.parent / ".comparison.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self.report.read_text()
            if begin in current:
                head, _, tail = current.partition(begin)
                if end not in tail:
                    raise ValueError("Report section is damaged; preserve existing text")
                _, _, rest = tail.partition(end)
                updated = head + begin + "\n" + body + end + rest
            else:
                updated = current.rstrip() + "\n\n" + begin + "\n" + body + end + "\n"
            atomic(self.report, updated)
        atomic(self.output / "comparison.md", body)
        write(self.output / "summary.json", dict(complete=status["state"] == "complete",
            request=str(self.job / "request.json"), variants=status["variants"], report=str(self.report)))
        if completed:
            with (self.output / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["experiment", "iterations", *METRICS])
                writer.writeheader()
                targets = {v["name"]: v["config"]["iterations"] for v in req["variants"]}
                writer.writerows(dict(experiment=v["name"], iterations=targets[v["name"]], **v["metrics"]) for v in completed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--iterations", type=int, nargs="+", default=list(ITERATIONS))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", default="/home/member/.conda/envs/paintmesh/bin/python")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--paper-comparison", type=Path, help="Paper metric rows JSON; independent of the actual training baseline")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    # Build manifests and LPIPS provenance in the same interpreter as training.
    if not args.resume and Path(sys.executable).resolve() != Path(args.python).resolve():
        return subprocess.call([args.python, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            cwd=REPO, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    job = args.resume.expanduser().resolve(strict=True) if args.resume else prepare(args)
    if args.detach:
        with (job / "run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("This batch is already running; inspect status.json")
        with (job / "run.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--resume", str(job)],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(job / "run.pid", str(child.pid) + "\n")
        req = read(job / "request.json")
        info = dict(pid=child.pid, job=str(job), log=str(job / "run.log"), status=str(job / "status.json"),
                    output=req["output"], report=req["report"])
        write(REPO / "logs" / f"original_supervision_iteration_ablation_gpu{req['gpu']}.latest.json", info)
        print(json.dumps(info, ensure_ascii=False))
        return 0
    try:
        IterationBatch(job).run()
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
