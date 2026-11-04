#!/usr/bin/env python3
"""Official RGB appearance + preserved PaintMesh geometry; scale prior off/on."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_geometry_ablation import Batch, METRICS, LOWER, REPO, atomic, now, read, sha, write

DEFAULT_BASELINE = REPO / ("output/paintmesh/Inpaint360GS/inpaint360/bag/removal/"
    "table1_adaptive180_sweep/inpaint/edgs_depth_local_prune_perceptual")
OFFICIAL_CONFIG = REPO.parent / "Inpaint360GS/config/object_inpaint/common.json"
PRESERVED_LOSSES = ("alpha_preserve", "depth", "lama_normal", "depth_normal_consistency", "multiview")


def make_variants(reference, *, iterations=None, scale_weight=100., dssim=.8, lpips=.0005):
    if any(reference["loss"][key] <= 0 for key in PRESERVED_LOSSES):
        raise ValueError("All existing geometry and coverage supervision must remain enabled")
    if (isinstance(scale_weight, bool) or not isinstance(scale_weight, (int, float))
            or not math.isfinite(scale_weight) or scale_weight <= 0):
        raise ValueError("scale weight must be a finite positive number")
    if not 0 < dssim < 1 or not math.isfinite(lpips) or lpips <= 0:
        raise ValueError("invalid official appearance weights")
    common = copy.deepcopy(reference)
    if iterations is not None:
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be positive")
        common["iterations"] = iterations
    common["appearance"]["mode"] = "inpaint360gs"
    common["loss"].update(rgb_hole=0., rgb_known_preserve=1. - dssim, ssim=dssim, lpips=lpips)
    result = []
    for name, label, weight in (("00_scale_off", "尺度关闭", 0.), ("01_scale_on", "尺度开启", float(scale_weight))):
        cfg = copy.deepcopy(common)
        cfg["loss"]["scale_flatten"] = weight
        result.append(dict(name=name, label=label, config=cfg))
    return result


def prepare(args):
    from edgs_inpaint_io import appearance_weights, implementation, load_config, record, verify_record
    baseline = args.baseline_run.expanduser().resolve(strict=True)
    reference = args.reference_run.expanduser().resolve(strict=True) if args.reference_run else Path(str(baseline) + "_official_appearance_geo")
    cfg = load_config(baseline / "edgs_joint/config.resolved.json")
    official = read(args.official_config)
    variants = make_variants(cfg, iterations=args.iterations, scale_weight=args.scale_weight,
        dssim=official["lambda_dssim"], lpips=official["lambda_lpips"])
    joint = read(baseline / "manifests/edgs_joint_manifest.json")
    ref_joint = read(reference / "manifests/edgs_joint_manifest.json")
    published = read(baseline / "inpainted_3dgs/model_manifest.json")
    base_result = baseline / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
    ref_result = reference / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
    if (not all(obj.get("complete") for obj in (joint, ref_joint, published, read(base_result), read(ref_result)))
            or joint["inputs"] != ref_joint["inputs"]):
        raise ValueError("Baseline/reference must be complete and share initialization, LaMa and cameras")
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
        raise ValueError("Use the baseline paintmesh environment; LPIPS assets differ")
    for value in (*joint["inputs"].values(), *assets.values()):
        verify_record(value)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "official_appearance_scale_ablation"
    job = REPO / "logs" / f"{prefix}_gpu{args.gpu}_{stamp}"
    output = args.output_root.expanduser().resolve() if args.output_root else baseline.parent / f"{prefix}_{stamp}"
    report = args.report.expanduser().resolve(strict=True) if args.report else baseline / "comparison_paper_bag/comparison.md"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output contains files; use --resume JOB_DIR or another output root")
    if not Path(args.python).is_file() or not report.is_file():
        raise ValueError("Python executable or comparison report is missing")
    rows = read(baseline / "comparison_paper_bag/comparison.json")["rows"]
    metric_files = ("scripts/paintmesh/eval_inpaint360.py", "submodules/Inpaint360GS/utils/image_utils.py",
                    "submodules/Inpaint360GS/utils/loss_utils.py")
    orchestration = ("scripts/paintmesh/run_appearance_scale_ablation.py", "scripts/paintmesh/run_geometry_ablation.py")
    request = dict(created_at=now(), repo=str(REPO), baseline=str(baseline), reference=str(reference),
        output=str(output), report=str(report), gpu=args.gpu, python=str(Path(args.python).resolve()),
        baseline_results=str(base_result), reference_results=str(ref_result),
        baseline_results_sha256=sha(base_result), reference_results_sha256=sha(ref_result),
        paper={row["metric"]: row["paper_bag"] for row in rows},
        variants=variants, training_inputs=joint["inputs"], implementation=current, implementation_update=changed,
        appearance_weights=assets, publish_inputs=published["inputs"], publish_parameters=published["parameters"],
        metric_implementation={name: sha(REPO / name) for name in metric_files},
        orchestration_implementation={name: sha(REPO / name) for name in orchestration},
        official_config=record(args.official_config), official_config_values=official,
        source=read(base_result)["inputs"]["source_path"], frame_count=read(base_result)["frame_count"],
        reference_config=cfg, varying_fields=["loss.scale_flatten"],
        scale_definition="mean minimum activated Gaussian scale over visible rows; frozen rows constant; no ramp")
    job.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        root = output / variant["name"]
        (root / "manifests").mkdir(parents=True)
        write(root / "config.yaml", variant["config"])
    write(job / "request.json", request)
    write(job / "status.json", dict(state="prepared", created_at=now(), variants=[]))
    write(job / "configuration_diff.json", {"between_groups": {"loss.scale_flatten": [0., float(args.scale_weight)]},
        "preserved_supervision": {key: cfg["loss"][key] for key in PRESERVED_LOSSES},
        "iterations": variants[0]["config"]["iterations"], "appearance": variants[0]["config"]["appearance"],
        "appearance_weights": {key: variants[0]["config"]["loss"][key] for key in
                               ("rgb_hole", "rgb_known_preserve", "ssim", "lpips")}})
    atomic(job / "comparison.before.md", report.read_text())
    return job


class ScaleBatch(Batch):
    def commands(self, root):
        return super().commands(root, iterations=read(root / "config.yaml")["iterations"])

    def run(self):
        for name, expected in self.req["orchestration_implementation"].items():
            if sha(REPO / name) != expected:
                raise ValueError("Batch implementation changed; preserve this job: " + name)
        if sha(self.req["official_config"]["path"]) != self.req["official_config"]["sha256"]:
            raise ValueError("Recorded official appearance configuration changed")
        super().run()

    def report_progress(self):
        req, status = self.req, self.status
        common = req["variants"][0]["config"]
        states = {v["name"]: v for v in status["variants"]}
        completed = [v for v in status["variants"] if v.get("state") == "complete"]
        baseline, reference = [read(req[key])["metrics"] for key in ("baseline_results", "reference_results")]
        loss = common["loss"]
        lines = ["## 官方式外观＋保留几何监督：尺度扁平化消融", "",
            f"状态：{status['state']}；当前阶段：{status.get('phase', '准备')}。", "",
            f"两组均从同一初始化独立训练 {common['iterations']} 步，区别仅为 loss.scale_flatten=0 或 " + str(req["variants"][1]["config"]["loss"]["scale_flatten"]) + "。", "",
            f"RGB 外观目标统一为现有 LaMa 输出；洞内 L1=0，洞外 LaMa L1={loss['rgb_known_preserve']:.6g}，全图 SSIM={loss['ssim']:.6g}，原生 mask 包围框 2×2 LPIPS={loss['lpips']:.6g}。使用官方公开代码默认配置，非论文公式的洞内 L1。", "",
            "保留的监督权重：`" + json.dumps({key: loss[key] for key in PRESERVED_LOSSES}) + "`。", "",
            f"初始化 depth={common['init']['use_depth']}/normal={common['init']['use_normal']}；相同 LaMa、180 帧相机、随机种子、学习率、局部旧高斯集合及远处冻结。geometry_from_iter={common['geometry_from_iter']}、geometry_ramp_iters={common['geometry_ramp_iters']}；pruning={json.dumps(common['pruning'])}；adaptive_density.enabled={common['adaptive_density']['enabled']}。", "",
            "尺度项采用 PGSR 原定义：所有可见高斯最小激活尺度的均值；冻结背景仅贡献常数，不会获得梯度。尺度项从第一步启用，不随几何权重渐增；没有添加 NCC 或改动其余几何损失。", "",
            "| 实验 | 尺度权重 | 进度 |", "|---|---:|---|"]
        for variant in req["variants"]:
            lines.append(f"| {variant['label']} | {variant['config']['loss']['scale_flatten']} | {states.get(variant['name'], {}).get('state', '等待')} |")
        lines += ["", f"评估使用同一 {req['frame_count']} 个 test_* 相机、resolution=4、官方 unseen_mask 与七项指标实现。", "",
            "| 指标 | 论文 bag | 原始外观＋几何 | 历史官方式＋几何 | 本次尺度关闭 | 本次尺度开启 | 开启−关闭 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        off, on = [states.get(v["name"], {}) for v in req["variants"]]
        for key in METRICS:
            values = [f"{req['paper'][key]:.6f}", f"{baseline[key]:.6f}", f"{reference[key]:.6f}"]
            for value in (off, on):
                values.append(f"{value['metrics'][key]:.6f}" if value.get("state") == "complete" else "—")
            values.append(f"{on['metrics'][key]-off['metrics'][key]:+.6f}" if all(v.get("state") == "complete" for v in (off, on)) else "—")
            lines.append(f"| {key} {'↓' if key in LOWER else '↑'} | " + " | ".join(values) + " |")
        lines += ["", "历史结果仅作参考；两组新实验使用同一当前实现。配置差异及代码版本见 request.json 与 configuration_diff.json。论文列沿用发表值。", ""]
        if status.get("error"):
            lines += ["错误：`" + status["error"] + "`。", ""]
        lines += [f"结果目录：`{self.output}`", f"总日志：`{self.job / 'run.log'}`", f"状态：`{self.job / 'status.json'}`", "",
            f"恢复命令：`{sys.executable} {Path(__file__).resolve()} --resume {self.job} --detach`", "",
            "本批次训练、发布并评估 3DGS，不额外执行 TSDF mesh 后处理。"]
        body = "\n".join(lines) + "\n"
        begin, end = [f"<!-- APPEARANCE_SCALE_ABLATION:{self.job.name}:{tag} -->" for tag in ("BEGIN", "END")]
        with (self.report.parent / ".comparison.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self.report.read_text()
            if begin in current:
                head, _, tail = current.partition(begin)
                if end not in tail:
                    raise ValueError("Report section is damaged; preserve previous text")
                _, _, rest = tail.partition(end)
                updated = head + begin + "\n" + body + end + rest
            else:
                updated = current.rstrip() + "\n\n" + begin + "\n" + body + end + "\n"
            atomic(self.report, updated)
        atomic(self.output / "comparison.md", body)
        write(self.output / "summary.json", dict(complete=status["state"] == "complete", request=str(self.job / "request.json"),
              variants=status["variants"], report=str(self.report)))
        if completed:
            weights = {v["name"]: v["config"]["loss"]["scale_flatten"] for v in req["variants"]}
            with (self.output / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["experiment", "scale_weight", *METRICS])
                writer.writeheader()
                writer.writerows(dict(experiment=v["name"], scale_weight=weights[v["name"]], **v["metrics"]) for v in completed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--official-config", type=Path, default=OFFICIAL_CONFIG)
    parser.add_argument("--iterations", type=int, help="Default: preserve baseline 5000 iterations")
    parser.add_argument("--scale-weight", type=float, default=100.)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", default="/home/member/.conda/envs/paintmesh/bin/python")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
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
        info = dict(pid=child.pid, job=str(job), log=str(job / "run.log"), status=str(job / "status.json"), output=req["output"], report=req["report"])
        write(REPO / "logs" / f"official_appearance_scale_ablation_gpu{req['gpu']}.latest.json", info)
        print(json.dumps(info, ensure_ascii=False))
        return 0
    try:
        ScaleBatch(job).run()
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
