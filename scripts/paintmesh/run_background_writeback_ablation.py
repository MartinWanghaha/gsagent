#!/usr/bin/env python3
"""Original PaintMesh supervision; train all Gaussians and restore old background."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_geometry_ablation import Batch, METRICS, LOWER, REPO, atomic, now, read, sha, write
from run_iteration_ablation import DEFAULT_BASELINE

PREFIX = "original_supervision_background_writeback"
OFFICIAL = REPO.parent / "Inpaint360GS"


def make_variant(reference):
    if reference["appearance"]["mode"] != "masked":
        raise ValueError("This experiment requires original masked appearance supervision")
    if any(reference["loss"][key] <= 0 for key in ("rgb_hole", "rgb_known_preserve", "ssim", "lpips",
            "alpha_preserve", "depth", "lama_normal", "depth_normal_consistency", "multiview")):
        raise ValueError("Original appearance and geometry/coverage must remain enabled")
    if reference["adaptive_density"]["enabled"] or reference["background_writeback"]["enabled"]:
        raise ValueError("Reference must be the original prune-only, locally frozen experiment")
    if not reference["freeze_background"] or not reference["local_background"]["enabled"]:
        raise ValueError("Reference must use the original local-background policy")
    cfg = copy.deepcopy(reference)
    cfg["freeze_background"] = False
    cfg["local_background"]["enabled"] = False
    cfg["background_writeback"]["enabled"] = True
    return dict(name="train_all_restore_local", label="全点训练＋官方式局部写回", config=cfg)


def prepare(args):
    from edgs_inpaint_io import appearance_weights, implementation, load_config, verify_record
    baseline = args.baseline_run.expanduser().resolve(strict=True)
    cfg = load_config(baseline / "edgs_joint/config.resolved.json")
    variant = make_variant(cfg)
    joint = read(baseline / "manifests/edgs_joint_manifest.json")
    published = read(baseline / "inpainted_3dgs/model_manifest.json")
    result_path = baseline / "inpainted_3dgs/evaluation/inpaint360_r4/results.json"
    results = read(result_path)
    if not all(value.get("complete") for value in (joint, published, results)):
        raise ValueError("Baseline training/publication/evaluation must be complete")
    if cfg["iterations"] != results["inputs"]["iteration"]:
        raise ValueError("Baseline training/evaluation iterations differ")
    current = implementation()
    allowed = {"scripts/paintmesh/configs/edgs_inpaint.yaml", "scripts/paintmesh/edgs_inpaint_io.py",
        "submodules/EDGS/source/paintmesh_joint_losses.py", "submodules/EDGS/source/paintmesh_joint_data.py",
        "submodules/EDGS/tools/train_paintmesh_pgsr.py"}
    changed = {}
    for name, expected in joint["request"]["implementation"].items():
        if current.get(name) != expected:
            if name not in allowed:
                raise ValueError("Unaudited implementation difference: " + name)
            changed[name] = dict(baseline=expected, current=current.get(name))
    assets = appearance_weights(cfg)
    if assets != joint["request"]["appearance_weights"]:
        raise ValueError("Baseline LPIPS assets differ; use the paintmesh environment")
    for value in (*joint["inputs"].values(), *assets.values()):
        verify_record(value)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = REPO / "logs" / f"{PREFIX}_gpu{args.gpu}_{stamp}"
    output = args.output_root.expanduser().resolve() if args.output_root else baseline.parent / f"{PREFIX}_{stamp}"
    report = args.report.expanduser().resolve(strict=True) if args.report else baseline / "comparison_paper_bag/comparison.md"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output contains files; use --resume JOB_DIR or a new output root")
    if not Path(args.python).is_file() or not report.is_file():
        raise ValueError("Python executable or comparison report missing")
    rows = read(baseline / "comparison_paper_bag/comparison.json")["rows"]
    training_inputs = {key: value for key, value in joint["inputs"].items() if key != "local_selection"}
    official_files = ("edit_object_inpaint.py", "scene/gaussian_model.py", "utils/point_utils.py")
    request = dict(created_at=now(), repo=str(REPO), baseline=str(baseline), reference=str(baseline),
        baseline_iteration=cfg["iterations"], output=str(output), report=str(report), gpu=args.gpu,
        python=str(Path(args.python).resolve()), baseline_results=str(result_path), reference_results=str(result_path),
        baseline_results_sha256=sha(result_path), reference_results_sha256=sha(result_path),
        paper={row["metric"]: row["paper_bag"] for row in rows}, variants=[variant],
        training_inputs=training_inputs, baseline_training_inputs=joint["inputs"],
        implementation=current, implementation_update=changed,
        implementation_added=sorted(set(current) - set(joint["request"]["implementation"])),
        appearance_weights=assets, publish_inputs=published["inputs"], publish_parameters=published["parameters"],
        metric_implementation={name: sha(REPO / name) for name in ("scripts/paintmesh/eval_inpaint360.py",
            "submodules/Inpaint360GS/utils/image_utils.py", "submodules/Inpaint360GS/utils/loss_utils.py")},
        orchestration_implementation={name: sha(REPO / name) for name in (
            "scripts/paintmesh/run_background_writeback_ablation.py", "scripts/paintmesh/run_geometry_ablation.py")},
        official_source={str(OFFICIAL / name): sha(OFFICIAL / name) for name in official_files},
        seg_implementation={name: sha(REPO / name) for name in ("scripts/paintmesh/run_seg.sh",
            "submodules/EDGS/train.py", "submodules/EDGS/source/trainer.py")},
        source=results["inputs"]["source_path"], frame_count=results["frame_count"], reference_config=cfg,
        varying_fields=["freeze_background", "local_background.enabled", "background_writeback.enabled"],
        policy="train all old/new; retain old identities; first-view projection AND seed-neighborhood old writeback")
    job.mkdir(parents=True)
    (output / variant["name"] / "manifests").mkdir(parents=True)
    write(output / variant["name"] / "config.yaml", variant["config"])
    write(job / "request.json", request)
    write(job / "status.json", dict(state="prepared", created_at=now(), variants=[]))
    atomic(job / "comparison.before.md", report.read_text())
    write(job / "configuration_diff.json", {"freeze_background": dict(before=True, after=False),
        "local_background.enabled": dict(before=True, after=False),
        "background_writeback.enabled": dict(before=False, after=True)})
    return job


class WritebackBatch(Batch):
    def commands(self, root):
        # Parent command builder expects a pre-training local-selection argument.
        # Retain all other commands unchanged, remove only that obsolete input.
        shared = self.req["training_inputs"]
        self.req["training_inputs"] = {**shared, "local_selection": {"path": "unused"}}
        try:
            commands = super().commands(root)
        finally:
            self.req["training_inputs"] = shared
        train = commands[0][1]
        offset = train.index("--local-selection")
        del train[offset:offset + 2]
        return commands

    def run(self):
        for key in ("orchestration_implementation", "seg_implementation"):
            for name, expected in self.req[key].items():
                if sha(REPO / name) != expected:
                    raise ValueError("Implementation changed: " + name)
        super().run()

    def verify_result(self, root, variant):
        from edgs_inpaint_io import validate_joint
        validate_joint(root / "manifests/edgs_joint_manifest.json")
        for name, expected in self.req["seg_implementation"].items():
            if sha(REPO / name) != expected:
                raise ValueError("seg implementation changed during experiment: " + name)
        return super().verify_result(root, variant)

    def report_progress(self):
        req, status = self.req, self.status
        cfg, loss = req["reference_config"], req["reference_config"]["loss"]
        entry = next(iter(status["variants"]), {})
        baseline = read(req["baseline_results"])["metrics"]
        lines = ["## 原始外观＋几何监督：全点训练、恢复背景及官方式局部写回", "",
            f"状态：{status['state']}；当前阶段：{status.get('phase', '准备')}。", "",
            f"从相同初始化重新训练 {cfg['iterations']} 步；保持原始 masked 外观监督，不切换官方外观损失。", "",
            "唯一实验因素是旧高斯更新／最终合成策略：训练时全部旧点和新点可更新；训练结束后按第一虚拟视角的训练后投影选区，mask 椭圆膨胀至原面积约 110%，再与初始化新点的三维最近邻距离门控求交；阈值为 seed 三轴标准差均值的 3 倍。恢复选区外旧点的全部参数，保留选区内旧点更新及全部存活新点。", "",
            "为固定初始化，三维门控使用原实验相同的新点 seed，未换成官方单视角 00004.ply。这是更新策略消融，不是整个官方方法复现。", "",
            "保留原剪枝阈值与窗口、不增密；官方式恢复要求旧点身份始终存在，因此本策略只剪枝新点。基线会剪枝可训练的局部旧点，此项是更新／恢复策略的一部分。", "",
            "固定 RGB 洞内=" + str(loss["rgb_hole"]) + "、背景=" + str(loss["rgb_known_preserve"]) +
            "、SSIM=" + str(loss["ssim"]) + "、LPIPS=" + str(loss["lpips"]) +
            "、覆盖=" + str(loss["alpha_preserve"]) + "、深度=" + str(loss["depth"]) +
            "、法线=" + str(loss["lama_normal"]) + "、深度—法线=" + str(loss["depth_normal_consistency"]) +
            "、多视图=" + str(loss["multiview"]) + "；LaMa、180 帧相机、depth 混合初始化、种子、学习率及几何渐增进度均与基线相同。", "",
            f"相同 {req['frame_count']} 个 test_* 相机、resolution=4、官方 unseen_mask，评估最终局部写回后的模型。", "",
            "| 指标 | 论文 bag | 原始局部解冻 | 全点训练＋局部写回 | 差值（本次−原始） |",
            "|---|---:|---:|---:|---:|"]
        for key in METRICS:
            before = baseline[key]
            values = f"{entry['metrics'][key]:.6f} | {entry['metrics'][key] - before:+.6f}" if entry.get("state") == "complete" else "— | —"
            lines.append(f"| {key} {'↓' if key in LOWER else '↑'} | {req['paper'][key]:.6f} | {before:.6f} | {values} |")
        root = self.output / req["variants"][0]["name"]
        summary = root / "edgs_joint/writeback_summary.json"
        if summary.exists():
            details = read(summary)
            lines += ["", f"实际写回：第一视角 {details['first_view']}；保留更新的旧点 {details['selected_old']:,}，恢复旧点 {details['restored_old']:,}；训练中发生变化的旧点 {details['changed_old_before_restore']:,}。", "",
                f"三维阈值 {details['seed_distance_threshold']:.8g}，mask 面积 {details['mask_area']} → {details['expanded_area']}；范围外旧点通过 PLY 逐字段完全一致校验。"]
        if status.get("error"):
            lines += ["", "错误：`" + status["error"] + "`。"]
        lines += ["", "论文列沿用发表值；原始列为历史结果。兼容实现的哈希变化与新增模块记录在 request.json；seg 入口与 Trainer 哈希单独核验。", "",
            f"结果目录：`{root}`", f"总日志：`{self.job / 'run.log'}`", f"状态：`{self.job / 'status.json'}`", "",
            f"恢复命令：`{req['python']} {Path(__file__).resolve()} --resume {self.job} --detach`", "",
            "本次自动执行训练、局部写回、模型发布和七项 RGB 指标评估，不另执行 TSDF mesh 后处理。"]
        body = "\n".join(lines) + "\n"
        begin, end = [f"<!-- BACKGROUND_WRITEBACK:{self.job.name}:{tag} -->" for tag in ("BEGIN", "END")]
        with (self.report.parent / ".comparison.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self.report.read_text()
            if begin in current:
                head, _, tail = current.partition(begin)
                if end not in tail:
                    raise ValueError("Existing report section damaged")
                _, _, rest = tail.partition(end)
                updated = head + begin + "\n" + body + end + rest
            else:
                updated = current.rstrip() + "\n\n" + begin + "\n" + body + end + "\n"
            atomic(self.report, updated)
        atomic(self.output / "comparison.md", body)
        write(self.output / "summary.json", dict(complete=status["state"] == "complete",
            request=str(self.job / "request.json"), variants=status["variants"], report=str(self.report)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
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
                parser.error("This job is already running; inspect status.json")
        with (job / "run.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--resume", str(job)],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(job / "run.pid", str(child.pid) + "\n")
        req = read(job / "request.json")
        info = dict(pid=child.pid, job=str(job), log=str(job / "run.log"), status=str(job / "status.json"),
            output=req["output"], report=req["report"])
        write(REPO / "logs" / f"{PREFIX}_gpu{req['gpu']}.latest.json", info)
        print(json.dumps(info, ensure_ascii=False))
        return 0
    try:
        WritebackBatch(job).run()
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
