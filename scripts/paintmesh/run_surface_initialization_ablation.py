#!/usr/bin/env python3
"""Train/evaluate the original supervision with only RGB-anchored depth initialization changed."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_geometry_ablation import Batch, METRICS, LOWER, REPO, atomic, now, read, sha, write
from run_iteration_ablation import DEFAULT_BASELINE

PREFIX = "original_supervision_depth_anchor_init"


def make_variant(reference):
    if not reference["init"]["use_depth"] or reference["init"]["use_normal"]:
        raise ValueError("Depth-anchor ablation requires baseline depth=true, normal=false")
    if reference["appearance"]["mode"] != "masked":
        raise ValueError("Requires the original masked appearance supervision")
    if any(reference["loss"][key] <= 0 for key in ("rgb_hole", "rgb_known_preserve", "ssim", "lpips",
            "alpha_preserve", "depth", "lama_normal", "depth_normal_consistency", "multiview")):
        raise ValueError("Original appearance and geometry/coverage must remain enabled")
    if (reference["surface_init"]["mode"] != "none" or reference["adaptive_density"]["enabled"]
            or reference["background_writeback"]["enabled"] or not reference["pruning"]["enabled"]
            or not reference["freeze_background"] or not reference["local_background"]["enabled"]):
        raise ValueError("Requires the original mixed initialization, local updates and prune-only policy")
    cfg = copy.deepcopy(reference)
    cfg["surface_init"]["mode"] = "depth_anchor"
    return dict(name="depth_anchor", label="RGB 锚点深度初始化", config=cfg)


def rebind_arrays(old_editable, old_trainable, new_editable, old_votes):
    """Keep every old row's eligibility; a changed new population is always trainable."""
    import numpy as np
    for editable in (old_editable, new_editable):
        if editable.dtype != np.bool_ or editable.ndim != 1 or not editable.any():
            raise ValueError("Invalid editable row ownership")
    old_count = int((~old_editable).sum())
    if (old_count != int((~new_editable).sum()) or old_editable[:old_count].any()
            or new_editable[:old_count].any() or not old_editable[old_count:].all()
            or not new_editable[old_count:].all()):
        raise ValueError("Old background row ordering/count differs")
    if (old_trainable.dtype != np.bool_ or old_trainable.shape != old_editable.shape
            or not old_trainable[old_count:].all() or old_votes.shape != old_editable.shape):
        raise ValueError("Invalid baseline selection/votes")
    trainable = new_editable.copy()
    trainable[:old_count] = old_trainable[:old_count]
    if not trainable[:old_count].any() or trainable[:old_count].all():
        raise ValueError("Baseline must update some old rows and freeze distant rows")
    votes = np.zeros(len(new_editable), dtype=old_votes.dtype)
    votes[:old_count] = old_votes[:old_count]
    stats = dict(initial_old=old_count, initial_new=int(new_editable.sum()),
        trainable_old=int(trainable[:old_count].sum()), frozen_old=int((~trainable[:old_count]).sum()),
        old_trainable_sha256=hashlib.sha256(trainable[:old_count].tobytes()).hexdigest(),
        old_mask_bitwise_equal=bool(np.array_equal(trainable[:old_count], old_trainable[:old_count])),
        policy="baseline_old_rows_fixed; all new rows trainable; new overlap votes not recomputed")
    return trainable, votes, stats


def rebind_selection(baseline_inputs, new_path, config, root):
    import numpy as np
    from edgs_inpaint_io import (validate_init, validate_selection, record, seal, SELECTION_KIND,
                                implementation, atomic_write, verify_record)
    old = validate_init(baseline_inputs["initialization"]["path"])
    new = validate_init(new_path)
    if ("proposal_initialization" in new["inputs"] and
            new["inputs"]["proposal_initialization"] != baseline_inputs["initialization"]):
        raise ValueError("Depth-anchor initializer must derive from this exact baseline proposal")
    if {k: v for k, v in new["inputs"].items() if k != "proposal_initialization"} != old["inputs"]:
        raise ValueError("Depth-anchor initialization changes inputs other than the baseline proposal")
    for key in ("init", "matcher", "seed"):
        if new["request"]["config"][key] != old["request"]["config"][key]:
            raise ValueError("Depth-anchor initialization changed baseline " + key)
    if new["request"]["matcher_weights"] != old["request"]["matcher_weights"]:
        raise ValueError("Depth-anchor initializer uses different matcher weights")
    from edgs_inpaint_io import check_initialization_config
    check_initialization_config(new, config)
    manifest = root.parent / "manifests/local_selection_manifest.json"
    if manifest.exists():
        existing = validate_selection(manifest, initial=new, config=config["local_background"])
        if existing["inputs"].get("baseline_selection") != baseline_inputs["local_selection"]:
            raise ValueError("Existing selection belongs to another baseline")
        return record(manifest), read(existing["outputs"]["summary"]["path"]), new
    selection = validate_selection(baseline_inputs["local_selection"]["path"], initial=old,
                                   config=config["local_background"])
    arrays = [np.load(path, allow_pickle=False) for path in (
        old["outputs"]["editable_mask"]["path"], selection["outputs"]["trainable_mask"]["path"],
        new["outputs"]["editable_mask"]["path"], selection["outputs"]["overlap_votes"]["path"])]
    trainable, votes, stats = rebind_arrays(*arrays)
    stats["baseline_selection"] = baseline_inputs["local_selection"]
    stats["baseline_summary"] = read(selection["outputs"]["summary"]["path"])
    records = {**selection["inputs"], "initialization": record(new_path),
               "baseline_selection": baseline_inputs["local_selection"]}
    request = dict(inputs=records, config=config["local_background"], implementation=implementation(),
        output_root=str(root.resolve()), policy="preserve baseline old-row selection")
    root.mkdir(parents=True, exist_ok=True)
    for name, array in (("trainable_mask", trainable), ("overlap_votes", votes)):
        atomic_write(root / (name + ".npy"), lambda handle, a=array: np.save(handle, a, allow_pickle=False))
    write(root / "selection_summary.json", stats)
    for value in records.values():
        verify_record(value)
    receipt = seal(SELECTION_KIND, request=request, inputs=records, initialization_id=new["artifact_id"],
        outputs={key: record(root / name) for key, name in (("trainable_mask", "trainable_mask.npy"),
            ("overlap_votes", "overlap_votes.npy"), ("summary", "selection_summary.json"))})
    validate_selection(receipt, initial=new, config=config["local_background"])
    write(manifest, receipt)
    return record(manifest), stats, new


def initialization_command(python, initial, root):
    """Fresh RGB matching, including triangle-closing edges; never the lossy old cache."""
    command = [python, str(REPO / "submodules/EDGS/tools/initialize_paintmesh_edgs.py")]
    for option, key in (("removed-model", "removed_model"), ("inpaint-config", "inpaint_config"),
                        ("lama", "lama"), ("camera", "camera")):
        command += ["--" + option, initial["inputs"][key]["path"]]
    return command + ["--config", str(root / "config.yaml"),
        "--output-root", str(root / "edgs_init"),
        "--manifest", str(root / "manifests/edgs_init_manifest.json")]


def prepare(args):
    from edgs_inpaint_io import (appearance_weights, implementation, load_config, verify_record,
                                record, check_initialization_config)
    baseline = args.baseline_run.expanduser().resolve(strict=True)
    cfg = load_config(baseline / "edgs_joint/config.resolved.json")
    variant = make_variant(cfg)
    anchored = args.surface_initialization
    if anchored is not None:
        anchored = anchored.expanduser().resolve(strict=True)
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
        "submodules/EDGS/tools/train_paintmesh_pgsr.py", "submodules/EDGS/tools/initialize_paintmesh_edgs.py",
        "submodules/EDGS/source/paintmesh_surface_init.py", "submodules/EDGS/source/paintmesh_edgs_init.py"}
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
    from edgs_inpaint_io import validate_init, validate_selection
    initial = validate_init(joint["inputs"]["initialization"]["path"])
    validate_selection(joint["inputs"]["local_selection"]["path"], initial=initial, config=cfg["local_background"])
    if anchored is not None:
        check_initialization_config(validate_init(anchored), variant["config"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = REPO / "logs" / f"{PREFIX}_gpu{args.gpu}_{stamp}"
    output = args.output_root.expanduser().resolve() if args.output_root else baseline.parent / f"{PREFIX}_{stamp}"
    report = args.report.expanduser().resolve(strict=True) if args.report else baseline / "comparison_paper_bag/comparison.md"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output contains files; use --resume JOB_DIR or another output root")
    if not Path(args.python).is_file() or not report.is_file():
        raise ValueError("Python executable or comparison report missing")
    job.mkdir(parents=True)
    root = output / variant["name"]
    (root / "manifests").mkdir(parents=True)
    write(root / "config.yaml", variant["config"])
    rows = read(baseline / "comparison_paper_bag/comparison.json")["rows"]
    initialization = dict(mode="existing" if anchored else "fresh_matching",
        manifest=str(anchored or root / "manifests/edgs_init_manifest.json"),
        command=None if anchored else initialization_command(args.python, initial, root),
        existing_record=record(anchored) if anchored else None)
    request = dict(created_at=now(), repo=str(REPO), baseline=str(baseline), reference=str(baseline),
        baseline_iteration=cfg["iterations"], output=str(output), report=str(report), gpu=args.gpu,
        python=str(Path(args.python).resolve()), baseline_results=str(result_path), reference_results=str(result_path),
        baseline_results_sha256=sha(result_path), reference_results_sha256=sha(result_path),
        paper={row["metric"]: row["paper_bag"] for row in rows}, variants=[variant],
        training_inputs=None, baseline_training_inputs=joint["inputs"], local_selection_check=None,
        initialization=initialization,
        implementation=current, implementation_update=changed,
        implementation_added=sorted(set(current) - set(joint["request"]["implementation"])),
        appearance_weights=assets, publish_inputs=published["inputs"], publish_parameters=published["parameters"],
        metric_implementation={name: sha(REPO / name) for name in ("scripts/paintmesh/eval_inpaint360.py",
            "submodules/Inpaint360GS/utils/image_utils.py", "submodules/Inpaint360GS/utils/loss_utils.py")},
        orchestration_implementation={name: sha(REPO / name) for name in (
            "scripts/paintmesh/run_surface_initialization_ablation.py", "scripts/paintmesh/run_geometry_ablation.py",
            "scripts/paintmesh/run_iteration_ablation.py")},
        seg_implementation={name: sha(REPO / name) for name in ("scripts/paintmesh/run_seg.sh",
            "submodules/EDGS/train.py", "submodules/EDGS/source/trainer.py")},
        source=results["inputs"]["source_path"], frame_count=results["frame_count"], reference_config=cfg,
        surface_model=None,
        varying_fields=["surface_init.mode"], policy="original supervision; depth-anchor centers/sampling only; same old-row mask")
    write(job / "request.json", request)
    write(job / "status.json", dict(state="prepared", created_at=now(), variants=[]))
    atomic(job / "comparison.before.md", report.read_text())
    write(job / "configuration_diff.json", {"surface_init.mode": dict(before="none", after="depth_anchor")})
    return job


class SurfaceInitializationBatch(Batch):
    def commands(self, root, *, iterations=None):
        plan = self.req.get("initialization")
        if plan is not None:
            if plan["mode"] == "fresh_matching":
                yield "initialization", plan["command"], REPO
            else:
                from edgs_inpaint_io import verify_record
                verify_record(plan["existing_record"])
            # This runs only after a successful initializer process. Failure must
            # never fall through to training or reuse the baseline PLY.
            self.status.update(phase="depth_anchor: preserving baseline old-row eligibility")
            self.save()
            selection, stats, initial = rebind_selection(self.req["baseline_training_inputs"],
                Path(plan["manifest"]), self.req["variants"][0]["config"], root / "local_selection")
            from edgs_inpaint_io import record
            self.req["training_inputs"] = {**self.req["baseline_training_inputs"],
                "initialization": record(plan["manifest"]), "local_selection": selection}
            self.req["local_selection_check"] = stats
            self.req["surface_model"] = read(initial["outputs"]["surface_model"]["path"])
            # Keep request.json immutable; generated artifacts get a separate receipt.
            write(self.job / "resolved_initialization.json", dict(
                training_inputs=self.req["training_inputs"], local_selection_check=stats,
                surface_model=self.req["surface_model"]))
            self.report_progress()
        yield from super().commands(root, iterations=iterations)

    def run(self):
        for key in ("orchestration_implementation", "seg_implementation"):
            for name, expected in self.req[key].items():
                if sha(REPO / name) != expected:
                    raise ValueError("Implementation changed: " + name)
        resolved = self.job / "resolved_initialization.json"
        if resolved.exists():
            from edgs_inpaint_io import verify_record
            bindings = read(resolved)
            for value in bindings["training_inputs"].values():
                verify_record(value)
            self.req.update(bindings)
        super().run()

    def report_progress(self):
        req, status = self.req, self.status
        cfg, stats = req["reference_config"], req["local_selection_check"]
        completed = [v for v in status["variants"] if v.get("state") == "complete"]
        baseline = read(req["baseline_results"])["metrics"]
        metrics = completed[0]["metrics"] if completed else None
        lines = ["## 原始外观＋几何监督：RGB 锚点深度初始化消融", "",
            f"状态：{status['state']}；当前阶段：{status.get('phase', '准备')}。", "",
            "仅将 surface_init.mode 从 none 改为 depth_anchor；同一 LaMa/相机重新提取 RGB 匹配并补齐三角闭环，锚点约束非平面深度校正，按跨视角一致性筛选新中心，不使用 GT。尺度、旋转、opacity 的初始化规则保持原样；颜色仍由原 LaMa 图按修正中心投影采样，具体颜色值可随投影改变。", "",
            f"原始 masked 外观、几何与覆盖监督全部保留：`{json.dumps(cfg['loss'], ensure_ascii=False)}`。", "",
            f"固定 {cfg['iterations']} 步、seed={cfg['seed']}、同一 LaMa/180 帧虚拟相机、学习率和几何渐增安排；depth 初始化开启、normal 初始化关闭；仅剪枝、不克隆分裂；局部旧高斯更新、远处背景严格冻结。", "",
            (f"旧点 {stats['initial_old']:,} 个；旧点解冻名单逐位复用基线（{stats['trainable_old']:,} 可更新、{stats['frozen_old']:,} 冻结），不因新增点分布改变而重新选旧点。新增点由 {stats['baseline_summary']['initial_new']:,} 变为 {stats['initial_new']:,}，这是表面重采样的结果。" if stats else
             "初始化尚未完成。通过锚点和跨视角检查后，逐位复用基线旧点解冻名单，再开始训练；若支持不足则停止并保留诊断，不回退或降低门槛。"), "",
            "旧背景中已观测到目标形态残留，本实验按单变量范围保留；校正深度仅用于初始化，训练仍使用原始伪目标，优化后中心仍可能偏离初始表面。", "",
            f"评估使用相同 {req['frame_count']} 个 test_* 相机、resolution=4、官方 unseen_mask 和相同七项指标实现；论文列沿用已发表数值。", "",
            "| 指标 | 论文 bag | 原始初始化 | RGB 锚点深度初始化 | 差值（深度锚定−原始） |",
            "|---|---:|---:|---:|---:|"]
        for key in METRICS:
            value = f"{metrics[key]:.6f} | {metrics[key]-baseline[key]:+.6f}" if metrics else "— | —"
            lines.append(f"| {key} {'↓' if key in LOWER else '↑'} | {req['paper'][key]:.6f} | {baseline[key]:.6f} | {value} |")
        lines += ["", "原始初始化为历史基线；兼容代码的哈希变化及本次确切配置、输入哈希记录在 request.json。此次不重跑原始基线。", ""]
        if status.get("error"):
            lines += ["错误：`" + status["error"] + "`。", ""]
        lines += [f"结果目录：`{self.output / 'depth_anchor'}`", f"总日志：`{self.job / 'run.log'}`",
            f"状态：`{self.job / 'status.json'}`", "",
            f"恢复命令：`{req['python']} {Path(__file__).resolve()} --resume {self.job} --detach`", "",
            "依次训练、发布并评估 3DGS；不额外执行 TSDF mesh 后处理。"]
        body = "\n".join(lines) + "\n"
        begin, end = [f"<!-- DEPTH_ANCHOR_INITIALIZATION_ABLATION:{self.job.name}:{tag} -->" for tag in ("BEGIN", "END")]
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--surface-initialization", type=Path, help="Optional completed depth_anchor manifest; default runs fresh initialization first")
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
        info = dict(pid=child.pid, job=str(job), log=str(job / "run.log"), status=str(job / "status.json"),
            output=req["output"], report=req["report"])
        write(REPO / "logs" / f"{PREFIX}_gpu{req['gpu']}.latest.json", info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return 0
    try:
        SurfaceInitializationBatch(job).run()
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
