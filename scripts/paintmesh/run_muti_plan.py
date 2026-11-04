#!/usr/bin/env python3
"""Controlled all-scene supervision experiments from audited PaintMesh inputs.

All new artifacts live in --root. No test image is consumed by training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from run_geometry_ablation import REPO, METRICS, atomic, now, read, sha, write

BEST = REPO / "output/paintmesh/Inpaint360GS/best_results"
DEFAULT_ROOT = BEST.parent / "muti_plan"
PYTHON = "/home/member/.conda/envs/paintmesh/bin/python"
SCENES = ("bag", "truck", "cube", "garden_toys", "fruits", "redbull",
          "doppelherz", "toys", "car", "cone_red", "cone_yellow")
VARIANTS = ("00_control", "01_no_lama_normal", "02_uniform_rgb", "03_both")
LABELS = ("本轮对照", "仅关闭外部法线", "仅均匀 RGB 权重", "关闭外部法线＋均匀 RGB")


def changes(a, b, prefix=""):
    result = {}
    for key in sorted(set(a) | set(b)):
        path = prefix + key
        if key not in a or key not in b:
            result[path] = dict(before=a.get(key), after=b.get(key))
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            result.update(changes(a[key], b[key], path + "."))
        elif a[key] != b[key]:
            result[path] = dict(before=a[key], after=b[key])
    return result


def make_variants(cfg):
    if cfg["appearance"]["mode"] != "masked" or cfg["loss"]["lama_normal"] <= 0:
        raise ValueError("Controlled experiments require original appearance and enabled external normal loss")
    if cfg["background_writeback"]["enabled"] or cfg["adaptive_density"]["enabled"]:
        raise ValueError("This matrix preserves prune-only training without final background writeback")
    base = copy.deepcopy(cfg)
    base["iterations"] = 10000
    base["appearance"]["rgb_weight_mode"] = "continuity"
    # Make the new-round topology policy explicit, including historical cube.
    base["pruning"]["enabled"] = True
    result = []
    for i, name in enumerate(VARIANTS):
        value = copy.deepcopy(base)
        if i in (1, 3):
            value["loss"]["lama_normal"] = 0.0
        if i in (2, 3):
            value["appearance"]["rgb_weight_mode"] = "uniform"
        result.append(dict(name=name, label=LABELS[i], config=value, diff=changes(base, value)))
    return result


def verify_code(request):
    for name, expected in request["code"].items():
        if sha(REPO / name) != expected:
            raise ValueError("Code changed after this experiment was prepared: " + name)


def prepare(root, gpu):
    from edgs_inpaint_io import load_config, implementation
    root.mkdir(parents=True, exist_ok=True)
    if (root / "request.json").exists():
        raise ValueError("Existing experiment: use --run/--resume, never overwrite its request")
    rows = []
    for scene in SCENES:
        source = BEST / scene
        selected = read(source / "selection.json")
        joint = read(source / "joint_manifest.json")
        published_path = Path(selected["run_root"]) / "inpainted_3dgs/model_manifest.json"
        published = read(published_path)
        cfg = load_config(source / "training_config.json")
        variants = make_variants(cfg)
        # These files are a frozen record, not replacements for producer paths.
        inputs_dir = root / "inputs" / scene
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for name in ("selection.json", "training_config.json", "initialization_manifest.json",
                     "joint_manifest.json", "camera.json", "lama_manifest.json", "results.json"):
            shutil.copyfile(source / name, inputs_dir / name)
        shutil.copyfile(published_path, inputs_dir / "publish_reference.json")
        for v in variants:
            output = root / "scenes" / scene / v["name"]
            write(output / "config.json", v["config"])
            write(output / "config_diff.json", v["diff"])
        write(inputs_dir / "normalization_diff.json", changes(joint["request"]["config"], variants[0]["config"]))
        rows.append(dict(scene=scene, inputs=joint["inputs"], historical_config=joint["request"]["config"],
            historical_metrics=selected["metrics"], historical_iteration=selected["configuration"]["iteration"],
            historical_model=str((source / "model").resolve()), publish=published,
            source=read(source / "results.json")["inputs"]["source_path"],
            historical_evaluation=read(source / "results.json"), configuration=selected["configuration"],
            frame_count=selected["frame_count"], variants=variants))
    shutil.copyfile(BEST / "comparison.csv", root / "reference_comparison.csv")
    for name in ("OPTIMIZATION_FROM_MODULE_DEBUG.zh-CN.md", "module_debug/target_geometry_consistency.json"):
        dest = root / "evidence" / Path(name).name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(BEST / name, dest)
    code = implementation()
    for name in ("scripts/paintmesh/run_muti_plan.py", "scripts/paintmesh/run_geometry_ablation.py",
                 "scripts/paintmesh/eval_inpaint360.py", "scripts/paintmesh/run_inpaint360_full.py",
                 "submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py",
                 "submodules/Inpaint360GS/utils/image_utils.py", "submodules/Inpaint360GS/utils/loss_utils.py"):
        code[name] = sha(REPO / name)
    for name in code:
        dest = root / "code_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    req = dict(created_at=now(), root=str(root), gpu=gpu, gpu_uuid=uuid, python=PYTHON,
        iterations=10000, save_iterations=[5000, 10000], scene_order=list(SCENES),
        variants=list(VARIANTS), scenes=rows, code=code, resolution=4,
        policy="matched-input 2x2 supervision ablation; historical inputs selected post-hoc; test GT evaluation only",
        reference_comparison_sha256=sha(root / "reference_comparison.csv"))
    write(root / "request.json", req)
    write(root / "status.json", dict(state="prepared", updated_at=now(), cases={}))
    plan(root, req)
    report(root)


def plan(root, req):
    lines = ["# PaintMesh 全场景新一轮训练方案", "", f"创建：{req['created_at']}。运行设备：GPU {req['gpu']}（{req['gpu_uuid']}）。",
        "", "## 本轮要验证什么", "",
        "上一轮模块诊断发现：独立补全的深度和法线在洞内不完全兼容；当前 RGB 权重按单图梯度降权，可能抑制木纹、网格和叶片。先做可归因的 2×2 全场景监督实验，不把未经验证的新几何机制一起塞进训练。",
        "", "[PGSR](https://arxiv.org/html/2406.06521v2) 结合单视图几何、多视图几何与光度约束。本轮只消融外部 LaMa 法线，保留现有内部深度—法线一致性和跨视角深度约束，不能把它写成关闭所有几何。",
        "[3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 使用可见性合成和各向异性高斯。本轮保持局部旧点名单、参数化和剪枝方式，单独验证监督信号，不宣称实现新的贡献选择器或 AbsGS。",
        "", "## 实验矩阵", "", "11 场景 × 4 配置 = 44 次独立训练，均从相同的对应场景初始 PLY 开始，训练 10000 步；保存 5000 步快照和 10000 步最终模型。每个 run 完成后依次完整评估两个节点，共 88 套测试结果。中间快照不是从历史 5000 步模型续训。",
        "", "| 目录 | 外部 LaMa normal 系数 | 洞内 RGB 像素权重 |", "|---|---|---|",
        "| 00_control | 原值（0.05） | 原 continuity |", "| 01_no_lama_normal | 0 | 原 continuity |",
        "| 02_uniform_rgb | 原值（0.05） | 1 |", "| 03_both | 0 | 1 |",
        "", "RGB 的归一化分母同步使用均匀像素数；不是只乘常数改变整体 L1 强度。SSIM、LPIPS、洞外保持、alpha 覆盖、深度、内部一致性、多视图损失和学习率不随组变化。normal 目标仍读入用于记录，但关闭组的该损失及梯度为零。",
        "", "## 固定输入和边界", "",
        "分辨率 r4、RECURSIVE_GUIDE=false；复用 best_results 对应的 Stage 1/删除/虚拟相机/跟踪/LaMa/初始化及局部更新名单，不重跑已确认正确的上游。远处旧点逐字段保护，10 个场景更新既有局部旧点，cube 保持历史旧点全冻结。仅剪枝，不克隆/分裂。没有新增 500000 点截断，也不会重新截断复用的 PLY；历史初始化配置中的 max_points 仅作为输入溯源，不能改写已产生数据的身份。",
        "历史最佳不是统一配置：bag 原预算 20000；cube 较早的配置没有剪枝字段、没有 SSIM/LPIPS 项。当前加载器补全缺失默认值，本轮 cube 明确采用仅剪枝，SSIM/LPIPS 仍为 0。与历史列的差异不能当作单变量证据，应比较本轮同预算 00_control。每场景的完整补全差异保存在 inputs/<scene>/normalization_diff.json。",
        "", "| 场景 | 目标 ID | 相机 / 帧数 | MASK_DILATION | 初始化 | 旧点局部更新 | 洞内 RGB 系数 | 历史节点 |", "|---|---|---|---:|---|---|---:|---:|"]
    for s in req["scenes"]:
        c, cfg = s["configuration"], s["variants"][0]["config"]
        lines.append(f"| {s['scene']} | {s['publish']['parameters']['target_ids']} | {c['trajectory']} / {c['virtual_frames']} | {c['mask_dilation']} | {c['surface_init']}, depth={c['init_use_depth']} | {cfg['local_background']['enabled']} | {cfg['loss']['rgb_hole']} | {s['historical_iteration']} |")
    lines += ["", "## 执行、评估与可复现性", "",
        "串行执行，优先 bag/truck/cube/garden_toys 等诊断场景。每个场景按 control→no_normal→uniform→both 执行，之后进入下一个场景。不根据测试指标自动更换超参数或择优混合成主表。seed 保留各场景输入身份（当前均为 0），检查点恢复包含 optimizer、采样顺序和随机状态。训练和队列实现做 SHA256 快照；源代码变化时拒绝混用。",
        "使用全部 test_* 和官方 unseen_mask，仅评估进程读取 GT，训练不读测试图像。七项指标为 PSNR、masked_PSNR、SSIM、masked_SSIM、LPIPS、masked_LPIPS、FID；masked 指标沿用官方洞外归零、全图分母定义。每组均值只在 11 个场景完整时写出；FID 为各场景 FID 的算术平均，不冒充全数据集 pooled FID。三种本机对比方法和历史 best 的结果从 reference_comparison.csv 固定引用，未重训它们。",
        "历史 best 输入是事后择优的实验合集，本轮属于基于该合集的受控优化消融，不等于独立验证集选出的统一配置、也不保证复现论文原表数值。结果主表保留所有组和失败，不只显示最优项。",
        "每个输出有真实 PLY、配置、相机/LaMa 源 manifest、每 100 步诊断及训练视图图像、完整测试渲染和七项指标。5000 步使用独立 snapshot_manifest，10000 步使用经过完整校验的 model_manifest，不伪造中间训练已完成的最终收据。训练使用 PGSR 几何，此轮衡量 3DGS 渲染质量，不额外对 88 个节点执行 TSDF mesh 后处理。",
        "", "## 后续建议的处理", "",
        "可见 alpha 贡献选旧点、可信 patch NCC、真实训练相机保护区回放、沿光线厚度约束和局部新增点分裂仍列为下一阶段候选，不在本轮四组中声称实现。它们需要分别验证遮挡合成、误匹配和编辑区边界，且会改变优化变量或输入；先用本轮完整结果确认监督冲突是否是主要原因。前一轮已否定的 circle30/dilation10 方案不恢复。",
        "", "## 文件入口", "",
        "- request.json：固定实验矩阵、历史指标、输入路径与代码哈希。",
        "- status.json / logs/run.log：实时队列和主日志；logs/<scene>/<variant>/*：分阶段日志。",
        "- comparison.md / comparison.csv / comparison.json：每次评估后自动更新的完整对比。",
        "- scenes/<scene>/<variant>/config.json：实际训练配置；config_diff.json：相对本轮对照的唯一区别。",
        "- scenes/<scene>/<variant>/edgs_joint：检查点、PLY、debug、diagnostics。",
        "- scenes/<scene>/<variant>/inpainted_3dgs_5000：中间模型和 evaluation/inpaint360_r4 测试图片/指标。",
        "- scenes/<scene>/<variant>/inpainted_3dgs_10000：最终发布模型和完整评估。",
        "- inputs/、code_snapshot/、evidence/：来源、源码及上一轮依据；旧数据只读引用。",
        "", "恢复命令（锁保证不会重复启动）：", "", "```bash",
        f"{PYTHON} {Path(__file__).resolve()} --root {root} --resume --detach", "```", ""]
    atomic(root / "TRAINING_PLAN.zh-CN.md", "\n".join(lines))
    atomic(root / "README.md", "# PaintMesh muti_plan\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [实时比较](comparison.md) · [状态](status.json) · [主日志](logs/run.log)\n\n44 次训练，88 套节点评估；保存路径和参数见方案。\n")


def validate_evaluation(result, source):
    if not result.get("complete") or result["protocol"] != source["protocol"]:
        raise ValueError("Incomplete result or changed metric protocol")
    for key in ("frames", "source_path", "resolution", "renderer", "reference_images",
                "official_masks", "colmap_images", "colmap_cameras"):
        if result["inputs"][key] != source["inputs"][key]:
            raise ValueError("Evaluation changed held-out data: " + key)
    if result["frame_count"] != source["frame_count"] or not all(math.isfinite(result["metrics"][k]) for k in METRICS):
        raise ValueError("Evaluation frame count/metrics invalid")


def report(root):
    req, status = read(root / "request.json"), read(root / "status.json")
    entries = []
    for s in req["scenes"]:
        for v in s["variants"]:
            for step in req["save_iterations"]:
                path = root / "scenes" / s["scene"] / v["name"] / f"inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json"
                key = s["scene"] + "/" + v["name"]
                row = dict(scene=s["scene"], method=v["name"], iteration=step,
                           state=status.get("cases", {}).get(key, {}).get("state", "pending"), result=str(path))
                if path.exists():
                    result = read(path)
                    validate_evaluation(result, s["historical_evaluation"])
                    row.update(state="complete", metrics=result["metrics"])
                entries.append(row)
    references = list(csv.DictReader((root / "reference_comparison.csv").open()))
    refs = [dict(scene=r["scene"], method=r["method"], iteration="historical", state="reference",
                 metrics={k: float(r[k]) for k in METRICS}) for r in references]
    means = []
    for method, step in [(v, step) for v in VARIANTS for step in (5000, 10000)] + [(r, "historical") for r in ("PaintMesh best", "Inpaint360GS", "Gaussian Grouping", "AuraFusion360")]:
        group = [r for r in entries + refs if r["method"] == method and r["iteration"] == step and "metrics" in r]
        value = dict(method=method, iteration=step, completed=len(group), total=len(SCENES))
        if {r["scene"] for r in group} == set(SCENES) and len(group) == len(SCENES):
            value["metrics"] = {k: sum(r["metrics"][k] for r in group) / len(group) for k in METRICS}
        means.append(value)
    header = "| 场景 | 方法 | 步数 | PSNR | masked_PSNR | SSIM | masked_SSIM | LPIPS | masked_LPIPS | FID | 状态 |"
    lines = ["# PaintMesh muti_plan 全场景监督消融", "", f"更新：{now()}。队列：{status['state']}。已获得 {sum('metrics' in r for r in entries)}/88 套完整评估。",
        "", "比较本轮同场景、同节点 00_control，判断单变量效果。历史 best 是测试集事后择优且预算不同；其他三种方法为本机已完成复现的固定参考，不是论文原表。",
        "", header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in req["scenes"]:
        for r in [x for x in refs + entries if x["scene"] == s["scene"]]:
            nums = [f"{r['metrics'][k]:.6f}" if "metrics" in r else "—" for k in METRICS]
            lines.append(f"| {r['scene']} | {r['method']} | {r['iteration']} | " + " | ".join(nums) + f" | {r['state']} |")
    lines += ["", "## 全 11 场景算术均值", "", "不将部分完成场景的均值当作全场景结果。FID 这里为逐场景 FID 平均。", "",
              "| 方法 | 步数 | 完成 | PSNR | masked_PSNR | SSIM | masked_SSIM | LPIPS | masked_LPIPS | FID |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in means:
        nums = [f"{m['metrics'][k]:.6f}" if "metrics" in m else "—" for k in METRICS]
        lines.append(f"| {m['method']} | {m['iteration']} | {m['completed']}/11 | " + " | ".join(nums) + " |")
    for k, state in status.get("cases", {}).items():
        if state.get("error"):
            lines += ["", f"- {k}: {state['error']}"]
    atomic(root / "comparison.md", "\n".join(lines) + "\n")
    write(root / "comparison.json", dict(updated_at=now(), results=entries, means=means, references=refs))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["scene", "method", "iteration", "state", *METRICS])
    writer.writeheader()
    for r in refs + entries:
        writer.writerow({**{k: r[k] for k in ("scene", "method", "iteration", "state")}, **r.get("metrics", {})})
    atomic(root / "comparison.csv", stream.getvalue())


def preflight(root):
    from edgs_inpaint_io import (validate_init, validate_selection, check_initialization_config,
        verify_targets, verify_record, load_config, appearance_weights)
    req = read(root / "request.json")
    verify_code(req)
    result = dict(complete=False, checked_at=now(), scenes={})
    for s in req["scenes"]:
        print("PREFLIGHT", s["scene"], flush=True)
        inputs = s["inputs"]
        for value in inputs.values():
            verify_record(value)
        initial = validate_init(inputs["initialization"]["path"])
        cfg = s["variants"][0]["config"]
        check_initialization_config(initial, cfg)
        verify_targets(inputs["lama"]["path"], inputs["camera"]["path"], **cfg["supervision"])
        if cfg["local_background"]["enabled"]:
            validate_selection(inputs["local_selection"]["path"], initial=initial, config=cfg["local_background"])
        for record in appearance_weights(cfg).values():
            verify_record(record)
        for name, value in s["publish"]["inputs"].items():
            # Previous trained PLY and receipt are reference results, not new inputs.
            if name not in ("inpainted_gaussian_ply", "edgs_joint_manifest"):
                verify_record(value)
        for v in s["variants"]:
            if load_config(root / "scenes" / s["scene"] / v["name"] / "config.json") != v["config"]:
                raise ValueError("Config resolution changed: " + s["scene"] + "/" + v["name"])
        result["scenes"][s["scene"]] = dict(valid=True, points=initial["point_count"],
            virtual_frames=s["configuration"]["virtual_frames"], test_frames=s["frame_count"], time=now())
        write(root / "preflight.json", result)
    result.update(complete=True, finished_at=now())
    write(root / "preflight.json", result)


class Runner:
    def __init__(self, root):
        self.root = root
        self.req = read(root / "request.json")
        self.state = read(root / "status.json")
        self.child = None
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.req["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
            GPU=str(self.req["gpu"]), PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.env.pop("PYTHONPATH", None)

    def save(self, **values):
        self.state.update(values, updated_at=now())
        write(self.root / "status.json", self.state)

    def stop(self, *_):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        self.save(state="stopped")
        raise InterruptedError("Queue stopped; resume restores original checkpoints")

    def execute(self, scene, variant, stage, cmd, cwd, expected):
        cmd = list(map(str, cmd))
        receipt = self.root / "receipts" / scene / variant / (stage + ".json")
        if receipt.exists():
            old = read(receipt)
            if old["command"] != cmd or any(not p.exists() for p in expected):
                raise ValueError("Completed stage changed or lost output: " + str(receipt))
            for artifact in old.get("artifacts", []):
                if sha(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("Completed artifact changed: " + artifact["path"])
            return
        verify_code(self.req)
        if shutil.disk_usage(self.root).free < 40 * 1024**3:
            raise RuntimeError("Less than 40 GiB available; preserving outputs and stopping this case")
        logfile = self.root / "logs" / scene / variant / (stage + ".log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.save(state="running", scene=scene, variant=variant, stage=stage, log=str(logfile), command=cmd,
                  training_progress=None, child_pid=None)
        print(now(), scene, variant, stage, flush=True)
        with logfile.open("a") as handle:
            self.child = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self.save(child_pid=self.child.pid)
            for line in self.child.stdout:
                handle.write(line)
                handle.flush()
                if line.startswith("PGSR joint ") or "training complete:" in line or "error:" in line:
                    print(line.rstrip(), flush=True)
                    self.save(training_progress=line.rstrip())
            code = self.child.wait()
        self.save(child_pid=None)
        if code:
            raise RuntimeError(f"{stage} exited {code}: {logfile}")
        if any(not p.is_file() for p in expected):
            raise RuntimeError("Command succeeded without expected outputs: " + stage)
        write(receipt, dict(complete=True, command=cmd, finished_at=now(), log=str(logfile),
            artifacts=[dict(path=str(p), sha256=sha(p)) for p in expected]))

    def training_command(self, s, root, *, saves=True):
        cmd = [PYTHON, REPO / "submodules/EDGS/tools/train_paintmesh_pgsr.py"]
        for flag, key in (("initialization", "initialization"), ("lama", "lama"), ("camera", "camera"), ("edgs-config", "edgs_config")):
            cmd += ["--" + flag, s["inputs"][key]["path"]]
        if "local_selection" in s["inputs"]:
            cmd += ["--local-selection", s["inputs"]["local_selection"]["path"]]
        cmd += ["--config", root / "config.json", "--output-root", root / "edgs_joint",
                "--manifest", root / "manifests/edgs_joint_manifest.json"]
        if saves:
            cmd += ["--save-iterations", "5000"]
        return cmd

    def publish_command(self, s, root, iteration=10000):
        p = s["publish"]["parameters"]
        ids = lambda v: ",".join(map(str, v)) if v else "none"
        cmd = [PYTHON, "-m", "tools.publish_inpainted_edgs_model", "--inpainted-ply",
            root / f"edgs_joint/point_cloud/iteration_{iteration}/point_cloud.ply",
            "--source-iteration", p["source_iteration"], "--output-iteration", iteration,
            "--target-ids", ids(p["target_ids"]), "--surrounding-ids", ids(p["surrounding_ids"])]
        for flag, key in (("classifier", "classifier"), ("edgs-config", "edgs_config"), ("cfg-args", "cfg_args"),
                         ("removed-model-manifest", "removed_model_manifest"), ("removal-manifest", "removal_manifest"),
                         ("workspace-manifest", "workspace_manifest"), ("tracking-session", "tracking_session"),
                         ("lama-manifest", "lama_manifest"), ("inpaint-config", "inpaint_config")):
            cmd += ["--" + flag, s["publish"]["inputs"][key]["path"]]
        return cmd + ["--edgs-joint-manifest", root / "manifests/edgs_joint_manifest.json",
                      "--output", root / f"inpainted_3dgs_{iteration}"]

    def case(self, s, v):
        from run_inpaint360_full import export_snapshot
        root = self.root / "scenes" / s["scene"] / v["name"]
        self.execute(s["scene"], v["name"], "train_10000", self.training_command(s, root), REPO / "submodules/EDGS",
                     [root / "manifests/edgs_joint_manifest.json", root / "edgs_joint/snapshots/iteration_5000/manifest.json"])
        # Export the actual same-trajectory intermediate model, not the final PLY.
        export_snapshot(root, 5000)
        self.execute(s["scene"], v["name"], "publish_10000", self.publish_command(s, root), REPO / "submodules/Inpaint360GS",
                     [root / "inpainted_3dgs_10000/model_manifest.json"])
        for step in (5000, 10000):
            model = root / f"inpainted_3dgs_{step}"
            result = model / "evaluation/inpaint360_r4/results.json"
            self.execute(s["scene"], v["name"], f"evaluate_{step}", [PYTHON, REPO / "scripts/paintmesh/eval_inpaint360.py", "run",
                "--model-path", model, "--source-path", s["source"], "--resolution", "4", "--iteration", str(step), "--batch-size", "4"], REPO, [result])
            validate_evaluation(read(result), s["historical_evaluation"])
            report(self.root)

    def run(self):
        with (self.root / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_code(self.req)
            if not read(self.root / "preflight.json")["complete"]:
                raise ValueError("All-scene input preflight must finish first")
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
            self.save(state="running", pid=os.getpid(), gpu=self.req["gpu"], started_at=now())
            for s in self.req["scenes"]:
                for v in s["variants"]:
                    key = s["scene"] + "/" + v["name"]
                    self.state.setdefault("cases", {})[key] = dict(state="running", started_at=now())
                    self.save()
                    try:
                        self.case(s, v)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        traceback.print_exc()
                        self.state["cases"][key] = dict(state="failed", error=str(exc), time=now())
                    else:
                        self.state["cases"][key] = dict(state="complete", finished_at=now())
                    self.save()
                    report(self.root)
            failed = [k for k, v in self.state["cases"].items() if v["state"] != "complete"]
            self.save(state="completed_with_failures" if failed else "complete", failures=failed, finished_at=now(), child_pid=None)
            report(self.root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gpu", type=int, default=1)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", "--resume", dest="run", action="store_true")
    mode.add_argument("--report", action="store_true")
    p.add_argument("--detach", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    if a.prepare:
        prepare(root, a.gpu)
    elif a.preflight:
        preflight(root)
    elif a.report:
        report(root)
    elif a.detach:
        if not read(root / "preflight.json")["complete"]:
            raise ValueError("Preflight incomplete")
        with (root / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = root / "logs/run.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as handle:
            child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--root", str(root), "--run"],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1"))
        atomic(root / "run.pid", str(child.pid) + "\n")
        print(dict(pid=child.pid, log=str(log), root=str(root)), flush=True)
    else:
        Runner(root).run()


if __name__ == "__main__":
    main()
