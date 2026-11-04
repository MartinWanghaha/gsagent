#!/usr/bin/env python3
"""Serial, resumable GPU job: adaptive180 removal/tracking, 10k train, 5k/10k eval."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from run_geometry_ablation import read, write, sha, METRICS

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/home/member/.conda/envs/paintmesh/bin/python"
DATA = Path("/home/member/wyc/data/Inpaint360GS/inpaint360")
BASELINES = {
    "Inpaint360GS": Path("/home/member/wyc/Inpaint360GS/output/paper_r4_all_scenes/comparison/summary.json"),
    "Gaussian Grouping": Path("/home/member/wyc/gaussian-grouping/output/inpaint360/all_scene_evaluation_results.json"),
    "AuraFusion360": Path("/home/member/wyc/AuraFusion360_official/output/inpaint360/all_scene_evaluation_results.json"),
}


def create_job():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = REPO / "logs" / ("paintmesh_full_r4_gpu1_" + stamp)
    job.mkdir()
    name = "full_r4_adaptive180_" + stamp
    policy = read(REPO / "scripts/paintmesh/configs/inpaint360_full_scenes.json")
    scenes = []
    for scene, settings in policy.items():
        settings = dict(settings)
        if settings.get("adaptive_options"):
            camera = read(REPO / "scripts/paintmesh/configs/inpaint360_adaptive180.json")
            camera.update(settings["adaptive_options"])
            camera_path = job / "camera_configs" / f"{scene}.json"
            write(camera_path, camera)
            settings["adaptive_config"] = str(camera_path)
        removal = REPO / "output/paintmesh/Inpaint360GS/inpaint360" / scene / "removal" / name
        scenes.append(dict(name=scene, **settings, removal_root=str(removal),
            inpaint_root=str(removal / "inpaint" / name), reuse_removal=False))
    write(job / "request.json", dict(created_at=now(), gpu=1, run_name=name, scenes=scenes,
        config=str(REPO / "scripts/paintmesh/configs/inpaint360_balanced_10000.yaml"),
        iterations=10000, save_iterations=[5000,10000], resolution=4, report=str(job / "comparison.md")))
    write(job / "status.json", dict(state="prepared", scenes={}))
    write(REPO / "logs/paintmesh_full_r4_gpu1.latest.json", dict(job=str(job)))
    return job


def now():
    return datetime.now().astimezone().isoformat()


def stage_command(args, start):
    args = list(map(str, args))
    if start is not None and args[:1] == ["bash"] and Path(args[1]).name in ("run_remove.sh", "run_inpaint.sh"):
        # Both shell entrypoints prioritize positional argument 6 over START_STAGE.
        if len(args) != 8:
            raise ValueError("expected dataset, scene, resolution, target, surrounding, start stage")
        args[-1] = str(start)
    return args


def baseline_results():
    result, provenance = {}, {}
    for method, path in BASELINES.items():
        if not path.is_file():
            result[method] = {}; continue
        payload = read(path)
        provenance[method] = dict(path=str(path), sha256=sha(path), read_at=now())
        if method == "Inpaint360GS":
            result[method] = {s["scene"]: s["metrics"] for s in payload["per_scene"]}
        else:
            rows = {}
            for key, values in payload["per_scene_results"].items():
                scene = Path(key).parts[2]
                if scene in rows:
                    raise ValueError(f"duplicate baseline scene: {method}/{scene}")
                rows[scene] = {**{m: values[f"{m}_full"] for m in ("PSNR", "SSIM", "LPIPS")},
                              **{f"masked_{m}": values[f"{m}_masked"] for m in ("PSNR", "SSIM", "LPIPS")},
                              "FID": values["FID"]}
            result[method] = rows
    return result, provenance


def report_corrections(job, original):
    """Apply only this job's explicitly registered, source-verified repairs."""
    path = job / "report_corrections.json"
    methods = {method: dict(rows) for method, rows in original.items()}
    if not path.is_file():
        return methods, []
    records = read(path)["records"]
    seen = set()
    for item in records:
        method, scene = item["method"], item["scene"]
        if (method, scene) in seen or scene not in original.get(method, {}):
            raise ValueError(f"duplicate or unknown report correction: {method}/{scene}")
        seen.add((method, scene))
        source = Path(item["source"])
        if sha(source) != item["sha256"]:
            raise ValueError(f"report correction source changed: {source}")
        payload = read(source)
        if item["format"] == "paintmesh":
            if not payload.get("complete") or payload["inputs"]["iteration"] != item["iteration"]:
                raise ValueError(f"incomplete or wrong checkpoint: {source}")
            values = payload["metrics"]
        elif item["format"] == "native":
            candidates = [v for key, v in payload["per_scene_results"].items()
                          if key.split('/inpaint/')[0].split('/')[-1] == scene]
            if len(candidates) != 1:
                raise ValueError(f"ambiguous correction scene: {source}")
            native = candidates[0]
            values = {**{m: native[f"{m}_full"] for m in ("PSNR", "SSIM", "LPIPS")},
                      **{f"masked_{m}": native[f"{m}_masked"] for m in ("PSNR", "SSIM", "LPIPS")},
                      "FID": native["FID"]}
        else:
            raise ValueError(f"unknown correction format: {item['format']}")
        if not all(math.isfinite(values[m]) for m in METRICS):
            raise ValueError(f"nonfinite correction metrics: {source}")
        methods[method][scene] = {m: values[m] for m in METRICS}
    return methods, records


def export_snapshot(root, step):
    """A truthful standard render layout; never forge a completed joint receipt."""
    from local_geometry_io import read_receipt, record, verify_record
    manifest = root / "edgs_joint/snapshots" / f"iteration_{step}/manifest.json"
    snapshot = read_receipt(manifest, "paintmesh-joint-snapshot")
    final = read_receipt(root / "manifests/edgs_joint_manifest.json", "paintmesh-edgs-joint")
    if (snapshot["request"] != final["request"] or snapshot["iterations"] != step
            or snapshot["training_budget"] != 10000):
        raise ValueError("snapshot is not from this same 10000-step trajectory")
    model = root / f"inpainted_3dgs_{step}"
    model.mkdir(exist_ok=True)
    ply = verify_record(snapshot["outputs"]["ply"])
    target = model / "point_cloud" / f"iteration_{step}/point_cloud.ply"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != snapshot["outputs"]["ply"]["sha256"]:
            raise ValueError("existing exported snapshot changed")
    else:
        temporary = target.with_suffix(".tmp"); shutil.copyfile(ply, temporary); temporary.replace(target)
    cfg = verify_record(snapshot["inputs"]["edgs_config"])
    if (model / "config.yaml").exists() and sha(model / "config.yaml") != sha(cfg):
        raise ValueError("snapshot render configuration changed")
    shutil.copyfile(cfg, model / "config.yaml")
    write(model / "snapshot_manifest.json", dict(complete=True, kind="paintmesh-training-snapshot-export",
        producer=record(manifest), iterations=step, training_budget=10000,
        model=record(target), config=record(model / "config.yaml")))
    return model


class FullJob:
    def __init__(self, job, *, report_only=False):
        self.job = job.resolve(strict=True); self.req = read(job / "request.json")
        self.status = read(job / "status.json")
        self.child = None
        if report_only:
            # Reading a completed experiment must not relabel it as running,
            # remove finished_at, or require a training/conda environment.
            self.env = dict(VIRTUAL_ADAPTIVE_CONFIG=str(REPO / "scripts/paintmesh/configs/inpaint360_adaptive180.json"))
            return
        self.status.update(pid=os.getpid(), gpu=1, state="running", updated_at=now())
        self.status.setdefault("scenes", {})
        for key in ("finished_at", "failed_scenes", "error", "child_pid"):
            self.status.pop(key, None)
        raw = subprocess.check_output(["bash", "-c", 'source "$1"; env -0', "bash",
            str(REPO / "scripts/paintmesh/configs/inpaint360_recommended.env")], cwd=REPO)
        self.env = dict(entry.decode().split("=", 1) for entry in raw.split(b"\0") if b"=" in entry)
        self.env.update(GPU="1", CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
            DATA_ROOT="/home/member/wyc/data", OUTPUT_ROOT=str(REPO / "output"),
            EDGS_INPAINT_CONFIG=self.req["config"], EDGS_JOINT_SAVE_ITERATIONS="5000",
            PAINTMESH_INIT_MODE="mixed", EDGS_SURFACE_INIT_MODE="none",
            EDGS_INIT_USE_DEPTH="true", EDGS_INIT_USE_NORMAL="false",
            RECURSIVE_GUIDE="false", EDGS_INPAINT_ITERATIONS="10000",
            CONDA_BIN="/home/member/miniconda3/bin/conda")
        # Discover this installation; never rely on an interactive shell function.
        conda = shutil.which("conda")
        if conda: self.env["CONDA_BIN"] = conda
        if not Path(self.env["CONDA_BIN"]).is_file():
            self.env["CONDA_BIN"] = "/home/member/.conda/bin/conda"
        if not Path(self.env["CONDA_BIN"]).is_file():
            raise ValueError("conda executable not found")
        self.env.pop("PYTHONPATH", None)
        signal.signal(signal.SIGTERM, self.stop); signal.signal(signal.SIGINT, self.stop)

    def stop(self, signum, frame):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        raise InterruptedError(f"received signal {signum}; resume this job to continue")

    def save(self):
        self.status["updated_at"] = now(); write(self.job / "status.json", self.status)
        self.report()

    def report(self):
        active = self.job / "active_rerun.json"
        if active.is_file():
            from run_inpaint360_official_camera import publish_report
            publish_report(Path(read(active)["job"]))
            return
        methods, provenance = baseline_results()
        for step in (5000, 10000):
            methods[f"PaintMesh {step}"] = {}
            for scene in self.req["scenes"]:
                root = Path(scene["inpaint_root"])
                model = root / (f"inpainted_3dgs_{step}" if step == 5000 else "inpainted_3dgs")
                path = model / "evaluation/inpaint360_r4/results.json"
                if path.is_file():
                    result = read(path)
                    if result.get("complete") and result["inputs"]["iteration"] == step:
                        methods[f"PaintMesh {step}"][scene["name"]] = result["metrics"]
        original = methods
        methods, corrections = report_corrections(self.job, original)
        correction_file = self.job / "report_corrections.json"
        quality_retrain = read(correction_file).get("quality_retrain") if correction_file.is_file() else None
        counts = {s["name"]: sum(p.stem.lower().startswith("test_") for p in (DATA / s["name"] / "images").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")) for s in self.req["scenes"]}
        common = set.intersection(*(set(rows) for rows in methods.values()))
        def average(rows, names, weighted):
            if not names: return None
            return {m: sum(rows[s][m] * (counts[s] if weighted and m != "FID" else 1) for s in names)
                    / sum(counts[s] if weighted and m != "FID" else 1 for s in names) for m in METRICS}
        aggregates = {method: dict(available_scenes=sorted(rows),
            all_available_scene_macro=average(rows, set(rows), False),
            common_scene_macro=average(rows, common, False),
            common_official_code_average=average(rows, common, True)) for method, rows in methods.items()}
        summary = dict(updated_at=now(), status=self.status,
            common_scenes=sorted(common), metrics=methods, original_metrics=original,
            report_corrections=corrections, test_frame_counts=counts,
            aggregates=aggregates, baseline_sources=provenance)
        if quality_retrain:
            summary["quality_retrain"] = quality_retrain
        write(self.job / "comparison.json", summary)
        lines = ["# PaintMesh 全场景 1/4 分辨率实验", "",
            f"更新：{now()}；状态：{self.status['state']}；当前：{self.status.get('scene')} / {self.status.get('phase')}", "",
            ("原始批次基础配置：" if quality_retrain else "固定配置：") +
            "PGSR adaptive 180；depth=true、normal=false、surface_init=none；"
            "局部旧高斯更新，远处冻结；仅剪枝；原始外观+几何监督；RECURSIVE_GUIDE=false。",
            "初始化总点数上限以 resolved_config.json 中 init.max_points 为准；null 表示不设上限。"
            "用户调整前已完成的结果保留各自的实际配置，配置变更记录在任务 revisions 目录。",
            "同一次 10000 步训练在 5000 步保存 PLY；训练退出释放显存后分别评估两个快照。最终 10000 步模型继续执行 mesh/语义后处理。",
            "对象从训练图中核对；新洞 mask 由原图/移除图的 PGSR 差异标注，DeAOT 传播并记录几何漂移修正；不读取 test RGB 或 unseen_mask 进行训练/跟踪。全部场景创建独立 removal 与跟踪产物。",
            "基线列是本服务器实际复现实验，不能当作论文公布数值。基线缺失场景显示缺失；跨方法均值只比较相同场景。", "",
            "轨迹通过各场景配置声明的可见性、遮挡、背景支持、碰撞、连续性和立体匹配检查。"
            "相机参数调整与背景支持代理阈值的例外记录在逐场景 JSON 和认证结果中。", "",
            "| 场景 | ID | MASK_DILATION | 相机倾角 / ROI / 背景支持门槛 | 状态 |", "|---|---|---:|---|---|"]
        for s in self.req["scenes"]:
            state = self.status["scenes"].get(s["name"], {}).get("state", "queued")
            options = read(s.get("adaptive_config", self.env["VIRTUAL_ADAPTIVE_CONFIG"]))
            camera = (f"{options['inclination_range_deg']}° / {options['target_roi_fraction']:.0%} / "
                      f"{options['min_context_support']:.0%}")
            lines.append(f"| {s['name']} | {s['target_ids']} | {s['mask_dilation']} | {camera} | {state} |")
        if quality_retrain:
            source_dir = Path(quality_retrain["source"]).parent
            relative_source = os.path.relpath(source_dir, self.job)
            lines += ["", "当前 PaintMesh 5000 逐场景行已覆盖为本轮登记的选择结果：在原结果与完成的新实验之间，"
                "按完整测试集全图 PSNR 择优，每行七项指标全部来自同一个实验；未提升场景保留原结果。"
                "这是逐场景事后选择的探索结果及其汇总，不是统一固定配置的独立测试成绩。",
                "本轮仅重训 5000 步；PaintMesh 10000 保留此前结果，不能作为本轮新配置继续训练到 10000 步的对照。"
                "上方配置与训练说明描述原始批次，当前所选实验改动见下面的修复记录。",
                f"[全部消融与失败记录]({relative_source}/comparison.md) · "
                f"[配置、原因和诊断]({relative_source}/DIAGNOSIS.zh-CN.md) · "
                f"[新实验对照图片]({relative_source}/before_after/README.md)"]
        correction_labels = {(r["method"], r["scene"]): r["label"] for r in corrections}
        if corrections:
            lines += ["", "## 低 PSNR 复查与结果更新", "",
                "下面的逐场景指标和均值已采用登记的修复结果，全部七项一起更新，不按指标挑选最好数值。未修复场景沿用原始运行。",
                "AuraFusion360/redbull 是初始化修正版，不能当作未经修改的官方复现；汇总行中的 Aura 含这一个变体场景。",
                "[详细排查记录](../low_psnr_audit_20261026/AUDIT.zh-CN.md)；原始七项指标保存在 comparison.json 的 original_metrics。", "",
                "| 方法 | 场景 | 原始 PSNR | 更新 PSNR | 修复内容 |", "|---|---|---:|---:|---|"]
            for r in corrections:
                method, scene = r["method"], r["scene"]
                lines.append(f"| {method} | {scene} | {original[method][scene]['PSNR']:.6f} | {methods[method][scene]['PSNR']:.6f} | {r['reason']} |")
            lows = [f"{m}/{s} = {v['PSNR']:.6f}" for m, rows in methods.items()
                    if m != "PaintMesh 5000" for s, v in rows.items() if v['PSNR'] < 20]
            lines += ["", "当前最终 PSNR < 20：" + ("；".join(lows) or "无") + "。",
                "GG/cone_red 的跟踪修正已完成；同一组测试相机的洞外区域 PSNR 在原始模型约 19.60、修正后约 19.20，剩余误差主要涉及已有背景的新视角重建。该区域诊断值与论文式 masked PSNR 不同。"]
        issue_file = self.job / "evaluation_issues.json"
        if issue_file.is_file():
            for issue in read(issue_file)["issues"]:
                if issue.get("status") == "confirmed_target_mismatch_pending_repair":
                    audit = os.path.relpath(issue["audit"], self.job)
                    lines += ["", f"**已确认的对比问题：{issue['method']} / {issue['scene']} 删除目标不一致，误删了应保留的物体。"
                        f"该项不能作为同目标方法优劣的有效证据；当前表格及均值仍保留该历史问题运行，待修复重评。** [证据与修复范围]({audit})"]
        lines += ["", "## 每场景指标", "", "| 场景 | 方法 | " + " | ".join(METRICS) + " |",
                  "|---|---|" + "---:|" * len(METRICS)]
        for s in self.req["scenes"]:
            for method, rows in methods.items():
                values = rows.get(s["name"])
                label = method + (f"（{correction_labels[method, s['name']]}）" if (method, s['name']) in correction_labels else "")
                lines.append(f"| {s['name']} | {label} | " + " | ".join(f"{values[m]:.6f}" if values else "待完成" for m in METRICS) + " |")
        lines += ["", "## 相同场景均值", "", f"共同完成的场景：{', '.join(sorted(common)) or '尚无'}。",
            "本表采用官方代码式平均：前六项按测试帧数加权，FID 按场景平均（不是 pooled FID）。场景宏平均另存于 comparison.json。修复内容见上表。", "",
            "| 方法 | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for method, values in aggregates.items():
            row = values["common_official_code_average"]
            label = method + ("（含初始化修正版）" if method == "AuraFusion360" and any(r['method'] == method for r in corrections) else "")
            lines.append(f"| {label} | " + " | ".join(f"{row[m]:.6f}" if row else "待完成" for m in METRICS) + " |")
        if (self.job / "qualitative_pm5000_relative_joint_psnr/README.md").is_file():
            lines += ["", "## 四方法对比图（自身双指标较高、相对优势优先）", "",
                "使用当前登记的 PaintMesh 5000 模型，11 个场景各展示 3 个共同测试视角。"
                "先要求 PaintMesh 的 PSNR 与 masked_PSNR 均不低于该场景中位数，再优先最大化两项相对最强对手差值中的较小值。"
                "不足三张时的补充视角明确标注；无双指标全面领先的场景保留真实差距。"
                "这是偏向 PaintMesh 的择优展示，四方法使用同一相机和裁剪，完整测试集均值不变。",
                "[全场景高清总览](qualitative_pm5000_relative_joint_psnr/all_scenes.png) · "
                "[修复区域总览](qualitative_pm5000_relative_joint_psnr/all_scenes_crops.png) · "
                "[三页 PDF](qualitative_pm5000_relative_joint_psnr/comparison_all_scenes.pdf) · "
                "[11 场景三视角 PDF](qualitative_pm5000_relative_joint_psnr/comparison_multiview.pdf) · "
                "[分场景图与来源](qualitative_pm5000_relative_joint_psnr/README.md) · "
                "[逐帧指标差距与选图依据](qualitative_pm5000_relative_joint_psnr/selection/README.md)"]
        if (self.job / "qualitative_pm5000_joint_psnr/README.md").is_file():
            lines += ["", "## 四方法最新对比图（PSNR 与 masked_PSNR 联合择优）", "",
                "使用当前登记的 PaintMesh 5000 模型，11 个场景各选 3 个共同测试视角。"
                "按 PaintMesh 两项指标中较低的场景内百分位优先排序，再按平均百分位排序；"
                "总览采用各场景第一名。四方法和 GT 使用相同相机与局部区域，图中同时标注逐视角 PSNR / masked_PSNR。"
                "这是择优展示，完整测试集均值不变。",
                "[全场景高清总览](qualitative_pm5000_joint_psnr/all_scenes.png) · "
                "[修复区域总览](qualitative_pm5000_joint_psnr/all_scenes_crops.png) · "
                "[三页 PDF](qualitative_pm5000_joint_psnr/comparison_all_scenes.pdf) · "
                "[11 场景三视角 PDF](qualitative_pm5000_joint_psnr/comparison_multiview.pdf) · "
                "[分场景图与来源](qualitative_pm5000_joint_psnr/README.md) · "
                "[425 个视角的选图排名](qualitative_pm5000_joint_psnr/selection/README.md)"]
        if quality_retrain:
            lines += ["", "以下已有四方法拼图仍对应本轮更新前的模型，图中数值未同步重绘；"
                "当前所选新实验图像请使用上方“新实验对照图片”链接。"]
        if (self.job / "qualitative_pm5000/README.md").is_file():
            lines += ["", "## 四方法图像对照（PaintMesh 5000）", "",
                "全部 11 个场景，每场景 3 个统一测试视角，提供完整图像与同位置局部放大。",
                "[全场景总览](qualitative_pm5000/all_scenes.png) · [局部区域总览](qualitative_pm5000/all_scenes_crops.png) · "
                "[三页 PDF](qualitative_pm5000/comparison_all_scenes.pdf) · [分场景图与来源说明](qualitative_pm5000/README.md)"]
        if (self.job / "qualitative_pm5000_showcase/README.md").is_file():
            lines += ["", "## PaintMesh 优势视角展示（5000 步）", "",
                "全部 11 个场景，每场景 3 个共同测试视角。按修复区域质量与相对表现筛选并人工检查；这是择优展示，完整测试集指标不变。",
                "[全场景总览](qualitative_pm5000_showcase/all_scenes.png) · [局部区域总览](qualitative_pm5000_showcase/all_scenes_crops.png) · "
                "[三页 PDF](qualitative_pm5000_showcase/comparison_all_scenes.pdf) · "
                "[分场景图与来源说明](qualitative_pm5000_showcase/README.md) · [选图记录](qualitative_pm5000_showcase/selection/README.md)"]
        if (self.job / "qualitative_pm5000_high_psnr/README.md").is_file():
            lines += ["", "## PaintMesh 高 PSNR 视角展示（5000 步）", "",
                "按每个场景 PaintMesh 的逐帧全图 PSNR 严格降序取前三名；总览使用第一名。图中显示四种方法在相同视角的全图 PSNR，完整测试集均值不变。",
                "[全场景总览](qualitative_pm5000_high_psnr/all_scenes.png) · [局部区域总览](qualitative_pm5000_high_psnr/all_scenes_crops.png) · "
                "[三页 PDF](qualitative_pm5000_high_psnr/comparison_all_scenes.pdf) · "
                "[11 场景三视角 PDF](qualitative_pm5000_high_psnr/comparison_multiview.pdf) · "
                "[逐帧 PSNR 与选图依据](qualitative_pm5000_high_psnr/selection/README.md)"]
        (self.job / "comparison.md").write_text("\n".join(lines) + "\n")

    def command(self, scene, phase, args, *, start=None, end=None, extra=None):
        args = stage_command(args, start)
        name = scene["name"]; env = dict(self.env)
        env.update(REMOVAL_ROOT=scene["removal_root"], INPAINT_RUN_ROOT=scene["inpaint_root"],
            RUN_NAME=Path(scene["removal_root"]).name, INPAINT_RUN_NAME=self.req["run_name"],
            MASK_DILATION=str(scene["mask_dilation"]))
        if scene.get("adaptive_config"):
            env["VIRTUAL_ADAPTIVE_CONFIG"] = str(Path(scene["adaptive_config"]).resolve(strict=True))
        if start is not None: env["START_STAGE"] = str(start)
        if end is not None: env["END_STAGE"] = str(end)
        if extra: env.update(extra)
        self.status.update(scene=name, phase=phase)
        self.status["scenes"][name].update(phase=phase, state="running")
        log_path = self.job / f"{name}_{phase}.log"
        print(f"[{now()}] {name} {phase}: {shlex.join(map(str,args))}", flush=True)
        with log_path.open("ab", buffering=0) as log:
            self.child = subprocess.Popen(list(map(str,args)), cwd=REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.status.update(child_pid=self.child.pid, current_log=str(log_path)); self.save()
            code = self.child.wait()
        self.status.pop("child_pid", None)
        self.child = None
        if code: raise RuntimeError(f"{name}/{phase} exit {code}; {log_path}")

    def run(self):
        from edgs_inpaint_io import load_config, implementation
        cfg = load_config(self.req["config"])
        if not (cfg["iterations"] == 10000 and cfg["freeze_background"] and cfg["local_background"]["enabled"]
            and cfg["pruning"]["enabled"] and not cfg["adaptive_density"]["enabled"]
            and cfg["init"]["use_depth"] and not cfg["init"]["use_normal"] and cfg["surface_init"]["mode"] == "none"):
            raise ValueError("configuration differs from requested experiment")
        write(self.job / "resolved_config.json", cfg)
        code = implementation()
        for script in ("run_inpaint360_full.py", "track_holes_headless.py", "tune_inpaint360_cameras.py",
                       "run_inpaint.sh", "run_remove.sh", "eval_inpaint360.py"):
            code["scripts/paintmesh/" + script] = sha(REPO / "scripts/paintmesh" / script)
        planner = REPO / "submodules/Inpaint360GS/utils/adaptive_camera.py"
        code[str(planner.relative_to(REPO))] = sha(planner)
        for config in (Path(self.req["config"]), Path(self.env["VIRTUAL_ADAPTIVE_CONFIG"]),
                       REPO / "scripts/paintmesh/configs/inpaint360_full_scenes.json",
                       *(Path(s["adaptive_config"]) for s in self.req["scenes"] if s.get("adaptive_config"))):
            code[str(config.relative_to(REPO))] = sha(config)
        previous = self.job / "implementation.json"
        if previous.exists() and read(previous) != code:
            raise ValueError("implementation changed; review before resuming this recorded experiment")
        write(previous, code)
        # Preparation was detached before job installation; wait without claiming
        # the GPU or accidentally running a second LaMa process concurrently.
        pid_file = self.job / "prepare.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text())
            while Path(f"/proc/{pid}/stat").exists():
                if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z": break
                self.status.update(phase="bag LaMa preparation", scene="bag"); self.save(); time.sleep(5)
        for scene in self.req["scenes"]:
            name = scene["name"]
            entry = self.status["scenes"].setdefault(name, {})
            if entry.get("state") == "complete": continue
            entry.update(state="running", started_at=now())
            entry.pop("error", None); entry.pop("failed_at", None); self.save()
            try:
                if shutil.disk_usage(REPO).free < 30 * 1024**3:
                    raise RuntimeError("less than 30 GiB free; refusing to start another scene")
                import yaml
                edgs = REPO / "output/paintmesh/Inpaint360GS/inpaint360" / name / "edgs/config.yaml"
                saved = yaml.safe_load(edgs.read_text())["gs"]["dataset"]
                if int(saved["resolution"]) != 4 or saved["eval"] or not saved.get("exclude_test_images"):
                    raise ValueError("seg training is not quarter-resolution/non-test-only")
                remove = ["bash", "scripts/paintmesh/run_remove.sh", "Inpaint360GS/inpaint360", name, "4",
                          ",".join(map(str, scene["target_ids"])), "none", "1"]
                removal = Path(scene["removal_root"])
                if not scene["reuse_removal"]:
                    session_path = removal / "tracker/tracking_session.json"
                    session = read(session_path) if session_path.is_file() else {}
                    if not (session.get("input_archive") and session.get("input_cameras")):
                        # A Stage 4 interruption leaves an invalidation marker,
                        # not an active tracking session. Reuse the complete mesh
                        # without regenerating its camera-bound provenance.
                        removed = removal / "removal_manifest.json"
                        ready = removed.is_file() and read(removed).get("complete") is True
                        self.command(scene, "removal", remove, start=4 if ready else 1, end=4,
                                     extra={"LAUNCH_REFINER":"false"})
                    session = read(session_path)
                    if not session.get("complete"):
                        masks = removal / "tracker/results/images/images_masks"
                        if not masks.is_dir():
                            self.command(scene, "tracking", [PYTHON, "scripts/paintmesh/track_holes_headless.py",
                                "--removal-root", removal])
                self.command(scene, "commit_masks", remove, start=5, end=5, extra={"LAUNCH_REFINER":"true"})
                inpaint = ["bash", "scripts/paintmesh/run_inpaint.sh", "Inpaint360GS/inpaint360", name, "4",
                           ",".join(map(str,scene["target_ids"])), "none", "1"]
                root = Path(scene["inpaint_root"])
                common = root / "manifests/lama_completion_manifest.json"
                ready = common.is_file() and read(common).get("complete") is True
                self.command(scene, "inpaint_train", inpaint, start=4 if ready else 1, end=5)
                model = export_snapshot(root, 5000)
                self.command(scene, "eval_5000", [PYTHON, "scripts/paintmesh/eval_inpaint360.py", "run",
                    "--model-path", model, "--source-path", DATA / name, "--resolution", "4", "--iteration", "5000",
                    "--batch-size", "4"])
                self.command(scene, "publish_10000", inpaint, start=6, end=6)
                self.command(scene, "eval_10000", [PYTHON, "scripts/paintmesh/eval_inpaint360.py", "run",
                    "--model-path", root / "inpainted_3dgs", "--source-path", DATA / name,
                    "--resolution", "4", "--iteration", "10000", "--batch-size", "4"])
                self.command(scene, "mesh", inpaint, start=7, end=8)
                entry.update(state="complete", completed_at=now()); entry.pop("error", None)
            except InterruptedError:
                entry.update(state="interrupted"); self.status["state"] = "interrupted"; self.save(); raise
            except Exception as exc:
                entry.update(state="failed", error=str(exc), failed_at=now())
                print(traceback.format_exc(), flush=True)
            self.save()
        failed = [s for s, v in self.status["scenes"].items() if v["state"] != "complete"]
        self.status.update(state="complete" if not failed else "finished_with_errors", phase="finished",
                           failed_scenes=failed, finished_at=now()); self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help="resume an existing recorded job; omit to create a new one")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only and args.job is None:
        parser.error("--report-only requires --job")
    job = args.job.resolve(strict=True) if args.job else create_job()
    if args.detach:
        with (job / "run.log").open("ab",buffering=0) as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--job", str(job)],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (job / "run.pid").write_text(str(child.pid) + "\n"); print(f"PID {child.pid}; {job}"); return
    runner = FullJob(job, report_only=args.report_only)
    if args.report_only: runner.report(); return
    with (REPO / "logs/paintmesh_full_gpu1.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: runner.run()
        except BaseException as exc:
            runner.status.update(state="interrupted" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
            runner.save(); raise


if __name__ == "__main__":
    main()
