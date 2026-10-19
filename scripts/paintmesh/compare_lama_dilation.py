#!/usr/bin/env python3
"""Isolated RGB-only LaMa mask-dilation comparison on existing virtual views.

Run with the paintmesh Python. LaMa inference runs in its own conda environment.
Only dilation changes; removal, cameras, RGB, checkpoint and refinement stay fixed.
These artifacts are diagnostic RGB results, not pipeline completion manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
LAMA = REPO / "submodules/Inpaint360GS/LaMa"
sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
from tools.prepare_paintmesh_lama_data import _clean_mask, _read_index_mask

DILATIONS = (10, 20, 30)
COLORS = ("#00d8ff", "#ffac30", "#ff48da")
SCENES = ("bag", "car", "cone_red", "cone_yellow", "cube", "doppelherz", "redbull", "truck")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def preflight(scenes, sample_count=0):
    sources = {}
    for scene in scenes:
        base = REPO / "output/paintmesh/Inpaint360GS/inpaint360" / scene
        base /= "removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP"
        path = base / "manifests/lama_input_manifest.json"
        payload = json.loads(path.read_text())
        if payload.get("complete") is not True:
            raise ValueError(f"Expected complete prepared-input source: {path}")
        all_frames = payload["parameters"]["frame_names"]
        if len(all_frames) != 90 or set(all_frames) != set(payload["frames"]):
            raise ValueError(f"Expected exactly 90 matching virtual views: {path}")
        frames = all_frames
        if sample_count:
            if not 3 <= sample_count <= len(all_frames):
                raise ValueError("sample-count must be 0 (all views) or between 3 and 90")
            indices = np.linspace(0, len(all_frames) - 1, sample_count).astype(int).tolist()
            indices[len(indices) // 2] = len(all_frames) // 2
            frames = [all_frames[index] for index in sorted(set(indices))]
        sources[scene] = {"manifest": str(path), "sha256": digest(path),
                          "payload": payload, "frames": frames, "total_source_frames": len(all_frames)}
        for frame in frames:
            record = payload["frames"][frame]
            for key in ("mask", "removed_rgb"):
                item = record["inputs"][key]
                if digest(item["path"]) != item["sha256"]:
                    raise ValueError(f"Source changed: {item['path']}")
            raw = _read_index_mask(Path(record["inputs"]["mask"]["path"]))
            expected = _clean_mask(raw, payload["parameters"]["min_area"], payload["parameters"]["dilation"])
            actual = _read_index_mask(Path(record["outputs"]["color_mask"]["path"]))
            if not np.array_equal(expected, actual):
                raise ValueError(f"Prepared mask differs from pipeline: {scene}/{frame}")
        print(f"Preflight OK: {scene}, {len(frames)} selected frames, existing masks match", flush=True)
    return sources


def prepare(sources, root):
    rows = []
    for scene, source in sources.items():
        min_area = source["payload"]["parameters"]["min_area"]
        inputs = {d: root / scene / f"dilation_{d}/lama/input/color" for d in DILATIONS}
        for path in inputs.values():
            path.mkdir(parents=True)
        for frame in source["frames"]:
            record = source["payload"]["frames"][frame]
            rgb_path = Path(record["inputs"]["removed_rgb"]["path"])
            raw = _read_index_mask(Path(record["inputs"]["mask"]["path"]))
            if read_rgb(rgb_path).shape[:2] != raw.shape:
                raise ValueError(f"RGB/mask shape mismatch: {scene}/{frame}")
            previous = None
            for dilation, indir in inputs.items():
                mask = _clean_mask(raw, min_area, dilation)
                if previous is not None and np.any(previous & ~mask):
                    raise ValueError("Masks must grow monotonically")
                previous = mask
                (indir / f"{frame}.png").symlink_to(rgb_path)
                mask_path = indir / f"{frame}_mask.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                rows.append({"scene": scene, "frame": frame, "dilation": dilation,
                             "height": mask.shape[0], "width": mask.shape[1],
                             "raw_pixels": int(raw.sum()), "mask_pixels": int(mask.sum()),
                             "mask_fraction": float(mask.mean()), "rgb": str(rgb_path),
                             "rgb_sha256": record["inputs"]["removed_rgb"]["sha256"],
                             "raw_mask": record["inputs"]["mask"]["path"],
                             "raw_mask_sha256": record["inputs"]["mask"]["sha256"],
                             "prepared_mask_sha256": digest(mask_path)})
    with (root / "mask_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def verify_outputs(scene, dilation, source, root):
    base = root / scene / f"dilation_{dilation}/lama"
    outdir = base / "output/color"
    actual = {p.stem for p in outdir.glob("*.png")}
    if actual != set(source["frames"]):
        raise ValueError(f"Output frame mismatch: {outdir}")
    records = {}
    for frame in source["frames"]:
        a = read_rgb(base / "input/color" / f"{frame}.png")
        b = read_rgb(outdir / f"{frame}.png")
        mask = _read_index_mask(base / "input/color" / f"{frame}_mask.png")
        if a.shape != b.shape or not np.array_equal(a[~mask], b[~mask]):
            raise ValueError(f"Unexpected shape or outside-mask edit: {scene}/{dilation}/{frame}")
        records[frame] = {"path": str(outdir / f"{frame}.png"),
                          "sha256": digest(outdir / f"{frame}.png"),
                          "outside_mask_changed_pixels": 0}
    write_json(base.parent / "rgb_comparison_manifest.json", {
        "kind": "lama-rgb-dilation-diagnostic", "complete": True,
        "dilation": dilation, "recursive_guide": False, "frames": records})


def make_figures(sources, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"

    def draw(axes, scene, frame):
        bases = [root / scene / f"dilation_{d}/lama" for d in DILATIONS]
        a = read_rgb(bases[0] / "input/color" / f"{frame}.png")
        masks = [_read_index_mask(b / "input/color" / f"{frame}_mask.png") for b in bases]
        ys, xs = np.where(masks[-1])
        pad = 65
        y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad + 1)
        region = np.s_[y0:y1, x0:x1]
        axes[0].imshow(a[region], interpolation="nearest")
        for m, color in zip(masks, COLORS):
            axes[0].contour(m[region], levels=[.5], colors=[color], linewidths=.8)
        axes[0].set_title(f"{scene} / {frame} 输入 + {'/'.join(map(str, DILATIONS))} 边界", fontsize=10)
        for ax, b, d, m, color in zip(axes[1:], bases, DILATIONS, masks, COLORS):
            ax.imshow(read_rgb(b / "output/color" / f"{frame}.png")[region], interpolation="nearest")
            ax.contour(m[region], levels=[.5], colors=[color], linewidths=.6)
            ax.set_title(f"MASK_DILATION={d} · mask {m.mean():.1%}", fontsize=10)
        for ax in axes:
            ax.axis("off")

    examples = [(s, f) for s in sources for f in ("00000", "00045")]
    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 3.1 * len(examples)), squeeze=False)
    for row, (scene, frame) in zip(axes, examples):
        draw(row, scene, frame)
    fig.suptitle(f"LaMa mask 扩张对照：青色 {DILATIONS[0]} / 橙色 {DILATIONS[1]} / 品红 {DILATIONS[2]}；递归引导关闭", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(root / "comparison_overview.png", dpi=150)
    plt.close(fig)
    pages = []
    # Every selected view is available for visual shadow-coverage inspection.
    for scene, source in sources.items():
        for start in range(0, len(source["frames"]), 15):
            frames = source["frames"][start:start + 15]
            fig, axes = plt.subplots(len(frames), 4, figsize=(14, 2.6 * len(frames)), squeeze=False)
            for row, frame in zip(axes, frames):
                draw(row, scene, frame)
            fig.tight_layout()
            name = f"{scene}_views_{start:02d}_{start + len(frames) - 1:02d}.jpg"
            fig.savefig(root / name, dpi=110)
            plt.close(fig)
            pages.append((scene, name))
        print(f"Figures completed: {scene}, all {len(source['frames'])} views", flush=True)
        review_frames = [source["frames"][i] for i in sorted({0, len(source["frames"]) // 3,
                         2 * len(source["frames"]) // 3, len(source["frames"]) - 1})]
        fig, axes = plt.subplots(len(review_frames), 4, figsize=(16, 3.1 * len(review_frames)), squeeze=False)
        for row, frame in zip(axes, review_frames):
            draw(row, scene, frame)
        fig.suptitle(f"{scene}: mask {' / '.join(map(str, DILATIONS))}, RECURSIVE_GUIDE=false", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(root / f"{scene}_review.png", dpi=140)
        plt.close(fig)
    gallery = ["<!doctype html><meta charset='utf-8'><title>LaMa dilation comparison</title>",
               "<style>body{font-family:sans-serif;max-width:1600px;margin:24px auto;background:#eee}img{width:100%;height:auto}a{margin-right:16px}</style>",
               f"<h1>LaMa：MASK_DILATION={' / '.join(map(str, DILATIONS))}</h1>",
               f"<p>RECURSIVE_GUIDE=false；固定虚拟视角、RGB、权重与推理设置，仅改变椭圆膨胀半径。青色={DILATIONS[0]}，橙色={DILATIONS[1]}，品红={DILATIONS[2]}。</p>",
               "<p>这些是虚拟视角的 RGB 补全对照，不是 GT 相机视角；没有人工阴影标注，不能由 mask 面积自动判定阴影已全部覆盖。</p>",
               "<a href='mask_coverage.csv'>各帧 mask 面积 CSV</a><a href='report.md'>实验报告</a>",
               "<img src='comparison_overview.png' alt='对照总览'>"]
    for scene, name in pages:
        gallery += [f"<h2>{html.escape(scene)} / {name}</h2>", f"<a href='{name}'><img loading='lazy' src='{name}' alt='{name}'></a>"]
    (root / "index.html").write_text("\n".join(gallery), encoding="utf-8")


def main():
    global DILATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=["bag", "truck"])
    parser.add_argument("--sample-count", type=int, default=0,
                        help="Evenly sample this many views including 0, 45, 89; 0 means all 90")
    parser.add_argument("--dilations", type=int, nargs=3, default=DILATIONS,
                        help="Three increasing dilation radii to compare (default: 10 20 30)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lama-env", default="lama")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    DILATIONS = tuple(args.dilations)
    if len(set(DILATIONS)) != 3 or tuple(sorted(DILATIONS)) != DILATIONS or not all(0 <= d <= 256 for d in DILATIONS):
        raise ValueError("Provide three distinct increasing dilation radii in [0, 256]")
    root = args.output_root or (REPO / "output/evaluation" / ("lama_dilation_bag_truck_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    root = root.resolve()
    if len(set(args.scenes)) != len(args.scenes):
        raise ValueError("Duplicate scene")
    if root.exists():
        raise ValueError(f"Use a new output directory: {root}")
    print(f"Output: {root}; GPU={args.gpu}; RECURSIVE_GUIDE=false; RGB-only", flush=True)
    sources = preflight(args.scenes, args.sample_count)
    if args.dry_run:
        print("Dry run OK; no files written or inference launched", flush=True)
        return
    root.mkdir(parents=True)
    logdir = REPO / "logs" / root.name
    logdir.mkdir(parents=True, exist_ok=True)
    model = REPO / "ckpt/big-lama"
    state = {"status": "preparing", "recursive_guide": False, "dilations": DILATIONS,
             "scenes": args.scenes, "gpu": args.gpu, "rgb_only": True, "sample_count": args.sample_count,
             "mask_kernel": "ellipse", "log_dir": str(logdir),
             "checkpoint_sha256": digest(model / "models/best.ckpt"),
             "model_config_sha256": digest(model / "config.yaml"),
             "prediction_config_sha256": digest(LAMA / "configs/prediction/default.yaml"),
             "prediction_script_sha256": digest(LAMA / "bin/predict_color.py"),
             "mask_preparation_sha256": digest(REPO / "submodules/Inpaint360GS/tools/prepare_paintmesh_lama_data.py"),
             "sources": {s: {k: v for k, v in info.items() if k in ("manifest", "sha256", "frames")} for s, info in sources.items()},
             "completed": []}
    status_path = root / "experiment.json"
    write_json(status_path, state)
    try:
        rows = prepare(sources, root)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONPATH": str(LAMA),
                    "TORCH_HOME": str(LAMA), "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        conda = "/home/anaconda3/condabin/conda"
        prefix = subprocess.check_output(
            [conda, "run", "--no-capture-output", "-n", args.lama_env,
             "python", "-c", "import sys; print(sys.prefix)"],
            env=env, text=True).strip().splitlines()[-1]
        if not (Path(prefix) / "lib").is_dir():
            raise ValueError(f"Invalid LaMa environment prefix: {prefix}")
        env["LD_LIBRARY_PATH"] = str(Path(prefix) / "lib") + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        state["lama_env_prefix"] = prefix
        for scene, source in sources.items():
            for dilation in DILATIONS:
                base = root / scene / f"dilation_{dilation}/lama"
                command = [conda, "run", "--no-capture-output", "-n", args.lama_env,
                           "python", "bin/predict_color.py", "--input-dir", str(base / "input/color"),
                           "--output-dir", str(base / "output/color"), "--model-path", str(model)]
                log = logdir / f"{scene}_dilation_{dilation}.log"
                state.update(status="running", current=f"{scene}/dilation_{dilation}")
                write_json(status_path, state)
                print(f"Starting {state['current']} ({len(source['frames'])} frames), log={log}", flush=True)
                with log.open("w") as stream:
                    subprocess.run(command, cwd=LAMA, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                verify_outputs(scene, dilation, source, root)
                state["completed"].append(state["current"])
                write_json(status_path, state)
                print(f"Validated {state['current']}: {len(source['frames'])} RGB outputs, zero outside-mask changes", flush=True)
        state["status"] = "plotting"
        write_json(status_path, state)
        make_figures(sources, root)
        report = ["# LaMa mask 扩张对照", "", "场景：" + ", ".join(sources) + "；分辨率 4；GPU " + args.gpu + "。",
                  "每场景选择的虚拟视角数：" + ", ".join(f"{s}={len(v['frames'])}" for s, v in sources.items()) + "。",
                  "RECURSIVE_GUIDE=false。所有三组均重新运行 RGB LaMa，保持相同输入 RGB、Big-LaMa 权重及 refine 设置。",
                  "仅改变现有 PaintMesh 椭圆核的膨胀半径；不同时更改为官方方形核，以免混入第二个变量。",
                  "原有训练和补全结果未改动；深度、法线补全及 3D 训练未重跑。", "",
                  "| Scene | Dilation | Mean mask fraction | Min | Max |", "|---|---:|---:|---:|---:|"]
        for scene in sources:
            for dilation in DILATIONS:
                values = [r["mask_fraction"] for r in rows if r["scene"] == scene and r["dilation"] == dilation]
                report.append(f"| {scene} | {dilation} | {np.mean(values):.2%} | {min(values):.2%} | {max(values):.2%} |")
        output_count = sum(len(s["frames"]) for s in sources.values()) * len(DILATIONS)
        report += ["", f"核验：现有输入 mask 与原流水线逐像素一致；三组 mask 逐层包含；全部 {output_count} 张输出尺寸匹配，mask 外像素逐字节保持输入。",
                   "", "查看各场景 *_review.png 和 comparison_overview.png；index.html 包含全部所选虚拟视角的三组对照。",
                   "没有真实阴影标注，未计算自动阴影覆盖率；需按边界图人工检查。虚拟视角无配对 GT，未报告伪造的 PSNR/SSIM 排名。"]
        (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        state.update(status="complete", complete=True, total_outputs=sum(len(s["frames"]) for s in sources.values()) * len(DILATIONS))
        write_json(status_path, state)
        print(f"COMPLETE: {root / 'comparison_overview.png'}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), complete=False)
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
