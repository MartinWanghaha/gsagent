"""Evaluate the eight completed table1_hemi90 removal/inpaint runs."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from eval_inpaint360 import METRICS, aggregate_results, image_index, write_json


REPO = Path(__file__).resolve().parents[2]
SCENES = ("bag", "car", "cone_red", "cone_yellow", "cube", "doppelherz", "redbull", "truck")
OTHER_SCENES = ("fruits", "garden_toys", "toys")
EXPERIMENT = "inpaint360_table1_hemi90_edgs_rgbdn_joint_woDP_r4_8scenes"


def model_path(scene):
    return REPO / "output/paintmesh/Inpaint360GS/inpaint360" / scene / "removal/table1_hemi90/inpaint/edgs_rgbdn_joint_woDP/inpainted_3dgs"


def preflight():
    policies = []
    for scene in SCENES:
        model = model_path(scene)
        for path in [model.parent / "inpaint_manifest.json", model / "model_manifest.json"]:
            data = json.loads(path.read_text())
            if data.get("complete") is not True or data.get("status") != "complete":
                raise ValueError(f"incomplete inpaint result: {path}")
        if not (model / "point_cloud/iteration_5000/point_cloud.ply").is_file():
            raise ValueError(f"missing iteration 5000: {scene}")
        dataset = yaml.safe_load((model / "config.yaml").read_text())["gs"]["dataset"]
        policy = {key: dataset[key] for key in ("resolution", "eval", "exclude_test_images")}
        policies.append(policy)
        source = (REPO / "data/Inpaint360GS/inpaint360" / scene).resolve(strict=True)
        if Path(dataset["source_path"]).resolve() != source:
            raise ValueError(f"saved source mismatch: {scene}")
        references = image_index(source / "images", reference_only=True)
        masks = image_index(source / "unseen_mask", reference_only=True)
        if len(references) < 2 or set(references) != set(masks):
            raise ValueError(f"missing or mismatched official reference masks: {scene}")
        print(f"Ready: {scene}, {len(references)} reference views, policy={policy}", flush=True)
    if any(policy != policies[0] for policy in policies):
        raise ValueError("scene training policies differ")
    return policies[0]


def publish(paths, report_root, training_policy):
    summary = aggregate_results(paths)
    summary.update(experiment=EXPERIMENT, training_policy=training_policy,
                   evaluated_scenes=list(SCENES), not_evaluated_scenes=list(OTHER_SCENES),
                   model_iteration=5000)
    if summary["scene_count"] != len(SCENES):
        raise ValueError("batch scene count mismatch")
    write_json(report_root / "summary.json", summary)
    counts = {json.loads(path.read_text())["inputs"]["scene"]:
              json.loads(path.read_text())["frame_count"] for path in paths}
    rows = [(scene, counts[scene], summary["per_scene"][scene]) for scene in SCENES]
    rows += [("official_code_average", summary["frame_count"], summary["official_code_average"]),
             ("scene_macro_average", summary["frame_count"], summary["scene_macro_average"])]
    with (report_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "frames", *METRICS])
        for name, count, scores in rows:
            writer.writerow([name, count, *[scores[key] for key in METRICS]])
    lines = ["# Inpaint360GS: eight-scene evaluation", "",
             f"Scenes: {summary['scene_count']}; object-free reference views: {summary['frame_count']}.", "",
             f"Training policy: `{json.dumps(training_policy, sort_keys=True)}`. Evaluation resolution: 4.", "",
             "This is an eight-scene subset; fruits, garden_toys and toys are not included.", "",
             "| Scene | Views | PSNR ↑ | masked PSNR ↑ | SSIM ↑ | masked SSIM ↑ | LPIPS ↓ | masked LPIPS ↓ | FID ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, count, scores in rows:
        lines.append(f"| {name} | {count} | " + " | ".join(f"{scores[key]:.6f}" for key in METRICS) + " |")
    lines += ["", "Official-code average: PSNR/SSIM/LPIPS average all reference frames; FID averages per-scene FIDs.",
              "Scene macro average: all seven metrics average the eight per-scene values equally.", ""]
    (report_root / "summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not args.gpu.isdigit():
        raise ValueError("GPU must be one nonnegative index and batch size must be positive")
    training_policy = preflight()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = REPO / "logs" / f"eval_inpaint360_gpu{args.gpu}_{stamp}_{os.getpid()}"
    report_root = REPO / "output/evaluation" / EXPERIMENT
    paths = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if not args.dry_run:
        log_root.mkdir(parents=True)
        report_root.mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(SCENES, 1):
        model = model_path(scene)
        command = [sys.executable, str(REPO / "scripts/paintmesh/eval_inpaint360.py"), "run",
                   "--model-path", str(model), "--source-path", str(REPO / "data/Inpaint360GS/inpaint360" / scene),
                   "--resolution", "4", "--iteration", "5000", "--batch-size", str(args.batch_size)]
        result_path = model / "evaluation/inpaint360_r4/results.json"
        if args.dry_run:
            print(f"[{index}/{len(SCENES)}] GPU={args.gpu}: {scene} -> {result_path}")
            continue
        scene_log = log_root / f"{scene}.log"
        print(f"[{index}/{len(SCENES)}] Evaluating {scene} on GPU {args.gpu}; log: {scene_log}", flush=True)
        with scene_log.open("w") as output:
            completed = subprocess.run(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
        if completed.returncode:
            print("\n".join(scene_log.read_text(errors="replace").splitlines()[-25:]), flush=True)
            raise RuntimeError(f"evaluation failed for {scene}; see {scene_log}")
        paths.append(result_path)
        result = json.loads(result_path.read_text())
        print(f"Completed {scene}: {json.dumps(result['metrics'])}", flush=True)
    if not args.dry_run:
        publish(paths, report_root, training_policy)
        print(f"Reports saved: {report_root}", flush=True)


if __name__ == "__main__":
    main()
