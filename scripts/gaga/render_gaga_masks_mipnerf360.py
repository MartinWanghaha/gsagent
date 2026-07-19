#!/usr/bin/env python3
"""Render Gaga RGB images and instance masks for Mip-NeRF 360 scenes.

Examples:
    python scripts/gaga/render_gaga_masks_mipnerf360.py \
        --scene counter --seg-method entityseg --gpu 0

    python scripts/gaga/render_gaga_masks_mipnerf360.py \
        --scene counter \
        --model-path outputs/gaga_mipnerf360/counter/entityseg \
        --seg-method entityseg --split test --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .associate_gaga_masks_mipnerf360 import (
        discover_scenes,
        inspect_point_cloud,
        select_scenes,
    )
except ImportError:
    from associate_gaga_masks_mipnerf360 import (  # type: ignore[no-redef]
        discover_scenes,
        inspect_point_cloud,
        select_scenes,
    )


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_GAGA_ROOT = GSAGENT_ROOT / "submodules" / "Gaga"
DEFAULT_MODEL_ROOT = GSAGENT_ROOT / "outputs" / "gaga_mipnerf360"
ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Render RGB images, color previews, feature visualizations, and "
            "lossless predicted instance masks from lifted Gaga models."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name. Repeat the option or use comma-separated names. "
            "Default: all discovered scenes."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mip-NeRF 360 root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--gaga-root",
        type=Path,
        default=DEFAULT_GAGA_ROOT,
        help=f"Gaga checkout root. Default: {DEFAULT_GAGA_ROOT}",
    )
    parser.add_argument(
        "--model-path",
        "--model-root",
        dest="model_path",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=(
            "A lifted method directory, a scene directory, or a root containing "
            f"one directory per scene. Default: {DEFAULT_MODEL_ROOT}"
        ),
    )
    parser.add_argument(
        "--seg-method",
        "--mask-method",
        dest="seg_method",
        choices=("sam", "entityseg"),
        default="entityseg",
        help="Lifted model and associated-mask method. Default: entityseg.",
    )
    parser.add_argument(
        "--images",
        default="images",
        help="Image subdirectory matching the COLMAP cameras. Default: images.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Lift iteration to render. Default: -1 (latest complete iteration).",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        help=(
            "Render resolution/downsample argument. Default: inherit the Lift "
            "manifest, otherwise 4."
        ),
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the evaluation split. Default: inherit the Lift manifest.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Camera split to render. Default: all.",
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used to store source RGB images and GT masks. Default: cpu.",
    )
    parser.add_argument(
        "--gpu",
        help="Value assigned to CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Also render Gaga's interpolated RGB/mask video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate used with --render-video. Default: 30.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Gaga's random-state banner.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Render again even when all requested outputs are complete.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after a render failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered dataset scenes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without rendering.",
    )

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be -1 or a positive integer.")
    if args.resolution is not None and (
        args.resolution == 0 or args.resolution < -1
    ):
        parser.error("--resolution must be -1 or a positive integer.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def complete_iterations(model_dir: Path) -> list[int]:
    point_cloud_root = model_dir / "point_cloud"
    if not point_cloud_root.is_dir():
        return []
    iterations = []
    for iteration_dir in point_cloud_root.iterdir():
        match = ITERATION_PATTERN.match(iteration_dir.name)
        if not match or not iteration_dir.is_dir():
            continue
        if (
            (iteration_dir / "point_cloud.ply").is_file()
            and (iteration_dir / "classifier.pth").is_file()
        ):
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def model_method(model_dir: Path) -> str | None:
    manifest_path = model_dir / "lift_manifest.json"
    if manifest_path.is_file():
        try:
            method = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("seg_method")
            if method in ("sam", "entityseg"):
                return method
        except json.JSONDecodeError:
            pass
    if model_dir.name in ("sam", "entityseg"):
        return model_dir.name
    return None


def resolve_scene_model(
    source: Path,
    scene: str,
    seg_method: str,
    selected_scene_count: int,
) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"Gaga model path does not exist: {source}")

    candidates = [
        source / scene / seg_method,
        source / scene,
        source / seg_method,
    ]
    if selected_scene_count == 1:
        candidates.append(source)

    valid = []
    for candidate in candidates:
        if candidate in valid or not complete_iterations(candidate):
            continue
        method = model_method(candidate)
        if method is None or method == seg_method:
            valid.append(candidate.resolve())
    valid = list(dict.fromkeys(valid))
    if not valid:
        raise FileNotFoundError(
            f"No complete Gaga '{seg_method}' model found for scene '{scene}' "
            f"under: {source}"
        )
    if len(valid) > 1:
        raise ValueError(
            f"Multiple Gaga models match scene '{scene}': "
            + ", ".join(str(path) for path in valid)
        )
    return valid[0]


def load_lift_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "lift_manifest.json"
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Lift manifest: {path}") from exc
    return manifest if isinstance(manifest, dict) else {}


def resolve_iteration(model_dir: Path, requested: int) -> int:
    available = complete_iterations(model_dir)
    if requested == -1:
        return available[-1]
    if requested not in available:
        raise FileNotFoundError(
            f"Iteration {requested} is incomplete in {model_dir}; "
            f"available: {available}"
        )
    return requested


def validate_model_and_masks(
    scene_dir: Path,
    model_dir: Path,
    seg_method: str,
    iteration: int,
) -> tuple[int, int]:
    info_path = scene_dir / f"{seg_method}_mask" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Associated-mask metadata does not exist: {info_path}"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid associated-mask metadata: {info_path}") from exc
    num_masks = info.get("num_mask")
    if not isinstance(num_masks, int) or num_masks <= 0:
        raise ValueError(f"Invalid num_mask in {info_path}: {num_masks!r}")

    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    mask_dtypes = set()
    max_mask_label = 0
    for mask_path in (scene_dir / f"{seg_method}_mask").glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"OpenCV could not decode associated mask: {mask_path}")
        mask_dtypes.add(str(mask.dtype))
        max_mask_label = max(
            max_mask_label,
            int(mask.max()) if mask.size else 0,
        )
    if num_masks > np.iinfo(np.uint8).max and max_mask_label <= np.iinfo(np.uint8).max:
        print(
            f"[warning] {scene_dir.name}: info.json declares {num_masks} groups, "
            f"but associated masks use {sorted(mask_dtypes)} and stop at label "
            f"{max_mask_label}. These masks were likely truncated by the legacy "
            "uint8 association export; rerun association and Lift for lossless IDs.",
            file=sys.stderr,
        )

    iteration_dir = model_dir / "point_cloud" / f"iteration_{iteration}"
    point_cloud = iteration_dir / "point_cloud.ply"
    classifier_path = iteration_dir / "classifier.pth"
    point_count, sh_degree = inspect_point_cloud(point_cloud)

    import torch  # pylint: disable=import-outside-toplevel

    state = torch.load(classifier_path, map_location="cpu")
    weight = state.get("weight") if isinstance(state, dict) else None
    expected_classes = num_masks + 1
    if weight is None or tuple(weight.shape) != (expected_classes, 16, 1, 1):
        shape = tuple(weight.shape) if weight is not None else None
        raise ValueError(
            f"Classifier shape {shape} does not match {expected_classes} classes"
        )
    return point_count, sh_degree


def image_count(scene_dir: Path, images: str) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in (scene_dir / images).iterdir()
    )


def expected_split_counts(total: int, eval_mode: bool) -> dict[str, int]:
    test = (total + 7) // 8 if eval_mode else 0
    return {"train": total - test, "test": test}


def requested_splits(split: str, eval_mode: bool) -> list[str]:
    if split == "test" and not eval_mode:
        raise ValueError("--split test requires --eval")
    if split == "train":
        return ["train"]
    if split == "test":
        return ["test"]
    return ["train", "test"] if eval_mode else ["train"]


def split_output_dir(model_dir: Path, split: str, iteration: int) -> Path:
    return model_dir / split / f"ours_{iteration}"


def outputs_are_complete(
    model_dir: Path,
    iteration: int,
    splits: list[str],
    counts: dict[str, int],
) -> bool:
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    for split in splits:
        root = split_output_dir(model_dir, split, iteration)
        predicted_paths = list((root / "objects_test").glob("*.png"))
        predicted = len(predicted_paths)
        rendered = len(list((root / "renders").glob("*.png")))
        if predicted < counts[split] or rendered < counts[split]:
            return False
        for path in predicted_paths:
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.dtype != np.uint16:
                return False
    return True


def build_command(
    gaga_root: Path,
    scene_dir: Path,
    model_dir: Path,
    seg_method: str,
    sh_degree: int,
    iteration: int,
    resolution: int,
    eval_mode: bool,
    args: argparse.Namespace,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(gaga_root / "render.py"),
        "--source_path",
        str(scene_dir),
        "--model_path",
        str(model_dir),
        "--object_path",
        f"{seg_method}_mask",
        "--images",
        args.images,
        "--sh_degree",
        str(sh_degree),
        "--iteration",
        str(iteration),
        "--resolution",
        str(resolution),
        "--data_device",
        args.data_device,
    ]
    if eval_mode:
        command.append("--eval")
    if args.split == "train":
        command.append("--skip_test")
    elif args.split == "test":
        command.append("--skip_train")
    if args.render_video:
        command.extend(("--render_video", "--fps", str(args.fps)))
    if args.quiet:
        command.append("--quiet")
    command.extend(extra_args)
    return command


def verify_outputs(
    model_dir: Path,
    iteration: int,
    splits: list[str],
    counts: dict[str, int],
    class_count: int,
) -> dict[str, int]:
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    exported = {}
    for split in splits:
        output_dir = split_output_dir(model_dir, split, iteration) / "objects_test"
        paths = sorted(output_dir.glob("*.png"))
        if len(paths) < counts[split]:
            raise FileNotFoundError(
                f"Expected {counts[split]} predicted {split} masks, "
                f"found {len(paths)}"
            )

        max_label = 0
        for path in paths:
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"OpenCV could not decode predicted mask: {path}")
            if mask.dtype != np.uint16:
                raise ValueError(
                    f"Predicted mask is not uint16: {path} ({mask.dtype})"
                )
            max_label = max(max_label, int(mask.max()) if mask.size else 0)
        if max_label >= class_count:
            raise ValueError(
                f"Predicted {split} mask label {max_label} exceeds class count "
                f"{class_count}"
            )
        exported[split] = len(paths)
    return exported


def write_render_manifest(
    model_dir: Path,
    scene: str,
    scene_dir: Path,
    seg_method: str,
    iteration: int,
    resolution: int,
    eval_mode: bool,
    splits: list[str],
    exported: dict[str, int],
    command: list[str],
) -> None:
    manifest = {
        "scene": scene,
        "source_path": str(scene_dir),
        "model_path": str(model_dir),
        "seg_method": seg_method,
        "iteration": iteration,
        "resolution": resolution,
        "eval": eval_mode,
        "splits": splits,
        "exported_predicted_masks": exported,
        "predicted_mask_dtype": "uint16",
        "command": command,
    }
    (model_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    gaga_root = resolve_path(args.gaga_root)
    model_source = resolve_path(args.model_path)
    render_script = gaga_root / "render.py"
    if not render_script.is_file():
        raise FileNotFoundError(f"Gaga render.py does not exist: {render_script}")

    available = discover_scenes(data_root, args.images)
    if not available:
        raise RuntimeError(f"No Mip-NeRF 360 COLMAP scenes found in: {data_root}")
    if args.list_scenes:
        print("\n".join(available))
        return 0
    scenes = select_scenes(args.scenes, available)

    jobs: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        model_dir = resolve_scene_model(
            model_source,
            scene,
            args.seg_method,
            len(scenes),
        )
        iteration = resolve_iteration(model_dir, args.iteration)
        manifest = load_lift_manifest(model_dir)
        resolution = args.resolution
        if resolution is None:
            resolution = int(manifest.get("resolution", 4))
        eval_mode = args.eval
        if eval_mode is None:
            eval_mode = bool(manifest.get("eval", True))
        splits = requested_splits(args.split, eval_mode)
        total_images = image_count(scene_dir, args.images)
        counts = expected_split_counts(total_images, eval_mode)
        point_count, sh_degree = validate_model_and_masks(
            scene_dir,
            model_dir,
            args.seg_method,
            iteration,
        )
        info = json.loads(
            (scene_dir / f"{args.seg_method}_mask" / "info.json").read_text(
                encoding="utf-8"
            )
        )
        class_count = int(info["num_mask"]) + 1
        command = build_command(
            gaga_root,
            scene_dir,
            model_dir,
            args.seg_method,
            sh_degree,
            iteration,
            resolution,
            eval_mode,
            args,
            extra_args,
        )
        jobs.append(
            {
                "scene": scene,
                "scene_dir": scene_dir,
                "model_dir": model_dir,
                "iteration": iteration,
                "resolution": resolution,
                "eval_mode": eval_mode,
                "splits": splits,
                "counts": counts,
                "point_count": point_count,
                "class_count": class_count,
                "command": command,
            }
        )

        print(
            f"[resolve] {scene}: {model_dir} iteration={iteration}, "
            f"Gaussians={point_count}, classes={class_count}, "
            f"resolution={resolution}, splits={','.join(splits)}"
        )
        print(f"[command] {shlex.join(command)}")

    if args.dry_run:
        return 0

    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    failures: list[tuple[str, str]] = []
    for job in jobs:
        scene = job["scene"]
        if (
            outputs_are_complete(
                job["model_dir"],
                job["iteration"],
                job["splits"],
                job["counts"],
            )
            and not args.force
        ):
            print(f"[skip] {scene}: requested render outputs are complete")
            continue

        print(f"[render] {scene}")
        completed = subprocess.run(
            job["command"],
            cwd=gaga_root,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            message = f"Gaga render.py exited with return code {completed.returncode}"
            failures.append((scene, message))
            print(f"[failed] {scene}: {message}", file=sys.stderr)
            if not args.keep_going:
                break
            continue

        try:
            exported = verify_outputs(
                job["model_dir"],
                job["iteration"],
                job["splits"],
                job["counts"],
                job["class_count"],
            )
            write_render_manifest(
                job["model_dir"],
                scene,
                job["scene_dir"],
                args.seg_method,
                job["iteration"],
                job["resolution"],
                job["eval_mode"],
                job["splits"],
                exported,
                job["command"],
            )
            for split, count in exported.items():
                output = (
                    split_output_dir(
                        job["model_dir"],
                        split,
                        job["iteration"],
                    )
                    / "objects_test"
                )
                print(f"[done] {scene}/{split}: {count} masks -> {output}")
        except Exception as exc:  # Batch mode should identify the failing scene.
            failures.append((scene, str(exc)))
            print(f"[failed] {scene}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    if failures:
        print(
            "[summary] failed: "
            + "; ".join(f"{scene}: {message}" for scene, message in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
