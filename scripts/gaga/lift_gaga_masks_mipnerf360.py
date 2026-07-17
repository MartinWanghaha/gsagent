#!/usr/bin/env python3
"""Lift associated Gaga 2D instance masks into pretrained 3D Gaussians.

Examples:
    python scripts/gaga/lift_gaga_masks_mipnerf360.py \
        --scene counter \
        --point-cloud outputs/gaussian_wrapping_mipnerf360/counter/point_cloud/iteration_30000/point_cloud.ply \
        --seg-method entityseg --gpu 0

    python scripts/gaga/lift_gaga_masks_mipnerf360.py \
        --scene bicycle,counter \
        --point-cloud outputs/gaussian_wrapping_mipnerf360 \
        --seg-method sam --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .associate_gaga_masks_mipnerf360 import (
        discover_scenes,
        inspect_point_cloud,
        resolve_scene_point_cloud,
        select_scenes,
    )
except ImportError:
    from associate_gaga_masks_mipnerf360 import (  # type: ignore[no-redef]
        discover_scenes,
        inspect_point_cloud,
        resolve_scene_point_cloud,
        select_scenes,
    )


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_GAGA_ROOT = GSAGENT_ROOT / "submodules" / "Gaga"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "gaga_mipnerf360"
DEFAULT_CONFIG = DEFAULT_GAGA_ROOT / "config" / "train.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize per-Gaussian instance features from cross-view-associated "
            "Gaga masks on Mip-NeRF 360 scenes."
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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Lift output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--point-cloud",
        "--point-cloud-root",
        dest="point_cloud",
        type=Path,
        help=(
            "A point_cloud.ply for one selected scene, a standard 3DGS model "
            "directory, or a root containing one model directory per scene."
        ),
    )
    parser.add_argument(
        "--seg-method",
        "--mask-method",
        dest="seg_method",
        choices=("sam", "entityseg"),
        default="sam",
        help=(
            "Associated mask directory to use: sam_mask or entityseg_mask. "
            "Default: sam."
        ),
    )
    parser.add_argument(
        "--images",
        default="images",
        help="Image subdirectory matching the COLMAP cameras. Default: images.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Object-feature optimization iterations. Default: 10000.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=8,
        help=(
            "Gaga/3DGS camera resolution argument. Use 1, 2, 4, or 8 as a "
            "downsample factor, or a positive target width. Default: 4."
        ),
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used to store source images. Default: cpu.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Gaga Lift regularization config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Mip-NeRF 360 evaluation split. Default: enabled.",
    )
    parser.add_argument(
        "--gpu",
        help="Value assigned to CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable the optional Gaga Weights & Biases logger.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run again even when the requested final Lift output is complete.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after a Lift failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without launching Gaga.",
    )

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive integer.")
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_gaga(gaga_root: Path, config_file: Path) -> None:
    required = [
        gaga_root / "lift.py",
        gaga_root / "arguments" / "__init__.py",
        gaga_root / "gaussian_renderer" / "__init__.py",
        config_file,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gaga Lift is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )


def image_stems(image_dir: Path) -> set[str]:
    return {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_associated_masks(
    scene_dir: Path,
    images: str,
    seg_method: str,
) -> tuple[Path, int, int, dict[str, Any]]:
    image_dir = scene_dir / images
    mask_dir = scene_dir / f"{seg_method}_mask"
    info_path = mask_dir / "info.json"
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Associated mask directory does not exist: {mask_dir}. "
            "Run associate_gaga_masks_mipnerf360.py first."
        )
    if not info_path.is_file():
        raise FileNotFoundError(f"Associated-mask metadata does not exist: {info_path}")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata: {info_path}") from exc
    num_masks = info.get("num_mask")
    if not isinstance(num_masks, int) or num_masks <= 0:
        raise ValueError(f"info.json has an invalid num_mask value: {num_masks!r}")

    expected_stems = image_stems(image_dir)
    mask_paths = [path for path in mask_dir.glob("*.png") if path.is_file()]
    mask_stems = {path.stem for path in mask_paths}
    missing = sorted(expected_stems - mask_stems)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{mask_dir} is missing masks for {len(missing)} image(s): "
            f"{preview}{suffix}"
        )
    return mask_dir, len(mask_paths), num_masks, info


def output_paths(output_dir: Path, iteration: int) -> tuple[Path, Path]:
    iteration_dir = output_dir / "point_cloud" / f"iteration_{iteration}"
    return iteration_dir / "point_cloud.ply", iteration_dir / "classifier.pth"


def output_is_complete(output_dir: Path, iteration: int) -> bool:
    point_cloud, classifier = output_paths(output_dir, iteration)
    return point_cloud.is_file() and classifier.is_file()


def camera_resolution(width: int, height: int, resolution: int) -> tuple[int, int]:
    if resolution in (1, 2, 4, 8):
        return round(width / resolution), round(height / resolution)
    if resolution == -1:
        scale = width / 1600 if width > 1600 else 1.0
    else:
        scale = width / resolution
    return int(width / scale), int(height / scale)


def stage_scene_with_resized_masks(
    scene_dir: Path,
    mask_dir: Path,
    images: str,
    seg_method: str,
    resolution: int,
    staged_scene: Path,
) -> Path:
    import cv2  # pylint: disable=import-outside-toplevel

    staged_scene.mkdir(parents=True)
    (staged_scene / images).symlink_to(scene_dir / images, target_is_directory=True)
    (staged_scene / "sparse").symlink_to(
        scene_dir / "sparse",
        target_is_directory=True,
    )
    staged_mask_dir = staged_scene / f"{seg_method}_mask"
    staged_mask_dir.mkdir()
    (staged_mask_dir / "info.json").symlink_to(mask_dir / "info.json")

    resized_count = 0
    for image_path in sorted((scene_dir / images).iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image: {image_path}")
        if mask is None:
            raise RuntimeError(f"OpenCV could not decode associated mask: {mask_path}")
        source_height, source_width = image.shape[:2]
        if mask.shape[:2] != (source_height, source_width):
            raise ValueError(
                f"Mask/image size mismatch for {image_path.stem}: "
                f"mask={mask.shape[:2]}, image={(source_height, source_width)}"
            )

        target_width, target_height = camera_resolution(
            source_width,
            source_height,
            resolution,
        )
        staged_mask_path = staged_mask_dir / mask_path.name
        if (target_width, target_height) == (source_width, source_height):
            staged_mask_path.symlink_to(mask_path)
            continue
        resized = cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if not cv2.imwrite(str(staged_mask_path), resized):
            raise RuntimeError(f"OpenCV could not write staged mask: {staged_mask_path}")
        resized_count += 1

    print(
        f"[stage] {scene_dir.name}: resized {resized_count} associated masks "
        f"for Gaga resolution {resolution}"
    )
    return staged_scene


def build_command(
    gaga_root: Path,
    scene_dir: Path,
    output_dir: Path,
    staged_model: Path,
    config_file: Path,
    sh_degree: int,
    args: argparse.Namespace,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(gaga_root / "lift.py"),
        "--source_path",
        str(scene_dir),
        "--model_path",
        str(output_dir),
        "--trained_model_path",
        str(staged_model),
        "--object_path",
        f"{args.seg_method}_mask",
        "--images",
        args.images,
        "--sh_degree",
        str(sh_degree),
        "--iterations",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--config_file",
        str(config_file),
    ]
    if args.eval:
        command.append("--eval")
    if args.use_wandb:
        command.append("--use_wandb")
    command.extend(extra_args)
    return command


def write_manifest(
    output_dir: Path,
    scene: str,
    scene_dir: Path,
    point_cloud: Path,
    point_count: int,
    sh_degree: int,
    mask_dir: Path,
    mask_count: int,
    num_masks: int,
    args: argparse.Namespace,
    command: list[str],
) -> None:
    manifest = {
        "scene": scene,
        "source_path": str(scene_dir),
        "input_point_cloud": str(point_cloud),
        "input_gaussian_count": point_count,
        "sh_degree": sh_degree,
        "seg_method": args.seg_method,
        "associated_mask_path": str(mask_dir),
        "associated_mask_count": mask_count,
        "instance_group_count": num_masks,
        "class_count_with_background": num_masks + 1,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "eval": args.eval,
        "command": command,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lift_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def verify_output(
    output_dir: Path,
    iteration: int,
    expected_points: int,
) -> tuple[Path, Path]:
    point_cloud, classifier = output_paths(output_dir, iteration)
    missing = [path for path in (point_cloud, classifier) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gaga Lift did not create expected output(s): "
            + ", ".join(str(path) for path in missing)
        )

    from plyfile import PlyData  # pylint: disable=import-outside-toplevel

    ply = PlyData.read(str(point_cloud), mmap="c")
    vertex = ply["vertex"]
    properties = {prop.name for prop in vertex.properties}
    missing_features = [f"obj_dc_{index}" for index in range(16) if f"obj_dc_{index}" not in properties]
    if missing_features:
        raise ValueError(
            f"Lifted PLY is missing object features: {', '.join(missing_features)}"
        )
    if len(vertex.data) != expected_points:
        raise ValueError(
            f"Lift changed Gaussian count from {expected_points} to {len(vertex.data)}"
        )
    return point_cloud, classifier


def run_lift(
    scene: str,
    scene_dir: Path,
    point_cloud: Path,
    output_dir: Path,
    gaga_root: Path,
    config_file: Path,
    point_count: int,
    sh_degree: int,
    mask_dir: Path,
    mask_count: int,
    num_masks: int,
    args: argparse.Namespace,
    extra_args: list[str],
    environment: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"gaga-lift-{scene}-") as temp_dir:
        temporary_root = Path(temp_dir)
        staged_model = temporary_root / "model"
        staged_iteration = staged_model / "point_cloud" / "iteration_1"
        staged_iteration.mkdir(parents=True)
        (staged_iteration / "point_cloud.ply").symlink_to(point_cloud)
        staged_scene = stage_scene_with_resized_masks(
            scene_dir,
            mask_dir,
            args.images,
            args.seg_method,
            args.resolution,
            temporary_root / "scene",
        )

        command = build_command(
            gaga_root,
            staged_scene,
            output_dir,
            staged_model,
            config_file,
            sh_degree,
            args,
            extra_args,
        )
        print(f"[lift] {scene}")
        print(shlex.join(command))
        completed = subprocess.run(
            command,
            cwd=gaga_root,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Gaga lift.py exited with return code {completed.returncode}")

        lifted_ply, classifier = verify_output(
            output_dir,
            args.iterations,
            point_count,
        )
        write_manifest(
            output_dir,
            scene,
            scene_dir,
            point_cloud,
            point_count,
            sh_degree,
            mask_dir,
            mask_count,
            num_masks,
            args,
            command,
        )
        print(f"[done] {scene}: {lifted_ply}")
        print(f"[done] {scene}: {classifier}")


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    gaga_root = resolve_path(args.gaga_root)
    output_root = resolve_path(args.output_root)
    config_file = resolve_path(args.config_file)

    available = discover_scenes(data_root, args.images)
    if not available:
        raise RuntimeError(f"No Mip-NeRF 360 COLMAP scenes found in: {data_root}")
    if args.list_scenes:
        print("\n".join(available))
        return 0
    if args.point_cloud is None:
        raise ValueError("--point-cloud is required unless --list-scenes is used")

    validate_gaga(gaga_root, config_file)
    scenes = select_scenes(args.scenes, available)
    point_cloud_source = resolve_path(args.point_cloud)
    jobs: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        point_cloud = resolve_scene_point_cloud(
            point_cloud_source,
            scene,
            len(scenes),
        )
        point_count, sh_degree = inspect_point_cloud(point_cloud)
        mask_dir, mask_count, num_masks, _ = validate_associated_masks(
            scene_dir,
            args.images,
            args.seg_method,
        )
        output_dir = output_root / scene / args.seg_method
        jobs.append(
            {
                "scene": scene,
                "scene_dir": scene_dir,
                "point_cloud": point_cloud,
                "point_count": point_count,
                "sh_degree": sh_degree,
                "mask_dir": mask_dir,
                "mask_count": mask_count,
                "num_masks": num_masks,
                "output_dir": output_dir,
            }
        )
        print(
            f"[resolve] {scene}: {point_cloud} ({point_count} Gaussians, "
            f"SH degree {sh_degree}) + {mask_dir} ({mask_count} views, "
            f"{num_masks} groups) -> {output_dir}"
        )

    if args.dry_run:
        for job in jobs:
            staged_placeholder = Path("<temporary-3dgs-model>")
            scene_placeholder = Path("<temporary-scene-with-resized-masks>")
            command = build_command(
                gaga_root,
                scene_placeholder,
                job["output_dir"],
                staged_placeholder,
                config_file,
                job["sh_degree"],
                args,
                extra_args,
            )
            print(f"[dry-run] {shlex.join(command)}")
        return 0

    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    failures: list[tuple[str, str]] = []
    for job in jobs:
        scene = job["scene"]
        output_dir = job["output_dir"]
        if output_is_complete(output_dir, args.iterations) and not args.force:
            print(f"[skip] {scene}: Lift output is already complete: {output_dir}")
            continue
        try:
            run_lift(
                scene=scene,
                scene_dir=job["scene_dir"],
                point_cloud=job["point_cloud"],
                output_dir=output_dir,
                gaga_root=gaga_root,
                config_file=config_file,
                point_count=job["point_count"],
                sh_degree=job["sh_degree"],
                mask_dir=job["mask_dir"],
                mask_count=job["mask_count"],
                num_masks=job["num_masks"],
                args=args,
                extra_args=extra_args,
                environment=environment,
            )
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
