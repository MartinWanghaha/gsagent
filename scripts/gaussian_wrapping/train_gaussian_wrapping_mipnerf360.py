#!/usr/bin/env python3
"""Train Gaussian Wrapping on the Mip-NeRF 360 COLMAP scenes.

Examples:
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mipnerf360.py --scene bicycle --gpu 0
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mipnerf360.py --scene bonsai --rasterizer radegs
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mipnerf360.py --scene bicycle,garden --dry-run
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mipnerf360.py --scene treehill -- --log_interval 200
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GW_ROOT = GSAGENT_ROOT / "submodules" / "GaussianWrapping"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "gaussian_wrapping_mipnerf360"

BENCHMARK_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
)

OURS_DEFAULT_ARGS = [
    "--feature_dc_lr",
    "0.0013",
    "--feature_rest_lr",
    "0.00011",
]

RADEGS_DEFAULT_ARGS = [
    "--regularization_from_iter",
    "15000",
    "--multiview_config",
    "fast_late",
    "--multiview_factor",
    "0.05",
    "--use_max_size_threshold",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train Gaussian Wrapping on Mip-NeRF 360 scenes."
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name. Repeat or pass comma-separated names. "
            "Default: the 7 README benchmark scenes; use 'all' for all discovered scenes."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mip-NeRF 360 data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--gaussian-wrapping-root",
        "--gw-root",
        dest="gw_root",
        type=Path,
        default=DEFAULT_GW_ROOT,
        help=f"GaussianWrapping checkout root. Default: {DEFAULT_GW_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Training output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Number of training iterations. Default: 30000.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help="Image resolution/downsample argument. Default: -1 (automatic).",
    )
    parser.add_argument(
        "--rasterizer",
        choices=("ours", "radegs"),
        default="ours",
        help="Gaussian Wrapping training rasterizer. Default: ours.",
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used to store source images. Default: cpu.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=6_000_000,
        help="Maximum Gaussian count; use 0 to disable the cap. Default: 6000000.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6009,
        help="Base GUI port. Each scene adds its index. Default: 6009.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Hold out every eighth image for evaluation.",
    )
    parser.add_argument(
        "--depth-order",
        action="store_true",
        help="Enable optional Depth-Anything-V2 depth-order regularization.",
    )
    parser.add_argument(
        "--depth-order-config",
        default=None,
        help="Depth-order config name, used only with --depth-order.",
    )
    parser.add_argument(
        "--no-paper-defaults",
        action="store_true",
        help="Do not add the official Mip-NeRF360 benchmark training defaults.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even if the final point cloud already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without launching training.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after a training failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.max_gaussians < 0:
        parser.error("--max-gaussians cannot be negative.")
    if args.depth_order_config is not None and not args.depth_order:
        parser.error("--depth-order-config requires --depth-order.")
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_colmap_scene(scene_dir: Path) -> bool:
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = scene_dir / "sparse"
    has_images = (scene_dir / "images").is_dir()
    has_cameras = (sparse_dir / "cameras.bin").is_file() or (
        sparse_dir / "cameras.txt"
    ).is_file()
    has_images_metadata = (sparse_dir / "images.bin").is_file() or (
        sparse_dir / "images.txt"
    ).is_file()
    has_points = (sparse_dir / "points3D.bin").is_file() or (
        sparse_dir / "points3D.txt"
    ).is_file()
    return has_images and has_cameras and has_images_metadata and has_points


def discover_scenes(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    return [
        scene_dir.name
        for scene_dir in sorted(data_root.iterdir())
        if scene_dir.is_dir() and is_colmap_scene(scene_dir)
    ]


def expand_requested_scenes(
    requested: list[str] | None, available: list[str]
) -> list[str]:
    if not requested:
        benchmark = [scene for scene in BENCHMARK_SCENES if scene in available]
        missing = [scene for scene in BENCHMARK_SCENES if scene not in available]
        if missing:
            raise FileNotFoundError(
                "Missing default benchmark scene(s): " + ", ".join(missing)
            )
        return benchmark

    scenes = [
        part.strip()
        for item in requested
        for part in item.split(",")
        if part.strip()
    ]
    if any(scene.lower() == "all" for scene in scenes):
        return available

    missing = sorted(set(scenes) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )
    requested_set = set(scenes)
    return [scene for scene in available if scene in requested_set]


def validate_gw_root(gw_root: Path) -> Path:
    workdir = gw_root / "gaussian_wrapping"
    required = [
        workdir / "train.py",
        workdir / "configs" / "multiview" / "fast.yaml",
        workdir / "configs" / "multiview" / "fast_late.yaml",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "GaussianWrapping root is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )
    return workdir


def build_command(
    source_path: Path,
    model_path: Path,
    scene_index: int,
    args: argparse.Namespace,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iterations",
        str(args.iterations),
        "-r",
        str(args.resolution),
        "--port",
        str(args.port + scene_index),
        "--rasterizer",
        args.rasterizer,
        "--data_device",
        args.data_device,
    ]
    if args.max_gaussians:
        cmd.extend(["--N_max_gaussians", str(args.max_gaussians)])
    if args.eval:
        cmd.append("--eval")
    if args.depth_order:
        cmd.append("--depth_order")
        if args.depth_order_config is not None:
            cmd.extend(["--depth_order_config", args.depth_order_config])
    if not args.no_paper_defaults:
        defaults = OURS_DEFAULT_ARGS if args.rasterizer == "ours" else RADEGS_DEFAULT_ARGS
        cmd.extend(defaults)
        cmd.append("--no-exposure_compensation")
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    gw_root = resolve_path(args.gw_root)
    output_root = resolve_path(args.output_root)

    available_scenes = discover_scenes(data_root)
    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0
    if not available_scenes:
        raise RuntimeError(f"No Mip-NeRF 360 COLMAP scenes found in {data_root}")

    scenes = expand_requested_scenes(args.scenes, available_scenes)
    workdir = validate_gw_root(gw_root)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for scene_index, scene in enumerate(scenes):
        source_path = data_root / scene
        model_path = output_root / scene
        final_point_cloud = (
            model_path
            / "point_cloud"
            / f"iteration_{args.iterations}"
            / "point_cloud.ply"
        )
        if final_point_cloud.exists() and not args.force:
            print(f"[skip] {scene}: found {final_point_cloud}")
            continue

        cmd = build_command(source_path, model_path, scene_index, args, extra_args)
        print(f"[train] {scene}")
        print(shlex.join(cmd))
        if args.dry_run:
            continue

        model_path.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=workdir, env=env, check=False)
        if completed.returncode != 0:
            failures.append((scene, completed.returncode))
            if not args.keep_going:
                break

    if failures:
        for scene, returncode in failures:
            print(f"[failed] {scene}: return code {returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
