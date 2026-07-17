#!/usr/bin/env python3
"""Train Gaussian Wrapping on the gsagent Mushroom COLMAP scenes.

Examples:
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mushroom.py --scene classroom --gpu 0
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mushroom.py --scene kokko --rasterizer radegs
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mushroom.py --scene classroom --dry-run
    python scripts/gaussian_wrapping/train_gaussian_wrapping_mushroom.py --scene classroom -- --log_interval 200
"""

from __future__ import annotations

import argparse
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GW_ROOT = GSAGENT_ROOT / "submodules" / "GaussianWrapping"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "PlanarGS_dataset" / "mushroom"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "gaussian_wrapping_mushroom"

OURS_DEFAULT_ARGS = [
    "--feature_dc_lr",
    "0.0013",
    "--feature_rest_lr",
    "0.00011",
    "--exposure_compensation",
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
        description="Prepare and train gsagent Mushroom scenes with Gaussian Wrapping."
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Scene name. Repeat or pass comma-separated names. Default: all scenes.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mushroom data root. Default: {DEFAULT_DATA_ROOT}",
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
        "--prepared-root",
        type=Path,
        default=None,
        help="Prepared COLMAP data root. Default: <output-root>/_prepared_data",
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
        help="Image resolution/downsample argument. Default: -1.",
    )
    parser.add_argument(
        "--rasterizer",
        choices=["ours", "radegs"],
        default="ours",
        help="Gaussian Wrapping training rasterizer. Default: ours.",
    )
    parser.add_argument(
        "--data-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used to store input images. Default: cpu.",
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
        "--no-paper-defaults",
        action="store_true",
        help="Do not add the selected rasterizer's project-script defaults.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even if the final point cloud already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the prepared COLMAP scene folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without preparing data or launching training.",
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
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_scenes(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    return [
        scene_dir.name
        for scene_dir in sorted(data_root.iterdir())
        if scene_dir.is_dir()
        and (scene_dir / "images").is_dir()
        and (scene_dir / "sparse").is_dir()
    ]


def expand_requested_scenes(
    requested: list[str] | None, available: list[str]
) -> list[str]:
    if not requested:
        return available

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


def find_source_sparse_dir(scene_dir: Path) -> Path:
    sparse_zero = scene_dir / "sparse" / "0"
    if (sparse_zero / "images.bin").exists() or (sparse_zero / "images.txt").exists():
        return sparse_zero
    return scene_dir / "sparse"


def find_colmap_file(sparse_dir: Path, stem: str) -> Path:
    for suffix in (".bin", ".txt", ".ply"):
        candidate = sparse_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.bin/.txt in {sparse_dir}")


def symlink_to(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        if dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")

    dst.symlink_to(os.path.relpath(src, start=dst.parent), target_is_directory=src.is_dir())


_COLMAP_NUM_PARAMS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 14,
}
_SINGLE_F_MODELS = {0, 2, 3, 8, 9}


def force_pinhole_cameras_bin(src: Path, dst: Path) -> bool:
    cameras = []
    with src.open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        for _ in range(count):
            camera_id, model_id = struct.unpack("<Ii", source.read(8))
            width, height = struct.unpack("<QQ", source.read(16))
            parameter_count = _COLMAP_NUM_PARAMS[model_id]
            params = struct.unpack(
                f"<{parameter_count}d", source.read(8 * parameter_count)
            )
            cameras.append((camera_id, model_id, width, height, params))

    if all(model_id in (0, 1) for _, model_id, *_ in cameras):
        symlink_to(src, dst)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as output:
        output.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            if model_id in _SINGLE_F_MODELS:
                pinhole_params = (params[0], params[0], params[1], params[2])
            else:
                pinhole_params = params[:4]
            output.write(struct.pack("<Ii", camera_id, 1))
            output.write(struct.pack("<QQ", width, height))
            output.write(struct.pack("<4d", *pinhole_params))
    return True


def prepare_scene(scene_dir: Path, prepared_root: Path) -> Path:
    source_sparse = find_source_sparse_dir(scene_dir)
    cameras = find_colmap_file(source_sparse, "cameras")
    images = find_colmap_file(source_sparse, "images")
    points = find_colmap_file(source_sparse, "points3D")

    prepared_scene = prepared_root / scene_dir.name
    prepared_sparse = prepared_scene / "sparse" / "0"
    symlink_to(scene_dir / "images", prepared_scene / "images")

    if cameras.suffix == ".bin":
        changed = force_pinhole_cameras_bin(cameras, prepared_sparse / "cameras.bin")
        if changed:
            print(f"[prepare] {scene_dir.name}: forced cameras to PINHOLE")
    else:
        symlink_to(cameras, prepared_sparse / cameras.name)
    symlink_to(images, prepared_sparse / images.name)
    symlink_to(points, prepared_sparse / points.name)

    source_ply = source_sparse / "points3D.ply"
    prepared_ply = prepared_sparse / "points3D.ply"
    if source_ply.exists() and not prepared_ply.exists():
        symlink_to(source_ply, prepared_ply)
    return prepared_scene


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
    if not args.no_paper_defaults:
        defaults = OURS_DEFAULT_ARGS if args.rasterizer == "ours" else RADEGS_DEFAULT_ARGS
        cmd.extend(defaults)
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    gw_root = resolve_path(args.gw_root)
    output_root = resolve_path(args.output_root)
    prepared_root = resolve_path(args.prepared_root or output_root / "_prepared_data")

    available_scenes = discover_scenes(data_root)
    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0
    if not available_scenes:
        raise RuntimeError(f"No Mushroom scenes found in {data_root}")

    scenes = expand_requested_scenes(args.scenes, available_scenes)
    workdir = validate_gw_root(gw_root)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for scene_index, scene in enumerate(scenes):
        scene_dir = data_root / scene
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

        source_path = (
            prepared_root / scene
            if args.dry_run
            else prepare_scene(scene_dir, prepared_root)
        )
        if args.prepare_only:
            print(f"[prepared] {scene}: {source_path}")
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
