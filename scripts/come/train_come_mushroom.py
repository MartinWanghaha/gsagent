#!/usr/bin/env python3
"""Train CoMe on the gsagent mushroom COLMAP scenes.

Examples:
    python scripts/come/train_come_mushroom.py --scene classroom --gpu 0
    python scripts/come/train_come_mushroom.py --iterations 7000 --dry-run
    python scripts/come/train_come_mushroom.py --scene kokko -- --data_device cpu
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COME_ROOT = GSAGENT_ROOT / "submodules" / "CoMe"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "PlanarGS_dataset" / "mushroom"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "come_mushroom"

COME_DEFAULT_ARGS = [
    "--splatting_config",
    "configs/hierarchical.json",
    "--use_ssimdecoupled_appearance",
    "--color_confidence",
    "--color_confidence_max",
    "0.075",
    "--color_confidence_from_iter",
    "500",
    "--lambda_variance",
    "0.5",
    "--lambda_normal_variance",
    "0.005",
    "--lambda_distortion",
    "100",
    "--far_plane",
    "100.",
    "--detach_alpha",
    "False",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Prepare and train gsagent/data/PlanarGS_dataset/mushroom scenes with CoMe."
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Scene name to train. Repeat or pass comma-separated names. Default: all scenes.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mushroom data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--come-root",
        type=Path,
        default=DEFAULT_COME_ROOT,
        help=f"CoMe checkout root. Default: {DEFAULT_COME_ROOT}",
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
        help="Prepared COLMAP-compatible data root. Default: <output-root>/_prepared_data",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Number of CoMe training iterations.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help="CoMe image resolution/downsample argument.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, for example 0. Omit to keep the current environment.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6009,
        help="Base CoMe GUI port. Each scene adds its index to this value.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Pass --eval to CoMe and hold out every 8th image.",
    )
    parser.add_argument(
        "--no-come-defaults",
        action="store_true",
        help="Do not append the CoMe meshing defaults from the paper scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if point_cloud/iteration_<iterations>/point_cloud.ply already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the prepared COLMAP-compatible scene folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without preparing data or launching training.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes if a training command fails.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_scenes(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    scenes = []
    for scene_dir in sorted(data_root.iterdir()):
        if (
            scene_dir.is_dir()
            and (scene_dir / "images").is_dir()
            and (scene_dir / "sparse").is_dir()
        ):
            scenes.append(scene_dir.name)
    return scenes


def expand_requested_scenes(requested: list[str] | None, available: list[str]) -> list[str]:
    if not requested:
        return available

    scenes: list[str] = []
    for item in requested:
        scenes.extend(part.strip() for part in item.split(",") if part.strip())

    if any(scene.lower() == "all" for scene in scenes):
        return available

    missing = sorted(set(scenes) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )

    return [scene for scene in available if scene in set(scenes)]


def find_colmap_file(sparse_dir: Path, stem: str, required: bool = True) -> Path | None:
    for suffix in (".bin", ".txt", ".ply"):
        candidate = sparse_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Missing {stem}.bin/.txt in {sparse_dir}")
    return None


def find_source_sparse_dir(scene_dir: Path) -> Path:
    sparse_zero = scene_dir / "sparse" / "0"
    if (sparse_zero / "images.bin").exists() or (sparse_zero / "images.txt").exists():
        return sparse_zero
    return scene_dir / "sparse"


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

    target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(target, target_is_directory=src.is_dir())


def prepare_scene(scene_dir: Path, prepared_root: Path) -> Path:
    source_sparse = find_source_sparse_dir(scene_dir)
    cameras = find_colmap_file(source_sparse, "cameras")
    images = find_colmap_file(source_sparse, "images")
    points = find_colmap_file(source_sparse, "points3D")

    prepared_scene = prepared_root / scene_dir.name
    prepared_sparse_zero = prepared_scene / "sparse" / "0"
    symlink_to(scene_dir / "images", prepared_scene / "images")
    symlink_to(cameras, prepared_sparse_zero / cameras.name)
    symlink_to(images, prepared_sparse_zero / images.name)
    symlink_to(points, prepared_sparse_zero / points.name)

    source_ply = source_sparse / "points3D.ply"
    prepared_ply = prepared_sparse_zero / "points3D.ply"
    if source_ply.exists() and not prepared_ply.exists():
        symlink_to(source_ply, prepared_ply)

    return prepared_scene


def validate_come_root(come_root: Path) -> None:
    required = [
        come_root / "train.py",
        come_root / "configs" / "hierarchical.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CoMe root is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )


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
    ]
    if args.eval:
        cmd.append("--eval")
    if not args.no_come_defaults:
        cmd.extend(COME_DEFAULT_ARGS)
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    come_root = resolve_path(args.come_root)
    output_root = resolve_path(args.output_root)
    prepared_root = resolve_path(args.prepared_root or (output_root / "_prepared_data"))

    available_scenes = discover_scenes(data_root)
    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0

    if not available_scenes:
        raise RuntimeError(f"No mushroom scenes found in {data_root}")

    scenes = expand_requested_scenes(args.scenes, available_scenes)
    validate_come_root(come_root)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for scene_index, scene in enumerate(scenes):
        original_scene = data_root / scene
        prepared_scene = prepared_root / scene
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

        if args.dry_run:
            source_path = prepared_scene
        else:
            source_path = prepare_scene(original_scene, prepared_root)

        if args.prepare_only:
            print(f"[prepared] {scene}: {source_path}")
            continue

        cmd = build_command(source_path, model_path, scene_index, args, extra_args)
        print(f"[train] {scene}")
        print(shlex.join(cmd))

        if args.dry_run:
            continue

        model_path.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=come_root, env=env, check=False)
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
