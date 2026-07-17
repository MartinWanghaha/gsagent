#!/usr/bin/env python3
"""Train MILo on the gsagent mushroom COLMAP scenes.

Examples:
    python gsagent/scripts/milo/train_milo_mushroom.py --scene classroom --gpu 0
    python gsagent/scripts/milo/train_milo_mushroom.py --iterations 7000 --dry-run
    python gsagent/scripts/milo/train_milo_mushroom.py --scene classroom --debug-interval 500
    python gsagent/scripts/milo/train_milo_mushroom.py --scene classroom --log_interval 200
    python gsagent/scripts/milo/train_milo_mushroom.py --scene kokko --imp-metric outdoor -- --mesh_config highres
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
DEFAULT_MILO_ROOT = GSAGENT_ROOT / "submodules" / "MILo"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "PlanarGS_dataset" / "mushroom"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "milo_mushroom"
HIGH_RES_MESH_CONFIGS = {"highres", "veryhighres"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Prepare and train gsagent/data/PlanarGS_dataset/mushroom scenes with MILo."
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
        "--milo-root",
        type=Path,
        default=DEFAULT_MILO_ROOT,
        help=f"MILo checkout root. Default: {DEFAULT_MILO_ROOT}",
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
        help="Number of MILo training iterations.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help="MILo image resolution/downsample argument.",
    )
    parser.add_argument(
        "--imp-metric",
        type=str,
        default="indoor",
        choices=["indoor", "outdoor"],
        help="Importance metric for MILo densification. Default: indoor.",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=None,
        help=(
            "Save mesh/Gaussian debug panels every N MILo iterations to "
            "<output-root>/<scene>/debug. Default: disabled."
        ),
    )
    parser.add_argument(
        "--log-interval",
        "--log_interval",
        dest="log_interval",
        type=int,
        default=None,
        help="Pass MILo --log_interval to log images every N training iterations.",
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
        help="Base MILo GUI port. Each scene adds its index to this value.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Pass --eval to MILo and hold out every 8th image.",
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
    if args.debug_interval is not None and args.debug_interval <= 0:
        parser.error("--debug-interval must be positive.")
    if args.log_interval is not None and args.log_interval <= 0:
        parser.error("--log-interval/--log_interval must be positive.")
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


# Number of parameters per COLMAP camera model id.
_COLMAP_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 14}
# Models whose first parameter is a single shared focal length (f, cx, cy, ...).
_SINGLE_F_MODELS = {0, 2, 3, 8, 9}


def _force_pinhole_cameras_bin(src: Path, dst: Path) -> bool:
    """Create cameras.bin at dst with every model forced to PINHOLE(1).

    Returns True if any camera was changed, False if all were already PINHOLE/SIMPLE_PINHOLE.
    """
    cameras = []
    with open(src, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id = struct.unpack("<Ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            np_ = _COLMAP_NUM_PARAMS.get(model_id, 0)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cameras.append((cam_id, model_id, width, height, params))

    needs_change = any(m not in (0, 1) for _, m, *_ in cameras)
    if not needs_change:
        symlink_to(src, dst)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, model_id, width, height, params in cameras:
            if model_id in _SINGLE_F_MODELS:
                pinhole = (params[0], params[0], params[1], params[2])
            else:
                pinhole = params[:4]
            f.write(struct.pack("<Ii", cam_id, 1))  # model 1 = PINHOLE
            f.write(struct.pack("<QQ", width, height))
            f.write(struct.pack("<4d", *pinhole))
    return True


def prepare_scene(scene_dir: Path, prepared_root: Path) -> Path:
    source_sparse = find_source_sparse_dir(scene_dir)
    cameras = find_colmap_file(source_sparse, "cameras")
    images = find_colmap_file(source_sparse, "images")
    points = find_colmap_file(source_sparse, "points3D")

    prepared_scene = prepared_root / scene_dir.name
    prepared_sparse_zero = prepared_scene / "sparse" / "0"
    symlink_to(scene_dir / "images", prepared_scene / "images")
    if cameras.suffix == ".bin":
        changed = _force_pinhole_cameras_bin(cameras, prepared_sparse_zero / "cameras.bin")
        if changed:
            print(f"[prepare] {scene_dir.name}: forced cameras to PINHOLE")
    else:
        symlink_to(cameras, prepared_sparse_zero / cameras.name)
    symlink_to(images, prepared_sparse_zero / images.name)
    symlink_to(points, prepared_sparse_zero / points.name)

    source_ply = source_sparse / "points3D.ply"
    prepared_ply = prepared_sparse_zero / "points3D.ply"
    if source_ply.exists() and not prepared_ply.exists():
        symlink_to(source_ply, prepared_ply)

    return prepared_scene


def validate_milo_root(milo_root: Path) -> None:
    train_script = milo_root / "milo" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"MILo train script not found: {train_script}")


def mesh_config_from_extra_args(extra_args: list[str]) -> str | None:
    for index, arg in enumerate(extra_args):
        if arg == "--mesh_config" and index + 1 < len(extra_args):
            return extra_args[index + 1]
        if arg.startswith("--mesh_config="):
            return arg.split("=", 1)[1]
    return None


def add_dense_gaussians_for_high_res_mesh(extra_args: list[str]) -> list[str]:
    mesh_config = mesh_config_from_extra_args(extra_args)
    if (
        mesh_config is not None
        and mesh_config.lower() in HIGH_RES_MESH_CONFIGS
        and "--dense_gaussians" not in extra_args
    ):
        return [*extra_args, "--dense_gaussians"]
    return extra_args


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
        "--imp_metric",
        args.imp_metric,
    ]
    if args.eval:
        cmd.append("--eval")
    if args.debug_interval is not None:
        cmd.extend(["--mesh_debug_interval", str(args.debug_interval)])
    if args.log_interval is not None:
        cmd.extend(["--log_interval", str(args.log_interval)])
    extra_args = add_dense_gaussians_for_high_res_mesh(extra_args)
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    milo_root = resolve_path(args.milo_root)
    output_root = resolve_path(args.output_root)
    prepared_root = resolve_path(args.prepared_root or (output_root / "_prepared_data"))

    available_scenes = discover_scenes(data_root)
    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0

    if not available_scenes:
        raise RuntimeError(f"No mushroom scenes found in {data_root}")

    scenes = expand_requested_scenes(args.scenes, available_scenes)
    validate_milo_root(milo_root)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for scene_index, scene in enumerate(scenes):
        original_scene = data_root / scene
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
            source_path = prepared_root / scene
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
        completed = subprocess.run(cmd, cwd=milo_root / "milo", env=env, check=False)
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
