#!/usr/bin/env python3
"""Train GaussianWrappingGaga on Mip-NeRF 360 scenes.

Examples:
    # Stable geometry-first training followed by frozen-geometry semantic lifting.
    python scripts/GaussianWrappingGaga/train_gaussian_wrapping_gaga_mipnerf360.py \
        --scene counter --mode two-stage --mask-method entityseg --gpu 0

    # Synchronous RGB/geometry/semantic training with embedding-only semantic backward.
    python scripts/GaussianWrappingGaga/train_gaussian_wrapping_gaga_mipnerf360.py \
        --scene counter --mode joint --mask-method entityseg --gpu 0

    # Forward additional native train.py options after a standalone separator.
    python scripts/GaussianWrappingGaga/train_gaussian_wrapping_gaga_mipnerf360.py \
        --scene counter --mode joint --dry-run -- --log_interval 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from ._common import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PROJECT_ROOT,
        discover_scenes,
        execute,
        model_directories,
        point_cloud_path,
        resolve_path,
        run_directory,
        select_scenes,
        semantic_checkpoint_path,
        validate_masks,
        validate_project,
        validate_semantic_extension,
        write_manifest,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PROJECT_ROOT,
        discover_scenes,
        execute,
        model_directories,
        point_cloud_path,
        resolve_path,
        run_directory,
        select_scenes,
        semantic_checkpoint_path,
        validate_masks,
        validate_project,
        validate_semantic_extension,
        write_manifest,
    )


OURS_DEFAULTS = (
    "--feature_dc_lr",
    "0.0013",
    "--feature_rest_lr",
    "0.00011",
)
RADEGS_DEFAULTS = (
    "--regularization_from_iter",
    "15000",
    "--multiview_config",
    "fast_late",
    "--multiview_factor",
    "0.05",
    "--use_max_size_threshold",
)
GAGA_RENDER_DIRECTORIES = (
    "renders",
    "gt",
    "objects_feature16",
    "objects_pred",
    "objects_test",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Train GaussianWrappingGaga with either geometry-first two-stage "
            "training or synchronous joint training on Mip-NeRF 360."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name; repeat or use comma-separated values. Default: the "
            "seven Mip-NeRF 360 benchmark scenes. Use 'all' for every scene."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("two-stage", "joint"),
        default="two-stage",
        help="Training strategy. Default: two-stage.",
    )
    parser.add_argument(
        "--mask-method",
        "--seg-method",
        dest="mask_method",
        choices=("sam", "entityseg"),
        default="entityseg",
        help="Associated Gaga mask directory. Default: entityseg.",
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--project-root",
        "--gaussian-wrapping-gaga-root",
        dest="project_root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--geometry-iterations",
        "--iterations",
        dest="geometry_iterations",
        type=int,
        default=30_000,
    )
    parser.add_argument("--semantic-iterations", type=int, default=10_000)
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help="3DGS resolution/downsample argument. Default: automatic.",
    )
    parser.add_argument(
        "--rasterizer",
        choices=("ours", "radegs"),
        default="radegs",
        help="Gaussian Wrapping renderer used by both geometry and semantics.",
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--max-gaussians", type=int, default=6_000_000)
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the standard Mip-NeRF 360 train/test split.",
    )
    parser.add_argument("--semantic-lr", type=float, default=2.5e-3)
    parser.add_argument("--semantic-head-lr", type=float, default=5e-4)
    parser.add_argument("--semantic-num-classes", type=int)
    parser.add_argument("--lambda-semantic", type=float, default=1.0)
    parser.add_argument("--lambda-semantic-3d", type=float, default=0.0)
    parser.add_argument("--semantic-3d-interval", type=int, default=10)
    parser.add_argument("--semantic-3d-samples", type=int, default=10_000)
    parser.add_argument("--semantic-3d-neighbors", type=int, default=5)
    parser.add_argument("--allow-missing-masks", action="store_true")
    parser.add_argument(
        "--render-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render Gaga-compatible train/test images after training. Default: on.",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=2,
        help="Post-training render downsample factor. Default: 2 (Gaga Mip-NeRF 360 convention).",
    )
    parser.add_argument(
        "--render-output-profile",
        choices=("images", "full"),
        default="images",
        help="Render PNGs only, or also numerical GW tensors/logits. Default: images.",
    )
    parser.add_argument(
        "--render-class-chunk-size",
        type=int,
        default=32_768,
        help="Pixels per semantic classifier chunk during rendering.",
    )
    parser.add_argument("--depth-order", action="store_true")
    parser.add_argument("--depth-order-config")
    parser.add_argument("--no-paper-defaults", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, e.g. 0.")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-scenes", action="store_true")

    argv = sys.argv[1:]
    args, native_args = parser.parse_known_args(argv)
    if native_args:
        if "--" not in argv:
            parser.error(
                "Unexpected native arguments: "
                + " ".join(native_args)
                + ". Put train.py arguments after a standalone '--'."
            )
        if native_args[0] == "--":
            native_args = native_args[1:]
    if args.geometry_iterations <= 0 or args.semantic_iterations <= 0:
        parser.error("Iteration counts must be positive.")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive integer.")
    if args.render_resolution == 0 or args.render_resolution < -1:
        parser.error("--render-resolution must be -1 or a positive integer.")
    if args.render_class_chunk_size <= 0:
        parser.error("--render-class-chunk-size must be positive.")
    if args.max_gaussians < 0:
        parser.error("--max-gaussians cannot be negative.")
    if args.semantic_lr <= 0 or args.semantic_head_lr <= 0:
        parser.error("Semantic learning rates must be positive.")
    if args.semantic_num_classes is not None and args.semantic_num_classes < 2:
        parser.error("--semantic-num-classes must be at least 2.")
    if args.lambda_semantic < 0 or args.lambda_semantic_3d < 0:
        parser.error("Semantic loss weights cannot be negative.")
    if args.depth_order_config and not args.depth_order:
        parser.error("--depth-order-config requires --depth-order.")
    return args, native_args


def paper_defaults(rasterizer: str) -> list[str]:
    defaults = OURS_DEFAULTS if rasterizer == "ours" else RADEGS_DEFAULTS
    return [*defaults, "--no-exposure_compensation"]


def geometry_command(
    source: Path,
    model: Path,
    scene_index: int,
    args: argparse.Namespace,
    native_args: list[str],
    *,
    joint: bool,
    mask_dir: Path,
    inferred_num_classes: int,
) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(source),
        "-m",
        str(model),
        "--images",
        args.images,
        "--iterations",
        str(args.geometry_iterations),
        "-r",
        str(args.resolution),
        "--rasterizer",
        args.rasterizer,
        "--data_device",
        args.data_device,
        "--port",
        str(args.port + scene_index),
    ]
    if args.max_gaussians:
        command.extend(["--N_max_gaussians", str(args.max_gaussians)])
    if args.eval:
        command.append("--eval")
    if args.depth_order:
        command.append("--depth_order")
        if args.depth_order_config:
            command.extend(["--depth_order_config", args.depth_order_config])
    if not args.no_paper_defaults:
        command.extend(paper_defaults(args.rasterizer))
    if joint:
        command.extend(
            [
                "--semantic_masks",
                str(mask_dir),
                "--semantic_num_classes",
                str(args.semantic_num_classes or inferred_num_classes),
                "--semantic_lr",
                str(args.semantic_lr),
                "--semantic_head_lr",
                str(args.semantic_head_lr),
                "--lambda_semantic",
                str(args.lambda_semantic),
                "--lambda_semantic_3d",
                str(args.lambda_semantic_3d),
                "--semantic_3d_interval",
                str(args.semantic_3d_interval),
                "--semantic_3d_samples",
                str(args.semantic_3d_samples),
                "--semantic_3d_neighbors",
                str(args.semantic_3d_neighbors),
            ]
        )
        if args.allow_missing_masks:
            command.append("--allow_missing_semantic_masks")
    command.extend(native_args)
    return command


def lift_command(
    source: Path,
    geometry_model: Path,
    semantic_model: Path,
    mask_dir: Path,
    inferred_num_classes: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "semantic_lift.py",
        "-s",
        str(source),
        "-m",
        str(geometry_model),
        "--images",
        args.images,
        "-r",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--semantic_masks",
        str(mask_dir),
        "--semantic_output",
        str(semantic_model),
        "--semantic_iterations",
        str(args.semantic_iterations),
        "--semantic_lr",
        str(args.semantic_lr),
        "--head_lr",
        str(args.semantic_head_lr),
        "--num_classes",
        str(args.semantic_num_classes or inferred_num_classes),
        "--rasterizer",
        args.rasterizer,
        "--lambda_semantic_3d",
        str(args.lambda_semantic_3d),
        "--semantic_3d_interval",
        str(args.semantic_3d_interval),
        "--semantic_3d_samples",
        str(args.semantic_3d_samples),
        "--semantic_3d_neighbors",
        str(args.semantic_3d_neighbors),
    ]
    if args.eval:
        command.append("--eval")
    if args.allow_missing_masks:
        command.append("--allow_missing_masks")
    return command


def render_command(
    source: Path,
    model: Path,
    semantic_checkpoint: Path,
    output: Path,
    mask_dir: Path,
    iteration: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "semantic_render.py",
        "-s",
        str(source),
        "-m",
        str(model),
        "--images",
        args.images,
        "-r",
        str(args.render_resolution),
        "--data_device",
        args.data_device,
        "--semantic_checkpoint",
        str(semantic_checkpoint),
        "--semantic_masks",
        str(mask_dir),
        "--output",
        str(output),
        "--load_iteration",
        str(iteration),
        "--rasterizer",
        args.rasterizer,
        "--split",
        "all",
        "--output_profile",
        args.render_output_profile,
        "--class_chunk_size",
        str(args.render_class_chunk_size),
    ]
    if args.eval:
        command.append("--eval")
    return command


def render_outputs_complete(
    output: Path,
    *,
    iteration: int,
    resolution: int,
    output_profile: str,
    eval_mode: bool,
) -> bool:
    manifest_path = output / "render_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if (
        manifest.get("iteration") != iteration
        or manifest.get("resolution") != resolution
        or manifest.get("output_profile") != output_profile
    ):
        return False
    expected_splits = ("train", "test") if eval_mode else ("train",)
    split_metadata = manifest.get("splits", {})
    for split in expected_splits:
        count = split_metadata.get(split, {}).get("count", 0)
        if not isinstance(count, int) or count <= 0:
            return False
        root = output / split / f"ours_{iteration}"
        required = list(GAGA_RENDER_DIRECTORIES)
        if split == "train":
            required.extend(("gt_objects", "gt_objects_color"))
        for directory in required:
            if len(list((root / directory).glob("*.png"))) < count:
                return False
    return True


def train_scene(
    scene: str,
    scene_index: int,
    args: argparse.Namespace,
    native_args: list[str],
    *,
    data_root: Path,
    output_root: Path,
    workdir: Path,
    env: dict[str, str],
) -> int:
    source = data_root / scene
    mask_dir, inferred_num_classes = validate_masks(
        source,
        args.mask_method,
        args.images,
        allow_missing=args.allow_missing_masks,
    )
    run_dir = run_directory(output_root, scene, args.mode, args.mask_method)
    geometry_model, semantic_model = model_directories(run_dir, args.mode)
    commands: dict[str, list[str]] = {}

    train_cmd = geometry_command(
        source,
        geometry_model,
        scene_index,
        args,
        native_args,
        joint=args.mode == "joint",
        mask_dir=mask_dir,
        inferred_num_classes=inferred_num_classes,
    )
    commands["joint" if args.mode == "joint" else "geometry"] = train_cmd

    geometry_complete = point_cloud_path(
        geometry_model,
        args.geometry_iterations,
    ).is_file()
    if args.mode == "joint":
        geometry_complete = geometry_complete and semantic_checkpoint_path(
            geometry_model,
            args.geometry_iterations,
        ).is_file()

    geometry_ran = False
    semantic_ran = False
    if geometry_complete and not args.force:
        print(f"[skip:{scene}] complete {args.mode} model: {geometry_model}")
    else:
        if not args.dry_run:
            geometry_model.mkdir(parents=True, exist_ok=True)
        returncode = execute(
            train_cmd,
            stage=f"train:{scene}:{args.mode}",
            cwd=workdir,
            env=env,
            dry_run=args.dry_run,
        )
        if returncode:
            return returncode
        geometry_ran = not args.dry_run

    if args.mode == "two-stage":
        lift_cmd = lift_command(
            source,
            geometry_model,
            semantic_model,
            mask_dir,
            inferred_num_classes,
            args,
        )
        commands["semantic_lift"] = lift_cmd
        semantic_complete = (
            point_cloud_path(semantic_model, args.semantic_iterations).is_file()
            and semantic_checkpoint_path(
                semantic_model,
                args.semantic_iterations,
            ).is_file()
        )
        if semantic_complete and not args.force and not geometry_ran:
            print(f"[skip:{scene}] complete semantic model: {semantic_model}")
        else:
            if not args.dry_run:
                semantic_model.mkdir(parents=True, exist_ok=True)
            returncode = execute(
                lift_cmd,
                stage=f"train:{scene}:semantic-lift",
                cwd=workdir,
                env=env,
                dry_run=args.dry_run,
            )
            if returncode:
                return returncode
            semantic_ran = not args.dry_run

    semantic_iteration = (
        args.semantic_iterations
        if args.mode == "two-stage"
        else args.geometry_iterations
    )
    semantic_checkpoint = semantic_checkpoint_path(
        semantic_model,
        semantic_iteration,
    )
    if args.render_after_train:
        render_cmd = render_command(
            source,
            semantic_model,
            semantic_checkpoint,
            run_dir,
            mask_dir,
            semantic_iteration,
            args,
        )
        commands["render"] = render_cmd
        render_complete = render_outputs_complete(
            run_dir,
            iteration=semantic_iteration,
            resolution=args.render_resolution,
            output_profile=args.render_output_profile,
            eval_mode=args.eval,
        )
        if (
            render_complete
            and not args.force
            and not geometry_ran
            and not semantic_ran
        ):
            print(f"[skip:{scene}] complete train/test renders: {run_dir}")
        else:
            returncode = execute(
                render_cmd,
                stage=f"render:{scene}:{args.mode}",
                cwd=workdir,
                env=env,
                dry_run=args.dry_run,
            )
            if returncode:
                return returncode

    if not args.dry_run:
        manifest_path = run_dir / "run_manifest.json"
        manifest = {
            "format_version": 1,
            "dataset": "mipnerf360",
            "scene": scene,
            "mode": args.mode,
            "mask_method": args.mask_method,
            "mask_dir": str(mask_dir),
            "num_classes": args.semantic_num_classes or inferred_num_classes,
            "rasterizer": args.rasterizer,
            "resolution": args.resolution,
            "eval": args.eval,
            "geometry_iterations": args.geometry_iterations,
            "semantic_iterations": semantic_iteration,
            "geometry_model": str(geometry_model),
            "semantic_model": str(semantic_model),
            "render_after_train": args.render_after_train,
            "render_resolution": args.render_resolution,
            "render_output_profile": args.render_output_profile,
            "render_class_chunk_size": args.render_class_chunk_size,
            "commands": commands,
        }
        if not geometry_ran and not semantic_ran and manifest_path.is_file():
            try:
                existing = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("scene") == scene
                and existing.get("mode") == args.mode
                and existing.get("mask_method") == args.mask_method
            ):
                manifest = existing
                manifest.update(
                    {
                        "render_after_train": args.render_after_train,
                        "render_resolution": args.render_resolution,
                        "render_output_profile": args.render_output_profile,
                        "render_class_chunk_size": args.render_class_chunk_size,
                    }
                )
                if args.render_after_train:
                    manifest.setdefault("commands", {})["render"] = render_cmd
        write_manifest(manifest_path, manifest)
    status = "dry-run" if args.dry_run else "complete"
    print(f"[{status}:{scene}] {run_dir}")
    return 0


def main() -> int:
    args, native_args = parse_args()
    data_root = resolve_path(args.data_root)
    project_root = resolve_path(args.project_root)
    output_root = resolve_path(args.output_root)
    workdir = validate_project(project_root)
    validate_semantic_extension(project_root, args.rasterizer)
    available = discover_scenes(data_root, args.images)
    if args.list_scenes:
        print("\n".join(available))
        return 0
    scenes = select_scenes(args.scenes, available)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for index, scene in enumerate(scenes):
        try:
            returncode = train_scene(
                scene,
                index,
                args,
                native_args,
                data_root=data_root,
                output_root=output_root,
                workdir=workdir,
                env=env,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"[failed:{scene}] {error}", file=sys.stderr)
            returncode = 2
        if returncode:
            failures.append((scene, returncode))
            if not args.keep_going:
                break

    if failures:
        for scene, returncode in failures:
            print(
                f"[summary:failed] {scene}: return code {returncode}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
