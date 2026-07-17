#!/usr/bin/env python3
"""Infer Mushroom scenes with DVLT and export COLMAP-compatible datasets.

Examples:
    python scripts/dvlt/infer_dvlt_mushroom.py --scene classroom --gpu 0
    python scripts/dvlt/infer_dvlt_mushroom.py --scene classroom --gpu 0 \
        --max-frames 32 --confidence-percentile 65 --force
    python scripts/dvlt/infer_dvlt_mushroom.py --list-scenes

The output images are DVLT's resized/cropped/padded inputs. This keeps image
pixels, predicted intrinsics, depth maps, and COLMAP observations aligned.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DVLT_ROOT = GSAGENT_ROOT / "submodules" / "dvlt"
DVLT_SOURCE_ROOT = DVLT_ROOT / "src"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "PlanarGS_dataset" / "mushroom"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "data" / "mushroom_dvlt"
DEFAULT_CHECKPOINT = GSAGENT_ROOT / "ckpt" / "model.safetensors"

# Reuse the COLMAP records and serializers already exercised by the VGGT-Omega
# exporter. The inference and point filtering paths below remain DVLT-specific.
if str(GSAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSAGENT_ROOT))
from scripts.vggt_omega.infer_vggt_omega_mushroom import (  # noqa: E402
    PointCloud,
    build_colmap_records,
    discover_scenes,
    select_image_paths,
    select_scenes,
    write_colmap_model,
    write_processed_images,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DVLT on Mushroom images and export COLMAP models."
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
        help=f"Input Mushroom root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Local checkpoint/file or Hugging Face repo ID. Default: {DEFAULT_CHECKPOINT}.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=504,
        help="Longest input side in pixels; must be divisible by 14. Default: 504.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["bf16", "fp16", "no"],
        default="bf16",
        help="Accelerate inference precision. Default: bf16.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth sorted input image. Default: 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Uniformly limit each scene to this many frames; 0 keeps all. Default: 48.",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=50.0,
        help="Keep pixels at or above this global depth-confidence percentile. Default: 50.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=2,
        help="Sample one depth pixel every N pixels in x/y. Default: 2.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Maximum fused COLMAP point count; 0 disables the cap. Default: 500000.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional world-space voxel size before max-points sampling. Default: disabled.",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=99.5,
        help="Keep points within this distance-from-median percentile; 100 disables. Default: 99.5.",
    )
    parser.add_argument(
        "--depth-edge-rtol",
        type=float,
        default=0.03,
        help="Reject local relative depth jumps above this value. Default: 0.03.",
    )
    parser.add_argument(
        "--no-filter-depth-edges",
        action="store_true",
        help="Do not remove points around depth discontinuities.",
    )
    parser.add_argument(
        "--no-text-model",
        action="store_true",
        help="Only write COLMAP binary files, skipping cameras/images/points3D.txt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when limiting the point count. Default: 0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output scene after a new result is complete.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after a scene fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected input/output scenes without loading the model.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.image_size <= 0 or args.image_size % 14 != 0:
        parser.error("--image-size must be positive and divisible by DVLT's patch size 14.")
    if args.frame_step <= 0:
        parser.error("--frame-step must be positive.")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative.")
    if not 0 <= args.confidence_percentile <= 100:
        parser.error("--confidence-percentile must be in [0, 100].")
    if args.point_stride <= 0:
        parser.error("--point-stride must be positive.")
    if args.max_points < 0:
        parser.error("--max-points cannot be negative.")
    if args.voxel_size < 0:
        parser.error("--voxel-size cannot be negative.")
    if not 0 < args.spatial_percentile <= 100:
        parser.error("--spatial-percentile must be in (0, 100].")
    if args.depth_edge_rtol <= 0:
        parser.error("--depth-edge-rtol must be positive.")
    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_checkpoint(value: str) -> str:
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else value


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if not DVLT_SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"DVLT source tree does not exist: {DVLT_SOURCE_ROOT}")
    if str(DVLT_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(DVLT_SOURCE_ROOT))

    # HTTPX rejects the legacy socks:// spelling sometimes inherited by shells.
    for name in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(name, "").startswith("socks://"):
            os.environ.pop(name)

    import torch
    from accelerate import Accelerator
    from dvlt.config.schema import register_configs
    from dvlt.util.model_registry import ModelEntry, compose_experiment

    if not torch.cuda.is_available():
        raise RuntimeError("DVLT inference requires a CUDA GPU.")

    # trainer/default.yaml inherits trainer/base from Hydra's ConfigStore.
    # DVLT's official entrypoints register these structured bases explicitly.
    register_configs()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    config = compose_experiment("dvlt")
    config.trainer.ckpt_dir = resolve_checkpoint(args.checkpoint)
    config.data.image_size = args.image_size
    # The released checkpoint contains the complete patch encoder. Avoid an
    # unnecessary DINO download before those parameters are overwritten.
    config.model.load_patch_embed_weights = False
    entry = ModelEntry(label="DVLT", config_name="dvlt", config=config)

    print(f"[model] loading {config.trainer.ckpt_dir}")
    entry.ensure_loaded(accelerator)
    print(
        f"[model] ready on {torch.cuda.get_device_name(0)}, "
        f"precision={args.mixed_precision}, image_size={entry.img_size}"
    )
    return entry, accelerator, torch


def infer_scene(
    entry: Any,
    accelerator: Any,
    torch: Any,
    image_paths: list[Path],
) -> dict[str, np.ndarray]:
    from dvlt.common.constants import DataField, PredictionField
    from dvlt.common.pose import to4x4
    from dvlt.util.preprocess import preprocess_images

    frames: list[Image.Image] = []
    try:
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        batch = preprocess_images(
            frames,
            img_size=entry.img_size,
            patch_size=entry.patch_size,
            device=accelerator.device,
        )
        print(f"[infer] {len(image_paths)} images -> {tuple(batch[DataField.IMAGES].shape)}")

        with torch.inference_mode(), accelerator.autocast():
            predictions = entry.model.predict(batch, accelerator)

        cameras = predictions[PredictionField.CAMERAS][0]
        c2w = to4x4(cameras.camera_to_worlds).detach().float().cpu().numpy()
        extrinsics = np.linalg.inv(c2w).astype(np.float32)
        confidence = predictions.get(PredictionField.DEPTHS_CONF)
        if confidence is None:
            confidence = predictions.get(PredictionField.WORLD_POINTS_DIRECT_CONF)
        if confidence is None:
            raise KeyError("DVLT returned neither depth nor world-point confidence.")

        result = {
            "images": batch[DataField.IMAGES][0].detach().float().cpu().numpy(),
            "valid_pixels": batch["gradio_valid_pixels"][0].detach().cpu().numpy(),
            "depth": predictions[PredictionField.DEPTHS][0].detach().float().cpu().numpy(),
            "confidence": confidence[0].detach().float().cpu().numpy(),
            "world_points": predictions[PredictionField.WORLD_POINTS][0]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "extrinsics": extrinsics,
            "intrinsics": cameras.get_intrinsics_matrices().detach().float().cpu().numpy(),
        }
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "DVLT ran out of GPU memory. Lower --max-frames (for example 32/24) "
            "or --image-size (448/392/336, all divisible by 14)."
        ) from error
    finally:
        frames.clear()
        if "predictions" in locals():
            del predictions
        if "batch" in locals():
            del batch
        torch.cuda.empty_cache()

    expected_frames = len(image_paths)
    if any(result[key].shape[0] != expected_frames for key in result):
        shapes = {key: value.shape for key, value in result.items()}
        raise ValueError(f"Unexpected DVLT output shapes: {shapes}")
    return result


def depth_edge_mask(depth: np.ndarray, relative_tolerance: float) -> np.ndarray:
    import cv2

    kernel = np.ones((3, 3), dtype=np.uint8)
    depth = depth.astype(np.float32, copy=False)
    depth_max = cv2.dilate(depth, kernel)
    depth_min = cv2.erode(depth, kernel)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return relative_jump > relative_tolerance


def fuse_point_cloud(
    predictions: dict[str, np.ndarray],
    confidence_percentile: float,
    point_stride: int,
    max_points: int,
    voxel_size: float,
    spatial_percentile: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    seed: int,
) -> tuple[PointCloud, float]:
    images = predictions["images"]
    depth = predictions["depth"]
    confidence = predictions["confidence"]
    world_points = predictions["world_points"]
    valid_pixels = predictions["valid_pixels"]

    valid = (
        valid_pixels
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(confidence)
        & np.isfinite(world_points).all(axis=-1)
    )
    valid_confidence = confidence[valid]
    if valid_confidence.size == 0:
        raise ValueError("DVLT produced no finite valid depth/confidence values.")
    threshold = float(np.percentile(valid_confidence, confidence_percentile))

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    image_id_parts: list[np.ndarray] = []
    xy_parts: list[np.ndarray] = []

    for index in range(depth.shape[0]):
        mask = valid[index] & (confidence[index] >= threshold)
        if filter_depth_edges:
            mask &= ~depth_edge_mask(depth[index], depth_edge_rtol)

        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::point_stride, ::point_stride] = True
        ys, xs = np.nonzero(mask & stride_mask)
        if len(xs) == 0:
            continue

        colors = np.transpose(images[index], (1, 2, 0))[ys, xs]
        colors = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
        xyz_parts.append(world_points[index, ys, xs].astype(np.float32))
        rgb_parts.append(colors)
        image_id_parts.append(np.full(len(xs), index + 1, dtype=np.int32))
        xy_parts.append(np.column_stack((xs, ys)).astype(np.float64))

    if not xyz_parts:
        raise ValueError("No points survived confidence and depth-edge filtering.")

    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    image_ids = np.concatenate(image_id_parts)
    xys = np.concatenate(xy_parts)
    selection = np.arange(len(xyz), dtype=np.int64)

    if spatial_percentile < 100 and len(selection) >= 32:
        center = np.median(xyz, axis=0)
        distances = np.linalg.norm(xyz - center, axis=1)
        distance_limit = np.percentile(distances, spatial_percentile)
        selection = selection[distances <= distance_limit]
    if voxel_size > 0 and len(selection):
        voxel_keys = np.floor(xyz[selection].astype(np.float64) / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        selection = selection[np.sort(unique_indices)]
    if max_points and len(selection) > max_points:
        rng = np.random.default_rng(seed)
        selection = np.sort(rng.choice(selection, size=max_points, replace=False))

    xyz = xyz[selection]
    rgb = rgb[selection]
    image_ids = image_ids[selection]
    xys = xys[selection]
    point2d_idxs = np.empty(len(xyz), dtype=np.int32)
    for image_id in np.unique(image_ids):
        point_indices = np.flatnonzero(image_ids == image_id)
        point2d_idxs[point_indices] = np.arange(len(point_indices), dtype=np.int32)

    return (
        PointCloud(
            xyz=xyz,
            rgb=rgb,
            image_ids=image_ids,
            point2d_idxs=point2d_idxs,
            xys=xys,
        ),
        threshold,
    )


def write_metadata(
    path: Path,
    scene: str,
    image_paths: list[Path],
    args: argparse.Namespace,
    predictions: dict[str, np.ndarray],
    point_cloud: PointCloud,
    confidence_threshold: float,
) -> None:
    metadata = {
        "generator": "DVLT",
        "scene": scene,
        "checkpoint": resolve_checkpoint(args.checkpoint),
        "coordinate_system": "OpenCV/COLMAP camera-from-world",
        "image_count": len(image_paths),
        "source_images": [str(path) for path in image_paths],
        "output_image_size": {
            "width": int(predictions["images"].shape[-1]),
            "height": int(predictions["images"].shape[-2]),
        },
        "point_count": int(len(point_cloud.xyz)),
        "confidence_threshold": confidence_threshold,
        "parameters": {
            "image_size": args.image_size,
            "mixed_precision": args.mixed_precision,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
            "confidence_percentile": args.confidence_percentile,
            "point_stride": args.point_stride,
            "max_points": args.max_points,
            "voxel_size": args.voxel_size,
            "spatial_percentile": args.spatial_percentile,
            "filter_depth_edges": not args.no_filter_depth_edges,
            "depth_edge_rtol": args.depth_edge_rtol,
            "seed": args.seed,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def process_scene(
    scene: str,
    entry: Any,
    accelerator: Any,
    torch: Any,
    data_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    source_scene = data_root / scene
    output_scene = output_root / scene
    temporary_scene = output_root / f".{scene}.tmp"
    if output_scene.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_scene}. Pass --force to replace it.")

    image_paths = select_image_paths(source_scene / "images", args.frame_step, args.max_frames)
    predictions = infer_scene(entry, accelerator, torch, image_paths)
    point_cloud, confidence_threshold = fuse_point_cloud(
        predictions,
        args.confidence_percentile,
        args.point_stride,
        args.max_points,
        args.voxel_size,
        args.spatial_percentile,
        not args.no_filter_depth_edges,
        args.depth_edge_rtol,
        args.seed,
    )
    cameras, images = build_colmap_records(image_paths, predictions, point_cloud)
    print(
        f"[points] {len(point_cloud.xyz):,} points, "
        f"confidence threshold {confidence_threshold:.6g}"
    )

    if temporary_scene.exists():
        shutil.rmtree(temporary_scene)
    temporary_scene.mkdir(parents=True)
    try:
        write_processed_images(predictions["images"], image_paths, temporary_scene / "images")
        write_colmap_model(
            temporary_scene / "sparse" / "0",
            cameras,
            images,
            point_cloud,
            write_text=not args.no_text_model,
        )
        write_metadata(
            temporary_scene / "dvlt_metadata.json",
            scene,
            image_paths,
            args,
            predictions,
            point_cloud,
            confidence_threshold,
        )
        if output_scene.exists():
            shutil.rmtree(output_scene)
        temporary_scene.replace(output_scene)
    except Exception:
        shutil.rmtree(temporary_scene, ignore_errors=True)
        raise
    print(f"[done] {scene}: {output_scene}")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    available_scenes = discover_scenes(data_root)

    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0

    scenes = select_scenes(args.scenes, available_scenes)
    for scene in scenes:
        image_count = len(
            select_image_paths(data_root / scene / "images", args.frame_step, args.max_frames)
        )
        print(f"[scene] {scene}: {image_count} images -> {output_root / scene}")
    if args.dry_run:
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_root.mkdir(parents=True, exist_ok=True)
    entry, accelerator, torch = load_model(args)

    failures: list[tuple[str, Exception]] = []
    try:
        for scene in scenes:
            print(f"\n[infer] {scene}")
            try:
                process_scene(
                    scene,
                    entry,
                    accelerator,
                    torch,
                    data_root,
                    output_root,
                    args,
                )
            except Exception as error:
                print(f"[failed] {scene}: {error}", file=sys.stderr)
                failures.append((scene, error))
                if not args.keep_going:
                    raise
    finally:
        entry.unload()

    if failures:
        print("Failed scenes: " + ", ".join(scene for scene, _ in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
