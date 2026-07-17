#!/usr/bin/env python3
"""Generate Gaga 2D instance masks for Mip-NeRF 360 scenes.

Examples:
    python scripts/gaga/preprocess_gaga_masks_mipnerf360.py \
        --sam-checkpoint ckpt/sam_vit_h_4b8939.pth --gpu 0

    python scripts/gaga/preprocess_gaga_masks_mipnerf360.py \
        --scene bicycle,garden --sam-checkpoint ckpt/sam_vit_h_4b8939.pth

    python scripts/gaga/preprocess_gaga_masks_mipnerf360.py \
        --seg-method entityseg --entity-checkpoint ckpt/CropFormer_hornet_3x_03823a.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_GAGA_ROOT = GSAGENT_ROOT / "submodules" / "Gaga"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SceneResult:
    scene: str
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-image, class-agnostic instance masks for Gaga. "
            "Masks are written under each Mip-NeRF 360 scene."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name. Repeat the option or use comma-separated names. "
            "Default: all discovered COLMAP scenes."
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
        "--images",
        default="images",
        help="Image subdirectory in every scene. Default: images.",
    )
    parser.add_argument(
        "--image-resolution",
        "--max-image-size",
        dest="image_resolution",
        type=int,
        default=1024,
        help=(
            "Maximum image long edge used during segmentation. Masks are resized "
            "back to the source resolution with nearest-neighbor interpolation. "
            "Use 0 to disable resizing. Default: 1024."
        ),
    )
    parser.add_argument(
        "--seg-method",
        choices=("sam", "entityseg"),
        default="sam",
        help="2D instance segmenter. Default: sam.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="SAM checkpoint, required with --seg-method sam.",
    )
    parser.add_argument(
        "--sam-model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default="vit_h",
        help="SAM backbone matching --sam-checkpoint. Default: vit_h.",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=64,
        help="SAM prompt-grid density per image edge. Default: 64.",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help="SAM prompts evaluated per batch. Lower it on CUDA OOM. Default: 64.",
    )
    parser.add_argument(
        "--pred-iou-threshold",
        type=float,
        default=0.7,
        help="SAM predicted-IoU filter threshold. Default: 0.7.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Instance-score threshold after segmentation. Default: 0.5.",
    )
    parser.add_argument(
        "--entity-checkpoint",
        type=Path,
        help="EntitySeg checkpoint, required with --seg-method entityseg.",
    )
    parser.add_argument(
        "--entity-config",
        type=Path,
        help="EntitySeg CropFormer config. Required if Gaga's bundled path is unavailable.",
    )
    parser.add_argument(
        "--gpu",
        help="Value assigned to CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also write false-color masks to raw_<method>_mask_vis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate masks that already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first scene or image failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print work without loading a segmentation model.",
    )

    args = parser.parse_args()
    if args.points_per_side <= 0 or args.points_per_batch <= 0:
        parser.error("--points-per-side and --points-per-batch must be positive.")
    if args.image_resolution < 0:
        parser.error("--image-resolution cannot be negative.")
    for name in ("pred_iou_threshold", "confidence_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_colmap_scene(scene_dir: Path, image_dir: str) -> bool:
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = scene_dir / "sparse"
    has_cameras = (sparse_dir / "cameras.bin").is_file() or (sparse_dir / "cameras.txt").is_file()
    has_images = (sparse_dir / "images.bin").is_file() or (sparse_dir / "images.txt").is_file()
    return (scene_dir / image_dir).is_dir() and has_cameras and has_images


def discover_scenes(data_root: Path, image_dir: str) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    return [
        item.name
        for item in sorted(data_root.iterdir())
        if item.is_dir() and is_colmap_scene(item, image_dir)
    ]


def select_scenes(requested: list[str] | None, available: list[str]) -> list[str]:
    if not requested:
        return available

    selected = [
        name.strip()
        for item in requested
        for name in item.split(",")
        if name.strip()
    ]
    if any(name.lower() == "all" for name in selected):
        return available

    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )
    selected_set = set(selected)
    return [scene for scene in available if scene in selected_set]


def image_paths(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def load_gaga_mask_api(gaga_root: Path) -> dict[str, Any]:
    mask_dir = gaga_root / "mask"
    required = [
        mask_dir / "get_raw_mask.py",
        mask_dir / "automatic_mask_generator.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gaga mask code is missing required file(s): " + ", ".join(map(str, missing))
        )

    sys.path.insert(0, str(mask_dir))
    from get_raw_mask import (  # pylint: disable=import-outside-toplevel
        get_entityseg_mask,
        get_sam_mask,
        get_seg_model,
        visualize_mask,
    )

    return {
        "get_entityseg_mask": get_entityseg_mask,
        "get_sam_mask": get_sam_mask,
        "get_seg_model": get_seg_model,
        "visualize_mask": visualize_mask,
    }


def build_segmenter_config(args: argparse.Namespace, gaga_root: Path) -> dict[str, Any]:
    if args.seg_method == "sam":
        checkpoint = resolve_path(args.sam_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")
        return {
            "sam_encoder_version": args.sam_model_type,
            "sam_checkpoint_path": str(checkpoint),
            "sam_num_points_per_side": args.points_per_side,
            "sam_num_points_per_batch": args.points_per_batch,
            "sam_pred_iou_threshold": args.pred_iou_threshold,
            "confidence_threshold": args.confidence_threshold,
        }

    checkpoint = resolve_path(args.entity_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EntitySeg checkpoint does not exist: {checkpoint}")
    default_config = (
        gaga_root
        / "mask"
        / "detectron2"
        / "detectron2"
        / "projects"
        / "CropFormer"
        / "configs"
        / "entityv2"
        / "entity_segmentation"
        / "cropformer_hornet_3x.yaml"
    )
    config_path = resolve_path(args.entity_config) if args.entity_config else default_config
    if not config_path.is_file():
        raise FileNotFoundError(
            "EntitySeg config does not exist: "
            f"{config_path}. Install Gaga's optional EntitySeg dependencies or pass --entity-config."
        )
    return {
        "entityseg_config_file": str(config_path),
        "entityseg_checkpoint_path": str(checkpoint),
        "confidence_threshold": args.confidence_threshold,
    }


def run_scene(
    scene_name: str,
    scene_dir: Path,
    args: argparse.Namespace,
    api: dict[str, Any],
    segmenter: Any,
    device: Any,
) -> SceneResult:
    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    result = SceneResult(scene=scene_name)
    source_dir = scene_dir / args.images
    output_dir = scene_dir / f"raw_{args.seg_method}_mask"
    visualization_dir = scene_dir / f"raw_{args.seg_method}_mask_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(source_dir)
    if not paths:
        raise FileNotFoundError(f"No supported image files found in: {source_dir}")

    for image_path in tqdm(paths, desc=f"{scene_name} ({args.seg_method})", unit="image"):
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists() and not args.force:
            result.skipped += 1
            continue

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("OpenCV could not decode the image")

            source_height, source_width = image.shape[:2]
            inference_image = image
            longest_edge = max(source_height, source_width)
            if args.image_resolution and longest_edge > args.image_resolution:
                scale = args.image_resolution / longest_edge
                inference_width = max(1, round(source_width * scale))
                inference_height = max(1, round(source_height * scale))
                inference_image = cv2.resize(
                    image,
                    (inference_width, inference_height),
                    interpolation=cv2.INTER_AREA,
                )

            if args.seg_method == "sam":
                mask = api["get_sam_mask"](
                    segmenter, inference_image, args.confidence_threshold
                )
            else:
                mask = api["get_entityseg_mask"](
                    segmenter, inference_image, args.confidence_threshold
                )

            mask = np.asarray(mask)
            if mask.shape != inference_image.shape[:2]:
                raise RuntimeError(
                    f"Mask shape {mask.shape} does not match inference image shape "
                    f"{inference_image.shape[:2]}"
                )
            if mask.shape != (source_height, source_width):
                mask = cv2.resize(
                    mask,
                    (source_width, source_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if mask.size and int(mask.max()) > 255:
                raise RuntimeError(
                    "Gaga stores masks as uint8 PNGs and cannot represent more than 255 instances in one image"
                )
            mask = mask.astype(np.uint8, copy=False)
            if not cv2.imwrite(str(output_path), mask):
                raise RuntimeError(f"OpenCV could not write: {output_path}")

            if args.visualize:
                visualization = api["visualize_mask"](mask)
                visualization_path = visualization_dir / f"{image_path.stem}.png"
                if not cv2.imwrite(str(visualization_path), visualization):
                    raise RuntimeError(f"OpenCV could not write: {visualization_path}")
            result.processed += 1
        except Exception as exc:  # Keep batch jobs useful when one image is damaged.
            result.failed += 1
            print(f"[failed] {scene_name}/{image_path.name}: {exc}", file=sys.stderr)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                predictor = getattr(segmenter, "predictor", None)
                if predictor is not None:
                    predictor.reset_image()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if args.fail_fast:
                raise

    return result


def validate_model_args(args: argparse.Namespace) -> None:
    if args.seg_method == "sam" and args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required with --seg-method sam")
    if args.seg_method == "entityseg" and args.entity_checkpoint is None:
        raise ValueError("--entity-checkpoint is required with --seg-method entityseg")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    gaga_root = resolve_path(args.gaga_root)
    available = discover_scenes(data_root, args.images)
    if not available:
        raise FileNotFoundError(
            f"No COLMAP scenes containing '{args.images}' were found under: {data_root}"
        )
    if args.list_scenes:
        print("\n".join(available))
        return 0

    scenes = select_scenes(args.scenes, available)
    if args.dry_run:
        print(
            f"[dry-run] method={args.seg_method}, images={args.images}, "
            f"image_resolution={args.image_resolution or 'original'}"
        )
        for scene in scenes:
            image_count = len(image_paths(data_root / scene / args.images))
            output_dir = data_root / scene / f"raw_{args.seg_method}_mask"
            print(f"[dry-run] {scene}: {image_count} image(s) -> {output_dir}")
        return 0

    validate_model_args(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch  # pylint: disable=import-outside-toplevel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"[segmenter] {args.seg_method} on {device}")

    api = load_gaga_mask_api(gaga_root)
    config = build_segmenter_config(args, gaga_root)
    segmenter = api["get_seg_model"](config, args.seg_method, device)

    results: list[SceneResult] = []
    for scene in scenes:
        try:
            result = run_scene(scene, data_root / scene, args, api, segmenter, device)
            results.append(result)
            print(
                f"[done] {scene}: processed={result.processed}, "
                f"skipped={result.skipped}, failed={result.failed}"
            )
        except Exception as exc:
            print(f"[failed] {scene}: {exc}", file=sys.stderr)
            results.append(SceneResult(scene=scene, failed=1))
            if args.fail_fast:
                break
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failures = sum(result.failed for result in results)
    processed = sum(result.processed for result in results)
    skipped = sum(result.skipped for result in results)
    print(
        f"[summary] scenes={len(results)}, processed={processed}, "
        f"skipped={skipped}, failed={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
