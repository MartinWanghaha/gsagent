#!/usr/bin/env python3
"""Infer Mushroom scenes with VGGT-Omega and write COLMAP-compatible datasets.

Examples:
    python scripts/vggt_omega/infer_vggt_omega_mushroom.py \
        --scene classroom --checkpoint checkpoints/vggt_omega_1b_512.pt --gpu 0
    python scripts/vggt_omega/infer_vggt_omega_mushroom.py \
        --checkpoint checkpoints/vggt_omega_1b_512.pt --gpu 0 --force
    python scripts/vggt_omega/infer_vggt_omega_mushroom.py --list-scenes

The output images are VGGT-Omega's resized/padded inputs. This keeps the image
pixels, predicted intrinsics, depth maps, and COLMAP observations consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGGT_ROOT = GSAGENT_ROOT / "submodules" / "vggt-omega"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "PlanarGS_dataset" / "mushroom"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "data" / "mushroom_omega"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    image_ids: np.ndarray
    point2d_idxs: np.ndarray
    xys: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega on Mushroom images and export COLMAP models."
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
        "--vggt-root",
        type=Path,
        default=DEFAULT_VGGT_ROOT,
        help=f"VGGT-Omega checkout. Default: {DEFAULT_VGGT_ROOT}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the official VGGT-Omega .pt or .safetensors checkpoint.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=256,
        help="VGGT token-budget resolution. Default: 256 for a 16 GB GPU.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["balanced", "max_size"],
        default="balanced",
        help="VGGT image preprocessing mode. Default: balanced.",
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
        default=0,
        help="Uniformly limit each scene to this many frames; 0 keeps all frames.",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=50.0,
        help="Keep pixels at or above this global confidence percentile. Default: 50.",
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
        "--text-alignment",
        action="store_true",
        help="Enable the text-alignment head for the 256 text-aligned checkpoint.",
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
    if args.image_resolution <= 0 or args.image_resolution % 16 != 0:
        parser.error("--image-resolution must be positive and divisible by 16.")
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
    if args.depth_edge_rtol <= 0:
        parser.error("--depth-edge-rtol must be positive.")
    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_scenes(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    return [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir() and any(iter_image_paths(path / "images"))
    ]


def select_scenes(requested: list[str] | None, available: list[str]) -> list[str]:
    if not requested:
        return available
    selected: list[str] = []
    for value in requested:
        for scene in value.split(","):
            scene = scene.strip()
            if scene and scene not in selected:
                selected.append(scene)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown scenes: {', '.join(unknown)}. Available: {', '.join(available)}")
    return selected


def iter_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def select_image_paths(image_dir: Path, frame_step: int, max_frames: int) -> list[Path]:
    paths = iter_image_paths(image_dir)[::frame_step]
    if max_frames and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(np.int64)
        paths = [paths[int(index)] for index in np.unique(indices)]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return paths


def load_model(checkpoint: Path, vggt_root: Path, text_alignment: bool) -> tuple[Any, Any]:
    if not vggt_root.is_dir():
        raise FileNotFoundError(f"VGGT-Omega root does not exist: {vggt_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    sys.path.insert(0, str(vggt_root))
    import torch
    from vggt_omega.models import VGGTOmega

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires a CUDA GPU.")

    print(f"[model] loading {checkpoint}")
    model = VGGTOmega(enable_alignment=text_alignment).eval()
    if checkpoint.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint), device="cpu")
    else:
        try:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint, map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state dictionary.")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")
    print(f"[model] ready on {torch.cuda.get_device_name(0)}")
    return model, torch


def infer_scene(
    model: Any,
    torch: Any,
    image_paths: list[Path],
    vggt_root: Path,
    image_resolution: int,
    resize_mode: str,
) -> dict[str, np.ndarray]:
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images_cpu = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=resize_mode,
        image_resolution=image_resolution,
    )
    print(f"[infer] {len(image_paths)} images -> {tuple(images_cpu.shape)}")
    images = images_cpu.to("cuda", non_blocking=True)

    try:
        with torch.inference_mode():
            predictions = model(images)
            extrinsics, intrinsics = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

        result = {
            "images": images_cpu.numpy(),
            "depth": _remove_batch_and_channel(predictions["depth"]),
            "confidence": _remove_batch_and_channel(predictions["depth_conf"]),
            "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
            "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
        }
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "VGGT-Omega ran out of GPU memory. Lower --image-resolution to 192/128, "
            "or use --max-frames to infer a smaller uniformly sampled subset."
        ) from error
    finally:
        if "predictions" in locals():
            del predictions
        del images
        torch.cuda.empty_cache()

    expected = len(image_paths)
    if any(result[key].shape[0] != expected for key in result):
        shapes = {key: value.shape for key, value in result.items()}
        raise ValueError(f"Unexpected VGGT output shapes: {shapes}")
    return result


def _remove_batch_and_channel(tensor: Any) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


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
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    seed: int,
) -> tuple[PointCloud, float]:
    images = predictions["images"]
    depth = predictions["depth"]
    confidence = predictions["confidence"]
    extrinsics = predictions["extrinsics"]
    intrinsics = predictions["intrinsics"]

    valid_confidence = confidence[
        np.isfinite(confidence) & np.isfinite(depth) & (depth > 0) & (confidence > 1e-5)
    ]
    if valid_confidence.size == 0:
        raise ValueError("VGGT produced no finite positive depth/confidence values.")
    threshold = float(np.percentile(valid_confidence, confidence_percentile))

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    image_id_parts: list[np.ndarray] = []
    xy_parts: list[np.ndarray] = []

    for index in range(depth.shape[0]):
        frame_depth = depth[index]
        frame_confidence = confidence[index]
        mask = (
            np.isfinite(frame_depth)
            & np.isfinite(frame_confidence)
            & (frame_depth > 0)
            & (frame_confidence >= threshold)
            & (frame_confidence > 1e-5)
        )
        if filter_depth_edges:
            mask &= ~depth_edge_mask(frame_depth, depth_edge_rtol)

        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::point_stride, ::point_stride] = True
        ys, xs = np.nonzero(mask & stride_mask)
        if len(xs) == 0:
            continue

        z = frame_depth[ys, xs].astype(np.float64)
        intrinsic = intrinsics[index].astype(np.float64)
        camera_points = np.column_stack(
            (
                (xs - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                (ys - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                z,
            )
        )
        rotation = extrinsics[index, :3, :3].astype(np.float64)
        translation = extrinsics[index, :3, 3].astype(np.float64)
        world_points = (camera_points - translation) @ rotation

        colors = np.transpose(images[index], (1, 2, 0))[ys, xs]
        colors = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)

        xyz_parts.append(world_points.astype(np.float32))
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
    if voxel_size > 0:
        voxel_keys = np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        selection = np.sort(unique_indices)
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

    point_cloud = PointCloud(
        xyz=xyz,
        rgb=rgb,
        image_ids=image_ids,
        point2d_idxs=point2d_idxs,
        xys=xys,
    )
    return point_cloud, threshold


def build_colmap_records(
    image_paths: list[Path], predictions: dict[str, np.ndarray], point_cloud: PointCloud
) -> tuple[list[CameraRecord], list[ImageRecord]]:
    height, width = predictions["images"].shape[-2:]
    cameras: list[CameraRecord] = []
    images: list[ImageRecord] = []
    point_ids = np.arange(1, len(point_cloud.xyz) + 1, dtype=np.int64)
    xys = point_cloud.xys

    for index, image_path in enumerate(image_paths):
        image_id = index + 1
        intrinsic = predictions["intrinsics"][index]
        extrinsic = predictions["extrinsics"][index]
        cameras.append(
            CameraRecord(
                camera_id=image_id,
                width=width,
                height=height,
                params=np.array(
                    [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
                    dtype=np.float64,
                ),
            )
        )

        observation_indices = np.flatnonzero(point_cloud.image_ids == image_id)
        images.append(
            ImageRecord(
                image_id=image_id,
                qvec=rotmat_to_qvec(extrinsic[:3, :3]),
                tvec=extrinsic[:3, 3].astype(np.float64),
                camera_id=image_id,
                name=image_path.name,
                xys=xys[observation_indices],
                point3d_ids=point_ids[observation_indices],
            )
        )
    return cameras, images


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.reshape(-1)
    matrix = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_processed_images(images: np.ndarray, image_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for tensor, source_path in zip(images, image_paths):
        pixels = np.transpose(tensor, (1, 2, 0))
        pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
        output_path = output_dir / source_path.name
        image = Image.fromarray(pixels, mode="RGB")
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(output_path, quality=95, subsampling=0)
        else:
            image.save(output_path)


def write_colmap_model(
    sparse_dir: Path,
    cameras: list[CameraRecord],
    images: list[ImageRecord],
    point_cloud: PointCloud,
    write_text: bool,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(sparse_dir / "cameras.bin", cameras)
    write_images_binary(sparse_dir / "images.bin", images)
    write_points3d_binary(sparse_dir / "points3D.bin", point_cloud)
    write_points3d_ply(sparse_dir / "points3D.ply", point_cloud)
    if write_text:
        write_cameras_text(sparse_dir / "cameras.txt", cameras)
        write_images_text(sparse_dir / "images.txt", images)
        write_points3d_text(sparse_dir / "points3D.txt", point_cloud)


def write_cameras_binary(path: Path, cameras: list[CameraRecord]) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            file.write(struct.pack("<iiQQ", camera.camera_id, 1, camera.width, camera.height))
            file.write(struct.pack("<dddd", *camera.params))


def write_images_binary(path: Path, images: list[ImageRecord]) -> None:
    observation_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(images)))
        for image in images:
            file.write(struct.pack("<i", image.image_id))
            file.write(struct.pack("<dddd", *image.qvec))
            file.write(struct.pack("<ddd", *image.tvec))
            file.write(struct.pack("<i", image.camera_id))
            file.write(image.name.encode("utf-8") + b"\x00")
            file.write(struct.pack("<Q", len(image.point3d_ids)))
            observations = np.empty(len(image.point3d_ids), dtype=observation_dtype)
            observations["x"] = image.xys[:, 0]
            observations["y"] = image.xys[:, 1]
            observations["point_id"] = image.point3d_ids
            observations.tofile(file)


def write_points3d_binary(path: Path, point_cloud: PointCloud) -> None:
    point_dtype = np.dtype(
        [
            ("point_id", "<u8"),
            ("xyz", "<f8", (3,)),
            ("rgb", "u1", (3,)),
            ("error", "<f8"),
            ("track_length", "<u8"),
            ("image_id", "<i4"),
            ("point2d_idx", "<i4"),
        ]
    )
    records = np.empty(len(point_cloud.xyz), dtype=point_dtype)
    records["point_id"] = np.arange(1, len(records) + 1, dtype=np.uint64)
    records["xyz"] = point_cloud.xyz.astype(np.float64)
    records["rgb"] = point_cloud.rgb
    records["error"] = 0.0
    records["track_length"] = 1
    records["image_id"] = point_cloud.image_ids
    records["point2d_idx"] = point_cloud.point2d_idxs
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(records)))
        records.tofile(file)


def write_cameras_text(path: Path, cameras: list[CameraRecord]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# Camera list with one line of data per camera:\n")
        file.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        file.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras:
            params = " ".join(f"{value:.17g}" for value in camera.params)
            file.write(
                f"{camera.camera_id} PINHOLE {camera.width} {camera.height} {params}\n"
            )


def write_images_text(path: Path, images: list[ImageRecord]) -> None:
    mean_observations = np.mean([len(image.point3d_ids) for image in images]) if images else 0
    with path.open("w", encoding="utf-8") as file:
        file.write("# Image list with two lines of data per image:\n")
        file.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        file.write(
            f"# Number of images: {len(images)}, mean observations per image: "
            f"{mean_observations:.6f}\n"
        )
        for image in images:
            pose = [*image.qvec, *image.tvec]
            file.write(
                f"{image.image_id} "
                + " ".join(f"{value:.17g}" for value in pose)
                + f" {image.camera_id} {image.name}\n"
            )
            observations = (
                f"{xy[0]:.9g} {xy[1]:.9g} {int(point_id)}"
                for xy, point_id in zip(image.xys, image.point3d_ids)
            )
            file.write(" ".join(observations) + "\n")


def write_points3d_text(path: Path, point_cloud: PointCloud) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# 3D point list with one line of data per point:\n")
        file.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        file.write(f"# Number of points: {len(point_cloud.xyz)}, mean track length: 1\n")
        for index, (xyz, rgb, image_id, point2d_idx) in enumerate(
            zip(
                point_cloud.xyz,
                point_cloud.rgb,
                point_cloud.image_ids,
                point_cloud.point2d_idxs,
            ),
            start=1,
        ):
            file.write(
                f"{index} {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0 "
                f"{int(image_id)} {int(point2d_idx)}\n"
            )


def write_points3d_ply(path: Path, point_cloud: PointCloud) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(point_cloud.xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = point_cloud.xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = point_cloud.rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)


def write_metadata(
    path: Path,
    scene: str,
    image_paths: list[Path],
    checkpoint: Path,
    args: argparse.Namespace,
    predictions: dict[str, np.ndarray],
    point_cloud: PointCloud,
    confidence_threshold: float,
) -> None:
    metadata = {
        "generator": "VGGT-Omega",
        "scene": scene,
        "checkpoint": str(checkpoint),
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
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
            "confidence_percentile": args.confidence_percentile,
            "point_stride": args.point_stride,
            "max_points": args.max_points,
            "voxel_size": args.voxel_size,
            "filter_depth_edges": not args.no_filter_depth_edges,
            "depth_edge_rtol": args.depth_edge_rtol,
            "seed": args.seed,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def process_scene(
    scene: str,
    model: Any,
    torch: Any,
    checkpoint: Path,
    data_root: Path,
    output_root: Path,
    vggt_root: Path,
    args: argparse.Namespace,
) -> None:
    source_scene = data_root / scene
    output_scene = output_root / scene
    temporary_scene = output_root / f".{scene}.tmp"
    if output_scene.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_scene}. Pass --force to replace it.")

    image_paths = select_image_paths(source_scene / "images", args.frame_step, args.max_frames)
    predictions = infer_scene(
        model,
        torch,
        image_paths,
        vggt_root,
        args.image_resolution,
        args.resize_mode,
    )
    point_cloud, confidence_threshold = fuse_point_cloud(
        predictions,
        args.confidence_percentile,
        args.point_stride,
        args.max_points,
        args.voxel_size,
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
            temporary_scene / "omega_metadata.json",
            scene,
            image_paths,
            checkpoint,
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
    vggt_root = resolve_path(args.vggt_root)
    available_scenes = discover_scenes(data_root)

    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0

    scenes = select_scenes(args.scenes, available_scenes)
    for scene in scenes:
        image_count = len(select_image_paths(data_root / scene / "images", args.frame_step, args.max_frames))
        print(f"[scene] {scene}: {image_count} images -> {output_root / scene}")
    if args.dry_run:
        return 0
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --list-scenes or --dry-run is used.")

    checkpoint = resolve_path(args.checkpoint)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_root.mkdir(parents=True, exist_ok=True)
    model, torch = load_model(checkpoint, vggt_root, args.text_alignment)

    failures: list[tuple[str, Exception]] = []
    for scene in scenes:
        print(f"\n[infer] {scene}")
        try:
            process_scene(
                scene,
                model,
                torch,
                checkpoint,
                data_root,
                output_root,
                vggt_root,
                args,
            )
        except Exception as error:
            print(f"[failed] {scene}: {error}", file=sys.stderr)
            failures.append((scene, error))
            if not args.keep_going:
                raise

    if failures:
        print("Failed scenes: " + ", ".join(scene for scene, _ in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
