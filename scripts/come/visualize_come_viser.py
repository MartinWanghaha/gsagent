#!/usr/bin/env python3
"""Viser viewer for CoMe point_cloud.ply + appearance_embedding.pth outputs.

This is a headless-friendly web viewer. It shows the trained Gaussian PLY as a
3D point cloud or as Viser Gaussian splats, and visualizes the appearance
embedding as colored COLMAP camera positions. The selected camera is shown as a
frustum; all frustums can be enabled with --all-frustums.

Examples:
    python gsagent/scripts/visualize_come_viser.py gsagent/outputs/come_mushroom/classroom
    python gsagent/scripts/visualize_come_viser.py gsagent/outputs/come_mushroom/classroom --mode splats
    python gsagent/scripts/visualize_come_viser.py --ply path/to/point_cloud.ply --appearance path/to/appearance_embedding.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


SH_C0 = 0.28209479177387814
PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass
class GaussianData:
    points: np.ndarray
    rgb_u8: np.ndarray
    opacity: np.ndarray | None
    confidence: np.ndarray | None
    scale: np.ndarray | None
    rotation_wxyz: np.ndarray | None
    covariances: np.ndarray | None
    total_vertices: int
    sampled_vertices: int


@dataclass
class AppearanceData:
    embeddings: np.ndarray
    class_path: str
    init_kwargs: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CoMe point_cloud.ply and appearance_embedding.pth with viser."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        help=(
            "CoMe model directory, iteration directory, or point_cloud.ply. "
            "For example: gsagent/outputs/come_mushroom/classroom"
        ),
    )
    parser.add_argument("--ply", type=Path, default=None, help="Explicit point_cloud.ply path.")
    parser.add_argument(
        "--appearance",
        type=Path,
        default=None,
        help="Explicit appearance_embedding.pth path.",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=None,
        help="Explicit cameras.json path. Default: inferred from model directory.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional directory used to resolve camera img_name files.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to load when model_path is a model directory. Default: latest.",
    )
    parser.add_argument(
        "--mode",
        choices=("points", "splats"),
        default="points",
        help="Render PLY as point cloud or Viser Gaussian splats.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Maximum Gaussian vertices to send to the browser.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.012,
        help="Point size for --mode points.",
    )
    parser.add_argument(
        "--splat-scale",
        type=float,
        default=1.0,
        help="Extra scale factor for --mode splats.",
    )
    parser.add_argument(
        "--opacity-min",
        type=float,
        default=0.0,
        help="Filter sampled Gaussians by sigmoid(opacity) before sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for downsampling the PLY.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.08,
        help="Camera frustum scale.",
    )
    parser.add_argument(
        "--frustum-images",
        action="store_true",
        help="Attach thumbnails to all camera frustums. This can be heavier.",
    )
    parser.add_argument(
        "--all-frustums",
        action="store_true",
        help="Draw frustums for many/all cameras. Default shows only camera positions plus the selected frustum.",
    )
    parser.add_argument(
        "--max-frustums",
        type=int,
        default=64,
        help="Maximum camera frustums to draw when --all-frustums is enabled.",
    )
    parser.add_argument(
        "--camera-point-size",
        type=float,
        default=0.035,
        help="Point size for colored camera positions.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not load camera images in the GUI.",
    )
    parser.add_argument(
        "--up",
        default="+z",
        help="Viser up direction, for example +z, +y, -y.",
    )
    return parser.parse_args()


def resolve_existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def find_latest_iteration(model_root: Path) -> int:
    point_cloud_dir = model_root / "point_cloud"
    if not point_cloud_dir.is_dir():
        raise FileNotFoundError(f"Missing point_cloud directory: {point_cloud_dir}")

    iterations: list[int] = []
    for path in point_cloud_dir.iterdir():
        match = re.match(r"iteration_(\d+)$", path.name)
        if match and path.is_dir():
            iterations.append(int(match.group(1)))
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories in {point_cloud_dir}")
    return max(iterations)


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    model_input = resolve_existing(args.model_path)
    ply_path = resolve_existing(args.ply)
    appearance_path = resolve_existing(args.appearance)
    cameras_path = resolve_existing(args.cameras)

    model_root: Path | None = None
    iteration_dir: Path | None = None

    if ply_path is None:
        if model_input is None:
            raise ValueError("Pass either model_path or --ply.")

        if model_input.is_file():
            ply_path = model_input
        elif (model_input / "point_cloud.ply").is_file():
            iteration_dir = model_input
            ply_path = iteration_dir / "point_cloud.ply"
        elif (model_input / "point_cloud").is_dir():
            model_root = model_input
            iteration = args.iteration if args.iteration >= 0 else find_latest_iteration(model_root)
            iteration_dir = model_root / "point_cloud" / f"iteration_{iteration}"
            ply_path = iteration_dir / "point_cloud.ply"
        else:
            raise FileNotFoundError(
                f"Could not infer point_cloud.ply from {model_input}"
            )

    if iteration_dir is None and ply_path is not None:
        if ply_path.name == "point_cloud.ply":
            iteration_dir = ply_path.parent
            if iteration_dir.parent.name == "point_cloud":
                model_root = iteration_dir.parent.parent

    if model_root is None and iteration_dir is not None and iteration_dir.parent.name == "point_cloud":
        model_root = iteration_dir.parent.parent

    if appearance_path is None:
        appearance_path = ply_path.with_name("appearance_embedding.pth")
    if cameras_path is None and model_root is not None:
        cameras_path = model_root / "cameras.json"

    for required in (ply_path, appearance_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    return ply_path, appearance_path, cameras_path, model_root or ply_path.parent


def read_ply_vertex_memmap(path: Path) -> tuple[np.memmap, list[str], int]:
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt: str | None = None
        vertex_count: int | None = None
        in_vertex = False
        properties: list[tuple[str, str]] = []

        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError(f"PLY header missing end_header: {path}")
            line = line_bytes.decode("ascii", errors="replace").strip()
            if line == "end_header":
                offset = f.tell()
                break
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("List properties in vertex elements are not supported.")
                ply_type, name = parts[1], parts[2]
                if ply_type not in PLY_TYPE_TO_DTYPE:
                    raise ValueError(f"Unsupported PLY property type {ply_type!r}")
                properties.append((name, PLY_TYPE_TO_DTYPE[ply_type]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt!r}")
    if vertex_count is None:
        raise ValueError("PLY has no vertex element.")

    dtype = np.dtype(properties)
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,))
    return vertices, [name for name, _ in properties], vertex_count


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def scalar_colormap(values: np.ndarray) -> np.ndarray:
    """A small dependency-free blue/cyan/yellow/red colormap."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1e-6)
    t = np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    stops = np.array(
        [
            [25, 35, 90],
            [30, 150, 190],
            [245, 215, 90],
            [190, 45, 35],
        ],
        dtype=np.float32,
    )
    x = t * (len(stops) - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    frac = (x - i0)[:, None]
    return np.clip(stops[i0] * (1.0 - frac) + stops[i1] * frac, 0, 255).astype(np.uint8)


def robust_rgb_from_components(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] < 3:
        values = np.pad(values, ((0, 0), (0, 3 - values.shape[1])))
    values = values[:, :3]
    out = np.zeros_like(values)
    for dim in range(3):
        lo, hi = np.percentile(values[:, dim], [2.0, 98.0])
        out[:, dim] = np.clip((values[:, dim] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def sh_dc_to_rgb_u8(vertices: np.ndarray) -> np.ndarray:
    rgb = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1)
    rgb = np.clip(0.5 + SH_C0 * rgb.astype(np.float32), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def normalize_quaternions(wxyz: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(wxyz, axis=1, keepdims=True)
    return wxyz / np.maximum(norms, 1e-8)


def quaternion_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(wxyz.astype(np.float32))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mats = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - w * z)
    mats[:, 0, 2] = 2.0 * (x * z + w * y)
    mats[:, 1, 0] = 2.0 * (x * y + w * z)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - w * x)
    mats[:, 2, 0] = 2.0 * (x * z - w * y)
    mats[:, 2, 1] = 2.0 * (y * z + w * x)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


def compute_covariances(scales: np.ndarray, wxyz: np.ndarray, scale_modifier: float) -> np.ndarray:
    scales = np.exp(scales.astype(np.float32)) * float(scale_modifier)
    rotations = quaternion_wxyz_to_matrix(wxyz)
    covariances = np.einsum(
        "nij,nj,nkj->nik",
        rotations,
        scales * scales,
        rotations,
        optimize=True,
    )
    return covariances.astype(np.float32)


def load_gaussian_data(path: Path, max_points: int, seed: int, opacity_min: float, mode: str, splat_scale: float) -> GaussianData:
    vertices, fields, total_vertices = read_ply_vertex_memmap(path)
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"PLY is missing required field(s): {', '.join(missing)}")

    candidate_indices = np.arange(total_vertices)
    opacity_all = None
    if opacity_min > 0.0 and "opacity" in fields:
        opacity_all = sigmoid(np.asarray(vertices["opacity"], dtype=np.float32))
        candidate_indices = candidate_indices[opacity_all >= opacity_min]
        if candidate_indices.size == 0:
            raise ValueError(f"No Gaussians passed --opacity-min {opacity_min}")

    rng = np.random.default_rng(seed)
    if max_points > 0 and candidate_indices.size > max_points:
        indices = rng.choice(candidate_indices, size=max_points, replace=False)
        indices.sort()
    else:
        indices = candidate_indices

    sampled = vertices[indices]
    points = np.stack([sampled["x"], sampled["y"], sampled["z"]], axis=1).astype(np.float32)
    rgb_u8 = sh_dc_to_rgb_u8(sampled)

    opacity = None
    if "opacity" in fields:
        opacity = sigmoid(np.asarray(sampled["opacity"], dtype=np.float32))

    confidence = None
    if "confidence" in fields:
        confidence = np.exp(np.clip(np.asarray(sampled["confidence"], dtype=np.float32), -30.0, 30.0))

    scale = None
    if all(f"scale_{i}" in fields for i in range(3)):
        scale = np.stack([sampled[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)

    rotation_wxyz = None
    if all(f"rot_{i}" in fields for i in range(4)):
        rotation_wxyz = np.stack([sampled[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)

    covariances = None
    if mode == "splats":
        if scale is None or rotation_wxyz is None or opacity is None:
            raise ValueError("--mode splats requires scale_*, rot_*, and opacity fields.")
        covariances = compute_covariances(scale, rotation_wxyz, splat_scale)

    return GaussianData(
        points=points,
        rgb_u8=rgb_u8,
        opacity=opacity,
        confidence=confidence,
        scale=scale,
        rotation_wxyz=rotation_wxyz,
        covariances=covariances,
        total_vertices=total_vertices,
        sampled_vertices=points.shape[0],
    )


def load_appearance(path: Path) -> AppearanceData:
    capture = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(capture, dict) or "state_dict" not in capture:
        raise ValueError(f"Unexpected appearance capture format: {path}")
    state_dict = capture["state_dict"]
    if "_appearance_embeddings" not in state_dict:
        raise ValueError(f"No _appearance_embeddings tensor in {path}")
    embeddings = state_dict["_appearance_embeddings"].detach().cpu().numpy().astype(np.float32)
    return AppearanceData(
        embeddings=embeddings,
        class_path=str(capture.get("class_path", "unknown")),
        init_kwargs=dict(capture.get("init_kwargs", {})),
    )


def load_cameras(path: Path | None, num_embeddings: int) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        print("No cameras.json found; camera frustums will not be shown.")
        return []
    with path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)

    train_cameras = [c for c in cameras if c.get("is_train_camera", True)]
    if len(train_cameras) == num_embeddings:
        return train_cameras
    if len(cameras) == num_embeddings:
        return cameras

    usable = min(len(train_cameras), num_embeddings)
    print(
        f"Warning: {num_embeddings} embeddings but {len(train_cameras)} train cameras. "
        f"Using first {usable} train cameras."
    )
    return train_cameras[:usable]


def rotation_matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / max(float(np.linalg.norm(q)), 1e-8)


def camera_pose(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    position = np.asarray(camera["position"], dtype=np.float32)
    return rotation_matrix_to_wxyz(rotation), position


def make_embedding_colors(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:3].T
    norms = np.linalg.norm(embeddings, axis=1)
    indices = np.arange(embeddings.shape[0], dtype=np.float32)
    return {
        "embedding_pca": robust_rgb_from_components(components),
        "embedding_norm": scalar_colormap(norms),
        "camera_index": scalar_colormap(indices),
    }


def gaussian_color_modes(data: GaussianData) -> dict[str, np.ndarray]:
    modes = {"rgb_dc": data.rgb_u8}
    if data.opacity is not None:
        modes["opacity"] = scalar_colormap(data.opacity)
    if data.confidence is not None:
        modes["confidence"] = scalar_colormap(data.confidence)
    if data.scale is not None:
        modes["scale_mean"] = scalar_colormap(np.exp(data.scale).mean(axis=1))
    return modes


def embedding_heatmap(vector: np.ndarray, cell_size: int = 18) -> np.ndarray:
    values = vector.reshape(8, 8)
    max_abs = max(float(np.max(np.abs(values))), 1e-8)
    t = np.clip(values / max_abs, -1.0, 1.0)
    image = np.zeros((8, 8, 3), dtype=np.float32)
    positive = np.clip(t, 0.0, 1.0)
    negative = np.clip(-t, 0.0, 1.0)
    image[..., 0] = 0.18 + 0.82 * positive
    image[..., 1] = 0.18 + 0.45 * (1.0 - np.abs(t))
    image[..., 2] = 0.18 + 0.82 * negative
    image = (image * 255.0).astype(np.uint8)
    image = np.repeat(np.repeat(image, cell_size, axis=0), cell_size, axis=1)
    image[::cell_size, :, :] = 30
    image[:, ::cell_size, :] = 30
    return image


def resolve_camera_image_path(camera: dict[str, Any], image_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    img_path = camera.get("img_path")
    img_name = camera.get("img_name")
    if img_path:
        candidates.append(Path(img_path))
    if image_root is not None and img_name:
        candidates.extend(
            [
                image_root / img_name,
                image_root / f"{img_name}.jpg",
                image_root / f"{img_name}.png",
                image_root / f"{img_name}.jpeg",
            ]
        )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate
    return None


def load_image_uint8(path: Path | None, max_width: int = 640) -> np.ndarray:
    if path is None:
        return np.zeros((16, 16, 3), dtype=np.uint8) + 40
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def make_camera_options(cameras: list[dict[str, Any]]) -> list[str]:
    return [f"{idx:04d}: {camera.get('img_name', f'camera_{idx}')}" for idx, camera in enumerate(cameras)]


def selected_index_from_option(option: str) -> int:
    return int(option.split(":", 1)[0])


def make_camera_markdown(
    index: int,
    camera: dict[str, Any],
    embedding: np.ndarray,
    color: np.ndarray,
    image_path: Path | None,
) -> str:
    pos = np.asarray(camera.get("position", [0, 0, 0]), dtype=np.float32)
    return (
        f"### Selected camera\n"
        f"- index: `{index}`\n"
        f"- image: `{camera.get('img_name', '')}`\n"
        f"- image path: `{image_path or 'not found'}`\n"
        f"- position: `[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]`\n"
        f"- embedding mean/std: `{embedding.mean():.5f}` / `{embedding.std():.5f}`\n"
        f"- embedding L2: `{np.linalg.norm(embedding):.5f}`\n"
        f"- camera color RGB: `({int(color[0])}, {int(color[1])}, {int(color[2])})`"
    )


def install_hint_and_exit(error: Exception) -> None:
    print("Could not import viser.", file=sys.stderr)
    print(f"{type(error).__name__}: {error}", file=sys.stderr)
    print("Install it in your active environment with: pip install viser", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()

    try:
        import viser
    except Exception as exc:
        install_hint_and_exit(exc)

    ply_path, appearance_path, cameras_path, model_root = infer_paths(args)
    print(f"PLY: {ply_path}")
    print(f"Appearance: {appearance_path}")
    if cameras_path is not None:
        print(f"Cameras: {cameras_path}")

    appearance = load_appearance(appearance_path)
    cameras = load_cameras(cameras_path, appearance.embeddings.shape[0])
    embeddings = appearance.embeddings[: len(cameras)] if cameras else appearance.embeddings

    gaussian_data = load_gaussian_data(
        ply_path,
        max_points=args.max_points,
        seed=args.seed,
        opacity_min=args.opacity_min,
        mode=args.mode,
        splat_scale=args.splat_scale,
    )
    point_color_modes = gaussian_color_modes(gaussian_data)
    camera_color_modes = make_embedding_colors(embeddings) if embeddings.size else {}
    camera_positions = (
        np.stack([np.asarray(camera["position"], dtype=np.float32) for camera in cameras], axis=0)
        if cameras
        else np.zeros((0, 3), dtype=np.float32)
    )

    server = viser.ViserServer(host=args.host, port=args.port, label="CoMe Viser")
    server.gui.configure_theme(
        titlebar_content="CoMe: point_cloud.ply + appearance_embedding.pth",
        control_layout="collapsible",
        control_width="large",
        dark_mode=True,
    )
    try:
        server.scene.set_up_direction(args.up)
    except Exception:
        pass

    center = gaussian_data.points.mean(axis=0)
    spread = float(np.percentile(np.linalg.norm(gaussian_data.points - center[None, :], axis=1), 95))
    server.initial_camera.look_at = tuple(center.tolist())
    server.initial_camera.position = tuple((center + np.array([0.0, -2.0 * spread, 0.7 * spread])).tolist())

    if args.mode == "splats":
        opacities = gaussian_data.opacity
        assert opacities is not None and gaussian_data.covariances is not None
        scene_handle = server.scene.add_gaussian_splats(
            "/come/gaussian_splats",
            centers=gaussian_data.points,
            covariances=gaussian_data.covariances,
            rgbs=gaussian_data.rgb_u8.astype(np.float32) / 255.0,
            opacities=opacities[:, None].astype(np.float32),
        )
    else:
        scene_handle = server.scene.add_point_cloud(
            "/come/point_cloud",
            points=gaussian_data.points,
            colors=gaussian_data.rgb_u8,
            point_size=args.point_size,
            point_shape="circle",
        )

    frustum_handles = []
    first_image = np.zeros((16, 16, 3), dtype=np.uint8) + 40
    image_paths: list[Path | None] = []
    camera_options = make_camera_options(cameras)
    selected_frustum_handle = None
    selected_label_handle = None
    camera_positions_handle = None

    if cameras:
        active_camera_colors = camera_color_modes["embedding_pca"]
        image_paths = [
            None if args.no_images else resolve_camera_image_path(camera, resolve_existing(args.image_root))
            for camera in cameras
        ]
        if image_paths and image_paths[0] is not None:
            first_image = load_image_uint8(image_paths[0])

        camera_positions_handle = server.scene.add_point_cloud(
            "/come/camera_positions",
            points=camera_positions,
            colors=active_camera_colors,
            point_size=args.camera_point_size,
            point_shape="sparkle",
            point_shading="flat",
        )

        frustum_count = min(len(cameras), args.max_frustums) if args.all_frustums else 0
        for idx, camera in enumerate(cameras[:frustum_count]):
            wxyz, position = camera_pose(camera)
            color = tuple(int(x) for x in active_camera_colors[idx])
            image_path = image_paths[idx]
            image = None
            if args.frustum_images and image_path is not None:
                image = load_image_uint8(image_path, max_width=256)

            handle = server.scene.add_camera_frustum(
                f"/come/cameras/{idx:04d}",
                fov=float(camera.get("fov_y", 2.0 * math.atan2(camera["height"], 2.0 * camera["fy"]))),
                aspect=float(camera["width"] / camera["height"]),
                scale=args.frustum_scale,
                line_width=1.5,
                color=color,
                image=image,
                wxyz=wxyz,
                position=position,
            )

            @handle.on_click
            def _(_, option=camera_options[idx]) -> None:
                camera_dropdown.value = option
                update_selected_camera()

            frustum_handles.append(handle)

    with server.gui.add_folder("Files", expand_by_default=True):
        server.gui.add_markdown(
            f"- model root: `{model_root}`\n"
            f"- ply: `{ply_path.name}`\n"
            f"- appearance: `{appearance_path.name}`\n"
            f"- appearance class: `{appearance.class_path}`\n"
            f"- init kwargs: `{appearance.init_kwargs}`\n"
            f"- PLY vertices: `{gaussian_data.total_vertices:,}`\n"
            f"- shown vertices: `{gaussian_data.sampled_vertices:,}`\n"
            f"- cameras: `{len(cameras):,}`\n"
            f"- all frustums: `{args.all_frustums}`\n"
            f"- frustums drawn: `{len(frustum_handles):,}`"
        )

    with server.gui.add_folder("Scene", expand_by_default=True):
        point_visible = server.gui.add_checkbox("Show PLY", True)
        point_color = server.gui.add_dropdown(
            "PLY color",
            tuple(point_color_modes.keys()),
            initial_value="rgb_dc",
        )
        camera_visible = server.gui.add_checkbox("Show cameras", True, disabled=not bool(cameras))
        frustum_visible = server.gui.add_checkbox(
            "Show all frustums",
            bool(args.all_frustums),
            disabled=not bool(frustum_handles),
        )
        camera_color = server.gui.add_dropdown(
            "Camera color",
            tuple(camera_color_modes.keys()) if camera_color_modes else ("none",),
            initial_value="embedding_pca" if camera_color_modes else "none",
            disabled=not bool(camera_color_modes),
        )

    with server.gui.add_folder("Appearance embedding", expand_by_default=True):
        if camera_options:
            camera_dropdown = server.gui.add_dropdown(
                "Camera",
                tuple(camera_options),
                initial_value=camera_options[0],
            )
        else:
            camera_dropdown = server.gui.add_dropdown(
                "Camera",
                ("no cameras.json",),
                initial_value="no cameras.json",
                disabled=True,
            )
        camera_info = server.gui.add_markdown("No camera selected.")
        camera_image = server.gui.add_image(first_image, label="Camera image", format="jpeg")
        embedding_image = server.gui.add_image(
            embedding_heatmap(embeddings[0] if embeddings.size else np.zeros(64, dtype=np.float32)),
            label="64D embedding heatmap",
            format="png",
        )

    def update_point_colors() -> None:
        colors = point_color_modes[point_color.value]
        if args.mode == "splats":
            scene_handle.rgbs = colors.astype(np.float32) / 255.0
        else:
            scene_handle.colors = colors

    def update_camera_colors() -> None:
        if camera_color.value == "none":
            return
        colors = camera_color_modes[camera_color.value]
        if camera_positions_handle is not None:
            camera_positions_handle.colors = colors
        for idx, handle in enumerate(frustum_handles):
            handle.color = tuple(int(x) for x in colors[idx])
        update_selected_camera()

    def update_selected_camera() -> None:
        nonlocal selected_frustum_handle, selected_label_handle
        if not cameras:
            return
        idx = selected_index_from_option(camera_dropdown.value)
        camera = cameras[idx]
        embedding = embeddings[idx]
        colors = camera_color_modes.get(camera_color.value, camera_color_modes["embedding_pca"])
        color = colors[idx]
        image_path = image_paths[idx] if idx < len(image_paths) else None
        image = load_image_uint8(image_path) if not args.no_images else np.zeros((16, 16, 3), dtype=np.uint8) + 40

        camera_info.content = make_camera_markdown(idx, camera, embedding, color, image_path)
        camera_image.image = image
        embedding_image.image = embedding_heatmap(embedding)

        if selected_frustum_handle is not None:
            selected_frustum_handle.remove()
        if selected_label_handle is not None:
            selected_label_handle.remove()

        wxyz, position = camera_pose(camera)
        selected_frustum_handle = server.scene.add_camera_frustum(
            "/come/selected_camera/frustum",
            fov=float(camera.get("fov_y", 2.0 * math.atan2(camera["height"], 2.0 * camera["fy"]))),
            aspect=float(camera["width"] / camera["height"]),
            scale=args.frustum_scale * 1.8,
            line_width=4.0,
            color=(255, 255, 255),
            image=image if not args.no_images else None,
            wxyz=wxyz,
            position=position,
        )
        selected_label_handle = server.scene.add_label(
            "/come/selected_camera/label",
            text=f"{idx}: {camera.get('img_name', '')}",
            position=position,
        )

    @point_visible.on_update
    def _(_) -> None:
        scene_handle.visible = point_visible.value

    @point_color.on_update
    def _(_) -> None:
        update_point_colors()

    @camera_visible.on_update
    def _(_) -> None:
        if camera_positions_handle is not None:
            camera_positions_handle.visible = camera_visible.value
        for handle in frustum_handles:
            handle.visible = camera_visible.value and frustum_visible.value

    @frustum_visible.on_update
    def _(_) -> None:
        for handle in frustum_handles:
            handle.visible = camera_visible.value and frustum_visible.value

    @camera_color.on_update
    def _(_) -> None:
        update_camera_colors()

    @camera_dropdown.on_update
    def _(_) -> None:
        update_selected_camera()

    update_selected_camera()
    server.flush()

    host_for_print = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Open http://{host_for_print}:{server.get_port()} in a browser.")
    print("If this is a remote machine, forward the port, for example:")
    print(f"  ssh -L {server.get_port()}:127.0.0.1:{server.get_port()} <user>@<host>")

    if hasattr(server, "sleep_forever"):
        server.sleep_forever()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
