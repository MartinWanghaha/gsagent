#!/usr/bin/env python3
"""Render mesh diagnostics from the calibrated training cameras.

The output montage contains, from left to right: RGB reference, shaded mesh,
silhouette overlay, and a face-edge heat map.  The heat map is normalized by
the median visible face edge, so red regions expose the long triangles that
ordinary aggregate mesh statistics tend to hide.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from scene.dataset_readers import readColmapSceneInfo
from utils.config_utils import load_config
from utils.graphics_utils import get_world_to_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--camera-indices",
        type=int,
        nargs="*",
        help="explicit indices into the sorted train-camera list",
    )
    return parser


def _load_mesh(path: Path):
    import open3d as o3d

    legacy = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(legacy.vertices) == 0 or len(legacy.triangles) == 0:
        raise RuntimeError(f"{path} contains no triangle mesh")
    vertices = np.asarray(legacy.vertices, dtype=np.float32)
    faces = np.asarray(legacy.triangles, dtype=np.int64)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene, vertices, faces


def _camera_intrinsic(camera, width: int, height: int) -> np.ndarray:
    sx = width / float(camera.width)
    sy = height / float(camera.height)
    fx = (
        float(camera.fx)
        if camera.fx is not None
        else 0.5 * float(camera.width) / math.tan(0.5 * float(camera.FovX))
    )
    fy = (
        float(camera.fy)
        if camera.fy is not None
        else 0.5 * float(camera.height) / math.tan(0.5 * float(camera.FovY))
    )
    cx = float(camera.cx if camera.cx is not None else (camera.width - 1) * 0.5)
    cy = float(camera.cy if camera.cy is not None else (camera.height - 1) * 0.5)
    return np.asarray(
        (
            (fx * sx, 0.0, (cx + 0.5) * sx - 0.5),
            (0.0, fy * sy, (cy + 0.5) * sy - 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _reference(camera, width: int, height: int) -> np.ndarray:
    with Image.open(camera.image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def _turbo(values: np.ndarray) -> np.ndarray:
    import matplotlib

    rgba = matplotlib.colormaps["turbo"](np.clip(values, 0.0, 1.0))
    return np.asarray(np.rint(rgba[..., :3] * 255.0), dtype=np.uint8)


def _render_view(
    scene,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float]]:
    import open3d as o3d
    from scipy.ndimage import binary_erosion

    intrinsic = _camera_intrinsic(camera, width, height)
    extrinsic = get_world_to_view(camera.R, camera.T).astype(np.float32)
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(intrinsic),
        o3d.core.Tensor(extrinsic),
        width,
        height,
    )
    result = scene.cast_rays(rays)
    depth = result["t_hit"].numpy()
    primitive = result["primitive_ids"].numpy().astype(np.int64)
    normal = result["primitive_normals"].numpy()
    ray = rays.numpy()[..., 3:]
    hit = np.isfinite(depth) & (primitive != np.iinfo(np.uint32).max)

    normal_length = np.linalg.norm(normal, axis=2, keepdims=True)
    ray_length = np.linalg.norm(ray, axis=2, keepdims=True)
    unit_normal = normal / np.maximum(normal_length, 1e-8)
    unit_ray = ray / np.maximum(ray_length, 1e-8)
    light = np.clip(
        np.abs(np.sum(unit_normal * unit_ray, axis=2)),
        0.0,
        1.0,
    )
    shaded = np.zeros((height, width, 3), dtype=np.uint8)
    value = np.asarray(np.rint((0.18 + 0.82 * light) * 255), dtype=np.uint8)
    shaded[hit] = value[hit, None]

    reference = _reference(camera, width, height)
    boundary = hit & ~binary_erosion(hit)
    overlay = reference.copy()
    overlay[hit] = np.asarray(
        0.55 * overlay[hit] + 0.45 * np.array((30, 210, 80)),
        dtype=np.uint8,
    )
    overlay[boundary] = (255, 40, 30)

    triangle = vertices[faces]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangle[:, 1] - triangle[:, 0], axis=1),
            np.linalg.norm(triangle[:, 2] - triangle[:, 1], axis=1),
            np.linalg.norm(triangle[:, 0] - triangle[:, 2], axis=1),
        ),
        axis=1,
    )
    face_max_edge = edge_lengths.max(axis=1)
    visible_faces = primitive[hit]
    visible_edges = face_max_edge[visible_faces]
    median_edge = (
        float(np.median(visible_edges)) if len(visible_edges) else 1.0
    )
    ratio = np.zeros((height, width), dtype=np.float32)
    ratio[hit] = face_max_edge[primitive[hit]] / max(median_edge, 1e-8)
    heat = np.zeros((height, width, 3), dtype=np.uint8)
    heat[hit] = _turbo(np.clip((ratio[hit] - 1.0) / 7.0, 0.0, 1.0))
    heat[boundary] = (255, 255, 255)

    montage = np.concatenate((reference, shaded, overlay, heat), axis=1)
    stats = {
        "coverage": float(hit.mean()),
        "visible_faces": int(len(np.unique(visible_faces))),
        "visible_median_edge": median_edge,
        "pixels_edge_ratio_gt_4": float((ratio[hit] > 4.0).mean())
        if np.any(hit)
        else 0.0,
        "pixels_edge_ratio_gt_8": float((ratio[hit] > 8.0).mean())
        if np.any(hit)
        else 0.0,
    }
    return montage, stats


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model_path).expanduser().resolve()
    mesh_path = Path(args.mesh).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(model_path / "config.yaml")
    data = config["data"]
    source_path = config.get("runtime", config.get("_runtime", {})).get(
        "source_path"
    )
    if not source_path:
        raise ValueError("experiment config does not contain runtime.source_path")
    scene_info = readColmapSceneInfo(
        source_path,
        images=data.get("images", "images"),
        eval=bool(data.get("eval", False)),
        semantic_path=None,
        llffhold=int(data.get("holdout", 8)),
        defer_camera_loading=True,
    )
    cameras = list(scene_info.train_cameras)
    if args.camera_indices:
        indices = args.camera_indices
    else:
        indices = np.linspace(
            0, len(cameras) - 1, min(args.views, len(cameras)), dtype=int
        ).tolist()
    if any(index < 0 or index >= len(cameras) for index in indices):
        raise ValueError("camera index is outside the train-camera list")

    scene, vertices, faces = _load_mesh(mesh_path)
    records = []
    for index in indices:
        camera = cameras[index]
        height = max(1, int(round(args.width * camera.height / camera.width)))
        montage, statistics = _render_view(
            scene, vertices, faces, camera, args.width, height
        )
        image = Image.fromarray(montage)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, montage.shape[1], 24), fill=(0, 0, 0))
        draw.text(
            (8, 5),
            "RGB reference | shaded mesh | silhouette overlay | long-edge heat",
            fill=(255, 255, 255),
        )
        output = output_dir / f"{index:03d}_{camera.image_name}.png"
        image.save(output)
        records.append(
            {
                "camera_index": int(index),
                "image_name": camera.image_name,
                "output": str(output),
                **statistics,
            }
        )
        print(f"[render] {output}", flush=True)
    manifest = {
        "mesh": str(mesh_path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "views": records,
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
