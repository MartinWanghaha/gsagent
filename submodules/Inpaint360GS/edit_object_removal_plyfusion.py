# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/

"""Fuse completed virtual RGB-D frames into run-local point clouds."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.fusion_manifest_identity import (
    FUSION_IDENTITY_VERSION,
    FUSION_MANIFEST_KIND,
    fusion_artifact_id,
)
from utils.graphics_utils import getWorld2View2
from utils.point_utils import create_point_cloud, get_intrinsics, ply_color_fusion
from utils.pose_utils import generate_ellipse_path
from utils.virtual_camera_manifest import (
    load_virtual_camera_manifest,
    virtual_views_from_manifest,
)

FRAME_COUNT = 30
IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"artifact is missing or empty: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


def _load_upstream_manifest(
    path: str | os.PathLike[str] | None,
    label: str,
    *,
    expected_kind: str,
):
    if not path:
        return None
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {resolved}")
    schemas = (1, 2) if expected_kind == "inpaint360gs-virtual-cameras" else (1,)
    if payload.get("schema_version") not in schemas or payload.get("kind") != expected_kind:
        raise ValueError(
            f"unexpected {label} schema/kind in {resolved}: "
            f"{payload.get('schema_version')!r}/{payload.get('kind')!r}"
        )
    if payload.get("complete") is not True or payload.get("status") != "complete":
        raise ValueError(f"{label} is incomplete: {resolved}")
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"{label} has no artifact_id: {resolved}")
    return {"artifact": _artifact(resolved), "artifact_id": artifact_id}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_image(directory: Path, image_name: str, label: str) -> Path:
    matches = [directory / f"{image_name}{suffix}" for suffix in IMAGE_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"{label} is missing for frame {image_name} in {directory}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{label} is ambiguous for frame {image_name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _existing_mask(directory: Path, image_name: str) -> Path:
    """Prefer LaMa's explicit mask name, with a legacy-name fallback.

    A LaMa color-input directory intentionally contains both ``00000.png``
    (RGB) and ``00000_mask.png`` (mask).  The basename-only convention is
    therefore considered only when no explicit ``*_mask`` file exists.
    """

    canonical = [
        directory / f"{image_name}_mask.png",
        directory / f"{image_name}_mask.PNG",
    ]
    legacy = [
        directory / f"{image_name}.png",
        directory / f"{image_name}.PNG",
    ]
    matches = [path for path in canonical if path.is_file()]
    convention = "canonical"
    if not matches:
        matches = [path for path in legacy if path.is_file()]
        convention = "legacy"
    if not matches:
        raise FileNotFoundError(
            f"inpainting mask is missing for frame {image_name} in {directory}; "
            f"expected {image_name}_mask.png (or legacy {image_name}.png)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"inpainting mask has multiple {convention} files for frame "
            f"{image_name}: " + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _read_color(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode RGB image: {path}")
    if image.shape != (*expected_shape, 3):
        raise ValueError(
            f"RGB shape mismatch for {path}: {image.shape}, "
            f"expected {(*expected_shape, 3)}"
        )
    return image


def _read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"inpainting mask is missing: {path}")
    with Image.open(path) as image:
        labels = np.asarray(image)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2 or labels.shape != expected_shape:
        raise ValueError(
            f"mask shape mismatch for {path}: {labels.shape}, expected {expected_shape}"
        )
    if not (np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_):
        raise ValueError(f"mask must contain integer labels: {path} ({labels.dtype})")
    mask = labels != 0
    if not mask.any():
        raise ValueError(f"inpainting mask is empty: {path}")
    if mask.all():
        raise ValueError(f"inpainting mask covers the full frame: {path}")
    return mask


def _read_depth(path: Path, expected_shape: tuple[int, int], label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    depth = np.load(path, allow_pickle=False)
    if depth.shape != expected_shape:
        raise ValueError(
            f"{label} shape mismatch for {path}: {depth.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(depth.dtype, np.number) or not np.isfinite(depth).all():
        raise ValueError(f"{label} must be a finite numeric array: {path}")
    if np.any(depth < 0):
        raise ValueError(f"{label} contains negative values: {path}")
    return np.asarray(depth)


def _fallback_virtual_views(views, circle_radius: float, frame_count: int):
    if circle_radius is None or not np.isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be finite and positive")
    if not views:
        raise ValueError("at least one training camera is required")
    base_view = views[0]
    poses = generate_ellipse_path(
        views,
        n_frames=frame_count,
        is_circle=True,
        circle_radius=circle_radius,
    )
    virtual_views = []
    for index, pose in enumerate(tqdm(poses, desc="Prepare virtual camera poses")):
        view = copy.deepcopy(base_view)
        view.R = pose[:3, :3].T
        view.T = pose[:3, 3]
        view.world_view_transform = torch.as_tensor(
            getWorld2View2(view.R, view.T, view.trans, view.scale),
            dtype=base_view.world_view_transform.dtype,
            device=base_view.world_view_transform.device,
        ).transpose(0, 1)
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0)
            .bmm(view.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        view.image_name = f"{index:05d}"
        virtual_views.append(view)
    return virtual_views


def _virtual_views(
    views,
    *,
    iteration: int,
    circle_radius: float | None,
    camera_manifest: str | os.PathLike[str] | None,
    frame_count: int,
):
    if camera_manifest:
        payload = load_virtual_camera_manifest(
            camera_manifest,
            expected_iteration=iteration,
        )
        if not views:
            raise ValueError("at least one training camera is required")
        return virtual_views_from_manifest(views[0], payload), payload["circle_radius"]
    return _fallback_virtual_views(views, circle_radius, frame_count), circle_radius


def _write_fusion_manifest(
    path: Path,
    *,
    iteration: int,
    frame_count: int,
    circle_radius: float | None,
    write_hole_ply: bool,
    camera_manifest: str | os.PathLike[str] | None,
    lama_manifest: str | os.PathLike[str] | None,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_frames = [f"{index:05d}" for index in range(frame_count)]
    input_frames = [record.get("frame") for record in inputs]
    output_frames = [record.get("frame") for record in outputs]
    if input_frames != expected_frames or output_frames != expected_frames:
        raise ValueError(
            "fusion manifest requires exactly one ordered input/output record for "
            f"each frame 00000..{frame_count - 1:05d}"
        )
    for record in outputs:
        if "fused_mask_ply" not in record:
            raise ValueError(
                f"missing fused_mask_ply output for frame {record['frame']}"
            )
        if write_hole_ply and "fused_hole_ply" not in record:
            raise ValueError(
                f"missing fused_hole_ply output for frame {record['frame']}"
            )
        if not write_hole_ply and "fused_hole_ply" in record:
            raise ValueError(
                f"unexpected fused_hole_ply output for frame {record['frame']}"
            )

    upstream = {
        "camera_manifest": (
            _load_upstream_manifest(
                camera_manifest,
                "virtual camera manifest",
                expected_kind="inpaint360gs-virtual-cameras",
            )
            if camera_manifest
            else None
        ),
        "lama_completion_manifest": _load_upstream_manifest(
            lama_manifest,
            "LaMa completion manifest",
            expected_kind="paintmesh-lama-completion",
        ),
    }
    parameters = {
        "iteration": int(iteration),
        "frame_count": int(frame_count),
        "circle_radius": float(circle_radius) if circle_radius is not None else None,
        "write_hole_ply": bool(write_hole_ply),
    }
    upstream_artifact_ids = {
        name: record["artifact_id"] if record is not None else None
        for name, record in upstream.items()
    }
    identity_payload = {
        "kind": FUSION_MANIFEST_KIND,
        "schema_version": 1,
        "identity_version": FUSION_IDENTITY_VERSION,
        "parameters": parameters,
        "upstream_artifact_ids": upstream_artifact_ids,
        "inputs": inputs,
        "outputs": outputs,
    }
    artifact_id = fusion_artifact_id(identity_payload)
    payload = {
        "schema_version": 1,
        "identity_version": FUSION_IDENTITY_VERSION,
        "kind": FUSION_MANIFEST_KIND,
        "parameters": parameters,
        "upstream": upstream,
        "upstream_artifact_ids": upstream_artifact_ids,
        "inputs": inputs,
        "outputs": outputs,
        "artifact_id": artifact_id,
        "complete": True,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"refusing to replace an unreadable fusion manifest {path}: {exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise ValueError(
                f"refusing to replace a non-object fusion manifest {path}; "
                "choose a new INPAINT_RUN_NAME"
            )
        if (
            existing.get("schema_version") != 1
            or existing.get("identity_version") != FUSION_IDENTITY_VERSION
            or existing.get("kind") != FUSION_MANIFEST_KIND
            or existing.get("complete") is not True
            or existing.get("status") != "complete"
        ):
            raise ValueError(
                f"refusing to replace an invalid fusion manifest {path}; "
                "choose a new INPAINT_RUN_NAME"
            )
        if existing.get("artifact_id") != artifact_id:
            raise ValueError(
                f"fusion manifest {path} belongs to different inputs or parameters; "
                "choose a new INPAINT_RUN_NAME"
            )
        return existing
    _atomic_json(path, payload)
    return payload


def fusion(
    dataset_path,
    model_path,
    name,
    iteration,
    views,
    *,
    circle_radius=None,
    virtual_data_root=None,
    inpainted_color_dir=None,
    inpaint_mask_dir=None,
    completed_depth_dir=None,
    removed_render_dir=None,
    removed_depth_dir=None,
    fused_output_dir=None,
    hole_output_dir=None,
    camera_manifest=None,
    lama_manifest=None,
    manifest=None,
    write_hole_ply=True,
    frame_count=None,
):
    """Fuse virtual views while preserving all historical path defaults."""

    if camera_manifest:
        count = load_virtual_camera_manifest(camera_manifest)["frame_count"]
        if frame_count is not None and frame_count != count:
            raise ValueError("requested fusion frame count differs from cameras")
        frame_count = count
    else:
        frame_count = FRAME_COUNT if frame_count is None else frame_count
    dataset_root = Path(dataset_path).expanduser().resolve()
    model_root = Path(model_path).expanduser().resolve()
    virtual_root = (
        Path(virtual_data_root).expanduser().resolve()
        if virtual_data_root
        else dataset_root
    )
    removal_root = model_root / name / f"ours_object_removal/iteration_{iteration}"
    color_root = (
        Path(inpainted_color_dir).expanduser().resolve()
        if inpainted_color_dir
        else virtual_root / "images_inpaint_unseen_virtual"
    )
    mask_root = (
        Path(inpaint_mask_dir).expanduser().resolve()
        if inpaint_mask_dir
        else virtual_root / "inpaint_2d_unseen_mask_virtual"
    )
    completed_root = (
        Path(completed_depth_dir).expanduser().resolve()
        if completed_depth_dir
        else removal_root / "depth_completed"
    )
    removed_render_root = (
        Path(removed_render_dir).expanduser().resolve()
        if removed_render_dir
        else removal_root / "renders"
    )
    removed_depth_root = (
        Path(removed_depth_dir).expanduser().resolve()
        if removed_depth_dir
        else removal_root / "depth"
    )
    fused_root = (
        Path(fused_output_dir).expanduser().resolve()
        if fused_output_dir
        else removal_root / "fused_mask_col_dep_ply"
    )
    hole_root = (
        Path(hole_output_dir).expanduser().resolve()
        if hole_output_dir
        else removal_root / "fused_hole_col_dep_ply"
    )

    fused_root.mkdir(parents=True, exist_ok=True)
    if write_hole_ply:
        hole_root.mkdir(parents=True, exist_ok=True)

    radius = float(circle_radius) if circle_radius is not None else None
    virtual_views, radius = _virtual_views(
        views,
        iteration=int(iteration),
        circle_radius=radius,
        camera_manifest=camera_manifest,
        frame_count=frame_count,
    )
    input_records = []
    output_records = []

    for view in tqdm(virtual_views, desc="Color-Depth-Fusion progress"):
        expected_shape = (int(view.image_height), int(view.image_width))
        color_path = _existing_image(color_root, view.image_name, "completed RGB")
        mask_path = _existing_mask(mask_root, view.image_name)
        completed_path = completed_root / f"{view.image_name}.npy"
        colors = _read_color(color_path, expected_shape)
        mask = _read_mask(mask_path, expected_shape)
        completed_depth = _read_depth(completed_path, expected_shape, "completed depth")

        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = np.asarray(view.R).transpose()
        world_to_camera[:3, 3] = np.asarray(view.T)
        camera_to_world = np.linalg.inv(world_to_camera)
        intrinsics = get_intrinsics(
            view.image_height, view.image_width, view.FoVx, view.FoVy
        )
        points = create_point_cloud(completed_depth, intrinsics, camera_to_world)
        fused_path = fused_root / f"{view.image_name}.ply"
        ply_color_fusion(
            points,
            colors.reshape(-1, 3),
            str(fused_path),
            mask=mask.reshape(-1),
        )
        frame_inputs = {
            "frame": view.image_name,
            "completed_rgb": _artifact(color_path),
            "inpaint_mask": _artifact(mask_path),
            "completed_depth": _artifact(completed_path),
        }
        frame_outputs = {
            "frame": view.image_name,
            "fused_mask_ply": _artifact(fused_path),
        }

        if write_hole_ply:
            render_path = _existing_image(
                removed_render_root, view.image_name, "removed-scene RGB"
            )
            depth_path = removed_depth_root / f"{view.image_name}.npy"
            hole_colors = _read_color(render_path, expected_shape)
            hole_depth = _read_depth(depth_path, expected_shape, "removed-scene depth")
            hole_points = create_point_cloud(hole_depth, intrinsics, camera_to_world)
            hole_path = hole_root / f"{view.image_name}.ply"
            ply_color_fusion(hole_points, hole_colors.reshape(-1, 3), str(hole_path))
            frame_inputs["removed_rgb"] = _artifact(render_path)
            frame_inputs["removed_depth"] = _artifact(depth_path)
            frame_outputs["fused_hole_ply"] = _artifact(hole_path)
        input_records.append(frame_inputs)
        output_records.append(frame_outputs)

    result = {
        "fused_output_dir": str(fused_root),
        "hole_output_dir": str(hole_root) if write_hole_ply else None,
        "frame_count": len(virtual_views),
    }
    if manifest:
        result["manifest"] = _write_fusion_manifest(
            Path(manifest),
            iteration=int(iteration),
            frame_count=len(virtual_views),
            circle_radius=radius,
            write_hole_ply=write_hole_ply,
            camera_manifest=camera_manifest,
            lama_manifest=lama_manifest,
            inputs=input_records,
            outputs=output_records,
        )
    return result


def removal(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args=None):
    del pipeline  # Kept in the public signature for backwards compatibility.
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    with torch.no_grad():
        return fusion(
            dataset.source_path,
            dataset.model_path,
            "virtual",
            scene.loaded_iter,
            scene.getTrainCameras(),
            circle_radius=getattr(args, "circle_radius", None),
            virtual_data_root=getattr(args, "virtual_data_root", None),
            inpainted_color_dir=getattr(args, "inpainted_color_dir", None),
            inpaint_mask_dir=getattr(args, "inpaint_mask_dir", None),
            completed_depth_dir=getattr(args, "completed_depth_dir", None),
            removed_render_dir=getattr(args, "removed_render_dir", None),
            removed_depth_dir=getattr(args, "removed_depth_dir", None),
            fused_output_dir=getattr(args, "fused_output_dir", None),
            hole_output_dir=getattr(args, "hole_output_dir", None),
            camera_manifest=getattr(args, "camera_manifest", None),
            lama_manifest=getattr(args, "lama_manifest", None),
            manifest=getattr(args, "manifest", None),
            write_hole_ply=not getattr(args, "skip_hole_ply", False),
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Fuse completed virtual RGB-D views")
    model = ModelParams(parser, sentinel=True)
    OptimizationParams(parser)  # Preserve legacy accepted optimization options.
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--source_iteration", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/object_removal/inpaint360/picnic.json",
    )
    parser.add_argument("--virtual_data_root", type=str, default=None)
    parser.add_argument("--inpainted_color_dir", type=str, default=None)
    parser.add_argument("--inpaint_mask_dir", type=str, default=None)
    parser.add_argument("--completed_depth_dir", type=str, default=None)
    parser.add_argument("--removed_render_dir", type=str, default=None)
    parser.add_argument("--removed_depth_dir", type=str, default=None)
    parser.add_argument("--fused_output_dir", type=str, default=None)
    parser.add_argument("--hole_output_dir", type=str, default=None)
    parser.add_argument("--camera_manifest", type=str, default=None)
    parser.add_argument("--lama_manifest", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--skip_hole_ply", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    with open(args.config_file, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    args.select_obj_id = config.get("select_obj_id")
    from utils.virtual_camera_manifest import require_declared_camera_manifest
    require_declared_camera_manifest(config, getattr(args, "camera_manifest", None))
    args.circle_radius = config.get("circle_radius")
    source_iteration = (
        args.source_iteration if args.source_iteration is not None else args.iteration
    )
    safe_state(args.quiet)
    removal(model.extract(args), source_iteration, pipeline.extract(args), args=args)
