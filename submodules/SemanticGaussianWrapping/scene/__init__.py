"""Scene orchestration with the public layout of the reference 3DGS project."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from .cameras import Camera, focal2fov, fov2focal
from .dataset_readers import CameraInfo, sceneLoadTypeCallbacks, storePly
from .gaussian_attributes import AttributeSpec, GaussianAttributeRegistry
from .gaussian_model import GaussianModel


def _latest_iteration(point_cloud_root: Path) -> int | None:
    if not point_cloud_root.is_dir():
        return None
    iterations = []
    for child in point_cloud_root.iterdir():
        match = re.fullmatch(r"iteration_(\d+)", child.name)
        if match and child.is_dir():
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def _resolution(camera: CameraInfo, args: Any, resolution_scale: float) -> tuple[int, int]:
    requested = int(getattr(args, "resolution", -1))
    if requested in (1, 2, 4, 8):
        factor = resolution_scale * requested
    elif requested == -1:
        factor = resolution_scale * (camera.width / 1600.0 if camera.width > 1600 else 1.0)
    elif requested > 0:
        factor = resolution_scale * camera.width / requested
    else:
        raise ValueError("resolution must be -1, a target width, or one of 1/2/4/8")
    return max(1, round(camera.width / factor)), max(1, round(camera.height / factor))


def camera_from_info(camera: CameraInfo, args: Any, resolution_scale: float = 1.0) -> Camera:
    width, height = _resolution(camera, args, resolution_scale)
    if camera.payload_loader is not None:
        payload = camera.payload_loader((height, width))
        sx = width / camera.width
        sy = height / camera.height
        source_fx = (
            camera.fx
            if camera.fx is not None
            else fov2focal(camera.FovX, camera.width)
        )
        source_fy = (
            camera.fy
            if camera.fy is not None
            else fov2focal(camera.FovY, camera.height)
        )
        source_cx = camera.cx if camera.cx is not None else (camera.width - 1) / 2.0
        source_cy = camera.cy if camera.cy is not None else (camera.height - 1) / 2.0
        fx = source_fx * sx
        fy = source_fy * sy
        return Camera(
            colmap_id=camera.uid,
            R=camera.R,
            T=camera.T,
            FoVx=focal2fov(fx, width),
            FoVy=focal2fov(fy, height),
            image=payload.image,
            gt_alpha_mask=payload.alpha,
            image_name=camera.image_name,
            uid=camera.uid,
            semantic_ids=payload.semantic_ids,
            semantic_confidence=payload.semantic_confidence,
            semantic_boundary=payload.semantic_boundary,
            data_device=getattr(args, "data_device", "cuda"),
            fx=fx,
            fy=fy,
            cx=(source_cx + 0.5) * sx - 0.5,
            cy=(source_cy + 0.5) * sy - 0.5,
            ignore_label=int(getattr(args, "semantic_ignore_label", -1)),
        )
    if camera.image is None:
        raise ValueError(f"camera {camera.image_name!r} has neither pixels nor a payload loader")
    result = Camera(
        colmap_id=camera.uid,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FovX,
        FoVy=camera.FovY,
        image=camera.image,
        gt_alpha_mask=camera.alpha,
        image_name=camera.image_name,
        uid=camera.uid,
        semantic_ids=camera.semantic_ids,
        semantic_confidence=camera.semantic_confidence,
        semantic_boundary=camera.semantic_boundary,
        data_device=getattr(args, "data_device", "cuda"),
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        ignore_label=int(getattr(args, "semantic_ignore_label", -1)),
    )
    return result if (width, height) == (camera.width, camera.height) else result.resized(width, height)


def cameraList_from_camInfos(
    camera_infos: Sequence[CameraInfo],
    resolution_scale: float,
    args: Any,
) -> list[Camera]:
    return [camera_from_info(info, args, resolution_scale) for info in camera_infos]


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: Any,
        gaussians: GaussianModel,
        load_iteration: int | None = None,
        shuffle: bool = True,
        resolution_scales: Sequence[float] = (1.0,),
    ) -> None:
        self.model_path = str(getattr(args, "model_path", ""))
        self.source_path = str(getattr(args, "source_path", ""))
        self.loaded_iter: int | None = None
        self.gaussians = gaussians
        model_root = Path(self.model_path) if self.model_path else None
        if load_iteration is not None:
            if load_iteration == -1:
                if model_root is None:
                    raise ValueError("model_path is required when loading the latest iteration")
                load_iteration = _latest_iteration(model_root / "point_cloud")
                if load_iteration is None:
                    raise FileNotFoundError(f"no saved point cloud under {model_root}")
            self.loaded_iter = int(load_iteration)

        source = Path(self.source_path)
        common = dict(
            semantic_path=getattr(args, "semantic_path", "sam_mask"),
            semantic_confidence_path=getattr(args, "semantic_confidence_path", None),
            semantic_boundary_path=getattr(args, "semantic_boundary_path", None),
            semantic_ignore_label=int(getattr(args, "semantic_ignore_label", -1)),
            semantic_background_label=int(getattr(args, "semantic_background_label", 0)),
            boundary_width=int(getattr(args, "boundary_width", 1)),
            # CameraInfo remains metadata-only.  Final RGB/semantic tensors are
            # decoded one view at a time at the actual training resolution.
            defer_camera_loading=True,
        )
        if (source / "sparse").is_dir():
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                source,
                getattr(args, "images", "images"),
                bool(getattr(args, "eval", False)),
                llffhold=int(getattr(args, "llffhold", 8)),
                **common,
            )
        elif (source / "transforms_train.json").is_file():
            scene_info = sceneLoadTypeCallbacks["Blender"](
                source,
                bool(getattr(args, "white_background", False)),
                bool(getattr(args, "eval", False)),
                random_points=int(getattr(args, "random_points", 100_000)),
                **common,
            )
        else:
            raise ValueError(f"could not recognize scene type at {source}")

        self.num_semantic_classes = int(scene_info.num_semantic_classes)
        self.semantic_label_mapping = scene_info.semantic_label_mapping

        train_infos = list(scene_info.train_cameras)
        test_infos = list(scene_info.test_cameras)
        if shuffle:
            random.shuffle(train_infos)
            random.shuffle(test_infos)
        self.cameras_extent = float(scene_info.nerf_normalization["radius"])
        self.train_cameras: dict[float, list[Camera]] = {}
        self.test_cameras: dict[float, list[Camera]] = {}
        for scale in resolution_scales:
            self.train_cameras[float(scale)] = cameraList_from_camInfos(train_infos, float(scale), args)
            self.test_cameras[float(scale)] = cameraList_from_camInfos(test_infos, float(scale), args)

        # Lazy Gaga adapters discover arbitrary raw labels while individual
        # views are materialized.  Resolve decoder cardinality and the exact
        # raw/compact mapping only after all requested cameras have been read.
        if scene_info.semantic_metadata_provider is not None:
            (
                self.num_semantic_classes,
                self.semantic_label_mapping,
            ) = scene_info.semantic_metadata_provider()
            self.num_semantic_classes = int(self.num_semantic_classes)
        if self.num_semantic_classes > 1:
            self.gaussians.configure_semantic_decoder(
                self.num_semantic_classes,
                float(getattr(args, "semantic_temperature", 1.0)),
            )

        if self.loaded_iter is not None:
            if model_root is None:
                raise ValueError("model_path is required to load an iteration")
            ply = model_root / "point_cloud" / f"iteration_{self.loaded_iter}" / "point_cloud.ply"
            self.gaussians.load_ply(ply)
        else:
            if scene_info.point_cloud is None:
                raise ValueError("scene has no initial point cloud")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            if model_root is not None:
                model_root.mkdir(parents=True, exist_ok=True)
                input_ply = model_root / "input.ply"
                if not input_ply.exists():
                    storePly(
                        input_ply,
                        scene_info.point_cloud.points,
                        scene_info.point_cloud.colors * 255.0,
                    )
                mapping_path = model_root / "semantic_labels.json"
                if self.semantic_label_mapping is not None:
                    with open(mapping_path, "w", encoding="utf8") as stream:
                        json.dump(self.semantic_label_mapping, stream, indent=2)

    @property
    def extent(self) -> float:
        return self.cameras_extent

    def save(self, iteration: int) -> None:
        if not self.model_path:
            raise ValueError("model_path is empty")
        output = Path(self.model_path) / "point_cloud" / f"iteration_{int(iteration)}"
        output.mkdir(parents=True, exist_ok=True)
        self.gaussians.save_ply(output / "point_cloud.ply")

    def getTrainCameras(self, scale: float = 1.0) -> list[Camera]:
        return self.train_cameras[float(scale)]

    def getTestCameras(self, scale: float = 1.0) -> list[Camera]:
        return self.test_cameras[float(scale)]


__all__ = [
    "AttributeSpec",
    "Camera",
    "GaussianAttributeRegistry",
    "GaussianModel",
    "Scene",
    "cameraList_from_camInfos",
    "camera_from_info",
]
