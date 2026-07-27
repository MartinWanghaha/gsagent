"""Dataset readers for COLMAP and Blender scenes with semantic evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:  # Standard 3DGS execution adds the project directory to sys.path.
    from semantic.gaga_adapter import GagaObservationAdapter
except ImportError:  # Package-style imports used by unit tests.
    from ..semantic.gaga_adapter import GagaObservationAdapter

from .colmap_loader import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)


@dataclass(frozen=True)
class BasicPointCloud:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True)
class CameraPayload:
    """Pixel observations materialized at one requested training resolution."""

    image: Any
    alpha: Any | None = None
    semantic_ids: Any | None = None
    semantic_confidence: Any | None = None
    semantic_boundary: Any | None = None


@dataclass(frozen=True)
class CameraInfo:
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Any | None
    image_path: str
    image_name: str
    width: int
    height: int
    alpha: np.ndarray | None = None
    semantic_ids: np.ndarray | None = None
    semantic_confidence: np.ndarray | None = None
    semantic_boundary: np.ndarray | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    # Training scenes keep only camera metadata here.  The loader decodes one
    # RGB/semantic observation directly at the requested resolution when the
    # final Camera is built, so native-resolution pixels never accumulate in
    # the SceneInfo list.
    payload_loader: Callable[[tuple[int, int]], CameraPayload] | None = None

    @property
    def objects(self) -> np.ndarray | None:
        """Gaga-compatible alias for instance labels."""

        return self.semantic_ids


@dataclass(frozen=True)
class SceneInfo:
    point_cloud: BasicPointCloud | None
    train_cameras: list[CameraInfo]
    test_cameras: list[CameraInfo]
    nerf_normalization: dict[str, Any]
    ply_path: str | None
    num_semantic_classes: int = 1
    semantic_label_mapping: dict[str, dict[int, int]] | None = None
    semantic_metadata_provider: (
        Callable[[], tuple[int, dict[str, dict[int, int]]]] | None
    ) = None


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def _load_camera_payload(
    image_path: str | Path,
    semantic_name: str,
    semantic_adapter: GagaObservationAdapter | None,
    target_size: tuple[int, int],
    *,
    always_alpha: bool,
) -> CameraPayload:
    """Decode and resize one view without retaining native-resolution data.

    RGB and alpha use the same PyTorch bilinear interpolation convention as
    :meth:`Camera.resized`; Gaga IDs remain nearest-neighbour while confidence
    and boundary maps remain bilinear.  This preserves the old training
    objective while reducing peak memory from O(number of full-resolution
    views) to O(one full-resolution view + final cameras).
    """

    target_height, target_width = map(int, target_size)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("camera target dimensions must be positive")
    with Image.open(image_path) as source:
        source_width, source_height = source.size
        rgb_array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        image = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div_(255.0)
        alpha = None
        if always_alpha or "A" in source.getbands():
            alpha_array = np.asarray(
                source.convert("RGBA").getchannel("A"), dtype=np.float32
            ).copy()
            alpha = torch.from_numpy(alpha_array).div_(255.0)
    if (source_height, source_width) != (target_height, target_width):
        image = F.interpolate(
            image[None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        if alpha is not None:
            alpha = F.interpolate(
                alpha[None, None],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

    observation = (
        None
        if semantic_adapter is None
        else semantic_adapter.load(
            semantic_name,
            target_size=(target_height, target_width),
        )
    )
    return CameraPayload(
        image=image.contiguous(),
        alpha=None if alpha is None else alpha.contiguous(),
        semantic_ids=None if observation is None else observation.ids,
        semantic_confidence=None if observation is None else observation.confidence,
        semantic_boundary=None if observation is None else observation.boundary,
    )


def getNerfppNorm(cam_info: Sequence[CameraInfo]) -> dict[str, Any]:
    if not cam_info:
        return {"translate": np.zeros(3, dtype=np.float32), "radius": 1.0}
    centers = []
    for camera in cam_info:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = camera.R.T
        w2c[:3, 3] = camera.T
        centers.append(np.linalg.inv(w2c)[:3, 3])
    centers_array = np.stack(centers, axis=0)
    center = centers_array.mean(axis=0)
    diagonal = np.linalg.norm(centers_array - center, axis=1).max(initial=0.0)
    return {"translate": -center.astype(np.float32), "radius": max(float(diagonal) * 1.1, 1e-6)}


def _intrinsics(camera: ColmapCamera) -> tuple[float, float, float, float]:
    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = map(float, params[:4])
    else:
        raise ValueError(
            f"COLMAP camera model {camera.model!r} still contains lens distortion. "
            "Run COLMAP image_undistorter (or convert.py) so cameras are PINHOLE/SIMPLE_PINHOLE."
        )
    return fx, fy, cx, cy


def readColmapCameras(
    cam_extrinsics: dict[int, ColmapImage],
    cam_intrinsics: dict[int, ColmapCamera],
    images_folder: str | Path,
    semantic_adapter: GagaObservationAdapter | None = None,
    defer_camera_loading: bool = False,
) -> list[CameraInfo]:
    camera_infos = []
    images_folder = Path(images_folder)
    for uid, key in enumerate(sorted(cam_extrinsics)):
        extrinsic = cam_extrinsics[key]
        intrinsic = cam_intrinsics[extrinsic.camera_id]
        image_path = images_folder / Path(extrinsic.name).name
        if not image_path.is_file():
            raise FileNotFoundError(f"COLMAP image does not exist: {image_path}")
        payload_loader = None
        if defer_camera_loading:
            with Image.open(image_path) as source:
                width, height = source.size
            image = None
            alpha = None
            observation = None
            payload_loader = partial(
                _load_camera_payload,
                image_path,
                extrinsic.name,
                semantic_adapter,
                always_alpha=False,
            )
        else:
            with Image.open(image_path) as source:
                if "A" in source.getbands():
                    rgba = source.convert("RGBA")
                    image = rgba.convert("RGB").copy()
                    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
                else:
                    image = source.convert("RGB").copy()
                    alpha = None
            width, height = image.size
            observation = None
            if semantic_adapter is not None:
                observation = semantic_adapter.load(
                    extrinsic.name,
                    target_size=(height, width),
                )
        fx, fy, cx, cy = _intrinsics(intrinsic)
        sx, sy = width / float(intrinsic.width), height / float(intrinsic.height)
        fx, fy = fx * sx, fy * sy
        cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        camera_infos.append(
            CameraInfo(
                uid=uid,
                R=qvec2rotmat(extrinsic.qvec).T.astype(np.float32),
                T=np.asarray(extrinsic.tvec, dtype=np.float32),
                FovY=focal2fov(fy, height),
                FovX=focal2fov(fx, width),
                image=image,
                image_path=str(image_path),
                image_name=Path(extrinsic.name).stem,
                width=width,
                height=height,
                alpha=alpha,
                semantic_ids=_numpy(observation.ids) if observation else None,
                semantic_confidence=_numpy(observation.confidence) if observation else None,
                semantic_boundary=_numpy(observation.boundary) if observation else None,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                payload_loader=payload_loader,
            )
        )
    return camera_infos


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _find_sparse_folder(path: Path) -> Path:
    candidates = (path / "sparse" / "0", path / "sparse")
    for candidate in candidates:
        if (candidate / "cameras.bin").is_file() or (candidate / "cameras.txt").is_file():
            return candidate
    raise FileNotFoundError(f"no COLMAP sparse model found in {path}")


def _read_sparse_model(folder: Path):
    if (folder / "cameras.bin").is_file():
        cameras = read_intrinsics_binary(folder / "cameras.bin")
        images = read_extrinsics_binary(folder / "images.bin")
    else:
        cameras = read_intrinsics_text(folder / "cameras.txt")
        images = read_extrinsics_text(folder / "images.txt")
    if (folder / "points3D.bin").is_file():
        points = read_points3D_binary(folder / "points3D.bin")
    elif (folder / "points3D.txt").is_file():
        points = read_points3D_text(folder / "points3D.txt")
    else:
        points = (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 1), dtype=np.float32),
        )
    return cameras, images, points


def _split_cameras(cameras: list[CameraInfo], evaluate: bool, llffhold: int) -> tuple[list[CameraInfo], list[CameraInfo]]:
    if not evaluate:
        return cameras, []
    if llffhold <= 0:
        raise ValueError("llffhold must be positive")
    train = [camera for index, camera in enumerate(cameras) if index % llffhold != 0]
    test = [camera for index, camera in enumerate(cameras) if index % llffhold == 0]
    return train, test


def readColmapSceneInfo(
    path: str | Path,
    images: str | None = None,
    eval: bool = False,
    semantic_path: str | None = "sam_mask",
    semantic_confidence_path: str | None = None,
    semantic_boundary_path: str | None = None,
    semantic_ignore_label: int = -1,
    semantic_background_label: int = 0,
    boundary_width: int = 1,
    llffhold: int = 8,
    defer_camera_loading: bool = False,
    **_: Any,
) -> SceneInfo:
    root = Path(path)
    sparse = _find_sparse_folder(root)
    intrinsics, extrinsics, (xyz, rgb, _errors) = _read_sparse_model(sparse)
    adapter = None
    if semantic_path:
        adapter = GagaObservationAdapter(
            root,
            semantic_path,
            semantic_confidence_path,
            semantic_boundary_path,
            semantic_ignore_label,
            semantic_background_label,
            boundary_width,
        )
    image_dir = root / (images or "images")
    camera_infos = sorted(
        readColmapCameras(
            extrinsics,
            intrinsics,
            image_dir,
            adapter,
            defer_camera_loading=defer_camera_loading,
        ),
        key=lambda camera: camera.image_name,
    )
    train, test = _split_cameras(camera_infos, eval, llffhold)
    point_cloud = BasicPointCloud(
        points=xyz.astype(np.float32),
        colors=rgb.astype(np.float32) / 255.0,
        normals=np.zeros_like(xyz, dtype=np.float32),
    )
    ply_path = sparse / "points3D.ply"
    return SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train,
        test_cameras=test,
        nerf_normalization=getNerfppNorm(train or camera_infos),
        ply_path=str(ply_path) if ply_path.is_file() else None,
        num_semantic_classes=adapter.num_classes if adapter is not None else 1,
        semantic_label_mapping=adapter.export_label_mapping() if adapter is not None else None,
        semantic_metadata_provider=(
            None
            if adapter is None
            else lambda adapter=adapter: (
                adapter.num_classes,
                adapter.export_label_mapping(),
            )
        ),
    )


def readCamerasFromTransforms(
    path: str | Path,
    transformsfile: str,
    white_background: bool,
    extension: str = ".png",
    semantic_adapter: GagaObservationAdapter | None = None,
    defer_camera_loading: bool = False,
) -> list[CameraInfo]:
    root = Path(path)
    with open(root / transformsfile, "r", encoding="utf8") as stream:
        contents = json.load(stream)
    fov_x = float(contents["camera_angle_x"])
    cameras = []
    for uid, frame in enumerate(contents["frames"]):
        relative = frame["file_path"]
        image_path = root / relative
        if not image_path.suffix:
            image_path = image_path.with_suffix(extension)
        payload_loader = None
        if defer_camera_loading:
            with Image.open(image_path) as source:
                width, height = source.size
            image = None
            alpha = None
            observation = None
            payload_loader = partial(
                _load_camera_payload,
                image_path,
                image_path.name,
                semantic_adapter,
                always_alpha=True,
            )
        else:
            with Image.open(image_path) as source:
                rgba = source.convert("RGBA")
                array = np.asarray(rgba, dtype=np.float32) / 255.0
            # Keep foreground RGB and alpha separate.  Training may choose a
            # random background per iteration, so reader-time compositing is
            # irreversible.
            image = Image.fromarray(
                np.round(array[..., :3] * 255.0).astype(np.uint8)
            )
            alpha = np.ascontiguousarray(array[..., 3], dtype=np.float32)
            width, height = image.size
            observation = (
                semantic_adapter.load(image_path.name, (height, width))
                if semantic_adapter
                else None
            )
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        fy = fov2focal(fov_x, width)
        fov_y = focal2fov(fy, height)
        cameras.append(
            CameraInfo(
                uid=uid,
                R=w2c[:3, :3].T.astype(np.float32),
                T=w2c[:3, 3].astype(np.float32),
                FovY=fov_y,
                FovX=fov_x,
                image=image,
                image_path=str(image_path),
                image_name=image_path.stem,
                width=width,
                height=height,
                alpha=alpha,
                semantic_ids=_numpy(observation.ids) if observation else None,
                semantic_confidence=_numpy(observation.confidence) if observation else None,
                semantic_boundary=_numpy(observation.boundary) if observation else None,
                fx=fov2focal(fov_x, width),
                fy=fov2focal(fov_y, height),
                cx=(width - 1) / 2.0,
                cy=(height - 1) / 2.0,
                payload_loader=payload_loader,
            )
        )
    return cameras


def readNerfSyntheticInfo(
    path: str | Path,
    white_background: bool,
    eval: bool,
    extension: str = ".png",
    semantic_path: str | None = "sam_mask",
    semantic_confidence_path: str | None = None,
    semantic_boundary_path: str | None = None,
    semantic_ignore_label: int = -1,
    semantic_background_label: int = 0,
    boundary_width: int = 1,
    random_points: int = 100_000,
    defer_camera_loading: bool = False,
    **_: Any,
) -> SceneInfo:
    root = Path(path)
    adapter = None
    if semantic_path:
        adapter = GagaObservationAdapter(
            root,
            semantic_path,
            semantic_confidence_path,
            semantic_boundary_path,
            semantic_ignore_label,
            semantic_background_label,
            boundary_width,
        )
    train = readCamerasFromTransforms(
        root,
        "transforms_train.json",
        white_background,
        extension,
        adapter,
        defer_camera_loading=defer_camera_loading,
    )
    test_file = root / "transforms_test.json"
    test = (
        readCamerasFromTransforms(
            root,
            "transforms_test.json",
            white_background,
            extension,
            adapter,
            defer_camera_loading=defer_camera_loading,
        )
        if test_file.is_file()
        else []
    )
    if not eval:
        train.extend(test)
        test = []
    generator = np.random.default_rng(0)
    xyz = generator.random((random_points, 3), dtype=np.float32) * 2.6 - 1.3
    colors = generator.random((random_points, 3), dtype=np.float32)
    cloud = BasicPointCloud(xyz, colors, np.zeros_like(xyz))
    return SceneInfo(
        cloud,
        train,
        test,
        getNerfppNorm(train),
        None,
        adapter.num_classes if adapter is not None else 1,
        adapter.export_label_mapping() if adapter is not None else None,
        (
            None
            if adapter is None
            else lambda adapter=adapter: (
                adapter.num_classes,
                adapter.export_label_mapping(),
            )
        ),
    )


def fetchPly(path: str | Path) -> BasicPointCloud:
    """Read the vertex subset of an ASCII PLY without a heavy dependency."""

    with open(path, "r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("not a PLY file")
        properties, count, fmt = [], None, None
        while True:
            line = stream.readline().strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property ") and count is not None:
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        if fmt != "ascii":
            raise ValueError("fetchPly's dependency-free fallback supports ASCII PLY only")
        rows = [stream.readline().split() for _ in range(count or 0)]
    values = np.asarray(rows, dtype=np.float32)
    lookup = {name: index for index, name in enumerate(properties)}
    points = values[:, [lookup[axis] for axis in ("x", "y", "z")]]
    if all(name in lookup for name in ("red", "green", "blue")):
        colors = values[:, [lookup[name] for name in ("red", "green", "blue")]] / 255.0
    else:
        colors = np.zeros_like(points)
    normals = (
        values[:, [lookup[name] for name in ("nx", "ny", "nz")]]
        if all(name in lookup for name in ("nx", "ny", "nz"))
        else np.zeros_like(points)
    )
    return BasicPointCloud(points, colors, normals)


def storePly(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            stream.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            stream.write(f"property uchar {name}\n")
        stream.write("end_header\n")
        for point, color in zip(xyz, rgb):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} 0 0 0 "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


sceneLoadTypeCallbacks: dict[str, Callable[..., SceneInfo]] = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}


__all__ = [
    "BasicPointCloud",
    "CameraInfo",
    "CameraPayload",
    "SceneInfo",
    "fetchPly",
    "getNerfppNorm",
    "readCamerasFromTransforms",
    "readColmapCameras",
    "readColmapSceneInfo",
    "readNerfSyntheticInfo",
    "sceneLoadTypeCallbacks",
    "storePly",
]
