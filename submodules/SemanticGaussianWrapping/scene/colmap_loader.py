"""Dependency-free readers for COLMAP sparse text and binary models.

The structures and binary layout mirror COLMAP's ``read_write_model.py`` but
the module deliberately contains only reading code needed by training.
"""

from __future__ import annotations

import collections
import os
import struct
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Tuple

import numpy as np


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = (
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
    CameraModel(11, "RAD_TAN_THIN_PRISM_FISHEYE", 16),
)
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {qvec.shape}")
    norm = np.linalg.norm(qvec)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("zero-length quaternion")
    w, x, y, z = qvec / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3,3), got {r.shape}")
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = r.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return -qvec if qvec[0] < 0 else qvec


class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


def read_next_bytes(
    fid: BinaryIO,
    num_bytes: int,
    format_char_sequence: str,
    endian_character: str = "<",
) -> Tuple:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"expected {num_bytes} bytes, received {len(data)}")
    return struct.unpack(endian_character + format_char_sequence, data)


def _data_lines(path: os.PathLike[str] | str) -> Iterable[str]:
    with open(path, "r", encoding="utf8") as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_intrinsics_text(path: os.PathLike[str] | str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    for line in _data_lines(path):
        elems = line.split()
        camera_id = int(elems[0])
        model = elems[1]
        if model not in CAMERA_MODEL_NAMES:
            raise ValueError(f"unknown COLMAP camera model {model!r}")
        expected = CAMERA_MODEL_NAMES[model].num_params
        params = np.asarray(elems[4:], dtype=np.float64)
        if len(params) != expected:
            raise ValueError(f"{model} expects {expected} parameters, received {len(params)}")
        cameras[camera_id] = Camera(
            id=camera_id,
            model=model,
            width=int(elems[2]),
            height=int(elems[3]),
            params=params,
        )
    return cameras


def read_intrinsics_binary(path: os.PathLike[str] | str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        count = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            model = CAMERA_MODEL_IDS[model_id]
            params = np.asarray(
                read_next_bytes(fid, 8 * model.num_params, "d" * model.num_params),
                dtype=np.float64,
            )
            cameras[camera_id] = Camera(camera_id, model.model_name, width, height, params)
    return cameras


def read_extrinsics_text(path: os.PathLike[str] | str) -> Dict[int, Image]:
    # Images occupy two data lines.  The second line may be empty; preserving
    # raw lines is therefore important instead of using ``_data_lines``.
    images: Dict[int, Image] = {}
    with open(path, "r", encoding="utf8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            if len(elems) < 10:
                raise ValueError(f"malformed COLMAP image line: {line!r}")
            image_id = int(elems[0])
            qvec = np.asarray(elems[1:5], dtype=np.float64)
            tvec = np.asarray(elems[5:8], dtype=np.float64)
            camera_id = int(elems[8])
            name = " ".join(elems[9:])
            points_line = fid.readline()
            if points_line is None:
                points_line = ""
            point_elems = points_line.split()
            if len(point_elems) % 3:
                raise ValueError(f"malformed 2D point line for image {image_id}")
            if point_elems:
                values = np.asarray(point_elems, dtype=np.float64).reshape(-1, 3)
                xys = values[:, :2]
                point_ids = values[:, 2].astype(np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point_ids = np.empty((0,), dtype=np.int64)
            images[image_id] = Image(
                image_id, qvec, tvec, camera_id, name, xys, point_ids
            )
    return images


def read_extrinsics_binary(path: os.PathLike[str] | str) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "rb") as fid:
        count = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(count):
            values = read_next_bytes(fid, 64, "idddddddi")
            image_id = values[0]
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = values[8]
            name_bytes = bytearray()
            while True:
                char = read_next_bytes(fid, 1, "c")[0]
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf8")
            num_points = read_next_bytes(fid, 8, "Q")[0]
            if num_points:
                point_values = read_next_bytes(fid, 24 * num_points, "ddq" * num_points)
                xys = np.column_stack((point_values[0::3], point_values[1::3])).astype(np.float64)
                point_ids = np.asarray(point_values[2::3], dtype=np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point_ids = np.empty((0,), dtype=np.int64)
            images[image_id] = Image(
                image_id, qvec, tvec, camera_id, name, xys, point_ids
            )
    return images


def read_points3D_text(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzs, rgbs, errors = [], [], []
    for line in _data_lines(path):
        elems = line.split()
        if len(elems) < 8:
            raise ValueError(f"malformed COLMAP point line: {line!r}")
        xyzs.append([float(value) for value in elems[1:4]])
        rgbs.append([int(value) for value in elems[4:7]])
        errors.append(float(elems[7]))
    return (
        np.asarray(xyzs, dtype=np.float32).reshape(-1, 3),
        np.asarray(rgbs, dtype=np.uint8).reshape(-1, 3),
        np.asarray(errors, dtype=np.float32).reshape(-1, 1),
    )


def read_points3D_binary(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzs, rgbs, errors = [], [], []
    with open(path, "rb") as fid:
        count = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(count):
            values = read_next_bytes(fid, 43, "QdddBBBd")
            xyzs.append(values[1:4])
            rgbs.append(values[4:7])
            errors.append(values[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            if track_length:
                read_next_bytes(fid, 8 * track_length, "ii" * track_length)
    return (
        np.asarray(xyzs, dtype=np.float32).reshape(-1, 3),
        np.asarray(rgbs, dtype=np.uint8).reshape(-1, 3),
        np.asarray(errors, dtype=np.float32).reshape(-1, 1),
    )


def read_model(
    path: os.PathLike[str] | str,
    extension: str | None = None,
) -> tuple[Dict[int, Camera], Dict[int, Image], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read cameras, images and sparse points from one COLMAP model folder."""

    root = Path(path)
    if extension is None:
        if (root / "cameras.bin").is_file():
            extension = ".bin"
        elif (root / "cameras.txt").is_file():
            extension = ".txt"
        else:
            raise FileNotFoundError(f"no COLMAP cameras.bin/cameras.txt under {root}")
    extension = extension if extension.startswith(".") else f".{extension}"
    if extension == ".bin":
        cameras = read_intrinsics_binary(root / "cameras.bin")
        images = read_extrinsics_binary(root / "images.bin")
        points = read_points3D_binary(root / "points3D.bin")
    elif extension == ".txt":
        cameras = read_intrinsics_text(root / "cameras.txt")
        images = read_extrinsics_text(root / "images.txt")
        points = read_points3D_text(root / "points3D.txt")
    else:
        raise ValueError(f"unsupported COLMAP extension {extension!r}")
    return cameras, images, points


__all__ = [
    "CAMERA_MODELS",
    "CAMERA_MODEL_IDS",
    "CAMERA_MODEL_NAMES",
    "Camera",
    "Image",
    "Point3D",
    "qvec2rotmat",
    "read_extrinsics_binary",
    "read_extrinsics_text",
    "read_intrinsics_binary",
    "read_intrinsics_text",
    "read_model",
    "read_next_bytes",
    "read_points3D_binary",
    "read_points3D_text",
    "rotmat2qvec",
]
