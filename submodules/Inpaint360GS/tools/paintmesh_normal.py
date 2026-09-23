"""CPU-only normal completion contract shared by preparation and LaMa.

Raw normals are camera-space float32 XYZ vectors, not visualization RGB.
There is deliberately no normal completion mode or enable switch.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ENCODING = "xyz_to_rgb_affine_v1"
METHOD = "lama"
EPS = 1e-6


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_mask(path):
    with Image.open(path) as image:
        if image.mode not in {"L", "1", "P"}:
            raise ValueError(f"normal mask must contain label indices: {path}")
        return np.asarray(image) != 0


def read_normal(path, validity_path):
    normal = np.load(path, allow_pickle=False)
    valid = read_mask(validity_path)
    if normal.dtype != np.float32 or normal.ndim != 3 or normal.shape[-1] != 3:
        raise ValueError(f"normal must be float32 HxWx3: {path}")
    if min(normal.shape[:2]) <= 0 or valid.shape != normal.shape[:2]:
        raise ValueError(f"normal/valid shape mismatch: {path}")
    if not np.isfinite(normal).all():
        raise ValueError(f"normal contains NaN or infinity: {path}")
    if not np.allclose(np.linalg.norm(normal[valid], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError(f"valid normal must have unit length: {path}")
    if np.any(normal[~valid] != 0):
        raise ValueError(f"invalid normal must be zero: {path}")
    return normal, valid


def read_cameras(path, stems):
    from utils.virtual_camera_manifest import load_virtual_camera_manifest
    load_virtual_camera_manifest(path, expected_frame_count=len(stems))
    payload = json.loads(Path(path).read_text())
    keys = ("schema_version", "kind", "frame_count", "iteration", "circle_radius", "cameras")
    if payload.get("schema_version") == 2:
        keys += ("trajectory",)
    if (
        payload.get("kind") != "inpaint360gs-virtual-cameras"
        or payload.get("schema_version") not in (1, 2)
        or payload.get("complete") is not True
        or payload.get("artifact_id") != identity({k: payload[k] for k in keys})
    ):
        raise ValueError("invalid normal camera manifest or identity")
    records = payload["cameras"]
    if payload["frame_count"] != len(stems) or [c["image_name"] for c in records] != list(stems):
        raise ValueError("normal camera frame set mismatch")
    for camera in records:
        for key in ("image_height", "image_width"):
            if type(camera[key]) is not int or camera[key] <= 0:
                raise ValueError("invalid normal camera dimensions")
        for key in ("FoVx", "FoVy"):
            if not math.isfinite(camera[key]) or not 0 < camera[key] < math.pi:
                raise ValueError("invalid normal camera FoV")
    return payload, {c["image_name"]: c for c in records}


def camera_rays(camera, shape):
    h, w = shape
    if (camera["image_height"], camera["image_width"]) != (h, w):
        raise ValueError("normal shape differs from camera")
    fx = w / (2 * math.tan(camera["FoVx"] / 2))
    fy = h / (2 * math.tan(camera["FoVy"] / 2))
    yy, xx = np.mgrid[:h, :w]
    return np.stack(((xx - w / 2) / fx, (yy - h / 2) / fy, np.ones((h, w))), -1).astype(np.float32)


def inference_mask(hole, valid):
    if hole.shape != valid.shape or not hole.any() or hole.all():
        raise ValueError("normal hole mask must match dimensions and be nonempty/nonfull")
    if not (valid & ~hole).any():
        raise ValueError("normal completion has no valid context outside the hole")
    return hole | ~valid


def compose_normal(source, valid, hole, prediction, camera):
    """Decode all three channels; modify ONLY the common hole mask."""
    inference_mask(hole, valid)
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != source.shape:
        raise ValueError("LaMa normal prediction dimensions differ from input")
    finite = np.isfinite(prediction).all(-1)
    vector = 2 * np.clip(np.where(finite[..., None], prediction, 0.5), 0, 1) - 1
    length = np.linalg.norm(vector, axis=-1)
    predicted_valid = finite & (length > EPS)
    unit = np.zeros_like(vector)
    unit[predicted_valid] = vector[predicted_valid] / length[predicted_valid, None]
    rays = camera_rays(camera, hole.shape)
    unit[np.sum(unit * rays, axis=-1) > 0] *= -1
    if not (predicted_valid & hole).any():
        raise ValueError("LaMa produced no valid normals inside the hole")
    completed = source.copy()
    completed[hole] = unit[hole]
    completed_valid = valid.copy()
    completed_valid[hole] = predicted_valid[hole]
    return completed, completed_valid


def normal_preview(normal, valid):
    return np.where(valid[..., None], np.clip((normal + 1) * 127.5, 0, 255), 0).astype(np.uint8)


class NormalInpaintingDataset:
    """Independent float32 loader; never invokes LaMa's .npy depth loader."""

    def __init__(self, root, stems, pad_out_to_modulo=8):
        self.root = Path(root)
        self.stems = list(stems)
        self.modulo = pad_out_to_modulo
        if self.modulo <= 0:
            raise ValueError("normal padding modulo must be positive")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        normal, valid = read_normal(self.root / f"{stem}.npy", self.root / "valid" / f"{stem}.png")
        hole = read_mask(self.root / f"{stem}_mask.png")
        mask = inference_mask(hole, valid)
        if not np.array_equal(mask, read_mask(self.root / "inference_mask" / f"{stem}.png")):
            raise ValueError(f"normal inference mask mismatch: {stem}")
        encoded = np.moveaxis((normal + 1) / 2, -1, 0)
        h, w = normal.shape[:2]
        padding = ((0, 0), (0, (-h) % self.modulo), (0, (-w) % self.modulo))
        return {
            "image": np.pad(encoded, padding, mode="symmetric"),
            "mask": np.pad(mask[None].astype(np.float32), padding, mode="symmetric"),
            "unpad_to_size": (h, w),
        }
