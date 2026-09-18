"""Camera and depth geometry shared by adaptive support sampling."""
import numpy as np


class DensityError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def camera(record):
    # Legacy RGB-D fusion ignores trans/scale. Do not silently change its frame.
    if not np.allclose(record.get("trans", [0, 0, 0]), 0) or record.get("scale", 1) != 1:
        raise DensityError("adaptive support requires identity camera trans/scale (legacy fusion contract)")
    w, h = record["image_width"], record["image_height"]
    if min(w,h) < 2:
        raise DensityError("density support requires image dimensions >= 2")
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = np.asarray(record["R"]).T, record["T"]
    c2w = np.linalg.inv(w2c)
    fx, fy = w / (2 * np.tan(record["FoVx"] / 2)), h / (2 * np.tan(record["FoVy"] / 2))
    return dict(w=w, h=h, fx=fx, fy=fy, w2c=w2c, c2w=c2w, center=c2w[:3, 3])


def backproject(depth, cam):
    y, x = np.indices(depth.shape)
    pc = np.stack(((x - cam["w"] / 2) / cam["fx"] * depth,
                   (y - cam["h"] / 2) / cam["fy"] * depth, depth), axis=-1)
    return pc @ cam["c2w"][:3, :3].T + cam["center"]


def project(xyz, cam):
    pc = xyz @ cam["w2c"][:3, :3].T + cam["w2c"][:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / np.maximum(z[:, None], 1e-12) * [cam["fx"], cam["fy"]] + [cam["w"] / 2, cam["h"] / 2]
    valid = np.isfinite(uv).all(1) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"] - 1)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"] - 1)
    pix = np.rint(np.nan_to_num(uv, nan=0, posinf=0, neginf=0)).astype(np.int64)
    pix[:, 0] = np.clip(pix[:, 0], 0, cam["w"] - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, cam["h"] - 1)
    return uv, pix, z, valid
