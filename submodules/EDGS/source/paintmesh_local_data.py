"""Existing-Ply local parameters and exact PaintMesh virtual-view targets."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from torch import nn
import torch.nn.functional as F

from source.pgsr_geometry import camera_rays

REPO = Path(__file__).resolve().parents[3]


def make_views(path, device="cuda"):
    # Only reuse the camera contract, not Inpaint360GS scene/renderer modules.
    from source.vendor import bootstrap_gaussian_splatting
    bootstrap_gaussian_splatting()
    spec = importlib.util.spec_from_file_location(
        "paintmesh_local_cameras", REPO / "submodules/Inpaint360GS/utils/virtual_camera_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from local_geometry_io import identity, read_json
    normalized = module.load_virtual_camera_manifest(path)
    payload = read_json(path)
    if identity(normalized) != payload["artifact_id"]:
        raise ValueError("virtual camera content does not match its identity")
    first = normalized["cameras"][0]
    base = SimpleNamespace(world_view_transform=torch.eye(4, device=device),
                           image_width=first["image_width"], image_height=first["image_height"])
    return module.virtual_views_from_manifest(base, normalized)


def read_targets(lama, inputs, views):
    targets = []
    for view in views:
        stem = view.image_name
        outputs = lama["frames"][stem]["outputs"]
        with Image.open(outputs["color"]["path"]) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(inputs["frames"][stem]["outputs"]["color_mask"]["path"]) as im:
            mask = np.array(im.convert("L")) > 0
        with Image.open(outputs["normal_valid"]["path"]) as im:
            valid = np.array(im.convert("L")) > 0
        depth = np.load(outputs["depth"]["path"], allow_pickle=False)
        normal = np.load(outputs["normal"]["path"], allow_pickle=False)
        shape = (view.image_height, view.image_width)
        if (rgb.shape != (*shape, 3) or depth.shape != shape or mask.shape != shape or
                normal.shape != (*shape, 3) or valid.shape != shape or
                depth.dtype != np.float32 or normal.dtype != np.float32):
            raise ValueError(f"target shape/dtype differs from camera: {stem}")
        if not np.isfinite(depth).all() or not np.isfinite(normal).all() or (depth < 0).any():
            raise ValueError(f"nonfinite/negative targets: {stem}")
        length = np.linalg.norm(normal, axis=-1)
        if (not np.allclose(length[valid], 1, atol=1e-4, rtol=0) or
                np.any(normal[~valid] != 0) or not (mask & valid & (depth > 0)).any() or
                not (~mask).any()):
            raise ValueError(f"invalid normal/mask target: {stem}")
        rays = camera_rays(view, device="cpu").numpy()
        if np.any((normal * rays).sum(-1)[valid & mask] > 1e-5):
            raise ValueError(f"LaMa normal is not camera-facing: {stem}")
        targets.append({
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
            "depth": torch.from_numpy(depth),
            "normal": torch.from_numpy(normal).permute(2, 0, 1),
            "mask": torch.from_numpy(mask), "normal_valid": torch.from_numpy(valid),
        })
    return targets


class LocalGaussians(nn.Module):
    """PGSR Gaussian interface with optimizer parameters ONLY for editable rows.

    Frozen rows are buffers. Rendering index_copy is differentiable, while
    inactive rows cannot acquire gradients or Adam state. No topology changes.
    """
    groups = {"xyz": ("x", "y", "z"),
              "scaling": ("scale_0", "scale_1", "scale_2"),
              "rotation": ("rot_0", "rot_1", "rot_2", "rot_3")}

    def __init__(self, ply_path, editable, device="cuda"):
        super().__init__()
        self.ply_path = Path(ply_path)
        self.ply = PlyData.read(str(ply_path), mmap="c")
        vertex = self.ply["vertex"].data
        editable = np.asarray(editable)
        if editable.dtype != np.bool_ or editable.shape != (len(vertex),) or not editable.any():
            raise ValueError("editable mask must select rows of the input PLY")
        self.editable = editable
        self.register_buffer("indices", torch.as_tensor(np.flatnonzero(editable), device=device))
        for key, fields in self.groups.items():
            array = np.stack([vertex[field] for field in fields], axis=1).astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"invalid PLY {key}")
            base = torch.from_numpy(array).to(device)
            self.register_buffer(key + "_base", base)
            setattr(self, key, nn.Parameter(base[self.indices].clone()))
        if (torch.linalg.vector_norm(self.rotation_base, dim=1) < 1e-6).any():
            raise ValueError("PLY has degenerate quaternion")
        dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None, :]
        rest_names = sorted((n for n in vertex.dtype.names if n.startswith("f_rest_")),
                            key=lambda n: int(n.rsplit("_", 1)[1]))
        degree = math.isqrt(len(rest_names) // 3 + 1) - 1
        if len(rest_names) != 3 * ((degree + 1) ** 2 - 1):
            raise ValueError("invalid SH coefficient count")
        rest = (np.stack([vertex[n] for n in rest_names], axis=1)
                .reshape(len(vertex), 3, -1).transpose(0, 2, 1)
                if rest_names else np.empty((len(vertex), 0, 3), dtype=np.float32))
        features = np.concatenate((dc, rest), axis=1).astype(np.float32)
        opacity = np.array(vertex["opacity"], dtype=np.float32)[:, None]
        if not np.isfinite(features).all() or not np.isfinite(opacity).all():
            raise ValueError("invalid frozen SH/opacity")
        self.register_buffer("features", torch.from_numpy(features).to(device))
        self.register_buffer("opacity", torch.from_numpy(opacity).to(device).sigmoid())
        self.active_sh_degree = self.max_sh_degree = degree

    def assembled(self, key):
        return getattr(self, key + "_base").index_copy(0, self.indices, getattr(self, key))

    @property
    def get_xyz(self):
        return self.assembled("xyz")

    @property
    def get_scaling(self):
        return self.assembled("scaling").exp()

    @property
    def get_rotation(self):
        return F.normalize(self.assembled("rotation"), dim=-1)

    @property
    def get_features(self):
        return self.features

    @property
    def get_opacity(self):
        return self.opacity

    def optimizer(self, config):
        return torch.optim.Adam([
            {"params": [self.xyz], "lr": config["position_lr"], "name": "xyz"},
            {"params": [self.scaling], "lr": config["scaling_lr"], "name": "scaling"},
            {"params": [self.rotation], "lr": config["rotation_lr"], "name": "rotation"},
        ], eps=1e-15)

    def local_state(self):
        return {key: getattr(self, key).detach().clone() for key in self.groups}

    @torch.no_grad()
    def restore_local(self, state):
        for key in self.groups:
            value = state[key].to(getattr(self, key))
            if value.shape != getattr(self, key).shape or not torch.isfinite(value).all():
                raise ValueError("invalid local checkpoint parameters")
            getattr(self, key).copy_(value)

    @torch.no_grad()
    def restore_rows(self, rejected):
        for key in self.groups:
            getattr(self, key)[rejected] = getattr(self, key + "_base")[self.indices[rejected]]

    @torch.no_grad()
    def save(self, path):
        from local_geometry_io import atomic_write, check_preserved
        # mmap='c' is copy-on-write: the source file is never modified.
        for key, fields in self.groups.items():
            values = getattr(self, key).detach().cpu().numpy()
            for column, field in enumerate(fields):
                self.ply["vertex"].data[field][self.editable] = values[:, column]
        atomic_write(Path(path), lambda stream: self.ply.write(stream))
        check_preserved(self.ply_path, path, self.editable)
