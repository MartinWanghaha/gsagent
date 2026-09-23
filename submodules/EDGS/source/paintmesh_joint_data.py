"""Optional targets and full editable-Gaussian parameters for the peer path."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from source.paintmesh_local_data import LocalGaussians, make_views
from source.pgsr_geometry import camera_rays


def reliability(value, valid):
    """Frozen within-modality continuity weights, NOT calibrated confidence.

    Preserve a nonzero floor at actual edges; never import another modality's
    validity. Robust training residuals remain the primary outlier protection.
    """
    value = value.float()
    if value.ndim == 2:
        value = value[None]
    penalty = torch.zeros_like(valid, dtype=torch.float32)
    count = torch.zeros_like(penalty)
    for axis in (-1, -2):
        delta = torch.diff(value, dim=axis).abs().mean(0)
        a, b = (valid[:, 1:], valid[:, :-1]) if axis == -1 else (valid[1:], valid[:-1])
        accepted = a & b
        residual = delta[accepted]
        scale = residual.median().clamp_min(1e-3) if residual.numel() else delta.new_tensor(1.)
        score = (delta / scale).clamp_max(20) * accepted
        if axis == -1:
            penalty[:, 1:] += score; penalty[:, :-1] += score
            count[:, 1:] += accepted; count[:, :-1] += accepted
        else:
            penalty[1:] += score; penalty[:-1] += score
            count[1:] += accepted; count[:-1] += accepted
    return ((.1 + .9 / (1 + penalty / count.clamp_min(1))) * valid).detach()


def read_targets(lama, inputs, views, *, use_depth=False, use_normal=False):
    result = []
    for view in views:
        stem = view.image_name
        outputs = lama["frames"][stem]["outputs"]
        with Image.open(outputs["color"]["path"]) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.float32) / 255
        with Image.open(inputs["frames"][stem]["outputs"]["color_mask"]["path"]) as im:
            mask = np.array(im.convert("L")) > 0
        shape = (view.image_height, view.image_width)
        if rgb.shape != (*shape, 3) or mask.shape != shape:
            raise ValueError(f"RGB/mask/camera shape mismatch: {stem}")
        target = dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1), mask=torch.from_numpy(mask))
        target["rgb_weight"] = reliability(target["rgb"], torch.ones_like(target["mask"]))
        if use_depth:
            depth = np.load(outputs["depth"]["path"], allow_pickle=False)
            if depth.dtype != np.float32 or depth.shape != shape or not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError(f"invalid depth: {stem}")
            target["depth"] = torch.from_numpy(depth)
            target["depth_weight"] = reliability(target["depth"].clamp_min(1e-6).log(), target["depth"] > 0)
        if use_normal:
            normal = np.load(outputs["normal"]["path"], allow_pickle=False)
            with Image.open(outputs["normal_valid"]["path"]) as im:
                valid = np.array(im.convert("L")) > 0
            if normal.dtype != np.float32 or normal.shape != (*shape, 3) or valid.shape != shape or not np.isfinite(normal).all():
                raise ValueError(f"invalid normal: {stem}")
            if not np.allclose(np.linalg.norm(normal, axis=-1)[valid], 1, atol=1e-4) or np.any(normal[~valid] != 0):
                raise ValueError(f"invalid unit-normal/validity: {stem}")
            if np.any((normal * camera_rays(view, device="cpu").numpy()).sum(-1)[valid & mask] > 1e-5):
                raise ValueError(f"normal is not camera-facing: {stem}")
            target.update(normal=torch.from_numpy(normal).permute(2, 0, 1), normal_valid=torch.from_numpy(valid))
            target["normal_weight"] = reliability(target["normal"], target["normal_valid"])
        result.append(target)
    if not any(t["mask"].any() for t in result):
        raise ValueError("no hole pixels in virtual cameras")
    for name in ("depth", "normal"):
        if (use_depth if name == "depth" else use_normal):
            if not any((t["mask"] & ((t["depth"] > 0) if name == "depth" else t["normal_valid"])).any() for t in result):
                raise ValueError(f"no valid hole {name} targets")
    return result


class JointGaussians(LocalGaussians):
    """Only new rows acquire Adam state, including color and opacity."""
    def __init__(self, ply_path, editable, device="cuda"):
        super().__init__(ply_path, editable, device)
        features = self.features.detach().clone()
        del self.features
        self.register_buffer("features_base", features)
        self.features = nn.Parameter(features[self.indices].clone())
        # Read logits without a lossy sigmoid -> logit round trip.
        opacity = torch.from_numpy(np.array(self.ply["vertex"].data["opacity"], dtype=np.float32)[:, None]).to(device)
        del self.opacity
        self.register_buffer("opacity_base", opacity)
        self.opacity = nn.Parameter(opacity[self.indices].clone())

    @property
    def get_features(self):
        return self.assembled("features")

    @property
    def get_opacity(self):
        return self.assembled("opacity").sigmoid()

    def optimizer(self, cfg):
        keys = {"xyz": "position_lr", "scaling": "scaling_lr", "rotation": "rotation_lr",
                "features": "feature_lr", "opacity": "opacity_lr"}
        return torch.optim.Adam([{"params": [getattr(self, k)], "lr": cfg[v], "name": k}
                                 for k, v in keys.items()], eps=1e-15)

    def local_state(self):
        return {k: getattr(self, k).detach().clone() for k in (*self.groups, "features", "opacity")}

    @torch.no_grad()
    def restore_local(self, state):
        if set(state) != set(self.local_state()):
            raise ValueError("invalid joint checkpoint fields")
        for key, value in state.items():
            parameter = getattr(self, key)
            if parameter.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError("invalid joint checkpoint parameters")
            parameter.copy_(value.to(parameter))

    @torch.no_grad()
    def save(self, path):
        from edgs_inpaint_io import atomic_write, check_preserved
        vertex = self.ply["vertex"].data
        for key, fields in self.groups.items():
            values = getattr(self, key).cpu().numpy()
            for i, field in enumerate(fields):
                vertex[field][self.editable] = values[:, i]
        features = self.features.cpu().numpy()
        for i in range(3):
            vertex[f"f_dc_{i}"][self.editable] = features[:, 0, i]
        rest = features[:, 1:].transpose(0, 2, 1).reshape(len(features), -1)
        for i in range(rest.shape[1]):
            vertex[f"f_rest_{i}"][self.editable] = rest[:, i]
        vertex["opacity"][self.editable] = self.opacity[:, 0].cpu().numpy()
        atomic_write(Path(path), lambda f: self.ply.write(f))
        check_preserved(self.ply_path, path, self.editable)
