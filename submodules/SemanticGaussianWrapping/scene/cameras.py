"""Camera objects with pixel-aligned RGB and semantic observations."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def get_world_to_view(
    rotation_transposed: np.ndarray,
    translation: np.ndarray,
    translate: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    rt = np.zeros((4, 4), dtype=np.float32)
    rt[:3, :3] = np.asarray(rotation_transposed, dtype=np.float32).T
    rt[:3, 3] = np.asarray(translation, dtype=np.float32)
    rt[3, 3] = 1.0
    c2w = np.linalg.inv(rt)
    center = (c2w[:3, 3] + np.asarray(translate, dtype=np.float32)) * float(scale)
    c2w[:3, 3] = center
    return np.linalg.inv(c2w).astype(np.float32)


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> Tensor:
    tan_half_y = math.tan(fov_y / 2.0)
    tan_half_x = math.tan(fov_x / 2.0)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    projection = torch.zeros(4, 4, dtype=torch.float32)
    projection[0, 0] = 2.0 * znear / (right - left)
    projection[1, 1] = 2.0 * znear / (top - bottom)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[3, 2] = 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    return projection


def _resolve_device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return result


def _image_tensor(image: Any) -> tuple[Tensor, Optional[Tensor]]:
    if isinstance(image, Tensor):
        tensor = image.detach() if not image.is_floating_point() else image
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                pass
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
    else:
        array = np.asarray(image)
        if not array.flags.writeable:
            array = array.copy()
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3, 4):
        raise ValueError(f"image must be HxWxC or CxHxW, got {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and tensor.max() > 1.0:
        tensor = tensor / 255.0
    alpha = tensor[3:4] if tensor.shape[0] == 4 else None
    rgb = tensor[:3] if tensor.shape[0] >= 3 else tensor.repeat(3, 1, 1)
    return rgb.clamp(0.0, 1.0).contiguous(), alpha


def _map_tensor(value: Any, dtype: torch.dtype) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(np.asarray(value))
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError(f"pixel map must be two-dimensional, got {tuple(tensor.shape)}")
    return tensor.to(dtype=dtype)


def semantic_boundary_from_ids(ids: Tensor, ignore_label: int = -1) -> Tensor:
    """Return a one-pixel, four-connected instance boundary map."""

    if ids.ndim != 2:
        raise ValueError("semantic ids must have shape [H,W]")
    valid = ids != ignore_label
    boundary = torch.zeros_like(ids, dtype=torch.bool)
    horizontal = (ids[:, 1:] != ids[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (ids[1:, :] != ids[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary.float()


def resize_observations(
    image: Tensor,
    semantic_ids: Tensor,
    semantic_confidence: Tensor,
    semantic_boundary: Tensor,
    size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resize all observations together; ``size`` is ``(height, width)``."""

    image_r = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
    ids_r = F.interpolate(semantic_ids[None, None].float(), size=size, mode="nearest")[0, 0].long()
    confidence_r = F.interpolate(
        semantic_confidence[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0].clamp(0.0, 1.0)
    boundary_r = F.interpolate(
        semantic_boundary[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0].clamp(0.0, 1.0)
    return image_r, ids_r, confidence_r, boundary_r


class Camera(nn.Module):
    """A standard 3DGS camera with aligned semantic evidence."""

    def __init__(
        self,
        colmap_id: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image: Any,
        gt_alpha_mask: Any | None,
        image_name: str,
        uid: int,
        semantic_ids: Any | None = None,
        semantic_confidence: Any | None = None,
        semantic_boundary: Any | None = None,
        trans: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        data_device: str | torch.device = "cuda",
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        ignore_label: int = -1,
    ) -> None:
        super().__init__()
        device = _resolve_device(data_device)
        rgb, embedded_alpha = _image_tensor(image)
        height, width = rgb.shape[-2:]

        if semantic_ids is None:
            ids = torch.full((height, width), ignore_label, dtype=torch.long)
        else:
            ids = _map_tensor(semantic_ids, torch.long)
        if semantic_confidence is None:
            confidence = (ids != ignore_label).float()
        else:
            confidence = _map_tensor(semantic_confidence, torch.float32).clamp(0.0, 1.0)
        if semantic_boundary is None:
            boundary = semantic_boundary_from_ids(ids, ignore_label)
        else:
            boundary = _map_tensor(semantic_boundary, torch.float32).clamp(0.0, 1.0)

        if ids.shape != (height, width) or confidence.shape != (height, width) or boundary.shape != (height, width):
            _, ids, confidence, boundary = resize_observations(
                rgb, ids, confidence, boundary, (height, width)
            )
        confidence = confidence * (ids != ignore_label).float()

        alpha = gt_alpha_mask if gt_alpha_mask is not None else embedded_alpha
        if alpha is not None:
            alpha = _map_tensor(alpha, torch.float32)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            elif alpha.ndim != 3 or alpha.shape[0] != 1:
                raise ValueError("alpha mask must have shape [H,W] or [1,H,W]")
            if alpha.shape[-2:] != (height, width):
                alpha = F.interpolate(alpha[None], (height, width), mode="bilinear", align_corners=False)[0]
            alpha = alpha.clamp(0.0, 1.0)

        self.uid = int(uid)
        self.colmap_id = int(colmap_id)
        self.R = np.asarray(R, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32)
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.image_name = str(image_name)
        self.data_device = device
        self.image_width = width
        self.image_height = height
        self.znear = 0.01
        self.zfar = 100.0
        self.trans = np.asarray(trans, dtype=np.float32)
        self.scale = float(scale)
        self.Fx = float(fx if fx is not None else fov2focal(FoVx, width))
        self.Fy = float(fy if fy is not None else fov2focal(FoVy, height))
        self.Cx = float(cx if cx is not None else (width - 1) / 2.0)
        self.Cy = float(cy if cy is not None else (height - 1) / 2.0)
        self.ignore_label = int(ignore_label)

        self.register_buffer("original_image", rgb.to(device))
        self.register_buffer("semantic_ids", ids.to(device))
        self.register_buffer("semantic_confidence", confidence.to(device))
        self.register_buffer("semantic_boundary", boundary.to(device))
        self.register_buffer("gt_mask", None if alpha is None else alpha.to(device))

        world_view = torch.from_numpy(get_world_to_view(self.R, self.T, self.trans, self.scale)).T
        projection = get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).T
        self.register_buffer("world_view_transform", world_view.to(device))
        self.register_buffer("projection_matrix", projection.to(device))
        self.register_buffer("full_proj_transform", (world_view @ projection).to(device))
        self.register_buffer("camera_center", torch.linalg.inv(world_view)[3, :3].to(device))

    @property
    def gray_image(self) -> Tensor:
        weights = self.original_image.new_tensor([0.299, 0.587, 0.114])[:, None, None]
        return (self.original_image * weights).sum(dim=0, keepdim=True)

    @property
    def has_semantics(self) -> bool:
        return bool((self.semantic_confidence > 0).any().item())

    def resized(self, width: int, height: int) -> "Camera":
        if width <= 0 or height <= 0:
            raise ValueError("camera dimensions must be positive")
        image, ids, confidence, boundary = resize_observations(
            self.original_image,
            self.semantic_ids,
            self.semantic_confidence,
            self.semantic_boundary,
            (height, width),
        )
        sx, sy = width / self.image_width, height / self.image_height
        alpha = None
        if self.gt_mask is not None:
            alpha = F.interpolate(self.gt_mask[None], (height, width), mode="bilinear", align_corners=False)[0]
        return Camera(
            self.colmap_id,
            self.R,
            self.T,
            focal2fov(self.Fx * sx, width),
            focal2fov(self.Fy * sy, height),
            image,
            alpha,
            self.image_name,
            self.uid,
            ids,
            confidence,
            boundary,
            self.trans,
            self.scale,
            self.original_image.device,
            self.Fx * sx,
            self.Fy * sy,
            (self.Cx + 0.5) * sx - 0.5,
            (self.Cy + 0.5) * sy - 0.5,
            self.ignore_label,
        )

    def crop(self, left: int, top: int, width: int, height: int) -> "Camera":
        if left < 0 or top < 0 or left + width > self.image_width or top + height > self.image_height:
            raise ValueError("crop lies outside the camera image")
        slices = (slice(top, top + height), slice(left, left + width))
        alpha = None if self.gt_mask is None else self.gt_mask[:, slices[0], slices[1]]
        return Camera(
            self.colmap_id,
            self.R,
            self.T,
            focal2fov(self.Fx, width),
            focal2fov(self.Fy, height),
            self.original_image[:, slices[0], slices[1]],
            alpha,
            self.image_name,
            self.uid,
            self.semantic_ids[slices],
            self.semantic_confidence[slices],
            self.semantic_boundary[slices],
            self.trans,
            self.scale,
            self.original_image.device,
            self.Fx,
            self.Fy,
            self.Cx - left,
            self.Cy - top,
            self.ignore_label,
        )


class MiniCam:
    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: Tensor,
        full_proj_transform: Tensor,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.linalg.inv(world_view_transform)[3, :3]


__all__ = [
    "Camera",
    "MiniCam",
    "focal2fov",
    "fov2focal",
    "get_projection_matrix",
    "get_world_to_view",
    "resize_observations",
    "semantic_boundary_from_ids",
]
