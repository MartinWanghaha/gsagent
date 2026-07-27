"""Minimal calibrated camera contract used by offline mesh extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _float_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.to(dtype=torch.float32).contiguous()


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class MeshCamera:
    """Camera calibration and transforms without image ownership."""

    uid: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_view_transform: Tensor
    full_proj_transform: Tensor
    camera_center: Tensor
    image_name: str = ""
    gt_mask: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("camera dimensions must be positive")
        intrinsics = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(float(value)) for value in intrinsics):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

        view = _float_tensor(self.world_view_transform)
        projection = _float_tensor(self.full_proj_transform, device=view.device)
        center = _float_tensor(self.camera_center, device=view.device)
        if view.shape != (4, 4):
            raise ValueError("world_view_transform must have shape [4,4]")
        if projection.shape != (4, 4):
            raise ValueError("full_proj_transform must have shape [4,4]")
        if center.shape != (3,):
            raise ValueError("camera_center must have shape [3]")
        _require_finite(view, "world_view_transform")
        _require_finite(projection, "full_proj_transform")
        _require_finite(center, "camera_center")
        object.__setattr__(self, "world_view_transform", view)
        object.__setattr__(self, "full_proj_transform", projection)
        object.__setattr__(self, "camera_center", center)
        object.__setattr__(self, "image_name", str(self.image_name))

        if self.gt_mask is not None:
            mask = _float_tensor(self.gt_mask, device=view.device)
            if mask.ndim == 2:
                mask = mask[None]
            if mask.shape != (1, self.height, self.width):
                raise ValueError("gt_mask must have shape [1,H,W]")
            _require_finite(mask, "gt_mask")
            object.__setattr__(self, "gt_mask", mask.clamp(0.0, 1.0))

    @classmethod
    def from_camera(
        cls,
        camera: Any,
        *,
        device: str | torch.device | None = None,
    ) -> "MeshCamera":
        target = (
            torch.as_tensor(camera.world_view_transform).device
            if device is None
            else torch.device(device)
        )
        width = int(camera.image_width)
        height = int(camera.image_height)
        fx = float(
            getattr(
                camera,
                "Fx",
                0.5 * width / math.tan(0.5 * float(camera.FoVx)),
            )
        )
        fy = float(
            getattr(
                camera,
                "Fy",
                0.5 * height / math.tan(0.5 * float(camera.FoVy)),
            )
        )
        return cls(
            uid=int(camera.uid),
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=float(getattr(camera, "Cx", 0.5 * (width - 1))),
            cy=float(getattr(camera, "Cy", 0.5 * (height - 1))),
            world_view_transform=torch.as_tensor(
                camera.world_view_transform,
                device=target,
                dtype=torch.float32,
            ),
            full_proj_transform=torch.as_tensor(
                camera.full_proj_transform,
                device=target,
                dtype=torch.float32,
            ),
            camera_center=torch.as_tensor(
                camera.camera_center,
                device=target,
                dtype=torch.float32,
            ),
            image_name=str(getattr(camera, "image_name", camera.uid)),
            gt_mask=(
                None
                if getattr(camera, "gt_mask", None) is None
                else torch.as_tensor(
                    camera.gt_mask,
                    device=target,
                    dtype=torch.float32,
                )
            ),
        )

    def to(self, device: str | torch.device) -> "MeshCamera":
        target = torch.device(device)
        return MeshCamera(
            uid=self.uid,
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            world_view_transform=self.world_view_transform.to(target),
            full_proj_transform=self.full_proj_transform.to(target),
            camera_center=self.camera_center.to(target),
            image_name=self.image_name,
            gt_mask=None if self.gt_mask is None else self.gt_mask.to(target),
        )

    @property
    def image_width(self) -> int:
        return self.width

    @property
    def image_height(self) -> int:
        return self.height

    @property
    def FoVx(self) -> float:  # noqa: N802 - standard 3DGS camera contract
        return 2.0 * math.atan(0.5 * self.width / self.fx)

    @property
    def FoVy(self) -> float:  # noqa: N802 - standard 3DGS camera contract
        return 2.0 * math.atan(0.5 * self.height / self.fy)

    @property
    def Fx(self) -> float:  # noqa: N802
        return self.fx

    @property
    def Fy(self) -> float:  # noqa: N802
        return self.fy

    @property
    def Cx(self) -> float:  # noqa: N802
        return self.cx

    @property
    def Cy(self) -> float:  # noqa: N802
        return self.cy

    def scaled(self, factor: float) -> "MeshCamera":
        """Return identical rays sampled on a scaled image lattice."""

        if not math.isfinite(float(factor)) or factor <= 0.0:
            raise ValueError("camera scale factor must be finite and positive")
        width = max(1, round(self.width * factor))
        height = max(1, round(self.height * factor))
        sx = width / self.width
        sy = height / self.height
        mask = None
        if self.gt_mask is not None:
            mask = F.interpolate(
                self.gt_mask[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        return MeshCamera(
            uid=self.uid,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            world_view_transform=self.world_view_transform,
            full_proj_transform=self.full_proj_transform,
            camera_center=self.camera_center,
            image_name=self.image_name,
            gt_mask=mask,
        )


__all__ = ["MeshCamera"]
