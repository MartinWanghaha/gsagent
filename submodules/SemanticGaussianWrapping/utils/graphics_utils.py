"""Camera, projection, point-cloud, and depth geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import tan

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_NORMAL_ALPHA_THRESHOLD = 0.5


@dataclass(frozen=True)
class BasicPointCloud:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray | None = None


def get_world_to_view(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = R.transpose()
    matrix[:3, 3] = T
    return matrix


def get_world_to_view2(
    R: np.ndarray,
    T: np.ndarray,
    translate: np.ndarray = np.zeros(3),
    scale: float = 1.0,
) -> np.ndarray:
    world_to_view = get_world_to_view(R, T)
    camera_to_world = np.linalg.inv(world_to_view)
    camera_to_world[:3, 3] = (camera_to_world[:3, 3] + translate) * scale
    return np.linalg.inv(camera_to_world).astype(np.float32)


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    top = tan(fov_y / 2.0) * znear
    right = tan(fov_x / 2.0) * znear
    matrix = torch.zeros(4, 4)
    matrix[0, 0] = znear / right
    matrix[1, 1] = znear / top
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    matrix[3, 2] = 1.0
    return matrix


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * np.tan(fov / 2.0))


def depth_to_points(camera, depth: torch.Tensor) -> torch.Tensor:
    """Back-project a ``[1,H,W]`` depth image into world space."""

    _, height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = float(getattr(camera, "Fx", fov2focal(float(camera.FoVx), width)))
    fy = float(getattr(camera, "Fy", fov2focal(float(camera.FoVy), height)))
    # Camera Cx/Cy use ordinary pixel-index coordinates. Raster samples live at
    # cell centers, so both receive the same +0.5 shift and cancel here.
    cx = float(getattr(camera, "Cx", (width - 1) * 0.5))
    cy = float(getattr(camera, "Cy", (height - 1) * 0.5))
    directions = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)), -1)
    points_cam = directions * depth[0, ..., None]
    # Cameras may intentionally keep calibration tensors on CPU while renderer
    # outputs live on CUDA.  Geometry derived from a depth map owns the runtime
    # device/dtype contract, so move the constant camera transform at this
    # boundary before any linear algebra.
    world_view_transform = torch.as_tensor(
        camera.world_view_transform,
        device=depth.device,
        dtype=depth.dtype,
    )
    if world_view_transform.shape != (4, 4):
        raise ValueError(
            "camera.world_view_transform must have shape [4,4], got "
            f"{tuple(world_view_transform.shape)}"
        )
    c2w = torch.linalg.inv(world_view_transform.transpose(0, 1))
    return points_cam @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]


def depth_normal_validity(
    depth: torch.Tensor,
    alpha: torch.Tensor | None = None,
    *,
    alpha_threshold: float = DEFAULT_NORMAL_ALPHA_THRESHOLD,
) -> torch.Tensor:
    """Return pixels whose complete 3x3 depth stencil is foreground.

    A central-difference normal at ``(y, x)`` reads the four axial neighbours.
    Requiring the stricter 3x3 neighbourhood keeps the same validity rule for
    every consumer and prevents zero background depth from becoming silhouette
    geometry.  When alpha is unavailable, positive finite depth provides a
    conservative fallback for callers outside the joint renderer.
    """

    if not 0.0 <= float(alpha_threshold) <= 1.0:
        raise ValueError("alpha_threshold must be in [0,1]")
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth_map = depth[0]
    elif depth.ndim == 2:
        depth_map = depth
    else:
        raise ValueError(f"depth must have shape [1,H,W] or [H,W], got {tuple(depth.shape)}")

    sample_valid = torch.isfinite(depth_map) & (depth_map > 0)
    if alpha is not None:
        if alpha.ndim == 3 and alpha.shape[0] == 1:
            alpha_map = alpha[0]
        elif alpha.ndim == 2:
            alpha_map = alpha
        else:
            raise ValueError(f"alpha must have shape [1,H,W] or [H,W], got {tuple(alpha.shape)}")
        if alpha_map.shape != depth_map.shape:
            raise ValueError(
                f"alpha and depth must share HxW, got {tuple(alpha_map.shape)} and {tuple(depth_map.shape)}"
            )
        sample_valid &= torch.isfinite(alpha_map) & (alpha_map >= float(alpha_threshold))

    # A binary 3x3 erosion. Zero padding deliberately invalidates image borders,
    # which cannot support a centred derivative in any case.
    support_count = F.conv2d(
        sample_valid[None, None].to(dtype=torch.float32),
        torch.ones((1, 1, 3, 3), device=depth_map.device),
        padding=1,
    )[0, 0]
    return support_count == 9


def depth_to_normal(
    camera,
    depth: torch.Tensor,
    alpha: torch.Tensor | None = None,
    *,
    alpha_threshold: float = DEFAULT_NORMAL_ALPHA_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert expected depth to normals with silhouette-safe validity."""

    points = depth_to_points(camera, depth)
    dx = points[:, 2:] - points[:, :-2]
    dy = points[2:, :] - points[:-2, :]
    normals = torch.zeros_like(points)
    cross = torch.cross(dx[1:-1], dy[:, 1:-1], dim=-1)
    finite_cross = torch.isfinite(cross).all(-1)
    safe_cross = torch.where(finite_cross[..., None], cross, torch.zeros_like(cross))
    valid = finite_cross & (safe_cross.norm(dim=-1) > 1e-8)
    normals[1:-1, 1:-1] = F.normalize(safe_cross, dim=-1, eps=1e-8)
    valid_full = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
    valid_full[1:-1, 1:-1] = valid
    valid_full &= depth_normal_validity(
        depth,
        alpha,
        alpha_threshold=alpha_threshold,
    )
    normals = torch.where(valid_full[..., None], normals, torch.zeros_like(normals))
    return normals.permute(2, 0, 1), valid_full


def depth_normal_residual(
    camera,
    expected_depth: torch.Tensor,
    rendered_normal: torch.Tensor,
    alpha: torch.Tensor | None = None,
    *,
    alpha_threshold: float = DEFAULT_NORMAL_ALPHA_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a masked, per-pixel depth/rendered-normal disagreement map."""

    depth_normal, valid = depth_to_normal(
        camera,
        expected_depth,
        alpha,
        alpha_threshold=alpha_threshold,
    )
    if rendered_normal.shape != depth_normal.shape:
        raise ValueError(
            "rendered_normal and depth normal must share [3,H,W], got "
            f"{tuple(rendered_normal.shape)} and {tuple(depth_normal.shape)}"
        )
    finite_rendered = torch.isfinite(rendered_normal).all(dim=0)
    safe_rendered = torch.where(
        finite_rendered[None],
        rendered_normal,
        torch.zeros_like(rendered_normal),
    )
    dot = (F.normalize(safe_rendered, dim=0, eps=1e-8) * depth_normal).sum(0).abs()
    # Do not mutate the validity tensor returned by ``depth_to_normal``: it was
    # saved by the preceding ``where`` for autograd's depth branch.
    valid = valid & finite_rendered
    residual = (1.0 - dot).clamp(0.0, 1.0)
    return torch.where(valid, residual, torch.zeros_like(residual)), valid
