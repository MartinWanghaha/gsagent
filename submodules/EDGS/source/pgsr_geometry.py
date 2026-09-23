"""Geometry adapters shared by the EDGS/PGSR renderer and losses.

The helpers in this module deliberately operate on the camera attributes that
already exist in EDGS.  They do not import PGSR's ``scene`` or ``utils``
packages, and they never assume that tensors live on a particular CUDA device.

EDGS stores transforms for row-vector multiplication::

    point_camera = point_world @ world_view[:3, :3] + world_view[3, :3]

That convention is kept explicit below, which makes the multi-view loss usable
without replacing EDGS's Camera or GaussianModel classes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def _positive_scale(scale: float) -> float:
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    return scale


def _camera_size(camera: Any, scale: float) -> tuple[int, int]:
    scale = _positive_scale(scale)
    width = max(1, int(int(camera.image_width) / scale))
    height = max(1, int(int(camera.image_height) / scale))
    return height, width


def _camera_tensor(camera: Any) -> torch.Tensor | None:
    for name in ("world_view_transform", "camera_center", "original_image"):
        value = getattr(camera, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _tensor_spec(
    camera: Any,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    reference = _camera_tensor(camera)
    if device is None:
        device = reference.device if reference is not None else torch.device("cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        if reference is not None and reference.is_floating_point():
            dtype = reference.dtype
        else:
            dtype = torch.float32
    return device, dtype


def camera_intrinsics(
    camera: Any,
    scale: float = 1.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the centered-pinhole intrinsic matrix for an EDGS camera.

    Explicit ``Fx/Fy/Cx/Cy`` attributes are honored when present.  Standard
    EDGS cameras only carry field of view, in which case focal lengths and the
    centered principal point are derived from image dimensions.
    """

    scale = _positive_scale(scale)
    device, dtype = _tensor_spec(camera, device=device, dtype=dtype)
    width = float(camera.image_width)
    height = float(camera.image_height)

    fx = getattr(camera, "Fx", None)
    fy = getattr(camera, "Fy", None)
    cx = getattr(camera, "Cx", None)
    cy = getattr(camera, "Cy", None)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5)) if fx is None else float(fx)
    fy = (
        height / (2.0 * math.tan(float(camera.FoVy) * 0.5)) if fy is None else float(fy)
    )
    cx = width * 0.5 if cx is None else float(cx)
    cy = height * 0.5 if cy is None else float(cy)

    return torch.tensor(
        ((fx / scale, 0.0, cx / scale), (0.0, fy / scale, cy / scale), (0.0, 0.0, 1.0)),
        device=device,
        dtype=dtype,
    )


def camera_rays(
    camera: Any,
    scale: float = 1.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Return camera-space rays with shape ``[H, W, 3]``.

    By default the ray z component is one, so multiplying by a z-depth map
    directly unprojects pixels.  Set ``normalize=True`` only for direction-only
    calculations such as angular comparisons.
    """

    height, width = _camera_size(camera, scale)
    device, dtype = _tensor_spec(camera, device=device, dtype=dtype)
    intrinsics = camera_intrinsics(camera, scale, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    rays = torch.stack(
        (
            (grid_x - intrinsics[0, 2]) / intrinsics[0, 0],
            (grid_y - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(grid_x),
        ),
        dim=-1,
    )
    if normalize:
        rays = F.normalize(rays, p=2, dim=-1)
    return rays


def image_gray(
    camera: Any,
    scale: float = 1.0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the camera image as a ``[1, H, W]`` luminance tensor."""

    image = getattr(camera, "original_image", None)
    if image is None:
        get_image = getattr(camera, "get_image", None)
        if get_image is None:
            raise AttributeError(
                "camera provides neither original_image nor get_image()"
            )
        image = get_image()
        if isinstance(image, (tuple, list)):
            image = image[0]
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"expected image shaped [C,H,W], got {tuple(image.shape)}")
    if device is not None:
        image = image.to(device=device)
    if not image.is_floating_point():
        image = image.float().div(255.0)

    height, width = _camera_size(camera, scale)
    if image.shape[-2:] != (height, width):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if image.shape[0] == 1:
        return image
    if image.shape[0] < 3:
        raise ValueError(
            f"expected one or at least three image channels, got {image.shape[0]}"
        )
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (image[:3] * weights).sum(dim=0, keepdim=True)


def _depth_2d(depth: torch.Tensor) -> torch.Tensor:
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.ndim != 2:
        raise ValueError(
            f"expected depth shaped [H,W] or singleton-prefixed, got {tuple(depth.shape)}"
        )
    if not depth.is_floating_point():
        depth = depth.float()
    return depth


def _depth_at_scale(camera: Any, depth: torch.Tensor, scale: float) -> torch.Tensor:
    depth = _depth_2d(depth)
    target_size = _camera_size(camera, scale)
    if depth.shape != target_size:
        depth = F.interpolate(
            depth[None, None],
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return depth


def _world_view(camera: Any, reference: torch.Tensor) -> torch.Tensor:
    transform = getattr(camera, "world_view_transform", None)
    if transform is None:
        raise AttributeError("camera must provide world_view_transform")
    return torch.as_tensor(transform, device=reference.device, dtype=reference.dtype)


def points_from_depth(
    camera: Any, depth: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Unproject a plane z-depth map to flattened world-space points."""

    depth_map = _depth_at_scale(camera, depth, scale)
    rays = camera_rays(
        camera,
        scale,
        device=depth_map.device,
        dtype=depth_map.dtype,
    )
    points_camera = (rays * depth_map[..., None]).reshape(-1, 3)
    world_view = _world_view(camera, points_camera)
    camera_to_world = torch.linalg.inv(world_view)
    homogeneous = torch.cat(
        (points_camera, torch.ones_like(points_camera[:, :1])),
        dim=-1,
    )
    points_world_h = homogeneous @ camera_to_world
    denominator = points_world_h[:, 3:4]
    denominator = torch.where(
        denominator.abs() < 1e-8,
        torch.ones_like(denominator),
        denominator,
    )
    return points_world_h[:, :3] / denominator


def sample_depth(
    camera: Any,
    depth: torch.Tensor,
    points_camera: torch.Tensor,
    scale: float = 1.0,
    min_depth: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample ``depth`` at projected camera-space 3D points.

    Returns ``(sampled_depth, valid_mask)`` with one entry per input point.  The
    mask covers both the image bounds and positive camera-space z.
    """

    if points_camera.ndim != 2 or points_camera.shape[-1] != 3:
        raise ValueError(
            f"expected points_camera shaped [N,3], got {tuple(points_camera.shape)}"
        )
    depth_map = _depth_at_scale(camera, depth, scale)
    points_camera = points_camera.to(device=depth_map.device, dtype=depth_map.dtype)
    count = points_camera.shape[0]
    if count == 0:
        return depth_map.new_empty((0,)), torch.empty(
            (0,), device=depth_map.device, dtype=torch.bool
        )

    height, width = depth_map.shape
    intrinsics = camera_intrinsics(
        camera,
        scale,
        device=depth_map.device,
        dtype=depth_map.dtype,
    )
    z = points_camera[:, 2]
    safe_z = z.clamp_min(torch.finfo(depth_map.dtype).eps)
    pixel_x = points_camera[:, 0] * intrinsics[0, 0] / safe_z + intrinsics[0, 2]
    pixel_y = points_camera[:, 1] * intrinsics[1, 1] / safe_z + intrinsics[1, 2]
    valid = (
        torch.isfinite(points_camera).all(dim=-1)
        & (z > min_depth)
        & (pixel_x >= 0)
        & (pixel_x <= width - 1)
        & (pixel_y >= 0)
        & (pixel_y <= height - 1)
    )

    if width > 1:
        grid_x = 2.0 * pixel_x / (width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(pixel_x)
    if height > 1:
        grid_y = 2.0 * pixel_y / (height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(pixel_y)
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        depth_map[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    return sampled, valid


def depth_to_normal(
    camera: Any, depth: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Convert plane z-depth to camera-space normals shaped ``[3,H,W]``.

    Normals face the camera, matching PGSR's oriented Gaussian normals.  Border
    and invalid-depth pixels are zeroed rather than filled with arbitrary unit
    vectors.
    """

    depth_map = _depth_at_scale(camera, depth, scale)
    rays = camera_rays(
        camera,
        scale,
        device=depth_map.device,
        dtype=depth_map.dtype,
    )
    points = rays * depth_map[..., None]
    height, width = depth_map.shape
    normal = torch.zeros_like(points)
    if height < 3 or width < 3:
        return normal.permute(2, 0, 1)

    left_to_right = points[1:-1, 2:] - points[1:-1, :-2]
    bottom_to_top = points[:-2, 1:-1] - points[2:, 1:-1]
    interior = torch.cross(left_to_right, bottom_to_top, dim=-1)
    interior = F.normalize(interior, p=2, dim=-1, eps=1e-8)
    valid = (
        torch.isfinite(depth_map[1:-1, 2:])
        & torch.isfinite(depth_map[1:-1, :-2])
        & torch.isfinite(depth_map[:-2, 1:-1])
        & torch.isfinite(depth_map[2:, 1:-1])
        & (depth_map[1:-1, 2:] > 0)
        & (depth_map[1:-1, :-2] > 0)
        & (depth_map[:-2, 1:-1] > 0)
        & (depth_map[2:, 1:-1] > 0)
    )
    normal[1:-1, 1:-1] = torch.where(valid[..., None], interior, 0.0)
    return normal.permute(2, 0, 1)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError(
            f"expected quaternion shaped [N,4], got {tuple(quaternion.shape)}"
        )
    q = F.normalize(quaternion, p=2, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(dim=-1)
    matrix = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - w * z)
    matrix[:, 0, 2] = 2 * (x * z + w * y)
    matrix[:, 1, 0] = 2 * (x * y + w * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - w * x)
    matrix[:, 2, 0] = 2 * (x * z - w * y)
    matrix[:, 2, 1] = 2 * (y * z + w * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def smallest_axis_normal(gaussians: Any, camera: Any) -> torch.Tensor:
    """Derive PGSR normals from an unmodified EDGS GaussianModel.

    The column corresponding to each Gaussian's smallest scale is its surface
    normal.  Its sign is oriented toward the current camera without mutating the
    Gaussian model.
    """

    scaling = gaussians.get_scaling
    rotation = gaussians.get_rotation
    xyz = gaussians.get_xyz
    matrices = _quaternion_to_matrix(rotation)
    axis_index = scaling.argmin(dim=-1)
    gather_index = axis_index[:, None, None].expand(-1, 3, 1)
    normals = matrices.gather(2, gather_index).squeeze(2)
    center = torch.as_tensor(
        camera.camera_center,
        device=xyz.device,
        dtype=xyz.dtype,
    )
    to_camera = center[None] - xyz
    flip = (normals * to_camera).sum(dim=-1, keepdim=True) < 0
    return torch.where(flip, -normals, normals)


def patch_offsets(
    radius: int,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return square patch offsets shaped ``[1,(2r+1)^2,2]`` in x/y order."""

    radius = int(radius)
    if radius < 0:
        raise ValueError(f"patch radius must be non-negative, got {radius}")
    dtype = torch.float32 if dtype is None else dtype
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    offset_x, offset_y = torch.meshgrid(coordinates, coordinates, indexing="xy")
    return torch.stack((offset_x, offset_y), dim=-1).reshape(1, -1, 2)


def patch_warp(homography: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Warp x/y patch coordinates with batched 3x3 homographies."""

    if homography.ndim == 2:
        homography = homography.unsqueeze(0)
    if pixels.ndim == 2:
        pixels = pixels.unsqueeze(0)
    if homography.ndim != 3 or homography.shape[-2:] != (3, 3):
        raise ValueError(
            f"expected homography shaped [B,3,3], got {tuple(homography.shape)}"
        )
    if pixels.ndim != 3 or pixels.shape[-1] != 2:
        raise ValueError(f"expected pixels shaped [B,P,2], got {tuple(pixels.shape)}")
    homography_batch = homography.shape[0]
    pixel_batch = pixels.shape[0]
    if homography_batch != pixel_batch and homography_batch != 1 and pixel_batch != 1:
        raise ValueError("homography and pixel batch dimensions are not broadcastable")
    batch = max(homography_batch, pixel_batch)
    homography = homography.expand(batch, -1, -1)
    pixels = pixels.expand(batch, -1, -1)
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    warped = torch.einsum("bij,bpj->bpi", homography, homogeneous)
    denominator = warped[..., 2:]
    eps = torch.finfo(warped.dtype).eps
    safe_denominator = torch.where(
        denominator.abs() < eps,
        torch.where(denominator < 0, -eps, eps),
        denominator,
    )
    return warped[..., :2] / safe_denominator


def _config_value(config: Any, names: Sequence[str], default: Any) -> Any:
    if config is None:
        return default
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _camera_forward_world(
    camera: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    transform = torch.as_tensor(camera.world_view_transform, device=device, dtype=dtype)
    camera_to_world = torch.linalg.inv(transform)
    forward = camera_to_world[2, :3]
    return F.normalize(forward, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def build_neighbor_map(
    cameras: Sequence[Any],
    config: Any = None,
    *,
    max_neighbors: int | None = None,
    max_angle_deg: float | None = None,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> dict[str, list[Any]]:
    """Build an image-name keyed neighbor map from the post-split cameras.

    ``config`` may be a dictionary, OmegaConf ``DictConfig`` or an object.  The
    accepted neighbor-count keys are ``neighbor_num`` and ``max_neighbors``.
    Explicit keyword arguments take precedence over config values.
    """

    cameras = list(cameras)
    max_neighbors = int(
        _config_value(config, ("neighbor_num", "max_neighbors"), 8)
        if max_neighbors is None
        else max_neighbors
    )
    max_angle_deg = float(
        _config_value(config, ("max_angle_deg",), 30.0)
        if max_angle_deg is None
        else max_angle_deg
    )
    min_distance = float(
        _config_value(config, ("min_distance",), 0.01)
        if min_distance is None
        else min_distance
    )
    max_distance = float(
        _config_value(config, ("max_distance",), 1.5)
        if max_distance is None
        else max_distance
    )
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if min_distance < 0 or max_distance <= min_distance:
        raise ValueError("neighbor distance bounds must satisfy 0 <= min < max")

    names = [
        str(getattr(camera, "image_name", index))
        for index, camera in enumerate(cameras)
    ]
    if len(set(names)) != len(names):
        raise ValueError("camera image_name values must be unique")
    result: dict[str, list[Any]] = {name: [] for name in names}
    if len(cameras) < 2 or max_neighbors == 0:
        return result

    reference = _camera_tensor(cameras[0])
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = (
        reference.dtype
        if reference is not None and reference.is_floating_point()
        else torch.float32
    )
    centers = torch.stack(
        [
            torch.as_tensor(camera.camera_center, device=device, dtype=dtype)
            for camera in cameras
        ]
    )
    forwards = torch.stack(
        [
            _camera_forward_world(camera, device=device, dtype=dtype)
            for camera in cameras
        ]
    )
    distances = torch.linalg.vector_norm(centers[:, None] - centers[None], dim=-1)
    cosine = (forwards[:, None] * forwards[None]).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))

    for index, name in enumerate(names):
        valid = (
            (distances[index] > min_distance)
            & (distances[index] < max_distance)
            & (angles[index] < max_angle_deg)
        )
        candidates = torch.arange(len(cameras), device=device)[valid]
        if candidates.numel() == 0:
            continue
        order = torch.argsort(distances[index, candidates], stable=True)
        selected = candidates[order[:max_neighbors]].tolist()
        result[name] = [cameras[neighbor_index] for neighbor_index in selected]
    return result


__all__ = [
    "build_neighbor_map",
    "camera_intrinsics",
    "camera_rays",
    "depth_to_normal",
    "image_gray",
    "patch_offsets",
    "patch_warp",
    "points_from_depth",
    "sample_depth",
    "smallest_axis_normal",
]
