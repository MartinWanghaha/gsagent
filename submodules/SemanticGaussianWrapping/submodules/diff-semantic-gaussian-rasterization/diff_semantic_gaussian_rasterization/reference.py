"""Differentiable, memory-bounded reference Gaussian rasterizer.

This module intentionally contains no extension-specific code.  Besides being
the CPU fallback it is the executable specification used to validate the CUDA
front end.  All per-pixel attributes share exactly the same front-to-back
compositing weights.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


SEMANTIC_DIM = 16
_ALPHA_CAP = 0.99
_ALPHA_EPS = 1.0 / 255.0
NEAR_PLANE = 0.2


class ProjectedGaussians(NamedTuple):
    means: Tensor
    depth: Tensor
    conic: Tensor
    normals: Tensor
    radii: Tensor
    visible: Tensor


class ReferenceRasterizationResult(NamedTuple):
    color: Tensor
    semantic: Tensor
    expected_depth: Tensor
    alpha: Tensor
    normal: Tensor
    radii: Tensor
    dominant_index: Tensor


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert scalar-first quaternions ``[w, x, y, z]`` to matrices."""

    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError("rotations must have shape [N,4] in (w,x,y,z) order")
    q = F.normalize(quaternions, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _transform_points_row(points: Tensor, matrix: Tensor) -> Tensor:
    """Apply Graphdeco's transposed (row-vector) camera convention."""

    ones = torch.ones_like(points[:, :1])
    return torch.cat((points, ones), dim=-1) @ matrix.to(points)


def project_gaussians(
    means3d: Tensor,
    means2d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    viewmatrix: Tensor,
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    scale_modifier: float = 1.0,
    antialias_sigma: float = 0.3,
    cx: float = math.nan,
    cy: float = math.nan,
) -> ProjectedGaussians:
    """Project anisotropic 3D covariance into an EWA screen-space conic."""

    if means3d.ndim != 2 or means3d.shape[-1] != 3:
        raise ValueError("means3d must have shape [N,3]")
    if scales.shape != means3d.shape:
        raise ValueError("scales must have shape [N,3]")
    if means2d.numel() and (means2d.ndim != 2 or means2d.shape[0] != means3d.shape[0] or means2d.shape[1] < 2):
        raise ValueError("means2d must be empty or have shape [N,2+] ")
    if viewmatrix.shape != (4, 4):
        raise ValueError("viewmatrix must have shape [4,4]")

    n = means3d.shape[0]
    if n == 0:
        empty = means3d.new_empty((0,))
        return ProjectedGaussians(
            means3d.new_empty((0, 2)),
            empty,
            means3d.new_empty((0, 3)),
            means3d.new_empty((0, 3)),
            empty,
            torch.empty((0,), dtype=torch.bool, device=means3d.device),
        )

    view = _transform_points_row(means3d, viewmatrix)[:, :3]
    x, y, z = view.unbind(-1)
    # Keep invisible near-camera outliers in a finite projection domain. This
    # matches the native and Graphdeco near-plane visibility contract.
    safe_z = z.clamp_min(NEAR_PLANE)
    fx = 0.5 * float(image_width) / float(tanfovx)
    fy = 0.5 * float(image_height) / float(tanfovy)
    principal_x = float(cx) if math.isfinite(float(cx)) else 0.5 * float(image_width)
    principal_y = float(cy) if math.isfinite(float(cy)) else 0.5 * float(image_height)
    screen_x = fx * x / safe_z + principal_x
    screen_y = fy * y / safe_z + principal_y
    if means2d.numel():
        screen_x = screen_x + means2d[:, 0]
        screen_y = screen_y + means2d[:, 1]
    screen_means = torch.stack((screen_x, screen_y), dim=-1)

    rotation = quaternion_to_matrix(rotations)
    scaled = scales * float(scale_modifier)
    covariance_world = rotation @ torch.diag_embed(scaled.square()) @ rotation.transpose(-1, -2)

    # A Graphdeco camera matrix maps row vectors as p_cam = p_world @ B.
    # Therefore covariance transforms as B^T Sigma B.
    camera_linear = viewmatrix[:3, :3].to(means3d)
    covariance_camera = camera_linear.transpose(0, 1) @ covariance_world @ camera_linear
    # Graphdeco evaluates the EWA covariance Jacobian in a 1.3x enlarged
    # camera frustum.  Keep the true projected mean above, but clamp x/z and
    # y/z for covariance propagation so far off-frustum points cannot create
    # unbounded screen-space footprints.
    covariance_x = (x / safe_z).clamp(-1.3 * float(tanfovx), 1.3 * float(tanfovx)) * safe_z
    covariance_y = (y / safe_z).clamp(-1.3 * float(tanfovy), 1.3 * float(tanfovy)) * safe_z
    zeros = torch.zeros_like(safe_z)
    jacobian = torch.stack(
        (
            fx / safe_z,
            zeros,
            -fx * covariance_x / safe_z.square(),
            zeros,
            fy / safe_z,
            -fy * covariance_y / safe_z.square(),
        ),
        dim=-1,
    ).reshape(-1, 2, 3)
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    variance_floor = float(antialias_sigma) ** 2
    covariance_2d = covariance_2d + torch.eye(2, device=means3d.device, dtype=means3d.dtype) * variance_floor

    a = covariance_2d[:, 0, 0]
    b = covariance_2d[:, 0, 1]
    c = covariance_2d[:, 1, 1]
    determinant = (a * c - b.square()).clamp_min(1e-12)
    conic = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)

    half_trace = 0.5 * (a + c)
    discriminant = (half_trace.square() - determinant).clamp_min(0).sqrt()
    max_eigenvalue = (half_trace + discriminant).clamp_min(0)
    radii = torch.ceil(3.0 * max_eigenvalue.sqrt())
    intersects = (
        (screen_x + radii >= 0)
        & (screen_x - radii < image_width)
        & (screen_y + radii >= 0)
        & (screen_y - radii < image_height)
    )
    visible = (z > NEAR_PLANE) & intersects & torch.isfinite(radii)
    radii = torch.where(visible, radii, torch.zeros_like(radii))

    # The local axis with minimum scale is the Gaussian's surface normal.
    normal_axis = scaled.detach().argmin(dim=-1)
    gather_index = normal_axis[:, None, None].expand(-1, 3, 1)
    normal_world = rotation.gather(2, gather_index).squeeze(-1)
    normal_camera = normal_world @ camera_linear
    normal_camera = F.normalize(normal_camera, dim=-1, eps=1e-12)
    # Orient normals toward a camera looking along +z; orientation selection is
    # intentionally discrete while rotation remains differentiable.
    normal_camera = torch.where(normal_camera[:, 2:3] > 0, -normal_camera, normal_camera)
    return ProjectedGaussians(screen_means, z, conic, normal_camera, radii, visible)


def rasterize_reference(
    means3d: Tensor,
    means2d: Tensor,
    colors: Tensor,
    semantic_features: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
    *,
    viewmatrix: Tensor,
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    background: Tensor,
    scale_modifier: float = 1.0,
    antialias_sigma: float = 0.3,
    chunk_size: int = 32,
    depth_order: Optional[Tensor] = None,
    cx: float = math.nan,
    cy: float = math.nan,
) -> ReferenceRasterizationResult:
    """Rasterize all attributes with shared front-to-back alpha weights.

    The implementation bounds temporary storage by
    ``chunk_size * image_height * image_width``.  Depth sorting is treated as a
    visibility decision (as in the CUDA rasterizer), while all continuous
    projection and compositing operations remain differentiable.
    """

    n = means3d.shape[0]
    if colors.shape != (n, 3):
        raise ValueError("colors must have shape [N,3]")
    if semantic_features.shape != (n, SEMANTIC_DIM):
        raise ValueError(f"semantic_features must have shape [N,{SEMANTIC_DIM}]")
    if opacities.numel() != n:
        raise ValueError("opacities must contain exactly N values")
    if background.numel() != 3:
        raise ValueError("background must contain exactly three RGB values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    projected = project_gaussians(
        means3d,
        means2d,
        scales,
        rotations,
        viewmatrix,
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        scale_modifier,
        antialias_sigma,
        cx,
        cy,
    )
    if depth_order is None:
        depth_order = torch.argsort(projected.depth.detach(), stable=True)
    else:
        depth_order = depth_order.to(device=means3d.device, dtype=torch.long)
    if depth_order.shape != (n,):
        raise ValueError("depth_order must have shape [N]")

    dtype, device = means3d.dtype, means3d.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=dtype) + 0.5,
        torch.arange(image_width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    transmittance = torch.ones((image_height, image_width), device=device, dtype=dtype)
    color_acc = torch.zeros((3, image_height, image_width), device=device, dtype=dtype)
    semantic_acc = torch.zeros((SEMANTIC_DIM, image_height, image_width), device=device, dtype=dtype)
    depth_acc = torch.zeros((image_height, image_width), device=device, dtype=dtype)
    normal_acc = torch.zeros((3, image_height, image_width), device=device, dtype=dtype)
    max_weight = torch.zeros((image_height, image_width), device=device, dtype=dtype)
    dominant = torch.full((image_height, image_width), -1, device=device, dtype=torch.long)
    opacity = opacities.reshape(-1)

    for start in range(0, n, chunk_size):
        ids = depth_order[start : start + chunk_size]
        mean = projected.means[ids]
        conic = projected.conic[ids]
        dx = x_grid[None] - mean[:, 0, None, None]
        dy = y_grid[None] - mean[:, 1, None, None]
        exponent = -0.5 * (
            conic[:, 0, None, None] * dx.square()
            + 2.0 * conic[:, 1, None, None] * dx * dy
            + conic[:, 2, None, None] * dy.square()
        )
        footprint = torch.exp(exponent.clamp(max=0))
        support = (dx.abs() <= projected.radii[ids, None, None]) & (dy.abs() <= projected.radii[ids, None, None])
        footprint = footprint * projected.visible[ids, None, None].to(dtype) * support.to(dtype)
        per_gaussian_alpha = (opacity[ids, None, None] * footprint).clamp(0.0, _ALPHA_CAP)
        per_gaussian_alpha = torch.where(
            per_gaussian_alpha >= _ALPHA_EPS,
            per_gaussian_alpha,
            torch.zeros_like(per_gaussian_alpha),
        )
        one_minus_alpha = (1.0 - per_gaussian_alpha).clamp_min(1e-8)
        prefix = torch.cumprod(
            torch.cat((torch.ones_like(one_minus_alpha[:1]), one_minus_alpha[:-1]), dim=0),
            dim=0,
        )
        weights = transmittance[None] * prefix * per_gaussian_alpha

        color_acc = color_acc + torch.einsum("khw,kc->chw", weights, colors[ids])
        semantic_acc = semantic_acc + torch.einsum("khw,kc->chw", weights, semantic_features[ids])
        depth_acc = depth_acc + torch.einsum("khw,k->hw", weights, projected.depth[ids])
        normal_acc = normal_acc + torch.einsum("khw,kc->chw", weights, projected.normals[ids])

        chunk_weight, chunk_local = weights.max(dim=0)
        update = chunk_weight > max_weight
        dominant = torch.where(update, ids[chunk_local], dominant)
        max_weight = torch.maximum(max_weight, chunk_weight)
        transmittance = transmittance * one_minus_alpha.prod(dim=0)

    alpha = 1.0 - transmittance
    denominator = alpha.clamp_min(1e-8)
    color = color_acc + transmittance[None] * background.to(device=device, dtype=dtype).reshape(3, 1, 1)
    semantic = semantic_acc / denominator[None]
    expected_depth = depth_acc / denominator
    normal = normal_acc / denominator[None]
    normal = F.normalize(normal, dim=0, eps=1e-8)
    foreground = alpha > 1e-8
    semantic = torch.where(foreground[None], semantic, torch.zeros_like(semantic))
    expected_depth = torch.where(foreground, expected_depth, torch.zeros_like(expected_depth))
    normal = torch.where(foreground[None], normal, torch.zeros_like(normal))
    dominant = torch.where(foreground, dominant, torch.full_like(dominant, -1))
    return ReferenceRasterizationResult(
        color,
        semantic,
        expected_depth[None],
        alpha[None],
        normal,
        projected.radii,
        dominant,
    )
