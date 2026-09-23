"""PGSR plane-aware rasterization for EDGS cameras and Gaussian models.

Only PGSR's ``diff_plane_rasterization`` extension is used.  In particular,
this module never places the PGSR repository root on ``sys.path`` and never
imports its colliding ``scene``, ``utils`` or ``gaussian_renderer`` packages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch
import torch.nn.functional as F

from source.pgsr_geometry import depth_to_normal, smallest_axis_normal

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
_SH_C4 = (
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
)


@lru_cache(maxsize=1)
def _load_plane_rasterizer():
    """Load the optional CUDA extension only when PGSR rendering is requested."""

    try:
        extension = import_module("diff_plane_rasterization")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PGSR rendering requires diff_plane_rasterization; install EDGS's "
            "submodules/PGSR/submodules/diff-plane-rasterization in the active environment"
        ) from error
    return extension.GaussianRasterizationSettings, extension.GaussianRasterizer


def _option(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _eval_sh(
    degree: int, coefficients: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Evaluate the SH basis used by the pinned EDGS Gaussian model."""

    if not 0 <= degree <= 4:
        raise ValueError(f"SH degree must be between zero and four, got {degree}")
    required = (degree + 1) ** 2
    if coefficients.shape[-1] < required:
        raise ValueError(
            f"degree {degree} needs {required} SH coefficients, got {coefficients.shape[-1]}"
        )
    result = _SH_C0 * coefficients[..., 0]
    if degree == 0:
        return result

    x, y, z = directions[..., 0:1], directions[..., 1:2], directions[..., 2:3]
    result = (
        result
        - _SH_C1 * y * coefficients[..., 1]
        + _SH_C1 * z * coefficients[..., 2]
        - _SH_C1 * x * coefficients[..., 3]
    )
    if degree == 1:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + _SH_C2[0] * xy * coefficients[..., 4]
        + _SH_C2[1] * yz * coefficients[..., 5]
        + _SH_C2[2] * (2.0 * zz - xx - yy) * coefficients[..., 6]
        + _SH_C2[3] * xz * coefficients[..., 7]
        + _SH_C2[4] * (xx - yy) * coefficients[..., 8]
    )
    if degree == 2:
        return result

    result = (
        result
        + _SH_C3[0] * y * (3 * xx - yy) * coefficients[..., 9]
        + _SH_C3[1] * xy * z * coefficients[..., 10]
        + _SH_C3[2] * y * (4 * zz - xx - yy) * coefficients[..., 11]
        + _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coefficients[..., 12]
        + _SH_C3[4] * x * (4 * zz - xx - yy) * coefficients[..., 13]
        + _SH_C3[5] * z * (xx - yy) * coefficients[..., 14]
        + _SH_C3[6] * x * (xx - 3 * yy) * coefficients[..., 15]
    )
    if degree == 3:
        return result

    return (
        result
        + _SH_C4[0] * xy * (xx - yy) * coefficients[..., 16]
        + _SH_C4[1] * yz * (3 * xx - yy) * coefficients[..., 17]
        + _SH_C4[2] * xy * (7 * zz - 1) * coefficients[..., 18]
        + _SH_C4[3] * yz * (7 * zz - 3) * coefficients[..., 19]
        + _SH_C4[4] * (zz * (35 * zz - 30) + 3) * coefficients[..., 20]
        + _SH_C4[5] * xz * (7 * zz - 3) * coefficients[..., 21]
        + _SH_C4[6] * (xx - yy) * (7 * zz - 1) * coefficients[..., 22]
        + _SH_C4[7] * xz * (xx - 3 * yy) * coefficients[..., 23]
        + _SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * coefficients[..., 24]
    )


class PGSRRenderer:
    """Plane-aware PGSR renderer backed by EDGS model and camera objects."""

    backend = "pgsr"

    def __init__(self, *, clamp_rgb: bool = True) -> None:
        self.clamp_rgb = bool(clamp_rgb)

    def render(
        self,
        viewpoint_camera: Any,
        gaussians: Any,
        pipe: Any,
        background: torch.Tensor,
        scaling_modifier: float = 1.0,
        override_color: torch.Tensor | None = None,
        *,
        return_plane: bool = False,
        return_depth_normal: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Render RGB and optional PGSR plane geometry.

        ``plane_depth`` is metric camera-space z-depth and is intentionally
        never exposed under the native renderer's ambiguous ``depth`` key.
        """

        if bool(_option(pipe, "antialiasing", False)):
            raise ValueError(
                "the PGSR plane rasterizer does not support antialiasing; "
                "set pipe.antialiasing=false"
            )
        if bool(_option(pipe, "debug", False)):
            raise ValueError(
                "the pinned PGSR plane rasterizer has no safe debug path; "
                "set pipe.debug=false"
            )
        if return_depth_normal:
            return_plane = True
        settings_type, rasterizer_type = _load_plane_rasterizer()

        means3d = gaussians.get_xyz
        device, dtype = means3d.device, means3d.dtype
        background = background.to(device=device, dtype=dtype)
        screenspace_points = torch.zeros_like(means3d, requires_grad=True)
        screenspace_points_abs = torch.zeros_like(means3d, requires_grad=True)
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()

        tan_fov_x = math.tan(float(viewpoint_camera.FoVx) * 0.5)
        tan_fov_y = math.tan(float(viewpoint_camera.FoVy) * 0.5)
        world_view = torch.as_tensor(
            viewpoint_camera.world_view_transform,
            device=device,
            dtype=dtype,
        )
        projection = torch.as_tensor(
            viewpoint_camera.full_proj_transform,
            device=device,
            dtype=dtype,
        )
        camera_center = torch.as_tensor(
            viewpoint_camera.camera_center,
            device=device,
            dtype=dtype,
        )
        raster_settings = settings_type(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background,
            scale_modifier=float(scaling_modifier),
            viewmatrix=world_view,
            projmatrix=projection,
            sh_degree=int(gaussians.active_sh_degree),
            campos=camera_center,
            prefiltered=False,
            render_geo=bool(return_plane),
            debug=False,
        )
        rasterizer = rasterizer_type(raster_settings=raster_settings)

        scales = rotations = covariance = None
        if bool(_option(pipe, "compute_cov3D_python", False)):
            covariance = gaussians.get_covariance(scaling_modifier)
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation

        spherical_harmonics = colors = None
        if override_color is not None:
            colors = override_color.to(device=device, dtype=dtype)
        elif bool(_option(pipe, "convert_SHs_python", False)):
            features = gaussians.get_features
            coefficients = features.transpose(1, 2).reshape(
                -1,
                3,
                (gaussians.max_sh_degree + 1) ** 2,
            )
            directions = means3d - camera_center[None]
            directions = F.normalize(directions, p=2, dim=-1, eps=1e-8)
            colors = (
                _eval_sh(
                    int(gaussians.active_sh_degree),
                    coefficients,
                    directions,
                )
                .add(0.5)
                .clamp_min(0.0)
            )
        else:
            spherical_harmonics = gaussians.get_features

        raster_arguments = dict(
            means3D=means3d,
            means2D=screenspace_points,
            means2D_abs=screenspace_points_abs,
            shs=spherical_harmonics,
            colors_precomp=colors,
            opacities=gaussians.get_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=covariance,
        )

        if return_plane:
            global_normal = smallest_axis_normal(gaussians, viewpoint_camera)
            local_normal = global_normal @ world_view[:3, :3]
            points_camera = means3d @ world_view[:3, :3] + world_view[3, :3]
            local_distance = (
                (local_normal * points_camera).sum(dim=-1, keepdim=True).abs()
            )
            all_map = torch.cat(
                (local_normal, torch.ones_like(local_distance), local_distance),
                dim=-1,
            )
            raster_arguments["all_map"] = all_map

        rendered, radii, observations, out_all_map, plane_depth = rasterizer(
            **raster_arguments
        )
        if self.clamp_rgb:
            rendered = rendered.clamp(0.0, 1.0)
        mask = radii > 0
        package = {
            "render": rendered,
            "viewspace_points": screenspace_points,
            "viewspace_points_abs": screenspace_points_abs,
            "visibility_filter": mask,
            "visible_mask": mask,
            "radii": radii,
            "out_observe": observations,
        }
        if not return_plane:
            return package

        rendered_normal = out_all_map[:3]
        rendered_alpha = out_all_map[3:4]
        rendered_distance = out_all_map[4:5]
        package.update(
            {
                "rendered_normal": rendered_normal,
                "rendered_alpha": rendered_alpha,
                "rendered_distance": rendered_distance,
                "plane_depth": plane_depth,
            }
        )
        if return_depth_normal:
            normal = depth_to_normal(viewpoint_camera, plane_depth)
            package["depth_normal"] = normal * rendered_alpha.detach()
        return package

    __call__ = render


__all__ = ["PGSRRenderer"]
