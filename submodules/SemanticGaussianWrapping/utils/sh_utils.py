"""Real spherical-harmonic helpers used by the Gaussian renderer.

The coefficient ordering follows the public 3D Gaussian Splatting reference.
"""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5


def eval_sh(degree: int, sh: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Evaluate degree <= 3 real SH.

    ``sh`` has shape ``[..., channels, (degree + 1) ** 2]`` and normalized
    ``directions`` has shape ``[..., 3]``.
    """

    if not 0 <= degree <= 3:
        raise ValueError(f"Only SH degrees 0..3 are supported, got {degree}")
    result = C0 * sh[..., 0]
    if degree == 0:
        return result

    x, y, z = directions.unbind(-1)
    result = result - C1 * y[..., None] * sh[..., 1]
    result = result + C1 * z[..., None] * sh[..., 2]
    result = result - C1 * x[..., None] * sh[..., 3]
    if degree == 1:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = result + C2[0] * xy[..., None] * sh[..., 4]
    result = result + C2[1] * yz[..., None] * sh[..., 5]
    result = result + C2[2] * (2.0 * zz - xx - yy)[..., None] * sh[..., 6]
    result = result + C2[3] * xz[..., None] * sh[..., 7]
    result = result + C2[4] * (xx - yy)[..., None] * sh[..., 8]
    if degree == 2:
        return result

    result = result + C3[0] * (y * (3 * xx - yy))[..., None] * sh[..., 9]
    result = result + C3[1] * (xy * z)[..., None] * sh[..., 10]
    result = result + C3[2] * (y * (4 * zz - xx - yy))[..., None] * sh[..., 11]
    result = result + C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[..., None] * sh[..., 12]
    result = result + C3[4] * (x * (4 * zz - xx - yy))[..., None] * sh[..., 13]
    result = result + C3[5] * (z * (xx - yy))[..., None] * sh[..., 14]
    result = result + C3[6] * (x * (xx - 3 * yy))[..., None] * sh[..., 15]
    return result


# Standard reference aliases.
RGB2SH = rgb_to_sh
SH2RGB = sh_to_rgb

