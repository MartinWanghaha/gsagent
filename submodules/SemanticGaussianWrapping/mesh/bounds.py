"""Scene-scale-aware bounds shared by training and offline meshing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sampling import Bounds


def _numpy(value: Any) -> np.ndarray:
    value = value() if callable(value) else value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
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
        axis=-1,
    ).reshape(-1, 3, 3)


def _selected_attribute(
    gaussians: Any,
    indices: Any,
    *,
    raw_name: str,
    getter_name: str | None,
    activation: Any | None = None,
):
    """Gather one evidence attribute without activating an N-wide tensor."""

    import torch

    registry = getattr(gaussians, "registry", None)
    source = None
    if registry is not None and raw_name in registry:
        source = registry[raw_name]
    elif getter_name is None:
        source = getattr(gaussians, raw_name, None)
    total = int(getattr(gaussians, "get_xyz").shape[0])
    if isinstance(source, torch.Tensor) and source.shape[0] == total:
        selected = source.detach().index_select(0, indices.to(source.device))
        selected = selected.to(indices.device)
        return selected if activation is None else activation(selected)
    if getter_name is None:
        return None
    source = getattr(gaussians, getter_name, None)
    source = source() if callable(source) else source
    if not isinstance(source, torch.Tensor) or source.shape[0] != total:
        return None
    return source.detach().index_select(0, indices.to(source.device)).to(indices.device)


@dataclass(frozen=True)
class MeshSupportPolicy:
    """Observation-aware Gaussian selection shared by online and offline meshes.

    A mesh lattice should cover the reconstructed scene, not every transient
    low-opacity splat or oversized outlier retained by the renderer.  The
    policy uses exactly the support evidence accepted by mesh feedback and a
    tiny robust trim, so offline evaluation measures the same scene domain as
    the geometry supervision used during training.
    """

    min_opacity: float = 0.05
    min_semantic_confidence: float = 0.35
    require_observation: bool = True
    trim_quantile: float = 0.001
    chunk_size: int = 65_536

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_opacity) <= 1.0:
            raise ValueError("min_opacity must lie in [0,1]")
        if not 0.0 <= float(self.min_semantic_confidence) <= 1.0:
            raise ValueError("min_semantic_confidence must lie in [0,1]")
        if not 0.0 <= float(self.trim_quantile) < 0.5:
            raise ValueError("trim_quantile must lie in [0,0.5)")
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be positive")

    def selected_indices(self, gaussians: Any):
        """Return trusted support indices on the Gaussian model device."""

        import torch

        xyz = getattr(gaussians, "get_xyz")
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Gaussian model must expose get_xyz with shape [N,3]")
        outputs = []
        count = int(xyz.shape[0])
        for start in range(0, count, int(self.chunk_size)):
            indices = torch.arange(
                start,
                min(start + int(self.chunk_size), count),
                dtype=torch.long,
                device=xyz.device,
            )
            opacity = _selected_attribute(
                gaussians,
                indices,
                raw_name="opacity",
                getter_name="get_opacity",
                activation=torch.sigmoid,
            )
            if opacity is None:
                opacity = torch.ones(len(indices), device=xyz.device)
            opacity = opacity.reshape(len(indices), -1).max(-1).values

            direct = _selected_attribute(
                gaussians,
                indices,
                raw_name="semantic_confidence",
                getter_name=None,
            )
            propagated = _selected_attribute(
                gaussians,
                indices,
                raw_name="propagated_semantic_confidence",
                getter_name=None,
            )
            if direct is None:
                confidence = _selected_attribute(
                    gaussians,
                    indices,
                    raw_name="__missing_semantic_confidence__",
                    getter_name="get_semantic_confidence",
                )
            else:
                confidence = direct if propagated is None else torch.maximum(direct, propagated)
            if confidence is None:
                confidence = torch.ones(len(indices), device=xyz.device)
            confidence = confidence.reshape(len(indices), -1).max(-1).values

            keep = (opacity >= float(self.min_opacity)) & (
                confidence.clamp(0.0, 1.0) >= float(self.min_semantic_confidence)
            )
            if self.require_observation:
                observed = _selected_attribute(
                    gaussians,
                    indices,
                    raw_name="observation_count",
                    getter_name=None,
                )
                # Foreign Gaussian models without observation evidence remain
                # valid; absence is not evidence that their support is unseen.
                if observed is not None:
                    keep &= observed.reshape(len(indices), -1).max(-1).values > 0
            outputs.append(indices[keep])
        return (
            torch.cat(outputs)
            if outputs
            else torch.empty(0, dtype=torch.long, device=xyz.device)
        )

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "min_opacity": float(self.min_opacity),
            "min_semantic_confidence": float(self.min_semantic_confidence),
            "require_observation": bool(self.require_observation),
            "trim_quantile": float(self.trim_quantile),
            "chunk_size": int(self.chunk_size),
        }


def trusted_gaussian_support_bounds(
    gaussians: Any,
    *,
    sigma: float = 3.0,
    relative_padding: float = 0.02,
    policy: MeshSupportPolicy | None = None,
) -> tuple[Bounds, int]:
    """Return the robust support bounds and trusted Gaussian count."""

    selected_policy = policy or MeshSupportPolicy()
    selection = selected_policy.selected_indices(gaussians)
    if int(selection.numel()) < 3:
        raise ValueError("fewer than three trusted Gaussian supports")
    return (
        gaussian_support_bounds(
            gaussians,
            sigma=sigma,
            relative_padding=relative_padding,
            selection=selection,
            trim_quantile=selected_policy.trim_quantile,
        ),
        int(selection.numel()),
    )


def gaussian_support_bounds(
    gaussians: Any,
    *,
    sigma: float = 3.0,
    relative_padding: float = 0.02,
    absolute_padding: float = 0.0,
    chunk_size: int = 65_536,
    selection: Any | None = None,
    trim_quantile: float = 0.0,
) -> Bounds:
    """Bound rotated anisotropic Gaussian support ellipsoids.

    ``selection`` lets asynchronous consumers restrict extraction to support
    that is actually trusted by their observation policy.  ``trim_quantile``
    is deliberately opt-in: offline meshing keeps the exact union, while a
    live feedback worker can keep a handful of unbounded-scene outliers from
    collapsing the useful grid resolution.
    """

    if (
        sigma <= 0
        or relative_padding < 0
        or absolute_padding < 0
        or chunk_size < 1
        or not 0.0 <= float(trim_quantile) < 0.5
    ):
        raise ValueError("sigma/chunk_size must be positive and padding non-negative")
    xyz = _numpy(getattr(gaussians, "get_xyz")).astype(np.float32, copy=False)
    scales = _numpy(getattr(gaussians, "get_scaling")).astype(np.float32, copy=False)
    rotations = _numpy(getattr(gaussians, "get_rotation")).astype(np.float32, copy=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or scales.shape != xyz.shape or rotations.shape != (len(xyz), 4):
        raise ValueError("Gaussian xyz/scaling/rotation must be [N,3]/[N,3]/[N,4]")
    if selection is not None:
        selected = _numpy(selection)
        if selected.dtype == np.bool_:
            if selected.ndim != 1 or len(selected) != len(xyz):
                raise ValueError("Gaussian bounds selection mask must have shape [N]")
            selected = np.flatnonzero(selected)
        else:
            selected = np.asarray(selected)
            if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
                raise ValueError("Gaussian bounds selection must be a 1D integer index or bool mask")
            selected = selected.astype(np.int64, copy=False)
            if len(selected) and (selected.min() < 0 or selected.max() >= len(xyz)):
                raise IndexError("Gaussian bounds selection index is out of range")
        xyz = xyz[selected]
        scales = scales[selected]
        rotations = rotations[selected]
    if not len(xyz):
        raise ValueError("cannot infer mesh bounds from an empty Gaussian model")
    lower_blocks: list[np.ndarray] = []
    upper_blocks: list[np.ndarray] = []
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(xyz), chunk_size):
        end = min(start + chunk_size, len(xyz))
        matrix = _quaternion_matrix(rotations[start:end])
        # The axis-aligned half extent of R diag(s) is abs(R) @ s.
        extent = float(sigma) * np.einsum("nij,nj->ni", np.abs(matrix), scales[start:end])
        lower = xyz[start:end] - extent
        upper = xyz[start:end] + extent
        if trim_quantile > 0.0 and len(xyz) >= 32:
            lower_blocks.append(lower)
            upper_blocks.append(upper)
        else:
            minimum = np.minimum(minimum, lower.min(axis=0))
            maximum = np.maximum(maximum, upper.max(axis=0))
    if trim_quantile > 0.0 and len(xyz) >= 32:
        lower = np.concatenate(lower_blocks, axis=0)
        upper = np.concatenate(upper_blocks, axis=0)
        minimum = np.quantile(lower, float(trim_quantile), axis=0)
        maximum = np.quantile(upper, 1.0 - float(trim_quantile), axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    margin = float(absolute_padding) + float(relative_padding) * max(diagonal, 1e-6)
    margin = max(margin, 1e-6)
    return Bounds((minimum - margin).astype(np.float32), (maximum + margin).astype(np.float32))


__all__ = [
    "MeshSupportPolicy",
    "gaussian_support_bounds",
    "trusted_gaussian_support_bounds",
]
