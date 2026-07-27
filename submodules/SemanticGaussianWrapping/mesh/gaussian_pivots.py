"""Gaussian-Wrapping pivots on a semantic-aware spatial scaffold.

This follows GaussianWrapping's default two-pivot construction: one Gaussian
center and one point displaced along its oriented minimum-covariance axis.
Physical edge support is derived from Gaussian covariance, never from semantic
nearest-neighbour distance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _checkpoint_tensor(gaussians: Any, name: str) -> Tensor:
    if not hasattr(gaussians, name):
        raise AttributeError(f"Gaussian checkpoint does not expose {name}")
    value = getattr(gaussians, name)
    value = value() if callable(value) else value
    if not isinstance(value, Tensor):
        raise TypeError(f"Gaussian checkpoint attribute {name} must be a torch.Tensor")
    return value


def _quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = quaternion.unbind(-1)
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


@dataclass(frozen=True)
class GaussianPivotConfig:
    """GaussianWrapping-compatible pivot and support scale."""

    std_factor: float = 3.0
    support_factor: float = 3.0
    sigma_factor: float = 1.5
    min_sigma_to_local: float = 0.05
    max_sigma_to_local: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "std_factor",
            "support_factor",
            "sigma_factor",
            "min_sigma_to_local",
            "max_sigma_to_local",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_sigma_to_local > self.max_sigma_to_local:
            raise ValueError("pivot sigma bounds must satisfy min <= max")


@dataclass(frozen=True)
class GaussianPivotSet:
    """Canonical center/outside pairs with scaffold metadata."""

    points: Tensor
    normals: Tensor
    normal_offset: Tensor
    support_radius: Tensor
    local_scale: Tensor
    quality: Tensor
    gaussian_indices: Tensor
    owner_chart_ids: Tensor
    membership_ids: Tensor
    membership_weights: Tensor
    membership_confidence: Tensor
    membership_tail: Tensor
    roles: Tensor

    def __post_init__(self) -> None:
        count = len(self.points)
        if self.points.shape != (count, 3) or self.normals.shape != (count, 3):
            raise ValueError("pivot points and normals must have shape [P,3]")
        vectors = (
            "normal_offset",
            "support_radius",
            "local_scale",
            "quality",
            "gaussian_indices",
            "owner_chart_ids",
            "membership_confidence",
            "membership_tail",
            "roles",
        )
        for name in vectors:
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(f"pivot {name} must have shape [P]")
            if value.device != self.points.device:
                raise ValueError("all pivot tensors must share one device")
        if self.membership_ids.ndim != 2 or self.membership_ids.shape[0] != count:
            raise ValueError("pivot membership_ids must have shape [P,K]")
        if self.membership_weights.shape != self.membership_ids.shape:
            raise ValueError("pivot membership_weights must have shape [P,K]")
        if (
            self.membership_ids.device != self.points.device
            or self.membership_weights.device != self.points.device
        ):
            raise ValueError("all pivot tensors must share one device")
        integer = (
            self.gaussian_indices,
            self.owner_chart_ids,
            self.membership_ids,
        )
        if any(value.dtype != torch.long for value in integer):
            raise TypeError("pivot indices and IDs must use torch.long")
        if self.roles.dtype != torch.int8:
            raise TypeError("pivot roles must use torch.int8")
        floats = (
            self.points,
            self.normals,
            self.normal_offset,
            self.support_radius,
            self.local_scale,
            self.quality,
            self.membership_weights,
            self.membership_confidence,
            self.membership_tail,
        )
        if any(not bool(torch.isfinite(value).all()) for value in floats):
            raise ValueError("pivot tensors must be finite")
        if count and bool(
            (self.normal_offset <= 0).any()
            or (self.support_radius <= 0).any()
            or (self.local_scale <= 0).any()
        ):
            raise ValueError("pivot physical scales must be positive")
        probabilities = (
            self.quality,
            self.membership_weights,
            self.membership_confidence,
            self.membership_tail,
        )
        if any(bool(((value < 0) | (value > 1)).any()) for value in probabilities):
            raise ValueError("pivot probabilities and quality must lie in [0,1]")
        total = self.membership_weights.sum(dim=1) + self.membership_tail
        if not torch.allclose(
            total,
            torch.ones_like(total),
            atol=4e-6,
            rtol=0.0,
        ):
            raise ValueError("pivot membership weights and tail must sum to one")
        if count % 2:
            raise ValueError("Gaussian pivots must contain complete pairs")
        if count:
            expected_roles = torch.tensor(
                (0, 1),
                device=self.points.device,
                dtype=torch.int8,
            ).repeat(count // 2)
            if not torch.equal(self.roles, expected_roles):
                raise ValueError("each Gaussian pivot pair must use roles [0,1]")
            pairs = self.gaussian_indices.reshape(-1, 2)
            if not bool((pairs == pairs[:, :1]).all()):
                raise ValueError("each pivot pair must share one Gaussian index")
            anchors = pairs[:, 0]
            if anchors.numel() > 1 and bool((anchors[1:] <= anchors[:-1]).any()):
                raise ValueError("pivot Gaussian pairs must be sorted and unique")
            if bool((self.owner_chart_ids < 0).any()):
                raise ValueError("every pivot must have one owner chart")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def gaussian_count(self) -> int:
        return len(self) // 2

    @property
    def anchor_gaussian_indices(self) -> Tensor:
        return self.gaussian_indices[::2].contiguous()

    def take(self, indices: Tensor | Sequence[int]) -> "GaussianPivotSet":
        selected = torch.as_tensor(
            indices,
            device=self.points.device,
            dtype=torch.long,
        ).reshape(-1)
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= len(self)
        ):
            raise IndexError("pivot index is out of range")
        return GaussianPivotSet(
            **{
                name: getattr(self, name).index_select(0, selected)
                for name in self.__dataclass_fields__
            }
        )

    def indices_for_gaussians(
        self,
        gaussian_indices: Tensor | Sequence[int],
    ) -> Tensor:
        query = torch.as_tensor(
            gaussian_indices,
            device=self.points.device,
            dtype=torch.long,
        ).reshape(-1)
        query = torch.unique(query, sorted=True)
        anchors = self.anchor_gaussian_indices
        rows = torch.searchsorted(anchors, query)
        valid = rows < len(anchors)
        if len(anchors):
            valid &= anchors[rows.clamp_max(len(anchors) - 1)] == query
        if not bool(valid.all()):
            raise IndexError("Gaussian index is outside the pivot set")
        roles = torch.arange(2, device=self.points.device, dtype=torch.long)
        return (rows[:, None] * 2 + roles[None]).reshape(-1)

    def for_chart(self, chart: Any) -> "GaussianPivotSet":
        return self.take(self.indices_for_gaussians(chart.gaussian_indices))


def _owner_chart_ids(scaffold: Any) -> Tensor:
    selected = scaffold.gaussian_indices
    owner = torch.full(
        (len(selected),),
        -1,
        device=selected.device,
        dtype=torch.long,
    )
    for chart in scaffold.charts:
        rows = torch.searchsorted(selected, chart.core_indices)
        if rows.numel() and not torch.equal(
            selected.index_select(0, rows),
            chart.core_indices,
        ):
            raise ValueError("chart core contains a Gaussian outside the scaffold")
        if rows.numel() and bool((owner.index_select(0, rows) >= 0).any()):
            raise ValueError("a scaffold Gaussian is owned by two charts")
        owner[rows] = int(chart.chart_id)
    if bool((owner < 0).any()):
        raise ValueError("the spatial scaffold left Gaussians without an owner")
    return owner


class GaussianWrappingPivotBuilder:
    """Build center/outside pivot pairs oriented toward observed cameras."""

    def __init__(self, config: GaussianPivotConfig | None = None) -> None:
        self.config = config or GaussianPivotConfig()

    def build(
        self,
        gaussians: Any,
        scaffold: Any,
        camera_centers: Tensor,
    ) -> GaussianPivotSet:
        xyz = _checkpoint_tensor(gaussians, "get_xyz")
        scaling = _checkpoint_tensor(gaussians, "get_scaling")
        rotation = _checkpoint_tensor(gaussians, "get_rotation")
        count = len(xyz)
        if xyz.shape != (count, 3) or scaling.shape != (count, 3):
            raise ValueError("Gaussian xyz and scaling must have shape [N,3]")
        if rotation.shape != (count, 4):
            raise ValueError("Gaussian rotation must have shape [N,4]")
        if count != scaffold.gaussian_count:
            raise ValueError("spatial scaffold and checkpoint Gaussian counts differ")
        if any(value.device != xyz.device for value in (scaling, rotation)):
            raise ValueError("Gaussian checkpoint tensors must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in (xyz, scaling, rotation)):
            raise ValueError("Gaussian checkpoint contains non-finite geometry")
        if bool((scaling <= 0).any()):
            raise ValueError("Gaussian activated scaling must be positive")
        if scaffold.gaussian_indices.device != xyz.device:
            raise ValueError("spatial scaffold and checkpoint must share one device")

        cameras = torch.as_tensor(
            camera_centers,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        if cameras.ndim != 2 or cameras.shape[1] != 3 or not len(cameras):
            raise ValueError("camera_centers must have shape [V,3], V > 0")
        if not bool(torch.isfinite(cameras).all()):
            raise ValueError("camera centers must be finite")

        selected = scaffold.gaussian_indices
        centers = xyz.index_select(0, selected)
        selected_scaling = scaling.index_select(0, selected)
        matrix = _quaternion_to_matrix(rotation.index_select(0, selected))
        minimum_axis = selected_scaling.argmin(dim=1)
        normal = matrix.gather(
            2,
            minimum_axis[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        normal = F.normalize(normal, dim=1, eps=1e-8)

        from scipy.spatial import cKDTree

        nearest = cKDTree(
            cameras.detach().cpu().float().numpy()
        ).query(
            centers.detach().cpu().float().numpy(),
            k=1,
            workers=-1,
        )[1]
        nearest_tensor = torch.as_tensor(
            nearest,
            device=xyz.device,
            dtype=torch.long,
        )
        towards_camera = cameras.index_select(0, nearest_tensor) - centers
        flip = torch.sum(normal * towards_camera, dim=1) < 0.0
        normal = torch.where(flip[:, None], -normal, normal)

        local_direction = torch.bmm(
            matrix.transpose(1, 2),
            normal.unsqueeze(-1),
        ).squeeze(-1)
        normal_sigma = torch.sqrt(
            (
                local_direction.square()
                * selected_scaling.square()
            ).sum(dim=1)
        )
        normal_offset = normal_sigma * float(self.config.std_factor)
        support_radius = (
            selected_scaling.max(dim=1).values
            * float(self.config.support_factor)
        )

        points = torch.stack(
            (centers, centers + normal_offset[:, None] * normal),
            dim=1,
        ).reshape(-1, 3)

        def repeat(value: Tensor) -> Tensor:
            return value[:, None].expand(-1, 2).reshape(-1)

        def repeat_matrix(value: Tensor) -> Tensor:
            return value[:, None, :].expand(-1, 2, -1).reshape(
                -1,
                value.shape[1],
            )

        owner = _owner_chart_ids(scaffold)
        membership = scaffold.membership
        return GaussianPivotSet(
            points=points,
            normals=normal[:, None].expand(-1, 2, -1).reshape(-1, 3),
            normal_offset=repeat(normal_offset),
            support_radius=repeat(support_radius),
            local_scale=repeat(scaffold.local_scale),
            quality=repeat(scaffold.quality),
            gaussian_indices=repeat(selected),
            owner_chart_ids=repeat(owner),
            membership_ids=repeat_matrix(membership.ids),
            membership_weights=repeat_matrix(membership.weights),
            membership_confidence=repeat(membership.confidence[:, 0]),
            membership_tail=repeat(membership.tail[:, 0]),
            roles=torch.tensor(
                (0, 1),
                device=xyz.device,
                dtype=torch.int8,
            ).repeat(len(selected)),
        )


@dataclass(frozen=True)
class GaussianAdaptivePivotSet:
    """Training-consistent negative/center/positive Gaussian pivots."""

    points: Tensor
    normals: Tensor
    normal_sigma: Tensor
    support_radius: Tensor
    local_scale: Tensor
    quality: Tensor
    gaussian_indices: Tensor
    owner_chart_ids: Tensor
    membership_ids: Tensor
    membership_weights: Tensor
    membership_confidence: Tensor
    membership_tail: Tensor
    roles: Tensor

    def __post_init__(self) -> None:
        count = len(self.points)
        if self.points.shape != (count, 3) or self.normals.shape != (count, 3):
            raise ValueError("pivot points and normals must have shape [P,3]")
        for name in (
            "normal_sigma",
            "support_radius",
            "local_scale",
            "quality",
            "gaussian_indices",
            "owner_chart_ids",
            "membership_confidence",
            "membership_tail",
            "roles",
        ):
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(f"pivot {name} must have shape [P]")
            if value.device != self.points.device:
                raise ValueError("all pivot tensors must share one device")
        if self.membership_ids.ndim != 2 or self.membership_ids.shape[0] != count:
            raise ValueError("pivot membership_ids must have shape [P,K]")
        if self.membership_weights.shape != self.membership_ids.shape:
            raise ValueError("pivot membership_weights must have shape [P,K]")
        if self.membership_ids.device != self.points.device:
            raise ValueError("all pivot tensors must share one device")
        if self.membership_weights.device != self.points.device:
            raise ValueError("all pivot tensors must share one device")
        if any(
            value.dtype != torch.long
            for value in (
                self.gaussian_indices,
                self.owner_chart_ids,
                self.membership_ids,
            )
        ):
            raise TypeError("pivot indices and IDs must use torch.long")
        if self.roles.dtype != torch.int8:
            raise TypeError("pivot roles must use torch.int8")
        floats = (
            self.points,
            self.normals,
            self.normal_sigma,
            self.support_radius,
            self.local_scale,
            self.quality,
            self.membership_weights,
            self.membership_confidence,
            self.membership_tail,
        )
        if any(not bool(torch.isfinite(value).all()) for value in floats):
            raise ValueError("pivot tensors must be finite")
        if count and bool(
            (self.normal_sigma <= 0).any()
            or (self.support_radius <= 0).any()
            or (self.local_scale <= 0).any()
        ):
            raise ValueError("pivot physical scales must be positive")
        if count % 3:
            raise ValueError("adaptive Gaussian pivots must contain complete triplets")
        if count:
            expected_roles = torch.tensor(
                (-1, 0, 1),
                device=self.points.device,
                dtype=torch.int8,
            ).repeat(count // 3)
            if not torch.equal(self.roles, expected_roles):
                raise ValueError("each adaptive pivot triplet must use roles [-1,0,1]")
            triplets = self.gaussian_indices.reshape(-1, 3)
            if not bool((triplets == triplets[:, :1]).all()):
                raise ValueError("each pivot triplet must share one Gaussian index")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def gaussian_count(self) -> int:
        return len(self) // 3

    @property
    def anchor_gaussian_indices(self) -> Tensor:
        return self.gaussian_indices[::3].contiguous()

    def take(self, indices: Tensor | Sequence[int]) -> "GaussianAdaptivePivotSet":
        selected = torch.as_tensor(
            indices,
            device=self.points.device,
            dtype=torch.long,
        ).reshape(-1)
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= len(self)
        ):
            raise IndexError("pivot index is out of range")
        return GaussianAdaptivePivotSet(
            **{
                name: getattr(self, name).index_select(0, selected)
                for name in self.__dataclass_fields__
            }
        )

    def indices_for_gaussians(
        self,
        gaussian_indices: Tensor | Sequence[int],
    ) -> Tensor:
        query = torch.as_tensor(
            gaussian_indices,
            device=self.points.device,
            dtype=torch.long,
        ).reshape(-1)
        query = torch.unique(query, sorted=True)
        anchors = self.anchor_gaussian_indices
        rows = torch.searchsorted(anchors, query)
        valid = rows < len(anchors)
        if len(anchors):
            valid &= anchors[rows.clamp_max(len(anchors) - 1)] == query
        if not bool(valid.all()):
            raise IndexError("Gaussian index is outside the pivot set")
        roles = torch.arange(3, device=self.points.device, dtype=torch.long)
        return (rows[:, None] * 3 + roles[None]).reshape(-1)

    def for_chart(self, chart: Any) -> "GaussianAdaptivePivotSet":
        return self.take(self.indices_for_gaussians(chart.gaussian_indices))


class GaussianAdaptivePivotBuilder:
    """Build the same center and signed normal probes used by surface training."""

    def __init__(self, config: GaussianPivotConfig | None = None) -> None:
        self.config = config or GaussianPivotConfig()

    def build(
        self,
        gaussians: Any,
        scaffold: Any,
        evidence: Any | None = None,
    ) -> GaussianAdaptivePivotSet:
        xyz = _checkpoint_tensor(gaussians, "get_xyz")
        scaling = _checkpoint_tensor(gaussians, "get_scaling")
        rotation = _checkpoint_tensor(gaussians, "get_rotation")
        selected = scaffold.gaussian_indices
        centers = xyz.index_select(0, selected)
        selected_scaling = scaling.index_select(0, selected)
        matrix = _quaternion_to_matrix(rotation.index_select(0, selected))
        minimum_axis = selected_scaling.argmin(dim=1)
        normal = matrix.gather(
            2,
            minimum_axis[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        normal = F.normalize(normal, dim=1, eps=1e-8)

        if evidence is not None:
            observed = torch.as_tensor(
                evidence.normal,
                device=xyz.device,
                dtype=xyz.dtype,
            ).index_select(0, selected)
            flip = torch.sum(normal * observed, dim=1) < 0.0
            normal = torch.where(flip[:, None], -normal, normal)

        normal_sigma = selected_scaling.gather(
            1,
            minimum_axis[:, None],
        ).squeeze(1)
        minimum_offset = (
            scaffold.local_scale * float(self.config.min_sigma_to_local)
        )
        maximum_offset = (
            scaffold.local_scale * float(self.config.max_sigma_to_local)
        )
        normal_offset = (
            normal_sigma * float(self.config.sigma_factor)
        ).clamp(min=minimum_offset, max=maximum_offset)
        support_radius = (
            selected_scaling.max(dim=1).values
            * float(self.config.support_factor)
        )
        roles = torch.tensor(
            (-1.0, 0.0, 1.0),
            device=xyz.device,
            dtype=xyz.dtype,
        )
        points = (
            centers[:, None, :]
            + roles[None, :, None]
            * normal_offset[:, None, None]
            * normal[:, None, :]
        ).reshape(-1, 3)

        def repeat(value: Tensor) -> Tensor:
            return value[:, None].expand(-1, 3).reshape(-1)

        def repeat_matrix(value: Tensor) -> Tensor:
            return value[:, None, :].expand(-1, 3, -1).reshape(
                -1,
                value.shape[1],
            )

        owner = _owner_chart_ids(scaffold)
        membership = scaffold.membership
        quality = scaffold.quality
        if evidence is not None:
            evidence_confidence = torch.as_tensor(
                evidence.confidence,
                device=xyz.device,
                dtype=xyz.dtype,
            ).index_select(0, selected)
            quality = torch.sqrt(
                quality.clamp(0.0, 1.0)
                * evidence_confidence.clamp(0.0, 1.0)
            )
        return GaussianAdaptivePivotSet(
            points=points,
            normals=normal[:, None, :].expand(-1, 3, -1).reshape(-1, 3),
            normal_sigma=repeat(normal_sigma),
            support_radius=repeat(support_radius),
            local_scale=repeat(scaffold.local_scale),
            quality=repeat(quality),
            gaussian_indices=repeat(selected),
            owner_chart_ids=repeat(owner),
            membership_ids=repeat_matrix(membership.ids),
            membership_weights=repeat_matrix(membership.weights),
            membership_confidence=repeat(membership.confidence[:, 0]),
            membership_tail=repeat(membership.tail[:, 0]),
            roles=torch.tensor(
                (-1, 0, 1),
                device=xyz.device,
                dtype=torch.int8,
            ).repeat(len(selected)),
        )


__all__ = [
    "GaussianAdaptivePivotBuilder",
    "GaussianAdaptivePivotSet",
    "GaussianPivotConfig",
    "GaussianPivotSet",
    "GaussianWrappingPivotBuilder",
]
