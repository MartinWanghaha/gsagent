"""Renderer-consistent global opacity field for Gaussian Wrapping.

The field streams one prepared camera at a time.  This keeps memory independent
of the number of training views while preserving the exact CUDA rasterizer
ordering and opacity convention used by image rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_renderer import prepare_point_integration

from .cameras import MeshCamera


@dataclass(frozen=True)
class OpacityFieldConfig:
    """One global scalar-field policy shared by pivots and root refinement."""

    occupancy_threshold: float = 0.5
    minimum_views: int = 1
    candidate_views: int = 2
    query_chunk_size: int = 65_536
    visibility_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must lie in (0,1)")
        if self.minimum_views < 1:
            raise ValueError("minimum_views must be positive")
        if self.candidate_views < 1:
            raise ValueError("candidate_views must be positive")
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must lie in [0,1]")


@dataclass(frozen=True)
class OpacityFieldSamples:
    """Global field values and sparse controlling-view ownership."""

    phi: Tensor
    occupancy: Tensor
    valid: Tensor
    support_views: Tensor
    confidence: Tensor
    view_ids: Tensor

    def __post_init__(self) -> None:
        count = len(self.phi)
        if self.phi.shape != (count,) or self.occupancy.shape != (count,):
            raise ValueError("phi and occupancy must have shape [P]")
        if self.valid.shape != (count,) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [P]")
        if self.support_views.shape != (count,):
            raise ValueError("support_views must have shape [P]")
        if self.confidence.shape != (count,):
            raise ValueError("confidence must have shape [P]")
        if self.view_ids.ndim != 2 or len(self.view_ids) != count:
            raise ValueError("view_ids must have shape [P,K]")
        if not bool(torch.isfinite(self.phi).all()):
            raise ValueError("phi must be finite")
        if not bool(torch.isfinite(self.occupancy).all()):
            raise ValueError("occupancy must be finite")
        if not bool(torch.isfinite(self.confidence).all()):
            raise ValueError("confidence must be finite")


@dataclass(frozen=True)
class RefinedFieldRoots:
    """Candidate-first roots refined within one controlling view each."""

    vertices: Tensor
    interpolation: Tensor
    valid: Tensor
    confidence: Tensor


class RendererOpacityField:
    """Conservative multi-view opacity field with bounded prepared-view memory."""

    def __init__(
        self,
        cameras: Sequence[MeshCamera],
        gaussians: Any,
        pipeline: Any,
        *,
        config: OpacityFieldConfig | None = None,
        progress_callback: Optional[Any] = None,
    ) -> None:
        if not cameras:
            raise ValueError("renderer opacity field requires at least one camera")
        ids = [int(camera.uid) for camera in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("mesh camera UIDs must be unique")
        self.cameras = tuple(cameras)
        self._camera_by_id = {
            int(camera.uid): camera for camera in self.cameras
        }
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.config = config or OpacityFieldConfig()
        if self.config.minimum_views > len(self.cameras):
            raise ValueError("minimum_views exceeds selected camera count")
        self.progress_callback = progress_callback
        self._visible_count: Optional[Tensor] = None
        self._oriented_normal_sum: Optional[Tensor] = None
        self._captured_view_ids: set[int] = set()

    @property
    def device(self) -> torch.device:
        return self.gaussians.get_xyz.device

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _prepare(self, camera: MeshCamera):
        return prepare_point_integration(
            camera,
            self.gaussians,
            self.pipeline,
            query_chunk_size=self.config.query_chunk_size,
        )

    @staticmethod
    def _camera_mask(camera: MeshCamera, points: Tensor) -> Tensor:
        if camera.gt_mask is None:
            return torch.ones(
                len(points),
                device=points.device,
                dtype=torch.bool,
            )
        ones = torch.ones(
            (len(points), 1),
            device=points.device,
            dtype=points.dtype,
        )
        camera_points = torch.cat((points, ones), dim=1) @ (
            camera.world_view_transform.to(points)
        )
        depth = camera_points[:, 2]
        safe_depth = depth.clamp_min(1e-8)
        sample_x = camera.fx * camera_points[:, 0] / safe_depth + camera.cx + 0.5
        sample_y = camera.fy * camera_points[:, 1] / safe_depth + camera.cy + 0.5
        grid = torch.stack(
            (
                2.0 * sample_x / camera.width - 1.0,
                2.0 * sample_y / camera.height - 1.0,
            ),
            dim=1,
        )
        sampled = F.grid_sample(
            camera.gt_mask.to(points)[None],
            grid[None, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0, 0]
        return (depth > 0.0) & (sampled > 0.5)

    @torch.no_grad()
    def _capture_gaussian_evidence(
        self,
        camera: MeshCamera,
        gaussian_visibility: Tensor,
    ) -> None:
        uid = int(camera.uid)
        if uid in self._captured_view_ids:
            return
        self._captured_view_ids.add(uid)
        count = len(self.gaussians.get_xyz)
        if self._visible_count is None:
            self._visible_count = torch.zeros(
                count,
                device=self.device,
                dtype=torch.int32,
            )
            self._oriented_normal_sum = torch.zeros(
                (count, 3),
                device=self.device,
                dtype=torch.float32,
            )
        visible = torch.as_tensor(
            gaussian_visibility,
            device=self.device,
            dtype=torch.bool,
        )
        self._visible_count[visible] += 1
        if bool(visible.any()):
            indices = torch.nonzero(visible, as_tuple=False).flatten()
            normal = self.gaussians.get_normal.index_select(0, indices)
            towards_camera = (
                camera.camera_center.to(self.device)
                - self.gaussians.get_xyz.index_select(0, indices)
            )
            flip = torch.sum(normal * towards_camera, dim=1) < 0.0
            oriented = torch.where(flip[:, None], -normal, normal)
            self._oriented_normal_sum.index_add_(0, indices, oriented)

    @torch.no_grad()
    def gaussian_evidence(self):
        """Return accumulated visibility evidence in the RegionAtlas contract."""

        from .region_atlas import GaussianEvidence

        count = len(self.gaussians.get_xyz)
        if self._visible_count is None or self._oriented_normal_sum is None:
            visible = torch.zeros(count, device=self.device, dtype=torch.int32)
            normal = self.gaussians.get_normal
        else:
            visible = self._visible_count
            accumulated = self._oriented_normal_sum
            fallback = self.gaussians.get_normal
            normal = torch.where(
                (torch.linalg.vector_norm(accumulated, dim=1) > 1e-8)[:, None],
                F.normalize(accumulated, dim=1, eps=1e-8),
                fallback,
            )
        denominator = max(len(self._captured_view_ids), 1)
        confidence = (visible.float() / float(denominator)).clamp(0.0, 1.0)
        return GaussianEvidence(
            visible_count=visible,
            normal=normal,
            confidence=confidence,
        )

    @torch.no_grad()
    def query(
        self,
        points: Tensor,
        *,
        candidate_view_ids: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> OpacityFieldSamples:
        """Evaluate ``threshold - min_view(alpha_to_point)``.

        A full query streams every selected camera and records the few views
        controlling the conservative minimum.  Candidate queries evaluate only
        those sparse owners and are intended for localized diagnostics; root
        refinement uses :meth:`refine_edges` to share each prepared context.
        """

        points = torch.as_tensor(points, device=self.device, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [P,3]")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("query points must be finite")
        selected_chunk = (
            self.config.query_chunk_size
            if chunk_size is None
            else int(chunk_size)
        )
        if selected_chunk < 1:
            raise ValueError("chunk_size must be positive")
        count = len(points)
        occupancy = torch.ones(count, device=self.device)
        support = torch.zeros(count, device=self.device, dtype=torch.int32)
        alpha_sum = torch.zeros(count, device=self.device)
        alpha_square_sum = torch.zeros(count, device=self.device)
        best_alpha = torch.full(
            (count, self.config.candidate_views),
            float("inf"),
            device=self.device,
        )
        best_views = torch.full(
            (count, self.config.candidate_views),
            -1,
            device=self.device,
            dtype=torch.long,
        )

        candidate = None
        if candidate_view_ids is not None:
            candidate = torch.as_tensor(
                candidate_view_ids,
                device=self.device,
                dtype=torch.long,
            )
            if candidate.ndim != 2 or len(candidate) != count:
                raise ValueError("candidate_view_ids must have shape [P,K]")
            unknown = set(candidate[candidate >= 0].unique().tolist()) - set(
                self._camera_by_id
            )
            if unknown:
                raise ValueError(f"unknown candidate camera IDs: {sorted(unknown)}")
            ordered_ids = [
                uid
                for uid in self._camera_by_id
                if bool((candidate == uid).any())
            ]
        else:
            ordered_ids = list(self._camera_by_id)

        for view_index, uid in enumerate(ordered_ids, start=1):
            camera = self._camera_by_id[uid]
            context = self._prepare(camera)
            if candidate is None:
                rows = None
                selected_points = points
                self._capture_gaussian_evidence(
                    camera,
                    context.gaussian_visibility,
                )
            else:
                rows = torch.nonzero(
                    (candidate == uid).any(dim=1),
                    as_tuple=False,
                ).flatten()
                selected_points = points.index_select(0, rows)
            if len(selected_points):
                result = context.query(
                    selected_points,
                    chunk_size=selected_chunk,
                    visibility_threshold=self.config.visibility_threshold,
                )
                accepted = result.inside & self._camera_mask(
                    camera,
                    selected_points,
                )
                alpha = result.alpha
                if rows is None:
                    occupancy = torch.where(
                        accepted,
                        torch.minimum(occupancy, alpha),
                        occupancy,
                    )
                    support += accepted.to(torch.int32)
                    alpha_sum += torch.where(accepted, alpha, 0.0)
                    alpha_square_sum += torch.where(
                        accepted,
                        alpha.square(),
                        0.0,
                    )
                    row_alpha = best_alpha
                    row_views = best_views
                else:
                    previous = occupancy.index_select(0, rows)
                    occupancy[rows] = torch.where(
                        accepted,
                        torch.minimum(previous, alpha),
                        previous,
                    )
                    support[rows] += accepted.to(torch.int32)
                    alpha_sum[rows] += torch.where(accepted, alpha, 0.0)
                    alpha_square_sum[rows] += torch.where(
                        accepted,
                        alpha.square(),
                        0.0,
                    )
                    row_alpha = best_alpha.index_select(0, rows)
                    row_views = best_views.index_select(0, rows)
                proposed_alpha = torch.cat(
                    (
                        row_alpha,
                        torch.where(
                            accepted,
                            alpha,
                            torch.full_like(alpha, float("inf")),
                        )[:, None],
                    ),
                    dim=1,
                )
                proposed_views = torch.cat(
                    (
                        row_views,
                        torch.full(
                            (len(selected_points), 1),
                            uid,
                            device=self.device,
                            dtype=torch.long,
                        ),
                    ),
                    dim=1,
                )
                values, order = torch.topk(
                    proposed_alpha,
                    k=self.config.candidate_views,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                selected_views = torch.gather(proposed_views, 1, order)
                if rows is None:
                    best_alpha = values
                    best_views = selected_views
                else:
                    best_alpha[rows] = values
                    best_views[rows] = selected_views
            del context
            if self.progress_callback is not None and (
                view_index == len(ordered_ids) or view_index % 10 == 0
            ):
                self._progress(
                    f"opacity field views {view_index}/{len(ordered_ids)}"
                )

        valid = support >= int(self.config.minimum_views)
        occupancy = torch.where(valid, occupancy, torch.zeros_like(occupancy))
        safe_support = support.clamp_min(1).float()
        mean = alpha_sum / safe_support
        variance = (
            alpha_square_sum / safe_support - mean.square()
        ).clamp_min(0.0)
        view_agreement = torch.exp(-4.0 * torch.sqrt(variance))
        coverage = (
            support.float() / float(self.config.minimum_views)
        ).clamp(max=1.0)
        confidence = torch.where(
            valid,
            view_agreement * coverage,
            torch.zeros_like(view_agreement),
        )
        best_views[~torch.isfinite(best_alpha)] = -1
        return OpacityFieldSamples(
            phi=float(self.config.occupancy_threshold) - occupancy,
            occupancy=occupancy,
            valid=valid,
            support_views=support.long(),
            confidence=confidence.clamp(0.0, 1.0),
            view_ids=best_views,
        )

    @torch.no_grad()
    def refine_edges(
        self,
        endpoints: Tensor,
        *,
        candidate_view_ids: Tensor,
        binary_steps: int,
        chunk_size: Optional[int] = None,
    ) -> RefinedFieldRoots:
        """Find one bracketing owner view, then refine all its roots in-place.

        Views are processed in controlling-rank order.  A prepared camera is
        retained for all binary steps of every edge it owns and immediately
        released, so the peak state is one camera regardless of scene size.
        """

        endpoints = torch.as_tensor(
            endpoints,
            device=self.device,
            dtype=torch.float32,
        )
        candidates = torch.as_tensor(
            candidate_view_ids,
            device=self.device,
            dtype=torch.long,
        )
        if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 3):
            raise ValueError("endpoints must have shape [E,2,3]")
        if candidates.ndim != 2 or len(candidates) != len(endpoints):
            raise ValueError("candidate_view_ids must have shape [E,K]")
        if binary_steps < 0:
            raise ValueError("binary_steps must be non-negative")
        selected_chunk = (
            self.config.query_chunk_size
            if chunk_size is None
            else int(chunk_size)
        )
        count = len(endpoints)
        vertices = endpoints.mean(dim=1)
        interpolation = torch.full(
            (count,),
            0.5,
            device=self.device,
        )
        valid = torch.zeros(count, device=self.device, dtype=torch.bool)
        confidence = torch.zeros(count, device=self.device)
        unresolved = torch.ones(count, device=self.device, dtype=torch.bool)

        ordered_ids: list[int] = []
        seen: set[int] = set()
        for rank in range(candidates.shape[1]):
            for uid in candidates[:, rank].detach().cpu().tolist():
                uid = int(uid)
                if uid >= 0 and uid not in seen:
                    if uid not in self._camera_by_id:
                        raise ValueError(f"unknown candidate camera ID {uid}")
                    seen.add(uid)
                    ordered_ids.append(uid)

        threshold = float(self.config.occupancy_threshold)
        for uid in ordered_ids:
            rows = torch.nonzero(
                unresolved & (candidates == uid).any(dim=1),
                as_tuple=False,
            ).flatten()
            if not len(rows):
                continue
            camera = self._camera_by_id[uid]
            context = self._prepare(camera)
            selected = endpoints.index_select(0, rows)
            endpoint_result = context.query(
                selected.reshape(-1, 3),
                chunk_size=selected_chunk,
                visibility_threshold=self.config.visibility_threshold,
            )
            endpoint_alpha = endpoint_result.alpha.reshape(-1, 2)
            endpoint_inside = endpoint_result.inside.reshape(-1, 2)
            mask = self._camera_mask(
                camera,
                selected.reshape(-1, 3),
            ).reshape(-1, 2)
            endpoint_phi = threshold - endpoint_alpha
            bracket = (
                endpoint_inside.all(dim=1)
                & mask.all(dim=1)
                & (endpoint_phi[:, 0] * endpoint_phi[:, 1] <= 0.0)
                & (endpoint_phi[:, 0] != endpoint_phi[:, 1])
            )
            owned_rows = rows[bracket]
            if len(owned_rows):
                owned_points = selected[bracket]
                owned_phi = endpoint_phi[bracket]
                low = owned_points[:, 0].clone()
                high = owned_points[:, 1].clone()
                low_phi = owned_phi[:, 0].clone()
                high_phi = owned_phi[:, 1].clone()
                swap = low_phi > 0.0
                if bool(swap.any()):
                    low[swap], high[swap] = (
                        high[swap].clone(),
                        low[swap].clone(),
                    )
                    low_phi[swap], high_phi[swap] = (
                        high_phi[swap].clone(),
                        low_phi[swap].clone(),
                    )
                step_valid = torch.ones(
                    len(owned_rows),
                    device=self.device,
                    dtype=torch.bool,
                )
                for _ in range(binary_steps):
                    middle = 0.5 * (low + high)
                    result = context.query(
                        middle,
                        chunk_size=selected_chunk,
                        visibility_threshold=self.config.visibility_threshold,
                    )
                    middle_valid = result.inside & self._camera_mask(
                        camera,
                        middle,
                    )
                    middle_phi = threshold - result.alpha
                    step_valid &= middle_valid
                    move_low = middle_valid & (middle_phi <= 0.0)
                    move_high = middle_valid & ~move_low
                    low[move_low] = middle[move_low]
                    low_phi[move_low] = middle_phi[move_low]
                    high[move_high] = middle[move_high]
                    high_phi[move_high] = middle_phi[move_high]
                root = 0.5 * (low + high)
                direction = owned_points[:, 1] - owned_points[:, 0]
                norm_squared = torch.sum(
                    direction * direction,
                    dim=1,
                ).clamp_min(1e-16)
                root_t = (
                    torch.sum(
                        (root - owned_points[:, 0]) * direction,
                        dim=1,
                    )
                    / norm_squared
                ).clamp(0.0, 1.0)
                vertices[owned_rows] = root
                interpolation[owned_rows] = root_t
                valid[owned_rows] = step_valid
                confidence[owned_rows] = (
                    endpoint_alpha[bracket, 0]
                    - endpoint_alpha[bracket, 1]
                ).abs().clamp(0.0, 1.0)
                unresolved[owned_rows] = False
            del context

        return RefinedFieldRoots(
            vertices=vertices,
            interpolation=interpolation,
            valid=valid,
            confidence=confidence,
        )


__all__ = [
    "OpacityFieldConfig",
    "OpacityFieldSamples",
    "RefinedFieldRoots",
    "RendererOpacityField",
]
