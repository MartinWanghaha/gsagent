"""Bounded, triangle-aware correspondences for lagged mesh supervision.

The CPU broad phase is deliberately discrete and detached.  Only a bounded
number of face candidates enters the exact Torch narrow phase, which runs on
the query device and avoids the vertex-nearest bias of a point-cloud lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TriangleProjection:
    """Detached closest-surface targets for one query batch."""

    closest_points: Tensor
    normals: Tensor
    semantic: Tensor | None
    uncertainty: Tensor | None
    local_spacing: Tensor
    distance: Tensor
    valid: Tensor
    face_indices: Tensor


def geman_mcclure(residual: Tensor, delta: float = 1.5) -> Tensor:
    """A finite, bounded robust penalty in ``[0, 1)`` for finite inputs."""

    if not isinstance(residual, Tensor) or not residual.is_floating_point():
        raise TypeError("residual must be a floating-point torch.Tensor")
    if not math.isfinite(float(delta)) or float(delta) <= 0:
        raise ValueError("delta must be finite and positive")
    # 1 - 1/(1+x^2) is stable when x is large, unlike x^2/(1+x^2),
    # whose numerator and denominator can both overflow.
    limit = 0.5 * math.sqrt(torch.finfo(residual.dtype).max)
    scaled = (residual.abs() / float(delta)).clamp_max(limit)
    return 1.0 - torch.reciprocal(1.0 + scaled.square())


def detached_local_scale(
    query_scale: Tensor | float,
    mesh_spacing: Tensor,
    *,
    mesh_weight: float = 0.5,
    eps: float = 1e-8,
) -> Tensor:
    """Combine Gaussian thickness and mesh spacing without scale cheating."""

    if not isinstance(mesh_spacing, Tensor) or not mesh_spacing.is_floating_point():
        raise TypeError("mesh_spacing must be a floating-point torch.Tensor")
    if mesh_weight < 0 or not math.isfinite(float(mesh_weight)):
        raise ValueError("mesh_weight must be finite and non-negative")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite and positive")
    scale = torch.as_tensor(query_scale, device=mesh_spacing.device, dtype=mesh_spacing.dtype)
    # Gaussian attributes commonly retain a trailing singleton channel while
    # mesh spacing is [Q].  Treat those as the same per-query scalar instead
    # of accidentally broadcasting them to a [Q,Q] matrix.
    if scale.ndim == mesh_spacing.ndim + 1 and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    scale, spacing = torch.broadcast_tensors(scale, mesh_spacing)
    result = torch.sqrt(scale.square() + (float(mesh_weight) * spacing).square())
    return result.detach().clamp_min(float(eps))


def robust_point_to_plane_loss(
    points: Tensor,
    projection: TriangleProjection,
    query_scale: Tensor | float,
    *,
    weights: Tensor | None = None,
    delta: float = 1.5,
    mesh_weight: float = 0.5,
) -> Tensor:
    """Scale-invariant robust point-to-plane loss over valid matches."""

    if points.ndim != 2 or points.shape[-1] != 3 or not points.is_floating_point():
        raise ValueError("points must be a floating-point tensor with shape [Q,3]")
    if projection.closest_points.shape != points.shape or projection.normals.shape != points.shape:
        raise ValueError("projection geometry must have the same shape as points")
    scale = detached_local_scale(
        query_scale,
        projection.local_spacing.to(points),
        mesh_weight=mesh_weight,
    )
    residual = (
        (points - projection.closest_points.to(points).detach())
        * projection.normals.to(points).detach()
    ).sum(-1) / scale
    penalty = geman_mcclure(residual, delta)
    valid = projection.valid.to(device=points.device)
    weight = valid.to(points.dtype)
    if weights is not None:
        supplied = torch.as_tensor(weights, device=points.device, dtype=points.dtype)
        if supplied.shape != weight.shape:
            raise ValueError("weights must have shape [Q]")
        supplied = torch.where(torch.isfinite(supplied), supplied, torch.zeros_like(supplied))
        weight = weight * supplied.detach().clamp_min(0)
    return (penalty * weight).sum() / weight.sum().clamp_min(1.0)


def _closest_on_segments(points: Tensor, start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    edge = end - start
    denominator = edge.square().sum(-1)
    parameter = ((points - start) * edge).sum(-1) / denominator.clamp_min(1e-20)
    parameter = torch.where(denominator > 1e-20, parameter, torch.zeros_like(parameter)).clamp(0, 1)
    return start + parameter[..., None] * edge, parameter


def _closest_points_on_triangles(points: Tensor, triangles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Exact closest points and barycentrics for corresponding point/triangle pairs."""

    a, b, c = triangles.unbind(-2)
    ab, ac = b - a, c - a
    face_cross = torch.cross(ab, ac, dim=-1)
    face_norm_sq = face_cross.square().sum(-1)

    plane_parameter = ((a - points) * face_cross).sum(-1) / face_norm_sq.clamp_min(1e-20)
    plane = points + plane_parameter[..., None] * face_cross
    relative = plane - a
    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)
    d20 = (relative * ab).sum(-1)
    d21 = (relative * ac).sum(-1)
    denominator = d00 * d11 - d01.square()
    bary_b = (d11 * d20 - d01 * d21) / denominator.clamp_min(1e-20)
    bary_c = (d00 * d21 - d01 * d20) / denominator.clamp_min(1e-20)
    bary_a = 1.0 - bary_b - bary_c
    plane_barycentric = torch.stack((bary_a, bary_b, bary_c), dim=-1)
    inside = (denominator > 1e-20) & (plane_barycentric >= -1e-6).all(-1)

    closest_ab, parameter_ab = _closest_on_segments(points, a, b)
    closest_bc, parameter_bc = _closest_on_segments(points, b, c)
    closest_ca, parameter_ca = _closest_on_segments(points, c, a)
    zero = torch.zeros_like(parameter_ab)
    barycentric_ab = torch.stack((1.0 - parameter_ab, parameter_ab, zero), dim=-1)
    barycentric_bc = torch.stack((zero, 1.0 - parameter_bc, parameter_bc), dim=-1)
    barycentric_ca = torch.stack((parameter_ca, zero, 1.0 - parameter_ca), dim=-1)

    candidates = torch.stack((plane, closest_ab, closest_bc, closest_ca), dim=-2)
    barycentrics = torch.stack(
        (plane_barycentric, barycentric_ab, barycentric_bc, barycentric_ca),
        dim=-2,
    )
    distance_sq = (candidates - points[..., None, :]).square().sum(-1)
    distance_sq[..., 0] = torch.where(
        inside,
        distance_sq[..., 0],
        torch.full_like(distance_sq[..., 0], torch.inf),
    )
    selected = distance_sq.argmin(-1)
    gather_point = selected[..., None, None].expand(*selected.shape, 1, 3)
    closest = candidates.gather(-2, gather_point).squeeze(-2)
    barycentric = barycentrics.gather(-2, gather_point).squeeze(-2)
    return closest, barycentric, face_norm_sq.sqrt()


class TriangleMeshProjector:
    """Candidate-first closest-point projector over a detached triangle mesh."""

    def __init__(
        self,
        vertices: Tensor | np.ndarray,
        faces: Tensor | np.ndarray,
        *,
        normals: Tensor | np.ndarray | None = None,
        semantic: Tensor | np.ndarray | None = None,
        uncertainty: Tensor | np.ndarray | None = None,
        k_candidates: int = 16,
        scipy_workers: int = 1,
        fallback_query_chunk: int = 1024,
        fallback_face_chunk: int = 8192,
    ) -> None:
        if k_candidates < 1 or fallback_query_chunk < 1 or fallback_face_chunk < 1:
            raise ValueError("candidate and fallback chunk sizes must be positive")
        if scipy_workers == 0 or scipy_workers < -1:
            raise ValueError("scipy_workers must be -1 or a positive integer")
        self.vertices = torch.as_tensor(vertices).detach()
        self.faces = torch.as_tensor(faces, dtype=torch.long).detach()
        if self.vertices.ndim != 2 or self.vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [V,3]")
        if not self.vertices.is_floating_point():
            self.vertices = self.vertices.float()
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3:
            raise ValueError("faces must have shape [F,3]")
        if self.faces.numel() and (int(self.faces.min()) < 0 or int(self.faces.max()) >= len(self.vertices)):
            raise IndexError("faces contain an out-of-range vertex index")
        if not bool(torch.isfinite(self.vertices).all()):
            raise ValueError("mesh vertices must be finite")
        self.normals = self._optional_attribute(normals, "normals", (3,))
        self.semantic = self._optional_attribute(semantic, "semantic", None)
        self.uncertainty = self._optional_attribute(uncertainty, "uncertainty", None)
        self.k_candidates = int(k_candidates)
        self.scipy_workers = int(scipy_workers)
        self.fallback_query_chunk = int(fallback_query_chunk)
        self.fallback_face_chunk = int(fallback_face_chunk)

        if len(self.faces):
            triangles = self.vertices.index_select(0, self.faces.reshape(-1)).reshape(-1, 3, 3)
            self._centroids = triangles.mean(1).float().cpu().contiguous()
        else:
            self._centroids = torch.empty((0, 3), dtype=torch.float32)
        self._tree = self._build_tree(self._centroids)

    def _optional_attribute(self, value, name: str, trailing: tuple[int, ...] | None) -> Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach()
        if not tensor.is_floating_point():
            tensor = tensor.float()
        if tensor.shape[0] not in {len(self.vertices), len(self.faces)}:
            raise ValueError(f"{name} must be defined per vertex or per face")
        if trailing is not None and tensor.shape[1:] != trailing:
            raise ValueError(f"{name} must have trailing shape {trailing}")
        if name == "semantic" and tensor.ndim != 2:
            raise ValueError("semantic must have shape [V,D] or [F,D]")
        if name == "uncertainty" and tensor.reshape(tensor.shape[0], -1).shape[1] != 1:
            raise ValueError("uncertainty must have one value per vertex or face")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"mesh {name} must be finite")
        return tensor

    @staticmethod
    def _build_tree(centroids: Tensor):
        if not len(centroids):
            return None
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return None
        return cKDTree(centroids.numpy().copy())

    def _fallback_candidates(self, points: Tensor, k: int) -> Tensor:
        centroids = self._centroids.to(device=points.device, dtype=points.dtype)
        outputs = []
        for query_start in range(0, len(points), self.fallback_query_chunk):
            query = points[query_start : query_start + self.fallback_query_chunk]
            best_distance = torch.full((len(query), k), torch.inf, device=query.device, dtype=query.dtype)
            best_indices = torch.zeros((len(query), k), device=query.device, dtype=torch.long)
            for face_start in range(0, len(centroids), self.fallback_face_chunk):
                block = centroids[face_start : face_start + self.fallback_face_chunk]
                distance = torch.cdist(query, block)
                indices = torch.arange(face_start, face_start + len(block), device=query.device)
                indices = indices.expand(len(query), -1)
                merged_distance = torch.cat((best_distance, distance), dim=1)
                merged_indices = torch.cat((best_indices, indices), dim=1)
                best_distance, selection = merged_distance.topk(k, largest=False, sorted=True)
                best_indices = merged_indices.gather(1, selection)
            outputs.append(best_indices)
        return torch.cat(outputs, dim=0)

    def _candidate_faces(self, points: Tensor, k: int) -> Tensor:
        detached = points.detach()
        if self._tree is None:
            # CPU cdist does not implement every low-precision dtype. Routing
            # is discrete, so FP32 is both portable and sufficient here.
            return self._fallback_candidates(detached.float(), k).to(points.device)
        query = detached.float().cpu().numpy()
        try:
            _, indices = self._tree.query(query, k=k, workers=self.scipy_workers)
        except TypeError:
            _, indices = self._tree.query(query, k=k)
        indices = np.asarray(indices, dtype=np.int64).reshape(len(points), k)
        return torch.as_tensor(indices, device=points.device, dtype=torch.long)

    def _candidate_attribute(
        self,
        attribute: Tensor | None,
        candidate_faces: Tensor,
        face_vertices: Tensor,
        barycentric: Tensor,
        reference: Tensor,
    ) -> Tensor | None:
        if attribute is None:
            return None
        value = attribute.to(device=reference.device, dtype=reference.dtype)
        if value.shape[0] == len(self.faces):
            return value.index_select(0, candidate_faces.reshape(-1)).reshape(*candidate_faces.shape, -1)
        values = value.index_select(0, face_vertices.reshape(-1)).reshape(*face_vertices.shape, -1)
        return (values * barycentric[..., None]).sum(-2)

    @torch.no_grad()
    def project(
        self,
        points: Tensor,
        *,
        query_semantic: Tensor | None = None,
        semantic_min_cosine: float | None = None,
        query_scale: Tensor | float | None = None,
        radius_factor: float | None = None,
        max_distance: float | None = None,
    ) -> TriangleProjection:
        if points.ndim != 2 or points.shape[-1] != 3 or not points.is_floating_point():
            raise ValueError("points must be a floating-point tensor with shape [Q,3]")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("query points must be finite")
        if semantic_min_cosine is not None and not -1.0 <= float(semantic_min_cosine) <= 1.0:
            raise ValueError("semantic_min_cosine must lie in [-1,1]")
        if radius_factor is not None and (query_scale is None or float(radius_factor) <= 0):
            raise ValueError("positive radius_factor requires query_scale")
        if max_distance is not None and (not math.isfinite(float(max_distance)) or float(max_distance) <= 0):
            raise ValueError("max_distance must be finite and positive")
        if query_semantic is not None:
            if query_semantic.ndim != 2 or query_semantic.shape[0] != len(points):
                raise ValueError("query_semantic must have shape [Q,D]")
            if self.semantic is None or query_semantic.shape[1] != self.semantic.shape[-1]:
                raise ValueError("query and mesh semantic dimensions must agree")
        if semantic_min_cosine is not None and query_semantic is None:
            raise ValueError("semantic gating requires query_semantic")
        if len(points) == 0 or len(self.faces) == 0:
            return self._empty(points)

        k = min(self.k_candidates, len(self.faces))
        candidate_faces = self._candidate_faces(points, k)
        faces = self.faces.to(points.device).index_select(0, candidate_faces.reshape(-1)).reshape(len(points), k, 3)
        vertices = self.vertices.to(device=points.device, dtype=points.dtype)
        triangles = vertices.index_select(0, faces.reshape(-1)).reshape(len(points), k, 3, 3)
        expanded_points = points[:, None, :].expand(-1, k, -1)
        closest, barycentric, double_area = _closest_points_on_triangles(expanded_points, triangles)
        distance = torch.linalg.vector_norm(points[:, None, :] - closest, dim=-1)

        edges = torch.stack(
            (
                triangles[..., 1, :] - triangles[..., 0, :],
                triangles[..., 2, :] - triangles[..., 1, :],
                triangles[..., 0, :] - triangles[..., 2, :],
            ),
            dim=-2,
        )
        spacing = edges.square().sum(-1).mean(-1).sqrt().clamp_min(1e-8)
        face_normal = F.normalize(
            torch.cross(edges[..., 0, :], -edges[..., 2, :], dim=-1),
            dim=-1,
            eps=1e-8,
        )
        normal = self._candidate_attribute(self.normals, candidate_faces, faces, barycentric, points)
        if normal is None:
            normal = face_normal
        else:
            normal = F.normalize(normal, dim=-1, eps=1e-8)
            normal = torch.where((normal.norm(dim=-1) > 1e-6)[..., None], normal, face_normal)
        semantic = self._candidate_attribute(self.semantic, candidate_faces, faces, barycentric, points)
        uncertainty = self._candidate_attribute(self.uncertainty, candidate_faces, faces, barycentric, points)
        if uncertainty is not None:
            uncertainty = uncertainty.squeeze(-1)

        candidate_valid = double_area > 1e-12
        if max_distance is not None:
            candidate_valid &= distance <= float(max_distance)
        if radius_factor is not None:
            scale = torch.as_tensor(query_scale, device=points.device, dtype=points.dtype)
            if scale.numel() not in {1, len(points)}:
                raise ValueError("query_scale must be scalar or have one value per query")
            scale = scale.reshape(1, 1) if scale.numel() == 1 else scale.reshape(-1, 1)
            local_scale = detached_local_scale(
                scale,
                spacing,
            )
            candidate_valid &= distance <= float(radius_factor) * local_scale
        if semantic_min_cosine is not None:
            assert semantic is not None and query_semantic is not None
            similarity = F.cosine_similarity(query_semantic[:, None, :].to(points), semantic, dim=-1, eps=1e-8)
            candidate_valid &= similarity >= float(semantic_min_cosine)

        gated_distance = torch.where(candidate_valid, distance, torch.full_like(distance, torch.inf))
        selected = gated_distance.argmin(-1)
        valid = candidate_valid.any(-1)

        def select(value: Tensor) -> Tensor:
            tail = value.shape[2:]
            gather = selected.reshape(len(points), 1, *([1] * len(tail))).expand(len(points), 1, *tail)
            return value.gather(1, gather).squeeze(1)

        selected_points = select(closest)
        selected_normals = select(normal)
        selected_spacing = select(spacing[..., None]).squeeze(-1)
        selected_distance = select(distance[..., None]).squeeze(-1)
        selected_faces = candidate_faces.gather(1, selected[:, None]).squeeze(1)
        finite_max = torch.finfo(points.dtype).max
        selected_points = torch.where(valid[:, None], selected_points, points.detach())
        selected_normals = torch.where(valid[:, None], selected_normals, torch.zeros_like(selected_normals))
        selected_spacing = torch.where(valid, selected_spacing, torch.full_like(selected_spacing, 1e-8))
        selected_distance = torch.where(valid, selected_distance, torch.full_like(selected_distance, finite_max))
        selected_faces = torch.where(valid, selected_faces, torch.full_like(selected_faces, -1))
        selected_semantic = None if semantic is None else select(semantic)
        if selected_semantic is not None:
            selected_semantic = torch.where(valid[:, None], selected_semantic, torch.zeros_like(selected_semantic))
        selected_uncertainty = None if uncertainty is None else select(uncertainty[..., None]).squeeze(-1)
        if selected_uncertainty is not None:
            selected_uncertainty = torch.where(
                valid,
                selected_uncertainty,
                torch.ones_like(selected_uncertainty),
            )
        return TriangleProjection(
            selected_points.detach(),
            selected_normals.detach(),
            None if selected_semantic is None else selected_semantic.detach(),
            None if selected_uncertainty is None else selected_uncertainty.detach(),
            selected_spacing.detach(),
            selected_distance.detach(),
            valid.detach(),
            selected_faces.detach(),
        )

    def _empty(self, points: Tensor) -> TriangleProjection:
        count = len(points)
        semantic = None
        if self.semantic is not None:
            semantic = points.new_zeros((count, self.semantic.shape[-1]))
        uncertainty = None if self.uncertainty is None else points.new_ones(count)
        return TriangleProjection(
            points.detach().clone(),
            torch.zeros_like(points),
            semantic,
            uncertainty,
            points.new_full((count,), 1e-8),
            points.new_full((count,), torch.finfo(points.dtype).max),
            torch.zeros(count, device=points.device, dtype=torch.bool),
            torch.full((count,), -1, device=points.device, dtype=torch.long),
        )


__all__ = [
    "TriangleMeshProjector",
    "TriangleProjection",
    "detached_local_scale",
    "geman_mcclure",
    "robust_point_to_plane_loss",
]
