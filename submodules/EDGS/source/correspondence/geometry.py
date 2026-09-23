"""Multi-view geometry for Gaussian-Splatting's row-vector camera matrices."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts import MultiViewTracks


@dataclass(frozen=True)
class TriangulationResult:
    points_h: torch.Tensor
    reprojection_error: torch.Tensor
    valid_observations: torch.Tensor
    accepted: torch.Tensor
    positive_depth: torch.Tensor
    triangulation_angle_deg: torch.Tensor

    @property
    def points(self) -> torch.Tensor:
        return self.points_h[:, :3]


def project_row_vector(
    points_h: torch.Tensor,
    projection_matrices: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[N,4]`` world points through EDGS ``X @ P`` matrices.

    Returns normalized x/y coordinates and the homogeneous denominator.  EDGS
    stores transposed Gaussian-Splatting matrices, hence this row-vector form is
    intentional.
    """

    if points_h.ndim != 2 or points_h.shape[-1] != 4:
        raise ValueError("points_h must have shape [N,4]")
    if projection_matrices.ndim != 3 or projection_matrices.shape[-2:] != (4, 4):
        raise ValueError("projection_matrices must have shape [V,4,4]")
    clip = torch.einsum("ni,vij->nvj", points_h, projection_matrices)
    denominator = clip[..., 3]
    safe = torch.where(
        denominator >= 0,
        denominator.clamp_min(eps),
        denominator.clamp_max(-eps),
    )
    return clip[..., :2] / safe[..., None], denominator


def _solve_weighted_dlt(
    projection_matrices: torch.Tensor,
    coordinates: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Solve homogeneous weighted DLT with EDGS' row-vector matrices."""

    # SVD is unsupported for fp16 on CUDA and is more stable in fp32.
    dtype = torch.float64 if coordinates.dtype == torch.float64 else torch.float32
    projections = projection_matrices.to(device=coordinates.device, dtype=dtype)
    xy = coordinates.to(dtype=dtype)
    weights = confidence.to(dtype=dtype).clamp_min(0.0)
    active = valid & torch.isfinite(xy).all(dim=-1) & torch.isfinite(weights)
    xy = torch.where(active[..., None], xy, torch.zeros_like(xy))
    weights = torch.where(active, weights, torch.zeros_like(weights)).sqrt()

    # With X represented as a row vector, x = (X P[:,0])/(X P[:,3]).
    # Therefore each observation contributes P[:,0]-xP[:,3] and
    # P[:,1]-yP[:,3].
    column_x = projections[None, :, :, 0]
    column_y = projections[None, :, :, 1]
    column_w = projections[None, :, :, 3]
    row_x = column_x - xy[..., 0, None] * column_w
    row_y = column_y - xy[..., 1, None] * column_w
    system = torch.stack((row_x, row_y), dim=2)
    system = system * weights[..., None, None]
    system = system.reshape(coordinates.shape[0], -1, 4)

    _, _, vh = torch.linalg.svd(system, full_matrices=False)
    points_h = vh[:, -1]
    denominator = points_h[:, 3:4]
    safe = torch.where(
        denominator >= 0,
        denominator.clamp_min(eps),
        denominator.clamp_max(-eps),
    )
    points_h = points_h / safe
    return points_h.to(dtype=coordinates.dtype)


def _triangulation_angles(
    points: torch.Tensor,
    camera_centers: torch.Tensor | None,
    valid: torch.Tensor,
) -> torch.Tensor:
    if camera_centers is None:
        return torch.full(
            (points.shape[0],),
            float("inf"),
            dtype=points.dtype,
            device=points.device,
        )
    centers = camera_centers.to(device=points.device, dtype=points.dtype)
    if centers.shape != (valid.shape[1], 3):
        raise ValueError(
            f"camera_centers must have shape {(valid.shape[1], 3)}, got {tuple(centers.shape)}"
        )
    rays = F.normalize(points[:, None] - centers[None], dim=-1, eps=1e-8)
    cosine = (rays[:, :1] * rays[:, 1:]).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    target_valid = valid[:, 1:]
    angle = angle.masked_fill(~target_valid, float("-inf"))
    maximum = angle.amax(dim=1)
    return torch.where(
        torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
    )


def weighted_multiview_dlt(
    projection_matrices: torch.Tensor,
    tracks: MultiViewTracks,
    *,
    camera_centers: torch.Tensor | None = None,
    min_views: int = 3,
    max_reprojection_error: float | None = 0.01,
    min_triangulation_angle_deg: float = 0.0,
    reject_outliers: bool = True,
    eps: float = 1e-8,
) -> TriangulationResult:
    """Triangulate multi-view tracks and reject geometrically invalid points.

    View zero is treated as the source observation and is never removed during
    robust refinement.  If enough target observations remain, the largest
    reprojection-error target is removed and DLT is solved again.
    """

    coordinates = tracks.coordinates
    views = coordinates.shape[1]
    if projection_matrices.shape != (views, 4, 4):
        raise ValueError(
            f"projection_matrices must have shape {(views, 4, 4)}, "
            f"got {tuple(projection_matrices.shape)}"
        )
    if not 2 <= min_views <= views:
        raise ValueError(f"min_views must be in [2,{views}], got {min_views}")
    if max_reprojection_error is not None and max_reprojection_error <= 0:
        raise ValueError("max_reprojection_error must be positive or None")
    if min_triangulation_angle_deg < 0:
        raise ValueError("min_triangulation_angle_deg must be non-negative")

    active = tracks.valid.clone()
    active &= torch.isfinite(coordinates).all(dim=-1)
    active &= torch.isfinite(tracks.confidence)
    active[:, 0] = True

    iterations = max(0, views - min_views) if reject_outliers else 0
    points_h = coordinates.new_empty((coordinates.shape[0], 4))
    errors = coordinates.new_full((coordinates.shape[0], views), float("inf"))
    depths = coordinates.new_full((coordinates.shape[0], views), float("-inf"))

    for iteration in range(iterations + 1):
        points_h = _solve_weighted_dlt(
            projection_matrices,
            coordinates,
            tracks.confidence,
            active,
            eps=eps,
        )
        projected, depths = project_row_vector(
            points_h,
            projection_matrices.to(device=points_h.device, dtype=points_h.dtype),
            eps=eps,
        )
        errors = torch.linalg.vector_norm(projected - coordinates, dim=-1)
        errors = errors.masked_fill(~active, float("inf"))

        if iteration == iterations or max_reprojection_error is None:
            break
        removable = active.sum(dim=1) > min_views
        current_error = errors.masked_fill(~active, float("-inf")).amax(dim=1)
        needs_refinement = removable & (current_error > max_reprojection_error)
        if not bool(needs_refinement.any().item()):
            break
        refinement_rows = torch.nonzero(
            needs_refinement, as_tuple=False
        ).squeeze(1)
        active_subset = active[refinement_rows]
        coordinates_subset = coordinates[refinement_rows]
        confidence_subset = tracks.confidence[refinement_rows]
        # A bad observation can pull the least-squares solution far enough that
        # another view has the largest residual.  Evaluate each leave-one-target
        # hypothesis and keep the deletion with the best remaining max error.
        candidate_scores = []
        for target in range(1, views):
            candidate_valid = active_subset[:, target]
            candidate_active = active_subset.clone()
            candidate_active[candidate_valid, target] = False
            candidate_points = _solve_weighted_dlt(
                projection_matrices,
                coordinates_subset,
                confidence_subset,
                candidate_active,
                eps=eps,
            )
            candidate_projection, _ = project_row_vector(
                candidate_points,
                projection_matrices.to(
                    device=candidate_points.device, dtype=candidate_points.dtype
                ),
                eps=eps,
            )
            candidate_error = torch.linalg.vector_norm(
                candidate_projection - coordinates_subset, dim=-1
            ).masked_fill(~candidate_active, float("-inf"))
            score = candidate_error.amax(dim=1)
            candidate_scores.append(
                score.masked_fill(~candidate_valid, float("inf"))
            )
        candidate_scores_tensor = torch.stack(candidate_scores, dim=1)
        best_error, best_target = candidate_scores_tensor.min(dim=1)
        remove_subset = (
            torch.isfinite(best_error)
            & (best_error < current_error[refinement_rows])
        )
        if not bool(remove_subset.any().item()):
            break
        rows = refinement_rows[remove_subset]
        active[rows, best_target[remove_subset] + 1] = False

    finite_point = torch.isfinite(points_h).all(dim=-1)
    enough_views = active.sum(dim=1) >= min_views
    positive_depth = ((depths > eps) | ~active).all(dim=1)
    active_errors = errors.masked_fill(~active, float("-inf"))
    maximum_error = active_errors.amax(dim=1)
    reprojection_ok = torch.isfinite(maximum_error)
    if max_reprojection_error is not None:
        reprojection_ok &= maximum_error <= max_reprojection_error

    angle = _triangulation_angles(points_h[:, :3], camera_centers, active)
    accepted = (
        finite_point
        & enough_views
        & positive_depth
        & reprojection_ok
        & (angle >= min_triangulation_angle_deg)
    )
    return TriangulationResult(
        points_h=points_h,
        reprojection_error=errors,
        valid_observations=active,
        accepted=accepted,
        positive_depth=positive_depth,
        triangulation_angle_deg=angle,
    )
