import pytest
import torch

from regularization.mesh_correspondence import (
    TriangleMeshProjector,
    geman_mcclure,
    robust_point_to_plane_loss,
)


def _unit_triangle(scale: float = 1.0):
    vertices = scale * torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    )
    return vertices, torch.tensor([[0, 1, 2]])


def test_triangle_projection_is_closer_than_nearest_vertex() -> None:
    vertices, faces = _unit_triangle()
    point = torch.tensor([[3.0, 3.0, 1.0]])
    result = TriangleMeshProjector(vertices, faces).project(point)

    nearest_vertex_distance = torch.cdist(point, vertices).min()
    assert result.valid.item()
    assert torch.allclose(result.closest_points, torch.tensor([[3.0, 3.0, 0.0]]))
    assert result.distance.item() == pytest.approx(1.0)
    assert result.distance.item() < nearest_vertex_distance.item()


def test_geman_mcclure_bounds_far_outliers() -> None:
    residual = torch.tensor([0.0, 1.0, 1e20])
    penalty = geman_mcclure(residual, delta=1.5)

    assert torch.isfinite(penalty).all()
    assert penalty[0].item() == 0.0
    assert 0.0 < penalty[1].item() < 1.0
    assert penalty[2].item() <= 1.0
    assert penalty[2].item() > 0.999


def test_normalized_point_to_plane_loss_is_coordinate_scale_invariant() -> None:
    def loss_for_scale(scale: float):
        vertices, faces = _unit_triangle(scale)
        points = torch.tensor([[3.0, 3.0, 1.0]]) * scale
        points.requires_grad_(True)
        projection = TriangleMeshProjector(vertices, faces).project(points)
        loss = robust_point_to_plane_loss(points, projection, query_scale=0.2 * scale)
        loss.backward()
        assert points.grad is not None and torch.isfinite(points.grad).all()
        return loss

    assert torch.allclose(loss_for_scale(1.0), loss_for_scale(100.0), atol=1e-6)


def test_column_query_scale_does_not_cross_broadcast_queries() -> None:
    vertices, faces = _unit_triangle()
    points = torch.tensor([[2.0, 2.0, 0.5], [3.0, 3.0, 1.0]], requires_grad=True)
    projection = TriangleMeshProjector(vertices, faces).project(points)

    loss = robust_point_to_plane_loss(
        points,
        projection,
        query_scale=torch.tensor([[0.1], [0.2]]),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_semantic_and_local_radius_gating_select_compatible_triangle() -> None:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    projector = TriangleMeshProjector(vertices, faces, semantic=semantic, k_candidates=2)
    point = torch.tensor([[0.25, 0.25, 0.05]])

    semantic_match = projector.project(
        point,
        query_semantic=torch.tensor([[0.0, 1.0]]),
        semantic_min_cosine=0.8,
    )
    assert semantic_match.valid.item()
    assert semantic_match.face_indices.item() == 1

    radius_reject = projector.project(point + torch.tensor([[0.0, 0.0, 10.0]]), query_scale=0.01, radius_factor=2.0)
    assert not radius_reject.valid.item()
    assert radius_reject.face_indices.item() == -1


def test_empty_mesh_returns_finite_invalid_projection() -> None:
    projector = TriangleMeshProjector(torch.empty(0, 3), torch.empty(0, 3, dtype=torch.long))
    points = torch.tensor([[1.0, 2.0, 3.0]])
    result = projector.project(points)

    assert not result.valid.any()
    assert torch.isfinite(result.closest_points).all()
    assert torch.isfinite(result.normals).all()
    assert torch.isfinite(result.local_spacing).all()
    assert torch.isfinite(result.distance).all()


def test_torch_fallback_keeps_candidate_width_bounded() -> None:
    vertices = []
    faces = []
    for index in range(10):
        offset = 2.0 * index
        vertices.extend(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]]
        )
        faces.append([3 * index, 3 * index + 1, 3 * index + 2])
    projector = TriangleMeshProjector(
        torch.tensor(vertices),
        torch.tensor(faces),
        k_candidates=3,
        fallback_query_chunk=1,
        fallback_face_chunk=2,
    )
    projector._tree = None

    candidates = projector._candidate_faces(torch.tensor([[0.1, 0.1, 0.2], [8.1, 0.1, 0.2]]), 3)
    result = projector.project(torch.tensor([[0.1, 0.1, 0.2], [8.1, 0.1, 0.2]]))

    assert candidates.shape == (2, 3)
    assert result.valid.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_projector_narrow_phase_runs_on_cuda() -> None:
    vertices, faces = _unit_triangle()
    result = TriangleMeshProjector(vertices, faces).project(
        torch.tensor([[2.0, 2.0, 0.5]], device="cuda")
    )
    assert result.closest_points.is_cuda
    assert result.valid.item()
