from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import mesh.opacity_field as field_module
from mesh.cameras import MeshCamera
from mesh.opacity_field import OpacityFieldConfig, RendererOpacityField


def _camera(uid: int) -> MeshCamera:
    return MeshCamera(
        uid=uid,
        width=32,
        height=24,
        fx=20.0,
        fy=20.0,
        cx=15.5,
        cy=11.5,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.tensor([0.0, 0.0, -2.0 - uid]),
    )


class _Context:
    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.gaussian_visibility = torch.tensor([True, uid == 0])

    def query(self, points, *, chunk_size, visibility_threshold):
        del chunk_size, visibility_threshold
        offset = 0.0 if self.uid == 0 else 0.1
        alpha = (0.5 + 0.25 * points[:, 0] + offset).clamp(0.0, 1.0)
        inside = torch.ones(len(points), dtype=torch.bool)
        return SimpleNamespace(
            alpha=alpha,
            transmittance=1.0 - alpha,
            inside=inside,
            visibility=inside,
        )


@pytest.fixture
def opacity_field(monkeypatch) -> RendererOpacityField:
    gaussians = SimpleNamespace(
        get_xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        get_normal=torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    monkeypatch.setattr(
        field_module,
        "prepare_point_integration",
        lambda camera, *args, **kwargs: _Context(camera.uid),
    )
    return RendererOpacityField(
        (_camera(0), _camera(1)),
        gaussians,
        SimpleNamespace(),
        config=OpacityFieldConfig(
            occupancy_threshold=0.5,
            candidate_views=2,
            query_chunk_size=2,
        ),
    )


def test_global_field_uses_conservative_renderer_alpha_and_sparse_owners(
    opacity_field,
) -> None:
    points = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    )

    samples = opacity_field.query(points)

    torch.testing.assert_close(samples.occupancy, torch.tensor([0.25, 0.5, 0.75]))
    torch.testing.assert_close(samples.phi, torch.tensor([0.25, 0.0, -0.25]))
    assert samples.valid.tolist() == [True, True, True]
    assert samples.support_views.tolist() == [2, 2, 2]
    assert samples.view_ids[:, 0].tolist() == [0, 0, 0]
    evidence = opacity_field.gaussian_evidence()
    assert evidence.visible_count.tolist() == [2.0, 1.0]


def test_candidate_first_refinement_shares_one_view_zero_set(
    opacity_field,
) -> None:
    endpoints = torch.tensor(
        [
            [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[-0.5, 1.0, 1.0], [0.75, 1.0, 1.0]],
        ]
    )
    candidates = torch.tensor([[0, 1], [0, 1]])

    roots = opacity_field.refine_edges(
        endpoints,
        candidate_view_ids=candidates,
        binary_steps=12,
    )

    assert roots.valid.tolist() == [True, True]
    assert torch.max(torch.abs(roots.vertices[:, 0])) < 5e-4
    assert torch.all((roots.interpolation > 0.0) & (roots.interpolation < 1.0))
