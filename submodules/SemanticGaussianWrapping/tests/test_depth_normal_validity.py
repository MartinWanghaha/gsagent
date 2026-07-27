from types import SimpleNamespace

import pytest
import torch

from regularization.losses import SemanticLossSystem
from scene.gaussian_model import SemanticDecoder
from training.engine import SemanticGaussianTrainer
from utils.graphics_utils import depth_normal_residual, depth_to_normal, depth_to_points


def _camera(height: int, width: int) -> SimpleNamespace:
    return SimpleNamespace(
        FoVx=1.0,
        FoVy=1.0,
        Fx=float(width),
        Fy=float(height),
        Cx=(width - 1) * 0.5,
        Cy=(height - 1) * 0.5,
        world_view_transform=torch.eye(4),
    )


def _silhouette_step(height: int = 9, width: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    depth = torch.zeros(1, height, width)
    alpha = torch.zeros(1, height, width)
    depth[:, :, :5] = 2.0
    alpha[:, :, :5] = 1.0
    return depth, alpha


def test_depth_to_points_uses_depth_device_and_dtype_for_camera_transform() -> None:
    height, width = 3, 5
    camera = _camera(height, width)
    camera.world_view_transform = torch.eye(4, dtype=torch.float64)
    depth = torch.full((1, height, width), 2.0, dtype=torch.float32, requires_grad=True)

    points = depth_to_points(camera, depth)

    assert points.device == depth.device
    assert points.dtype == depth.dtype
    torch.testing.assert_close(points[1, 2], torch.tensor([0.0, 0.0, 2.0]))
    points.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_depth_to_points_accepts_cpu_camera_transform_for_cuda_depth() -> None:
    height, width = 3, 5
    camera = _camera(height, width)
    camera.world_view_transform = torch.eye(4, dtype=torch.float64, device="cpu")
    depth = torch.full(
        (1, height, width),
        2.0,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    points = depth_to_points(camera, depth)

    assert points.device == depth.device
    assert points.dtype == depth.dtype
    torch.testing.assert_close(
        points[1, 2],
        torch.tensor([0.0, 0.0, 2.0], device=depth.device),
    )
    points.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


def test_depth_to_normal_erodes_silhouette_but_keeps_plane_interior() -> None:
    depth, alpha = _silhouette_step()
    camera = _camera(9, 9)
    normals, valid = depth_to_normal(camera, depth, alpha)
    _, depth_fallback_valid = depth_to_normal(camera, depth)

    assert valid[2:-2, 2:4].all()
    assert not valid[:, 4:].any()
    assert torch.equal(valid, depth_fallback_valid)
    assert torch.count_nonzero(normals[:, :, 4:]) == 0
    assert torch.isfinite(normals).all()


def test_normal_loss_ignores_wrong_normals_at_depth_silhouette() -> None:
    depth, alpha = _silhouette_step()
    camera = _camera(9, 9)
    depth_normals, valid = depth_to_normal(camera, depth, alpha)
    rendered_normal = depth_normals.clone()
    rendered_normal[:, ~valid] = torch.tensor([1.0, 0.0, 0.0])[:, None]
    rendered_normal.requires_grad_()
    depth.requires_grad_()
    system = SemanticLossSystem(16, 2, {}, SemanticDecoder(16, 2))

    loss = system.normal_consistency_loss(
        {
            "render": torch.zeros(3, 9, 9),
            "expected_depth": depth,
            "normal": rendered_normal,
            "alpha": alpha,
        },
        camera,
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)
    loss.backward()
    assert torch.count_nonzero(rendered_normal.grad[:, ~valid]) == 0
    assert torch.count_nonzero(depth.grad[:, :, 5:]) == 0


def test_low_alpha_has_zero_loss_gradient_and_zero_density_residual() -> None:
    height = width = 7
    camera = _camera(height, width)
    depth = torch.full((1, height, width), 2.0, requires_grad=True)
    rendered_normal = torch.randn(3, height, width, requires_grad=True)
    alpha = torch.full((1, height, width), 0.49)
    system = SemanticLossSystem(16, 2, {}, SemanticDecoder(16, 2))
    package = {
        "render": torch.zeros(3, height, width),
        "expected_depth": depth,
        "normal": rendered_normal,
        "alpha": alpha,
    }

    loss = system.normal_consistency_loss(package, camera)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(depth.grad) == 0
    assert torch.count_nonzero(rendered_normal.grad) == 0

    trainer = object.__new__(SemanticGaussianTrainer)
    trainer.loss_system = system
    residual = trainer._geometry_residual(package, camera)
    assert residual is not None
    assert torch.count_nonzero(residual) == 0


def test_depth_normal_residual_keeps_valid_plane_evidence() -> None:
    height = width = 7
    camera = _camera(height, width)
    depth = torch.full((1, height, width), 2.0)
    alpha = torch.ones(1, height, width)
    depth_normals, valid = depth_to_normal(camera, depth, alpha)
    wrong = depth_normals.clone()
    wrong[:, valid] = torch.tensor([1.0, 0.0, 0.0])[:, None]

    residual, residual_valid = depth_normal_residual(camera, depth, wrong, alpha)
    assert torch.equal(valid, residual_valid)
    assert torch.all(residual[valid] > 0.9)
    assert torch.count_nonzero(residual[~valid]) == 0
