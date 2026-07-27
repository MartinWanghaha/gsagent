from __future__ import annotations

import math

import pytest
import torch

from diff_semantic_gaussian_rasterization import (
    GaussianRasterizationSettings,
    has_point_integration_extension,
    prepare_point_integration,
)


def _settings(device: torch.device) -> GaussianRasterizationSettings:
    return GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=0.8,
        tanfovy=0.8,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        backend="cuda",
        antialias_sigma=0.3,
        cx=8.0,
        cy=8.0,
    )


@pytest.mark.skipif(
    not has_point_integration_extension(),
    reason="native CUDA point integration extension is not installed",
)
def test_point_integration_has_no_cpu_execution_path() -> None:
    device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        prepare_point_integration(
            torch.tensor([[0.0, 0.0, 2.0]], device=device),
            torch.ones((1, 1), device=device),
            torch.full((1, 3), 0.2, device=device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            _settings(device),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_point_integration_extension(),
    reason="native CUDA point integration is not available",
)
def test_point_integration_matches_single_gaussian_ray_and_frustum() -> None:
    device = torch.device("cuda")
    context = prepare_point_integration(
        torch.tensor([[0.0, 0.0, 2.0]], device=device),
        torch.ones((1, 1), device=device),
        torch.full((1, 3), 0.2, device=device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        _settings(device),
        query_chunk_size=2,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [100.0, 0.0, 2.0],
            [0.0, 0.0, 0.1],
        ],
        device=device,
    )
    result = context.query(
        points,
        chunk_size=2,
        visibility_threshold=0.02,
    )
    assert not any(field.requires_grad for field in result)
    expected_front_alpha = 0.99 * math.exp(-12.5)
    torch.testing.assert_close(
        result.alpha[:3],
        torch.tensor(
            [expected_front_alpha, 0.99, 0.99],
            device=device,
        ),
        rtol=2e-4,
        atol=2e-6,
    )
    assert torch.equal(
        result.inside,
        torch.tensor([True, True, True, False, False], device=device),
    )
    assert torch.equal(
        result.visibility,
        torch.tensor([True, False, False, False, False], device=device),
    )
    torch.testing.assert_close(
        result.alpha + result.transmittance,
        torch.ones_like(result.alpha),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_point_integration_extension(),
    reason="native CUDA point integration is not available",
)
def test_point_integration_is_chunk_invariant_and_preserves_shape() -> None:
    device = torch.device("cuda")
    context = prepare_point_integration(
        torch.tensor(
            [[-0.08, 0.0, 1.8], [0.11, -0.03, 2.4]],
            device=device,
        ),
        torch.tensor([[0.43], [0.67]], device=device),
        torch.tensor(
            [[0.08, 0.12, 0.05], [0.11, 0.07, 0.09]],
            device=device,
        ),
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.96, 0.1, -0.04, 0.02]],
            device=device,
        ),
        _settings(device),
    )
    points = torch.tensor(
        [
            [[-0.05, 0.0, 1.5], [0.0, 0.0, 1.9], [0.08, -0.02, 2.2]],
            [[0.12, -0.03, 2.4], [0.16, 0.01, 2.8], [0.0, 0.0, 3.1]],
        ],
        device=device,
    )
    single = context.query(points, chunk_size=1)
    batched = context.query(points, chunk_size=points.numel())
    for single_field, batched_field in zip(single, batched):
        assert single_field.shape == points.shape[:-1]
        if single_field.dtype == torch.bool:
            assert torch.equal(single_field, batched_field)
        else:
            torch.testing.assert_close(single_field, batched_field)
