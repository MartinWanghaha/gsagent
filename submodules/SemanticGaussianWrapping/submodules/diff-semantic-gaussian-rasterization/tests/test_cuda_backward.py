from __future__ import annotations

import pytest
import torch

from diff_semantic_gaussian_rasterization import (
    GaussianRasterizationSettings,
    has_cuda_extension,
    project_gaussians,
    rasterize_gaussians,
)


def _settings(device: torch.device, backend: str) -> GaussianRasterizationSettings:
    return GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=0.8,
        tanfovy=0.8,
        bg=torch.tensor([0.07, 0.11, 0.19], device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        backend=backend,
        chunk_size=2,
        antialias_sigma=0.3,
    )


def _inputs(device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1729)
    means3d = torch.tensor(
        [[-0.12, -0.04, 1.7], [0.09, 0.05, 2.1], [0.02, -0.11, 2.6]],
        device=device,
    )
    means2d = torch.zeros((3, 3), device=device)
    colors = torch.rand((3, 3), generator=generator, device=device) * 0.8 + 0.1
    semantics = torch.randn((3, 16), generator=generator, device=device) * 0.2
    opacities = torch.tensor([[0.37], [0.52], [0.44]], device=device)
    # A unique minimum axis avoids the deliberately discrete normal-axis tie.
    scales = torch.tensor(
        [[0.075, 0.042, 0.097], [0.064, 0.091, 0.039], [0.083, 0.046, 0.068]],
        device=device,
    )
    rotations = torch.tensor(
        [[1.0, 0.08, -0.03, 0.02], [0.96, -0.11, 0.17, 0.04], [0.91, 0.13, 0.07, -0.18]],
        device=device,
    )
    return [tensor.requires_grad_() for tensor in (
        means3d,
        means2d,
        colors,
        semantics,
        opacities,
        scales,
        rotations,
    )]


def _loss(outputs) -> torch.Tensor:
    # Nonuniform upstream gradients exercise every normalization and the
    # background-transmittance branch, rather than only sum reductions.
    terms = []
    for index, output in enumerate(outputs[:5]):
        weight = torch.linspace(
            -0.31 + index * 0.07,
            0.47 + index * 0.03,
            output.numel(),
            device=output.device,
            dtype=output.dtype,
        ).reshape_as(output)
        terms.append((output * weight).sum())
    return torch.stack(terms).sum()


def test_reference_all_continuous_inputs_receive_finite_gradients() -> None:
    inputs = _inputs(torch.device("cpu"))
    outputs = rasterize_gaussians(*inputs, _settings(torch.device("cpu"), "reference"))
    _loss(outputs).backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
        assert torch.isfinite(tensor.grad).all()


def test_reference_supports_off_center_principal_point_and_centered_default() -> None:
    device = torch.device("cpu")
    inputs = [tensor[:1].detach().clone() for tensor in _inputs(device)]
    inputs[0][0] = torch.tensor([0.0, 0.0, 2.0])
    inputs[1].zero_()

    default_settings = _settings(device, "reference")
    explicit_center = default_settings._replace(cx=8.0, cy=8.0)
    default_output = rasterize_gaussians(*inputs, default_settings)
    explicit_output = rasterize_gaussians(*inputs, explicit_center)
    for default_field, explicit_field in zip(default_output[:6], explicit_output[:6]):
        torch.testing.assert_close(default_field, explicit_field, rtol=0.0, atol=0.0)
    assert torch.equal(default_output.dominant_index, explicit_output.dominant_index)

    # Pixel samples lie at x/y + 0.5, so this principal point puts the peak
    # exactly at pixel (x=3, y=11), far from the image center.
    off_center = default_settings._replace(cx=3.5, cy=11.5)
    shifted = rasterize_gaussians(*inputs, off_center)
    peak = int(shifted.alpha.reshape(-1).argmax())
    peak_y, peak_x = divmod(peak, off_center.image_width)
    assert (peak_x, peak_y) == (3, 11)
    assert shifted.dominant_index[peak_y, peak_x].item() == 0


def test_reference_clamps_off_frustum_covariance_but_not_projected_mean() -> None:
    device = torch.device("cpu")
    settings = _settings(device, "reference")
    depth = 2.0
    limit = 1.3 * settings.tanfovx
    means3d = torch.tensor(
        [[limit * depth, 0.0, depth], [4.0 * limit * depth, 0.0, depth]],
        device=device,
    )
    means2d = torch.zeros_like(means3d)
    scales = torch.full_like(means3d, 0.12)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    projected = project_gaussians(
        means3d,
        means2d,
        scales,
        rotations,
        settings.viewmatrix,
        settings.image_height,
        settings.image_width,
        settings.tanfovx,
        settings.tanfovy,
    )

    # Covariance uses the same clamped camera coordinate, while the actual
    # pinhole means still distinguish the two off-frustum positions.
    torch.testing.assert_close(projected.conic[0], projected.conic[1])
    assert projected.means[1, 0] > projected.means[0, 0]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_extension(),
    reason="native CUDA rasterizer is not available",
)
def test_native_cuda_culls_near_camera_outlier_with_finite_gradients() -> None:
    device = torch.device("cuda")
    inputs = _inputs(device)
    with torch.no_grad():
        inputs[0][0] = torch.tensor([-4.43, 5.99, 4.0e-4], device=device)
        inputs[5][0] = 0.061

    outputs = rasterize_gaussians(*inputs, _settings(device, "cuda"))

    assert outputs.radii[0] == 0
    assert (outputs.radii[1:] > 0).all()
    _loss(outputs).backward()
    torch.cuda.synchronize()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(inputs[0].grad[0]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_extension(),
    reason="native CUDA rasterizer is not available",
)
def test_native_cuda_forward_and_backward_match_reference() -> None:
    device = torch.device("cuda")
    reference_inputs = _inputs(device)
    native_inputs = [tensor.detach().clone().requires_grad_() for tensor in reference_inputs]
    reference_settings = _settings(device, "reference")._replace(cx=6.25, cy=9.1)
    native_settings = _settings(device, "cuda")._replace(cx=6.25, cy=9.1)

    reference = rasterize_gaussians(
        *reference_inputs,
        reference_settings,
    )
    native = rasterize_gaussians(
        *native_inputs,
        native_settings,
    )
    _loss(reference).backward()
    _loss(native).backward()
    torch.cuda.synchronize()

    for reference_output, native_output in zip(reference[:6], native[:6]):
        torch.testing.assert_close(native_output, reference_output, rtol=2e-3, atol=3e-4)
    assert torch.equal(native.dominant_index, reference.dominant_index)
    for reference_input, native_input in zip(reference_inputs, native_inputs):
        torch.testing.assert_close(
            native_input.grad,
            reference_input.grad,
            rtol=8e-3,
            atol=8e-4,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_extension(),
    reason="native CUDA rasterizer is not available",
)
def test_native_cuda_off_frustum_covariance_clamp_matches_reference() -> None:
    device = torch.device("cuda")
    reference_inputs = _inputs(device)
    with torch.no_grad():
        # x/z=1.1 lies beyond the 1.3*tan(FoV)=1.04 covariance
        # domain, while this Gaussian is large enough to overlap the image.
        reference_inputs[0][0] = torch.tensor([2.2, 0.0, 2.0], device=device)
        reference_inputs[5][0] = torch.tensor([0.4, 0.3, 0.35], device=device)
    native_inputs = [tensor.detach().clone().requires_grad_() for tensor in reference_inputs]
    reference = rasterize_gaussians(
        *reference_inputs,
        _settings(device, "reference"),
    )
    native = rasterize_gaussians(
        *native_inputs,
        _settings(device, "cuda"),
    )
    assert reference.radii[0] > 0
    _loss(reference).backward()
    _loss(native).backward()
    torch.cuda.synchronize()

    for reference_output, native_output in zip(reference[:6], native[:6]):
        torch.testing.assert_close(native_output, reference_output, rtol=2e-3, atol=3e-4)
    assert torch.equal(native.dominant_index, reference.dominant_index)
    for reference_input, native_input in zip(reference_inputs, native_inputs):
        torch.testing.assert_close(
            native_input.grad,
            reference_input.grad,
            rtol=1e-2,
            atol=1e-3,
        )
