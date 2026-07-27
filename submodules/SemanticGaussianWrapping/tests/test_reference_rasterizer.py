from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTERIZER_ROOT = PROJECT_ROOT / "submodules" / "diff-semantic-gaussian-rasterization"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RASTERIZER_ROOT))

from diff_semantic_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
    has_cuda_extension,
)
from gaussian_renderer import render  # noqa: E402


def _inputs(device: str = "cpu"):
    kwargs = {"device": device, "dtype": torch.float32}
    means = torch.tensor(
        [[-0.10, 0.03, 2.0], [0.12, -0.05, 2.7], [0.02, 0.14, 3.4]],
        **kwargs,
        requires_grad=True,
    )
    means2d = torch.zeros((3, 3), **kwargs, requires_grad=True)
    colors = torch.tensor(
        [[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.9]],
        **kwargs,
        requires_grad=True,
    )
    semantic = torch.linspace(-0.7, 0.9, 48, **kwargs).reshape(3, 16).requires_grad_()
    opacity = torch.tensor([[0.75], [0.62], [0.53]], **kwargs, requires_grad=True)
    scales = torch.tensor(
        [[0.22, 0.11, 0.055], [0.13, 0.25, 0.075], [0.19, 0.09, 0.14]],
        **kwargs,
        requires_grad=True,
    )
    rotations = torch.tensor(
        [[0.94, 0.13, -0.19, 0.22], [0.88, -0.21, 0.31, 0.12], [0.91, 0.28, 0.08, -0.17]],
        **kwargs,
        requires_grad=True,
    )
    return means, means2d, colors, semantic, opacity, scales, rotations


def _settings(device: str = "cpu", *, backend: str = "reference", chunk_size: int = 2):
    return GaussianRasterizationSettings(
        image_height=13,
        image_width=15,
        tanfovx=0.7,
        tanfovy=0.65,
        bg=torch.tensor([0.03, 0.04, 0.05], device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        backend=backend,
        chunk_size=chunk_size,
    )


def _run(inputs, settings):
    means, means2d, colors, semantic, opacity, scales, rotations = inputs
    return GaussianRasterizer(settings)(
        means3D=means,
        means2D=means2d,
        colors_precomp=colors,
        semantic_features=semantic,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
    )


def test_reference_contract_and_joint_gradients():
    inputs = _inputs()
    output = _run(inputs, _settings())
    assert output.color.shape == (3, 13, 15)
    assert output.semantic.shape == (16, 13, 15)
    assert output.expected_depth.shape == (1, 13, 15)
    assert output.alpha.shape == (1, 13, 15)
    assert output.normal.shape == (3, 13, 15)
    assert output.radii.shape == (3,)
    assert output.dominant_index.shape == (13, 15)
    assert output.dominant_index.dtype == torch.long
    assert torch.all((output.alpha >= 0) & (output.alpha <= 1))
    assert (output.radii > 0).all()

    # Asymmetric probes prevent cancellation and exercise every rendered field.
    probe = torch.linspace(0.2, 1.1, 13 * 15).reshape(13, 15)
    loss = (
        (output.color * probe).mean()
        + (output.semantic * probe).square().mean()
        + (output.expected_depth * probe).mean()
        + (output.alpha * probe.flip(0)).mean()
        + (output.normal * probe.flip(1)).square().mean()
    )
    loss.backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    # Core trainable state must receive useful, not merely allocated gradients.
    for tensor in (inputs[0], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]):
        assert tensor.grad.abs().sum() > 0


def test_near_camera_outlier_is_culled_before_projection_backward():
    inputs = list(_inputs())
    inputs[0] = inputs[0].detach().clone()
    inputs[0][0] = torch.tensor([-4.43, 5.99, 4.0e-4])
    inputs[0].requires_grad_()
    inputs[5] = inputs[5].detach().clone()
    inputs[5][0] = 0.061
    inputs[5].requires_grad_()

    output = _run(tuple(inputs), _settings())

    assert output.radii[0] == 0
    assert (output.radii[1:] > 0).all()
    loss = sum(field.square().mean() for field in output[:5])
    loss.backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(inputs[0].grad[0]) == 0


def test_chunking_is_an_exact_memory_strategy():
    inputs_a = _inputs()
    inputs_b = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs_a)
    output_one = _run(inputs_a, _settings(chunk_size=1))
    output_all = _run(inputs_b, _settings(chunk_size=16))
    for a, b in zip(output_one[:6], output_all[:6]):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    assert torch.equal(output_one.dominant_index, output_all.dominant_index)


def test_semantic_dimension_is_a_strict_contract():
    inputs = list(_inputs())
    inputs[3] = torch.zeros((3, 15))
    with pytest.raises(ValueError, match=r"\[N,16\]"):
        _run(tuple(inputs), _settings())


def test_source_checkout_auto_falls_back_on_cpu():
    inputs = _inputs()
    output = _run(inputs, _settings(backend="auto"))
    assert output.color.device.type == "cpu"
    output.color.sum().backward()
    assert inputs[0].grad is not None


def test_public_renderer_returns_architecture_dictionary():
    means, _, _, semantic, opacity, scales, rotations = _inputs()
    sh_dc = torch.full((3, 1, 3), 0.35, requires_grad=True)

    class DummyGaussians:
        active_sh_degree = 0
        get_xyz = means
        get_scaling = scales
        get_rotation = rotations
        get_opacity = opacity
        get_semantic_embedding = semantic
        get_features = sh_dc

    camera = SimpleNamespace(
        FoVx=2 * torch.atan(torch.tensor(0.7)).item(),
        FoVy=2 * torch.atan(torch.tensor(0.65)).item(),
        image_height=13,
        image_width=15,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
    )
    package = render(camera, DummyGaussians(), SimpleNamespace(debug=False), torch.zeros(3), backend="reference")
    assert set(package) == {
        "render",
        "semantic",
        "expected_depth",
        "alpha",
        "normal",
        "dominant_index",
        "viewspace_points",
        "visibility_filter",
        "radii",
    }
    package["render"].mean().backward()
    assert package["viewspace_points"].grad is not None
    assert sh_dc.grad is not None


def test_public_renderer_preserves_graphdeco_viewport_gradient_contract():
    height, width = 13, 15
    tanfovx, tanfovy = 0.7, 0.65
    fovx, fovy = 2.0 * math.atan(tanfovx), 2.0 * math.atan(tanfovy)
    background = torch.tensor([0.03, 0.04, 0.05])

    direct_inputs = _inputs()
    direct_settings = _settings()._replace(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(0.5 * fovx),
        tanfovy=math.tan(0.5 * fovy),
        cx=0.5 * width,
        cy=0.5 * height,
    )
    direct = _run(direct_inputs, direct_settings)

    means, _, colors, semantic, opacity, scales, rotations = _inputs()

    class DummyGaussians:
        active_sh_degree = 0
        get_xyz = means
        get_scaling = scales
        get_rotation = rotations
        get_opacity = opacity
        get_semantic_embedding = semantic

    camera = SimpleNamespace(
        FoVx=fovx,
        FoVy=fovy,
        image_height=height,
        image_width=width,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
    )
    package = render(
        camera,
        DummyGaussians(),
        SimpleNamespace(debug=False, reference_chunk_size=2, antialias_sigma=0.3),
        background,
        backend="reference",
        override_color=colors,
    )

    direct_fields = (
        direct.color,
        direct.semantic,
        direct.expected_depth,
        direct.alpha,
        direct.normal,
        direct.radii,
    )
    public_fields = (
        package["render"],
        package["semantic"],
        package["expected_depth"],
        package["alpha"],
        package["normal"],
        package["radii"],
    )
    for public, expected in zip(public_fields, direct_fields):
        torch.testing.assert_close(public, expected, rtol=0.0, atol=0.0)
    assert torch.equal(package["dominant_index"], direct.dominant_index)
    assert torch.equal(package["visibility_filter"], direct.radii > 0)

    def spatial_loss(outputs) -> torch.Tensor:
        terms = []
        for index, output in enumerate(outputs):
            weight = torch.linspace(
                -0.37 + 0.05 * index,
                0.43 + 0.03 * index,
                output.numel(),
                dtype=output.dtype,
                device=output.device,
            ).reshape_as(output)
            terms.append((output * weight).sum())
        return torch.stack(terms).sum()

    spatial_loss(public_fields[:5]).backward()
    spatial_loss(direct_fields[:5]).backward()

    proxy_gradient = package["viewspace_points"].grad
    pixel_gradient = direct_inputs[1].grad
    assert proxy_gradient is not None
    assert pixel_gradient is not None
    expected_xy = pixel_gradient[:, :2] * pixel_gradient.new_tensor(
        (0.5 * width, 0.5 * height)
    )
    torch.testing.assert_close(proxy_gradient[:, :2], expected_xy, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(proxy_gradient[:, 2]) == 0


def test_public_renderer_uses_camera_principal_point_coordinates():
    class DummyGaussians:
        active_sh_degree = 0
        get_xyz = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
        get_scaling = torch.tensor([[0.12, 0.12, 0.12]], requires_grad=True)
        get_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
        get_opacity = torch.tensor([[0.9]], requires_grad=True)
        get_semantic_embedding = torch.zeros((1, 16), requires_grad=True)
        get_features = torch.full((1, 1, 3), 0.35, requires_grad=True)

    camera = SimpleNamespace(
        FoVx=2 * torch.atan(torch.tensor(0.7)).item(),
        FoVy=2 * torch.atan(torch.tensor(0.65)).item(),
        image_height=13,
        image_width=15,
        Cx=3.0,
        Cy=11.0,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
    )
    package = render(
        camera,
        DummyGaussians(),
        SimpleNamespace(debug=False),
        torch.zeros(3),
        backend="reference",
    )
    peak = int(package["alpha"].reshape(-1).argmax())
    peak_y, peak_x = divmod(peak, camera.image_width)
    assert (peak_x, peak_y) == (3, 11)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_cuda_extension(), reason="CUDA extension not built")
def test_cuda_forward_matches_reference_and_replay_backward():
    reference_inputs = _inputs("cuda")
    cuda_inputs = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in reference_inputs)
    expected = _run(reference_inputs, _settings("cuda", backend="reference", chunk_size=3))
    actual = _run(cuda_inputs, _settings("cuda", backend="cuda", chunk_size=3))
    for ref, cuda in zip(expected[:6], actual[:6]):
        assert torch.allclose(ref, cuda, atol=2e-4, rtol=2e-4)
    assert torch.equal(expected.dominant_index, actual.dominant_index)
    sum(t.mean() for t in actual[:5]).backward()
    for tensor in cuda_inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
