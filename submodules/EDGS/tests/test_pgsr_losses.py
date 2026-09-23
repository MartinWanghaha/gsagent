from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

EDGS_ROOT = Path(__file__).resolve().parents[1]
if str(EDGS_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGS_ROOT))

from source.losses import l1_loss, ssim  # noqa: E402
from source.pgsr_losses import PGSRLossComposer  # noqa: E402


def _config(**overrides):
    values = {
        "enabled": True,
        "scale_weight": 0.0,
        "single_view_weight": 0.0,
        "single_view_from_iter": 7,
        "image_weight": False,
        "multi_view_from_iter": 7,
        "geo_weight": 0.0,
        "ncc_weight": 0.0,
        "patch_size": 1,
        "sample_num": 32,
        "pixel_noise_th": 1.0,
        "ncc_scale": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _camera(name="ref", image=None):
    if image is None:
        image = torch.zeros(3, 5, 5)
    return SimpleNamespace(
        image_name=name,
        uid=name,
        colmap_id=name,
        original_image=image,
        image_width=image.shape[-1],
        image_height=image.shape[-2],
        FoVx=math.pi / 2.0,
        FoVy=math.pi / 2.0,
        world_view_transform=torch.eye(4, dtype=image.dtype, device=image.device),
    )


def _pkg(visibility, *, image=None):
    if image is None:
        image = torch.zeros(3, 5, 5, requires_grad=True)
    visibility = torch.as_tensor(visibility, dtype=torch.bool, device=image.device)
    return {
        "render": image,
        "visibility_filter": visibility,
        "radii": visibility.to(dtype=image.dtype),
    }


class _Gaussians:
    def __init__(self, scales):
        self.get_scaling = scales


def test_photo_terms_are_weighted_once():
    composer = PGSRLossComposer(_config(), lambda_dssim=0.25)
    image = torch.zeros(3, 5, 5, requires_grad=True)
    target = torch.ones_like(image)

    terms = composer.photo_terms(image, target)

    assert set(terms) == {"photo_l1", "photo_dssim"}
    torch.testing.assert_close(terms["photo_l1"], 0.75 * l1_loss(image, target))
    torch.testing.assert_close(terms["photo_dssim"], 0.25 * (1.0 - ssim(image, target)))
    sum(terms.values()).backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_scale_term_uses_only_visible_gaussians():
    composer = PGSRLossComposer(_config(scale_weight=2.0), lambda_dssim=0.2)
    scales = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, 5.0]], requires_grad=True)

    terms = composer.pgsr_terms(
        step=0,
        camera=_camera(),
        pkg=_pkg([True, False]),
        gaussians=_Gaussians(scales),
        neighbors=None,
        render_neighbor=lambda _: pytest.fail("neighbor render must not run"),
    )

    assert set(terms) == {
        "pgsr_scale",
        "pgsr_normal",
        "pgsr_geo",
        "pgsr_ncc",
    }
    torch.testing.assert_close(terms["pgsr_scale"], torch.tensor(2.0))
    assert all(torch.isfinite(term) for term in terms.values())


def test_single_view_normal_loss_has_strict_start_iteration():
    composer = PGSRLossComposer(
        _config(single_view_weight=0.2, single_view_from_iter=7),
        lambda_dssim=0.2,
    )
    package = _pkg([True])
    package.update(
        rendered_normal=torch.zeros(3, 5, 5, requires_grad=True),
        depth_normal=torch.ones(3, 5, 5),
    )
    gaussians = _Gaussians(torch.ones(1, 3))

    at_boundary = composer.pgsr_terms(
        7, _camera(), package, gaussians, None, lambda _: {}
    )
    after_boundary = composer.pgsr_terms(
        8, _camera(), package, gaussians, None, lambda _: {}
    )

    torch.testing.assert_close(at_boundary["pgsr_normal"], torch.tensor(0.0))
    torch.testing.assert_close(after_boundary["pgsr_normal"], torch.tensor(0.6))
    assert composer.required_outputs(7) == (False, False)
    assert composer.required_outputs(8) == (True, True)


def test_constant_image_weight_is_finite():
    composer = PGSRLossComposer(
        _config(single_view_weight=0.1, image_weight=True), lambda_dssim=0.2
    )
    camera = _camera(image=torch.full((3, 5, 5), 0.5))
    package = _pkg([True])
    package.update(
        rendered_normal=torch.zeros(3, 5, 5, requires_grad=True),
        depth_normal=torch.ones(3, 5, 5),
    )

    term = composer.pgsr_terms(
        8,
        camera,
        package,
        _Gaussians(torch.ones(1, 3)),
        None,
        lambda _: {},
    )["pgsr_normal"]

    assert torch.isfinite(term)
    assert term > 0


def test_empty_visibility_and_no_neighbors_return_graph_zeros():
    composer = PGSRLossComposer(
        _config(scale_weight=100.0, geo_weight=0.03, ncc_weight=0.15),
        lambda_dssim=0.2,
    )
    rendered = torch.zeros(3, 5, 5, requires_grad=True)
    package = _pkg([False, False], image=rendered)

    terms = composer.pgsr_terms(
        8,
        _camera(),
        package,
        _Gaussians(torch.ones(2, 3, requires_grad=True)),
        {"ref": []},
        lambda _: pytest.fail("empty neighbor set must not be rendered"),
    )

    assert composer.required_outputs(8) == (True, False)
    for term in terms.values():
        torch.testing.assert_close(term, torch.zeros_like(term))
        assert torch.isfinite(term)
    sum(terms.values()).backward()
    assert rendered.grad is not None


def test_identity_multiview_with_constant_images_does_not_nan():
    composer = PGSRLossComposer(
        _config(geo_weight=0.03, ncc_weight=0.15), lambda_dssim=0.2
    )
    reference = _camera("ref", torch.full((3, 5, 5), 0.5))
    neighbor = _camera("neighbor", torch.full((3, 5, 5), 0.5))
    depth = torch.ones(1, 5, 5, requires_grad=True)
    package = _pkg([False])
    package.update(
        plane_depth=depth,
        rendered_normal=torch.tensor([0.0, 0.0, -1.0])[:, None, None].expand(3, 5, 5),
        rendered_distance=torch.ones(1, 5, 5),
    )

    terms = composer.pgsr_terms(
        8,
        reference,
        package,
        _Gaussians(torch.ones(1, 3)),
        {"ref": [neighbor]},
        lambda _: {"plane_depth": torch.ones_like(depth)},
    )

    assert torch.isfinite(terms["pgsr_geo"])
    assert torch.isfinite(terms["pgsr_ncc"])
    assert terms["pgsr_geo"] < 1.0e-5
    torch.testing.assert_close(terms["pgsr_ncc"], torch.tensor(0.0))


def test_requested_diagnostics_capture_the_active_multiview_geometry():
    composer = PGSRLossComposer(
        _config(
            single_view_weight=0.1,
            image_weight=True,
            geo_weight=0.03,
            ncc_weight=0.0,
        ),
        lambda_dssim=0.2,
    )
    reference = _camera("ref", torch.full((3, 5, 5), 0.5))
    neighbor = _camera("neighbor", torch.full((3, 5, 5), 0.5))
    depth = torch.ones(1, 5, 5, requires_grad=True)
    package = _pkg([False])
    package.update(
        plane_depth=depth,
        rendered_normal=torch.tensor([0.0, 0.0, -1.0])[:, None, None].expand(3, 5, 5),
        rendered_distance=torch.ones(1, 5, 5),
        depth_normal=torch.tensor([0.0, 0.0, -1.0])[:, None, None].expand(3, 5, 5),
    )
    diagnostics = {}

    composer.pgsr_terms(
        8,
        reference,
        package,
        _Gaussians(torch.ones(1, 3)),
        {"ref": [neighbor]},
        lambda _: {"plane_depth": torch.ones_like(depth)},
        diagnostics=diagnostics,
    )

    assert diagnostics["neighbor_image_name"] == "neighbor"
    assert diagnostics["image_weight"].shape == (5, 5)
    assert diagnostics["reprojection_weight"].shape == (5, 5)
    assert diagnostics["reprojection_valid"].dtype == torch.bool
    assert diagnostics["pixel_noise"].shape == (5, 5)
    assert all(
        not value.requires_grad
        for value in diagnostics.values()
        if torch.is_tensor(value)
    )


def test_zero_sample_count_skips_ncc_image_sampling(monkeypatch):
    import source.pgsr_losses as losses_module

    composer = PGSRLossComposer(
        _config(geo_weight=0.0, ncc_weight=0.15, sample_num=0),
        lambda_dssim=0.2,
    )
    reference = _camera("ref")
    neighbor = _camera("neighbor")
    depth = torch.ones(1, 5, 5, requires_grad=True)
    package = _pkg([False])
    package.update(
        plane_depth=depth,
        rendered_normal=torch.tensor([0.0, 0.0, -1.0])[:, None, None].expand(3, 5, 5),
        rendered_distance=torch.ones(1, 5, 5),
    )
    monkeypatch.setattr(
        losses_module,
        "image_gray",
        lambda *args, **kwargs: pytest.fail("sample_num=0 must skip NCC sampling"),
    )

    terms = composer.pgsr_terms(
        8,
        reference,
        package,
        _Gaussians(torch.ones(1, 3)),
        {"ref": [neighbor]},
        lambda _: {"plane_depth": torch.ones_like(depth)},
    )

    torch.testing.assert_close(terms["pgsr_ncc"], torch.tensor(0.0))


def test_disabled_composer_requests_and_computes_no_geometry():
    composer = PGSRLossComposer(
        {"enabled": False, "single_view_weight": 1.0, "geo_weight": 1.0},
        lambda_dssim=0.2,
    )
    assert composer.enabled is False
    assert composer.required_outputs(100_000) == (False, False)

    terms = composer.pgsr_terms(
        100_000,
        _camera(),
        _pkg([]),
        _Gaussians(torch.empty(0, 3)),
        None,
        lambda _: pytest.fail("disabled composer must not render a neighbor"),
    )
    assert all(term.item() == 0.0 for term in terms.values())


def test_root_omegaconf_node_is_supported():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    config = OmegaConf.create(
        {
            "opt": {
                "pgsr_loss": {
                    "enabled": True,
                    "scale_weight": 4.0,
                    "single_view_weight": 0.0,
                    "geo_weight": 0.0,
                    "ncc_weight": 0.0,
                }
            }
        }
    )
    composer = PGSRLossComposer(config, lambda_dssim=0.2)

    terms = composer.pgsr_terms(
        0,
        _camera(),
        _pkg([True]),
        _Gaussians(torch.tensor([[3.0, 2.0, 1.0]])),
        None,
        lambda _: {},
    )

    torch.testing.assert_close(terms["pgsr_scale"], torch.tensor(4.0))
