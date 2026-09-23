from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from source.renderers import build_renderer, visible_mask  # noqa: E402
from source.renderers.native import NativeRenderer  # noqa: E402
from source.renderers.pgsr import PGSRRenderer  # noqa: E402


def test_visible_mask_normalizes_indices_and_boolean_masks():
    radii = torch.tensor((1.0, 0.0, 2.0))

    from_indices = visible_mask(
        {"radii": radii, "visibility_filter": torch.tensor(((0,), (2,)))}
    )
    from_bool = visible_mask(
        {"radii": radii, "visibility_filter": torch.tensor((True, False, True))}
    )

    assert from_indices.dtype == torch.bool
    assert from_indices.tolist() == [True, False, True]
    assert torch.equal(from_indices, from_bool)


def test_factory_returns_objects_with_stable_backend_interface():
    native = build_renderer({"backend": "native", "clamp_rgb": False})
    pgsr = build_renderer(SimpleNamespace(backend="pgsr", clamp_rgb=True))

    assert native.backend == "native" and callable(native.render)
    assert pgsr.backend == "pgsr" and callable(pgsr.render)
    with pytest.raises(ValueError, match="unknown renderer backend"):
        build_renderer("missing")


def test_native_adapter_adds_explicit_inverse_depth_and_mask():
    inverse_depth = torch.ones(1, 2, 2)

    def fake_render(*args, **kwargs):
        return {
            "render": torch.full((3, 2, 2), 2.0),
            "viewspace_points": torch.zeros(3, 3),
            "visibility_filter": torch.tensor(((0,), (2,))),
            "radii": torch.tensor((1.0, 0.0, 1.0)),
            "depth": inverse_depth,
        }

    package = NativeRenderer(render_fn=fake_render).render(
        None, None, None, torch.zeros(3)
    )

    assert package["render"].max().item() == 1.0
    assert package["inverse_depth"] is inverse_depth
    assert package["visible_mask"].tolist() == [True, False, True]
    with pytest.raises(ValueError, match="cannot return PGSR plane depth"):
        NativeRenderer(render_fn=fake_render).render(
            None, None, None, torch.zeros(3), return_plane=True
        )


class FakeSettings:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeRasterizer:
    def __init__(self, raster_settings):
        self.settings = raster_settings

    def __call__(self, **arguments):
        height = self.settings.image_height
        width = self.settings.image_width
        count = arguments["means3D"].shape[0]
        device = arguments["means3D"].device
        dtype = arguments["means3D"].dtype
        all_map = torch.zeros(5, height, width, device=device, dtype=dtype)
        all_map[2].fill_(-1.0)
        all_map[3].fill_(1.0)
        all_map[4].fill_(2.0)
        return (
            torch.full((3, height, width), 0.5, device=device, dtype=dtype),
            torch.tensor((1.0, 0.0), device=device, dtype=dtype)[:count],
            torch.ones(count, device=device, dtype=dtype),
            all_map,
            torch.full((1, height, width), 2.0, device=device, dtype=dtype),
        )


def test_pgsr_adapter_has_plane_contract_without_depth_alias(monkeypatch):
    import source.renderers.pgsr as pgsr_module

    monkeypatch.setattr(
        pgsr_module,
        "_load_plane_rasterizer",
        lambda: (FakeSettings, FakeRasterizer),
    )
    camera = SimpleNamespace(
        image_width=3,
        image_height=3,
        FoVx=1.0,
        FoVy=1.0,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
    )
    gaussians = SimpleNamespace(
        get_xyz=torch.tensor(((0.0, 0.0, 2.0), (1.0, 0.0, 2.0))),
        get_scaling=torch.tensor(((1.0, 1.0, 0.1), (1.0, 1.0, 0.1))),
        get_rotation=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2),
        get_opacity=torch.ones(2, 1),
        get_features=torch.zeros(2, 1, 3),
        active_sh_degree=0,
        max_sh_degree=0,
    )
    pipe = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )

    package = PGSRRenderer().render(
        camera,
        gaussians,
        pipe,
        torch.zeros(3),
        return_plane=True,
        return_depth_normal=True,
    )

    assert "plane_depth" in package and "depth_normal" in package
    assert "depth" not in package and "inverse_depth" not in package
    assert package["visible_mask"].tolist() == [True, False]
    assert package["depth_normal"].shape == (3, 3, 3)

    pipe.antialiasing = True
    with pytest.raises(ValueError, match="does not support antialiasing"):
        PGSRRenderer().render(camera, gaussians, pipe, torch.zeros(3))

    pipe.antialiasing = False
    pipe.debug = True
    with pytest.raises(ValueError, match="no safe debug path"):
        PGSRRenderer().render(camera, gaussians, pipe, torch.zeros(3))


def test_importing_pgsr_adapter_does_not_load_cuda_extension():
    # Reload from a clean module-cache entry: importing the Python adapter must
    # not attempt to resolve or initialize the optional CUDA extension.
    import source.renderers.pgsr as pgsr_module

    sys.modules.pop("diff_plane_rasterization", None)
    importlib.reload(pgsr_module)
    assert "diff_plane_rasterization" not in sys.modules
