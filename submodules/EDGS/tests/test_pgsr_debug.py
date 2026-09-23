from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from source.pgsr_debug import PGSRDebugVisualizer  # noqa: E402


def _camera(image_name: str = "nested/frame.PNG") -> SimpleNamespace:
    return SimpleNamespace(image_name=image_name)


def _render_package(height: int = 4, width: int = 6):
    normal = torch.zeros(3, height, width)
    normal[2] = 1.0
    return {
        "render": torch.full((3, height, width), 0.25),
        "rendered_normal": normal,
        "rendered_distance": torch.linspace(0.0, 2.0, height * width).reshape(
            1, height, width
        ),
        "plane_depth": torch.linspace(0.1, 3.0, height * width).reshape(
            1, height, width
        ),
        "depth_normal": normal.clone(),
    }


def test_disabled_visualizer_performs_no_writes(tmp_path):
    model_path = tmp_path / "model"
    visualizer = PGSRDebugVisualizer(
        None,
        model_path,
        default_from_iter=0,
    )

    result = visualizer.save(
        1,
        _camera(),
        render_pkg={},
        gt_image=torch.empty(0),
        diagnostics=None,
    )

    assert visualizer.enabled is False
    assert visualizer.interval == 200
    assert result is None
    assert not model_path.exists()


def test_schedule_uses_strict_from_iter_and_interval(tmp_path):
    visualizer = PGSRDebugVisualizer(
        {
            "enabled": True,
            "interval": 200,
            "from_iter": None,
            "output_dir": "debug/pgsr",
        },
        tmp_path / "model",
        default_from_iter=7000,
    )

    assert visualizer.from_iter == 7000
    assert visualizer.should_capture(7000) is False
    assert visualizer.should_capture(7199) is False
    assert visualizer.should_capture(7200) is True
    assert not visualizer.output_dir.exists()
    visualizer.prepare()
    assert visualizer.output_dir.is_dir()


def test_writes_atomic_two_by_four_montage_and_safe_filename(tmp_path):
    height, width = 5, 7
    visualizer = PGSRDebugVisualizer(
        {
            "enabled": True,
            "interval": 200,
            "from_iter": 0,
            "jpeg_quality": 100,
        },
        tmp_path / "model",
        default_from_iter=7000,
    )
    diagnostics = {
        "reprojection_weight": torch.linspace(0.0, 1.0, height * width),
        "image_weight": torch.ones(height, width),
    }

    output = visualizer.save(
        200,
        _camera("unsafe/path/frame one.PNG"),
        _render_package(height, width),
        torch.full((3, height, width), 0.75),
        diagnostics,
    )

    assert output is not None
    assert output.name == "00200_frame_one.jpg"
    assert output.parent == tmp_path / "model" / "debug"
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (2 * height, 4 * width, 3)
    assert not list(output.parent.glob(".*.tmp"))


def test_nan_and_infinity_inputs_are_encoded_stably(tmp_path):
    height, width = 3, 4
    package = _render_package(height, width)
    package["render"][0, 0, 0] = torch.nan
    package["render"][1, 0, 1] = torch.inf
    package["rendered_normal"][:, 1, 1] = torch.nan
    package["rendered_distance"][0, 0, 0] = torch.nan
    package["plane_depth"][0, 0, 1] = torch.inf
    diagnostics = {
        "reprojection_weight": torch.full((height * width,), torch.nan),
        "image_weight": torch.tensor(
            [[0.0, 0.5, 1.0, torch.inf]] * height,
        ),
    }
    visualizer = PGSRDebugVisualizer(
        {"enabled": True, "interval": 1, "from_iter": 0},
        tmp_path / "model",
        default_from_iter=0,
    )

    output = visualizer.save(
        1,
        _camera(),
        package,
        torch.full((3, height, width), torch.nan),
        diagnostics,
    )

    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape == (2 * height, 4 * width, 3)


def test_one_pixel_montage_keeps_scalar_maps_two_dimensional(tmp_path):
    visualizer = PGSRDebugVisualizer(
        {"enabled": True, "interval": 1, "from_iter": 0},
        tmp_path / "model",
        default_from_iter=0,
    )

    output = visualizer.save(
        1,
        _camera(),
        _render_package(1, 1),
        torch.zeros(3, 1, 1),
        {
            "reprojection_weight": torch.ones(1, 1),
            "image_weight": torch.ones(1, 1),
        },
    )

    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image.shape == (2, 4, 3)


def test_missing_diagnostics_produce_zero_panels(tmp_path):
    height, width = 24, 24
    visualizer = PGSRDebugVisualizer(
        {
            "enabled": True,
            "interval": 1,
            "from_iter": 0,
            "jpeg_quality": 100,
        },
        tmp_path / "model",
        default_from_iter=0,
    )

    output = visualizer.save(
        1,
        _camera(),
        _render_package(height, width),
        torch.zeros(3, height, width),
        diagnostics=None,
    )

    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    reprojection_panel = image[height:, :width]
    image_weight_panel = image[height:, 3 * width :]
    # JPEG can introduce a small amount of ringing at neighboring panel edges.
    assert reprojection_panel[:, 2:-2].mean() < 3.0
    assert image_weight_panel[:, 2:-2].mean() < 3.0


@pytest.mark.parametrize(
    "config,exception",
    (
        ({"enabled": "false"}, TypeError),
        ({"enabled": False, "interval": 0}, ValueError),
        ({"enabled": False, "interval": True}, TypeError),
        ({"enabled": False, "from_iter": -1}, ValueError),
        ({"enabled": False, "jpeg_quality": 0}, ValueError),
        ({"enabled": False, "jpeg_quality": 101}, ValueError),
        ({"enabled": False, "output_dir": ""}, ValueError),
        ({"enabled": False, "output_dir": "../outside"}, ValueError),
    ),
)
def test_rejects_invalid_configuration(tmp_path, config, exception):
    with pytest.raises(exception):
        PGSRDebugVisualizer(config, tmp_path / "model", default_from_iter=0)


def test_rejects_absolute_and_symlinked_output_escape(tmp_path):
    model_path = tmp_path / "model"
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="inside model_path"):
        PGSRDebugVisualizer(
            {"enabled": False, "output_dir": str(outside)},
            model_path,
            default_from_iter=0,
        )

    model_path.mkdir()
    (model_path / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside model_path"):
        PGSRDebugVisualizer(
            {"enabled": False, "output_dir": "escape/debug"},
            model_path,
            default_from_iter=0,
        )


def test_scheduled_capture_requires_pgsr_geometry(tmp_path):
    visualizer = PGSRDebugVisualizer(
        {"enabled": True, "interval": 1, "from_iter": 0},
        tmp_path / "model",
        default_from_iter=0,
    )

    with pytest.raises(KeyError, match="rendered_normal"):
        visualizer.save(
            1,
            _camera(),
            {"render": torch.zeros(3, 2, 2)},
            torch.zeros(3, 2, 2),
            diagnostics=None,
        )
