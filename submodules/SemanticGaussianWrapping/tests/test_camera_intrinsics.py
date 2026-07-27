import numpy as np
import pytest

from scene.colmap_loader import Camera as ColmapCamera
from scene.dataset_readers import _intrinsics


def test_off_center_pinhole_intrinsics_are_preserved() -> None:
    camera = ColmapCamera(1, "PINHOLE", 640, 480, np.array([500.0, 510.0, 301.0, 217.0]))
    assert _intrinsics(camera) == (500.0, 510.0, 301.0, 217.0)


def test_distorted_colmap_camera_is_rejected() -> None:
    camera = ColmapCamera(1, "OPENCV", 640, 480, np.zeros(8))
    with pytest.raises(ValueError, match="image_undistorter"):
        _intrinsics(camera)
