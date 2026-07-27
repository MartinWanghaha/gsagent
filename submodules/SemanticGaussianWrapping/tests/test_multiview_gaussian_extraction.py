import numpy as np
import pytest

from mesh.multiview_gaussian_extraction import (
    MultiviewGaussianMeshConfig,
    MultiviewGaussianMeshExtractor,
)


class _Camera:
    R = np.eye(3, dtype=np.float32)
    T = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)


def test_camera_geometry_uses_graphdeco_convention():
    world_to_view, center = (
        MultiviewGaussianMeshExtractor._camera_geometry(_Camera())
    )
    np.testing.assert_allclose(world_to_view[:3, 3], _Camera.T)
    np.testing.assert_allclose(center, (-1.0, -2.0, -3.0))


def test_config_rejects_invalid_visibility_policy():
    with pytest.raises(ValueError, match="minimum_visible_views"):
        MultiviewGaussianMeshConfig(minimum_visible_views=0)


def test_config_is_serializable():
    payload = MultiviewGaussianMeshConfig().as_dict()
    assert payload["poisson_depth"] == 9
    assert payload["view_count"] == 60
