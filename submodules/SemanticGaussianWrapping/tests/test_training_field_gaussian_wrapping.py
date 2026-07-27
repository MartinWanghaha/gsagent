import numpy as np
import pytest
import torch

from mesh.training_field_gaussian_wrapping import (
    TrainingFieldGaussianWrappingConfig,
    TrainingFieldGaussianWrappingExtractor,
    _quaternion_matrix,
)


def test_quaternion_matrix_identity_and_z_rotation():
    quaternion = torch.tensor(
        (
            (1.0, 0.0, 0.0, 0.0),
            (2**-0.5, 0.0, 0.0, 2**-0.5),
        )
    )
    matrix = _quaternion_matrix(quaternion)
    assert torch.allclose(matrix[0], torch.eye(3), atol=1e-6)
    expected = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    assert torch.allclose(matrix[1], expected, atol=1e-6)


def test_global_chart_owns_every_anchor_once():
    charts, owner = TrainingFieldGaussianWrappingExtractor._global_chart(7)
    assert len(charts) == 1
    np.testing.assert_array_equal(charts[0].core_rows, np.arange(7))
    np.testing.assert_array_equal(charts[0].all_rows, np.arange(7))
    np.testing.assert_array_equal(owner, np.zeros(7, dtype=np.int64))


def test_config_rejects_invalid_trim_quantile():
    with pytest.raises(ValueError, match="trim_quantile"):
        TrainingFieldGaussianWrappingConfig(trim_quantile=0.5)
