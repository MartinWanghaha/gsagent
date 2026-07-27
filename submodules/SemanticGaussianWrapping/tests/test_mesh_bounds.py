from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mesh import MeshSupportPolicy, gaussian_support_bounds, trusted_gaussian_support_bounds


def _axis_aligned_gaussians(points: np.ndarray, scale: float = 0.1) -> SimpleNamespace:
    xyz = torch.as_tensor(points, dtype=torch.float32)
    count = xyz.shape[0]
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1.0
    return SimpleNamespace(
        get_xyz=xyz,
        get_scaling=torch.full_like(xyz, scale),
        get_rotation=rotation,
    )


def test_bounds_include_rotated_three_sigma_support() -> None:
    # 90 degrees about z rotates the long local x axis into world y.
    half = np.sqrt(0.5)
    gaussians = SimpleNamespace(
        get_xyz=torch.tensor([[1.0, 2.0, 3.0]]),
        get_scaling=torch.tensor([[2.0, 0.1, 0.1]]),
        get_rotation=torch.tensor([[half, 0.0, 0.0, half]]),
    )
    bounds = gaussian_support_bounds(gaussians, sigma=3.0, relative_padding=0.0)
    assert bounds.minimum[1] < 2.0 - 6.0
    assert bounds.maximum[1] > 2.0 + 6.0
    assert bounds.minimum[0] < 1.0 - 0.3
    assert bounds.maximum[0] > 1.0 + 0.3


def test_bounds_selection_accepts_equivalent_indices_and_boolean_mask() -> None:
    gaussians = _axis_aligned_gaussians(
        np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )
    indices = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
        selection=torch.tensor([0, 1]),
    )
    mask = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
        selection=torch.tensor([True, True, False]),
    )
    full = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
    )

    assert np.allclose(indices.minimum, mask.minimum)
    assert np.allclose(indices.maximum, mask.maximum)
    assert indices.maximum[0] < 11.0
    assert full.maximum[0] > 100.0


def test_default_zero_trim_is_exact_and_positive_trim_removes_outlier() -> None:
    cluster = np.zeros((64, 3), dtype=np.float32)
    cluster[:, 0] = np.linspace(0.0, 1.0, len(cluster), dtype=np.float32)
    points = np.concatenate(
        (cluster, np.asarray([[1_000.0, 0.0, 0.0]], dtype=np.float32)),
        axis=0,
    )
    gaussians = _axis_aligned_gaussians(points)

    default = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
    )
    explicit_exact = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
        trim_quantile=0.0,
    )
    trimmed = gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
        trim_quantile=0.05,
    )

    assert np.array_equal(default.minimum, explicit_exact.minimum)
    assert np.array_equal(default.maximum, explicit_exact.maximum)
    assert default.maximum[0] > 999.0
    assert trimmed.maximum[0] < 2.0


def test_trusted_bounds_match_observation_aware_feedback_policy() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    rotation = torch.zeros(len(points), 4)
    rotation[:, 0] = 1.0
    gaussians = SimpleNamespace(
        get_xyz=points,
        get_scaling=torch.full_like(points, 0.1),
        get_rotation=rotation,
        get_opacity=torch.tensor([[0.9], [0.8], [0.95], [0.8]]),
        get_semantic_confidence=torch.tensor([[0.9], [0.8], [0.9], [0.1]]),
        observation_count=torch.tensor([[2.0], [1.0], [5.0], [3.0]]),
    )
    policy = MeshSupportPolicy(
        min_opacity=0.5,
        min_semantic_confidence=0.5,
        trim_quantile=0.0,
    )
    bounds, count = trusted_gaussian_support_bounds(
        gaussians,
        sigma=1.0,
        relative_padding=0.0,
        policy=policy,
    )

    assert count == 3
    assert bounds.minimum[0] < -0.09
    # The high-confidence far support stays included; unobserved / low-
    # confidence supports are intentionally excluded by the common policy.
    assert bounds.maximum[0] > 100.0


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        (torch.tensor([True, False]), ValueError),
        (torch.tensor([[0, 1]]), ValueError),
        (torch.tensor([0.0]), ValueError),
        (torch.tensor([3]), IndexError),
        (torch.tensor([-1]), IndexError),
    ],
)
def test_bounds_reject_invalid_selection(selection, error) -> None:
    gaussians = _axis_aligned_gaussians(np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(error, match="selection"):
        gaussian_support_bounds(gaussians, selection=selection)


@pytest.mark.parametrize("trim_quantile", [-0.01, 0.5, float("nan")])
def test_bounds_reject_invalid_trim_quantile(trim_quantile) -> None:
    gaussians = _axis_aligned_gaussians(np.zeros((32, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        gaussian_support_bounds(gaussians, trim_quantile=trim_quantile)
