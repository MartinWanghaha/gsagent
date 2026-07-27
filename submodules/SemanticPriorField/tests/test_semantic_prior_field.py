from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "semantic_prior_field"


@pytest.fixture(autouse=True)
def package_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))


def _make_head(num_classes=4, scale=10.0):
    from semantic.head import SemanticHead

    head = SemanticHead(16, num_classes)
    with torch.no_grad():
        head.classifier.weight.zero_()
        head.classifier.bias.zero_()
        for class_id in range(num_classes):
            head.classifier.weight[class_id, class_id, 0, 0] = scale
    return head


def _one_hot_features(labels, num_channels=16):
    features = torch.zeros(labels.shape[0], num_channels)
    features[torch.arange(labels.shape[0]), labels] = 1.0
    return features


def _default_config(**overrides):
    config = {
        "refresh_interval": 500,
        "min_label_confidence": 0.3,
        "min_instance_gaussians": 100,
        "max_instances": 8,
        "max_points_per_instance": 5000,
        "background_label": 0,
        "skip_background_instance": True,
        "ransac_iterations": 64,
        "planar_inlier_threshold_rel": 0.02,
        "planar_inlier_ratio": 0.8,
        "quadric_min_points": 64,
        "quadric_max_residual_rel": 0.05,
        "thin_anisotropy_threshold": 4.0,
        "prior_weight_sigma": 0.5,
        "densify_multiplier_planar": 1.5,
        "densify_multiplier_thin": 0.7,
    }
    config.update(overrides)
    return config


def test_head_logits_match_pixel_path():
    from semantic.prior_field import head_logits_per_gaussian

    head = _make_head()
    features = torch.randn(32, 16)
    per_gaussian = head_logits_per_gaussian(head, features)
    # Pixel path: treat each Gaussian as one pixel of a [16, N, 1] image
    pixel = head(features.t().unsqueeze(-1)).squeeze(-1).t()
    assert torch.allclose(per_gaussian, pixel, atol=1e-6)


def test_plane_ransac_recovers_normal():
    from semantic.prior_field import fit_plane_ransac

    torch.manual_seed(0)
    grid = torch.rand(600, 2) * 2.0 - 1.0
    points = torch.cat([grid, 1e-4 * torch.randn(600, 1)], dim=-1)
    normal, inlier_ratio, residual = fit_plane_ransac(
        points, n_iterations=64, inlier_threshold=0.01
    )
    assert inlier_ratio > 0.95
    assert abs(normal[2].item()) > 0.99
    assert residual < 0.01


def test_prior_field_planar_and_quadric_instances():
    from semantic.prior_field import (
        PRIOR_PLANAR,
        PRIOR_QUADRIC,
        SemanticPriorField,
    )

    torch.manual_seed(0)
    n_per_instance = 512

    # Instance 1: a plane patch in z = 0
    plane_xy = torch.rand(n_per_instance, 2) * 2.0 - 1.0
    plane_points = torch.cat([plane_xy, 1e-4 * torch.randn(n_per_instance, 1)], dim=-1)

    # Instance 2: a unit sphere, shifted away from the plane
    sphere_dirs = torch.nn.functional.normalize(torch.randn(n_per_instance, 3), dim=-1)
    sphere_center = torch.tensor([5.0, 0.0, 0.0])
    sphere_points = sphere_center + sphere_dirs

    xyz = torch.cat([plane_points, sphere_points], dim=0)
    labels = torch.cat(
        [
            torch.full((n_per_instance,), 1, dtype=torch.long),
            torch.full((n_per_instance,), 2, dtype=torch.long),
        ]
    )
    features = _one_hot_features(labels)
    # Disk-like scales: never needle-shaped
    scaling = torch.tensor([0.05, 0.04, 0.01]).expand(xyz.shape[0], 3)

    gaussians = SimpleNamespace(
        get_semantic_features=features,
        get_xyz=xyz,
        get_scaling=scaling,
    )
    field = SemanticPriorField(_default_config())
    field.refresh(gaussians, _make_head(), iteration=1000)

    assert field.valid
    assert set(field.instances.keys()) == {1, 2}
    assert field.instances[1].prior_type == PRIOR_PLANAR
    assert field.instances[2].prior_type == PRIOR_QUADRIC

    # Planar proxy normal is the plane normal
    plane_normal = field.instances[1].normal
    assert abs(plane_normal[2].item()) > 0.99

    # Quadric proxy normals are radial on the sphere
    sphere_mask = field.labels == 2
    proxy_normals = field.prior_normals[sphere_mask]
    radial = sphere_dirs
    cosine = (proxy_normals * radial).sum(dim=-1).abs()
    assert cosine.mean().item() > 0.9

    # Budget multipliers: planar instances densify less
    assert torch.all(field.densify_multiplier[field.labels == 1] == 1.5)

    # Prior weights are positive where a proxy exists
    assert (field.prior_weight[field.labels == 1] > 0).all()
    assert (field.prior_weight[sphere_mask] > 0).all()


def test_prior_field_thin_instance_is_protected():
    from semantic.prior_field import PRIOR_THIN, SemanticPriorField

    torch.manual_seed(0)
    n_points = 256
    # A spoke: points along a line, needle-shaped Gaussians
    t = torch.linspace(-1, 1, n_points).unsqueeze(-1)
    xyz = torch.cat([t, 1e-3 * torch.randn(n_points, 2)], dim=-1)
    labels = torch.full((n_points,), 1, dtype=torch.long)
    features = _one_hot_features(labels)
    scaling = torch.tensor([0.2, 0.01, 0.01]).expand(n_points, 3)  # needles

    gaussians = SimpleNamespace(
        get_semantic_features=features,
        get_xyz=xyz,
        get_scaling=scaling,
    )
    field = SemanticPriorField(_default_config(min_instance_gaussians=64))
    field.refresh(gaussians, _make_head(), iteration=0)

    assert field.instances[1].prior_type == PRIOR_THIN
    # No orientation prior on thin structures ...
    assert (field.prior_weight[field.labels == 1] == 0).all()
    # ... but they densify more
    assert torch.allclose(
        field.densify_multiplier[field.labels == 1], torch.tensor(0.7)
    )


def test_prior_field_invalidation():
    from semantic.prior_field import SemanticPriorField

    labels = torch.full((256,), 1, dtype=torch.long)
    gaussians = SimpleNamespace(
        get_semantic_features=_one_hot_features(labels),
        get_xyz=torch.randn(256, 3),
        get_scaling=torch.rand(256, 3) * 0.05,
    )
    field = SemanticPriorField(_default_config(min_instance_gaussians=64))
    field.refresh(gaussians, _make_head(), iteration=0)
    assert field.valid
    assert not field.needs_refresh(iteration=100)
    assert field.needs_refresh(iteration=600)
    field.invalidate()
    assert not field.valid
    assert field.needs_refresh(iteration=0)
    assert field.labels is None


def test_boundary_weight_map():
    from semantic.prior_field import compute_boundary_weight_map

    labels = torch.zeros(16, 16, dtype=torch.long)
    labels[:, 8:] = 1
    weight = compute_boundary_weight_map(
        labels, ignore_label=-1, boundary_radius=2, boundary_weight=0.25
    )
    # Boundary band around column 8 is down-weighted
    assert weight[8, 7].item() == pytest.approx(0.25)
    assert weight[8, 8].item() == pytest.approx(0.25)
    # Far interior keeps full weight
    assert weight[8, 0].item() == pytest.approx(1.0)
    assert weight[8, 15].item() == pytest.approx(1.0)


def test_boundary_weight_map_ignores_ignore_label():
    from semantic.prior_field import compute_boundary_weight_map

    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[:, 4:] = -1  # ignore region: no boundary generated
    weight = compute_boundary_weight_map(
        labels, ignore_label=-1, boundary_radius=1, boundary_weight=0.25
    )
    assert torch.all(weight == 1.0)


def test_semantic_edge_mask():
    from extraction.mesh import semantic_edge_mask

    labels = torch.tensor([1, 1, 2, 0, -1, 2])
    confidence = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    verts_idx = torch.tensor(
        [
            [0, 1],  # same instance -> keep
            [0, 2],  # different confident instances -> bad
            [2, 3],  # instance vs background -> keep
            [0, 4],  # unlabeled endpoint -> keep
            [0, 5],  # low-confidence endpoint -> keep
        ]
    )
    bad = semantic_edge_mask(labels, confidence, verts_idx, confidence_threshold=0.5)
    assert bad.tolist() == [False, True, False, False, False]


def test_expand_per_gaussian_to_pivots_layout():
    from extraction.semantic import expand_per_gaussian_to_pivots

    values = torch.tensor([10, 20, 30])
    expanded = expand_per_gaussian_to_pivots(values, n_pivots=2)
    assert expanded.tolist() == [10, 10, 20, 20, 30, 30]


def test_sh_outlier_decay():
    from regularization.regularizer.semantic_prior import _sh_outlier_decay

    features_rest = torch.ones(100, 15, 3) * 0.01
    labels = torch.zeros(100, dtype=torch.long)
    # Uniform energy: no outliers, zero decay
    assert _sh_outlier_decay(features_rest, labels).item() == pytest.approx(0.0)
    # One Gaussian with 100x the instance median energy is decayed
    features_rest[0] = 1.0
    assert _sh_outlier_decay(features_rest, labels).item() > 0.0


def test_sh_region_consistency_respects_labels():
    from regularization.regularizer.semantic_prior import _sh_region_consistency

    torch.manual_seed(0)
    n_points = 200
    positions = torch.randn(n_points, 3)
    labels = torch.cat(
        [torch.zeros(n_points // 2), torch.ones(n_points // 2)]
    ).long()
    # Identical SH inside each instance, very different across instances
    features = torch.zeros(n_points, 15, 3)
    features[labels == 1] = 5.0
    loss = _sh_region_consistency(features, positions, labels, sample_size=n_points)
    # Neighbourhoods never cross labels, so the loss stays at zero
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_densify_multiplier_threshold_selection():
    """The per-Gaussian multiplier scales the selection threshold."""
    grads = torch.tensor([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
    threshold = 0.4
    multipliers = torch.tensor([1.0, 1.5, 0.7])
    selected = torch.norm(grads, dim=-1) >= threshold * multipliers
    assert selected.tolist() == [True, False, True]
