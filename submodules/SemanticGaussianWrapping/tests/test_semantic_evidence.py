from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from scene.dataset_readers import BasicPointCloud
from scene.gaussian_model import GaussianModel
from semantic.geometry_policy import GeometryEvidenceProjector
from semantic.neighbor_index import GaussianNeighborIndex


def _coherent_model() -> GaussianModel:
    cloud = BasicPointCloud(
        points=np.array(
            [
                [0.00, 0.00, 0.00],
                [0.04, 0.00, 0.00],
                [0.08, 0.00, 0.00],
                [0.12, 0.00, 0.00],
            ],
            dtype=np.float32,
        ),
        colors=np.ones((4, 3), dtype=np.float32) * 0.5,
        normals=np.zeros((4, 3), dtype=np.float32),
    )
    model = GaussianModel(sh_degree=1, semantic_dim=4, device="cpu")
    model.create_from_pcd(cloud, 1.0)
    with torch.no_grad():
        model.registry["scaling"].fill_(math.log(0.10))
        model.registry["rotation"].zero_()
        model.registry["rotation"][:, 0] = 1.0
        model.registry["semantic_embedding"].copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1)
        )
        model.registry["semantic_confidence"][:2] = 0.9
    return model


def test_confidence_propagation_is_separate_bounded_and_coherent() -> None:
    model = _coherent_model()
    projector = GeometryEvidenceProjector(
        neighbor_index=GaussianNeighborIndex(model, backend="exact"),
        propagation_samples=4,
        propagation_neighbors=3,
        propagation_momentum=0.0,
    )

    report = projector.propagate_semantic_confidence(model)

    assert report["visited"] == 4
    assert report["supported"] == 4
    assert torch.equal(model.semantic_confidence[2:], torch.zeros(2, 1))
    assert torch.all(model.propagated_semantic_confidence[2:] > 0)
    assert float(model.propagated_semantic_confidence.max()) <= 0.85
    assert torch.all(model.get_semantic_confidence >= model.semantic_confidence)


def test_confidence_propagation_is_suppressed_at_semantic_boundaries() -> None:
    interior = _coherent_model()
    boundary = _coherent_model()
    with torch.no_grad():
        boundary.registry["boundary_score"][3] = 1.0

    def propagate(model: GaussianModel) -> float:
        projector = GeometryEvidenceProjector(
            neighbor_index=GaussianNeighborIndex(model, backend="exact"),
            propagation_samples=4,
            propagation_neighbors=3,
            propagation_momentum=0.0,
        )
        projector.propagate_semantic_confidence(model)
        return float(model.propagated_semantic_confidence[3])

    assert propagate(boundary) < propagate(interior)


def test_old_checkpoint_schema_migrates_propagated_confidence() -> None:
    model = _coherent_model()
    snapshot = copy.deepcopy(model.capture())
    registry = snapshot["registry"]
    registry["specs"] = [
        spec
        for spec in registry["specs"]
        if spec["name"] != "propagated_semantic_confidence"
    ]
    registry["tensors"].pop("propagated_semantic_confidence")

    restored = GaussianModel(sh_degree=1, semantic_dim=4, device="cpu")
    restored.restore(snapshot)

    assert "propagated_semantic_confidence" in restored.registry
    assert torch.equal(
        restored.propagated_semantic_confidence,
        torch.zeros_like(restored.semantic_confidence),
    )


def test_gaussian_model_rejects_unknown_future_checkpoint_schema() -> None:
    model = _coherent_model()
    snapshot = model.capture()
    snapshot["format_version"] = 4

    with pytest.raises(ValueError, match="newer Gaussian model schema"):
        model.restore(snapshot)


def test_inference_snapshot_is_optimizer_free_and_immutable() -> None:
    model = _coherent_model()
    snapshot = model.capture_inference("cpu")
    original = snapshot["registry"]["tensors"]["xyz"].clone()

    with torch.no_grad():
        model.registry["xyz"].add_(10.0)

    assert snapshot["optimizer"] is None
    assert torch.equal(snapshot["registry"]["tensors"]["xyz"], original)


def test_policy_confidence_floor_round_trips_and_legacy_restore_keeps_runtime_default() -> None:
    model = GaussianModel(
        sh_degree=1,
        semantic_dim=4,
        device="cpu",
        confidence_floor=0.2,
    )
    snapshot = model.capture_inference("cpu")
    restored = GaussianModel(
        sh_degree=1,
        semantic_dim=4,
        device="cpu",
        confidence_floor=0.05,
    )
    restored.restore(snapshot)
    assert math.isclose(restored.policy_bank.confidence_floor, 0.2)

    snapshot.pop("policy_confidence_floor")
    legacy = GaussianModel(
        sh_degree=1,
        semantic_dim=4,
        device="cpu",
        confidence_floor=0.3,
    )
    legacy.restore(snapshot)
    assert math.isclose(legacy.policy_bank.confidence_floor, 0.3)
