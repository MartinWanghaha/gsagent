from types import SimpleNamespace

import pytest
import torch

from training.engine import SemanticGaussianTrainer


def test_semantic_geometry_diagnostics_use_bounded_uniform_sample() -> None:
    count = 70_000
    direct = torch.zeros(count, 1)
    propagated = torch.zeros(count, 1)
    direct[::2] = 0.8
    propagated[1::2] = 0.6
    logits = torch.zeros(count, 5)
    logits[:, 2] = 4.0
    gaussians = SimpleNamespace(
        get_xyz=torch.zeros(count, 3),
        semantic_confidence=direct,
        propagated_semantic_confidence=propagated,
        get_semantic_confidence=torch.maximum(direct, propagated),
        get_geometry_logits=logits,
    )
    trainer = SimpleNamespace(
        gaussians=gaussians,
        device=torch.device("cpu"),
        config={"semantic": {"geometry_confidence_threshold": 0.35}},
    )

    values = SemanticGaussianTrainer._semantic_geometry_diagnostics(trainer)

    assert values["diagnostic_sample_size"] == 65_536
    assert values["semantic_direct_coverage"] == pytest.approx(0.5, abs=2e-5)
    assert values["semantic_propagated_coverage"] == pytest.approx(0.5, abs=2e-5)
    assert values["semantic_effective_coverage"] == 1.0
    assert 0.0 < values["expert_entropy"] < 1.0
    assert values["expert_max_probability"] > 0.9


def test_semantic_geometry_diagnostics_are_optional_for_plain_gaussians() -> None:
    trainer = SimpleNamespace(
        gaussians=SimpleNamespace(get_xyz=torch.zeros(3, 3)),
        device=torch.device("cpu"),
        config={},
    )
    values = SemanticGaussianTrainer._semantic_geometry_diagnostics(trainer)
    assert values == {"diagnostic_sample_size": 3.0}
