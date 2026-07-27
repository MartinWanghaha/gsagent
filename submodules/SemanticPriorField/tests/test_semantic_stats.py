from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "semantic_prior_field"


@pytest.fixture(autouse=True)
def package_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))


def _make_accumulator():
    from semantic.scatter import SemanticStatsAccumulator

    return SemanticStatsAccumulator(device="cpu")


def test_accumulator_conflict_identities():
    accumulator = _make_accumulator()
    n = 8
    grad = torch.randn(n, 16)

    # Aligned case: unsigned mass equals the signed norm -> zero conflict
    sink = {
        "semantic_abs_grad": grad.norm(dim=-1),
        "semantic_contribution": torch.ones(n),
        "semantic_grad": grad,
    }
    accumulator.update_from_sink(sink)
    assert accumulator.ready
    conflict = accumulator.conflict_score()
    assert torch.allclose(conflict, torch.zeros(n), atol=1e-6)

    # Conflicting case: unsigned mass exceeds the signed norm
    accumulator.reset(n)
    sink_conflict = {
        "semantic_abs_grad": 2.0 * grad.norm(dim=-1),
        "semantic_contribution": torch.full((n,), 2.0),
        "semantic_grad": grad,
    }
    accumulator.update_from_sink(sink_conflict)
    conflict = accumulator.conflict_score()
    expected = grad.norm(dim=-1) / 2.0
    assert torch.allclose(conflict, expected, atol=1e-5)


def test_accumulator_resets_on_count_change():
    accumulator = _make_accumulator()
    sink8 = {
        "semantic_abs_grad": torch.ones(8),
        "semantic_contribution": torch.ones(8),
        "semantic_grad": torch.zeros(8, 16),
    }
    accumulator.update_from_sink(sink8)
    assert accumulator.updates == 1
    # Gaussian count changed (topology change without explicit reset):
    # the accumulator re-sizes and starts over instead of mixing indices
    sink10 = {
        "semantic_abs_grad": torch.ones(10),
        "semantic_contribution": torch.ones(10),
        "semantic_grad": torch.zeros(10, 16),
    }
    accumulator.update_from_sink(sink10)
    assert accumulator.abs_grad.shape[0] == 10
    assert accumulator.updates == 1

    accumulator.reset()
    assert not accumulator.ready
    assert accumulator.conflict_score() is None


def test_accumulator_multiple_updates_accumulate():
    accumulator = _make_accumulator()
    sink = {
        "semantic_abs_grad": torch.ones(4),
        "semantic_contribution": torch.ones(4),
        "semantic_grad": torch.zeros(4, 16),
    }
    for _ in range(5):
        accumulator.update_from_sink({k: v.clone() for k, v in sink.items()})
    assert accumulator.updates == 5
    assert torch.allclose(accumulator.abs_grad, torch.full((4,), 5.0))
    # All mass unsigned with zero signed gradient -> conflict = abs/contrib = 1
    assert torch.allclose(accumulator.conflict_score(), torch.ones(4))
