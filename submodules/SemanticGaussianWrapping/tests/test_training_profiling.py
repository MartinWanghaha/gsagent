from __future__ import annotations

import pytest
import torch

import training.engine as engine
from training.engine import _should_profile_step, _StepProfiler


def test_profile_sampling_is_disabled_at_zero_interval() -> None:
    assert not any(_should_profile_step(iteration, 0) for iteration in range(1, 20))
    assert [
        iteration
        for iteration in range(1, 13)
        if _should_profile_step(iteration, 5)
    ] == [5, 10]


def test_cuda_step_profiler_synchronizes_only_at_sampled_boundaries(
    monkeypatch,
) -> None:
    synchronizations: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronizations.append(torch.device(device)),
    )
    ticks = iter([1.0, 1.01, 1.03, 1.07, 1.10])
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(ticks))

    profiler = _StepProfiler("cuda:0")
    profiler.start()
    profiler.mark("render")
    profiler.mark("surface")
    profiler.mark("backward")
    metrics = profiler.finish("topology")

    assert synchronizations == [torch.device("cuda:0")] * 5
    assert metrics == pytest.approx(
        {
            "time_render_ms": 10.0,
            "time_surface_ms": 20.0,
            "time_backward_ms": 40.0,
            "time_topology_ms": 30.0,
            "time_step_ms": 100.0,
        }
    )
