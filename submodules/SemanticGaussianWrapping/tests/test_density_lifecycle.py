from __future__ import annotations

from training.engine import _density_lifecycle


def _config(**overrides):
    values = {
        "from_iter": 500,
        "until_iter": 2_000,
        "interval": 100,
        "opacity_reset_interval": 1_000,
        "enable_size_pruning": False,
    }
    values.update(overrides)
    return values


def test_first_topology_step_uses_the_complete_warmup_window() -> None:
    lifecycle = [_density_lifecycle(iteration, _config()) for iteration in range(1, 601)]

    assert all(item.observe for item in lifecycle)
    assert not lifecycle[499].topology_step  # iteration 500 is still warm-up
    assert [index + 1 for index, item in enumerate(lifecycle) if item.topology_step] == [600]


def test_size_pruning_is_opt_in_and_delayed_until_after_opacity_reset() -> None:
    disabled = _config(enable_size_pruning=False)
    enabled = _config(enable_size_pruning=True)

    assert not _density_lifecycle(1_000, enabled).enable_size_pruning
    assert _density_lifecycle(1_100, enabled).enable_size_pruning
    assert not _density_lifecycle(1_100, disabled).enable_size_pruning


def test_density_observation_stops_at_exclusive_until_iteration() -> None:
    config = _config(until_iter=700)

    assert _density_lifecycle(699, config).observe
    assert not _density_lifecycle(700, config).observe
    assert not _density_lifecycle(700, config).topology_step


def _surface_config(**overrides):
    values = {
        "enabled": True,
        "topology_enabled": True,
        "topology_from": 2_400,
        "topology_until": 3_000,
        "topology_interval": 200,
        "topology_max_net_growth": 0,
        "topology_replacement_budget": 25,
        "topology_enable_size_pruning": True,
        "topology_protect_min_confidence": 0.5,
        "topology_protect_boundary": 0.25,
        "topology_protect_thin_probability": 0.5,
    }
    values.update(overrides)
    return values


def _surface_lifecycle(iteration: int, surface=None):
    return _density_lifecycle(
        iteration,
        _config(until_iter=2_000),
        surface_config=_surface_config() if surface is None else surface,
    )


def test_surface_topology_collects_a_complete_window_before_replacement() -> None:
    opening = _surface_lifecycle(2_400)
    before_first_step = _surface_lifecycle(2_599)
    first_step = _surface_lifecycle(2_600)
    closing_step = _surface_lifecycle(3_000)
    after = _surface_lifecycle(3_001)

    assert opening.observe and opening.window == "surface"
    assert not opening.topology_step
    assert before_first_step.observe and not before_first_step.topology_step
    assert first_step.topology_step and closing_step.topology_step
    assert first_step.enable_size_pruning
    assert first_step.topology_budget is not None
    assert first_step.topology_budget.max_net_growth == 0
    assert first_step.topology_budget.replacement_budget == 25
    assert not after.observe and after.window is None


def test_surface_topology_switch_disables_observation_and_mutation() -> None:
    disabled = _surface_config(topology_enabled=False)
    lifecycle = _surface_lifecycle(2_600, disabled)
    assert not lifecycle.observe
    assert not lifecycle.topology_step
    assert lifecycle.topology_budget is None


def test_standard_window_wins_if_surface_window_overlaps() -> None:
    overlapping = _surface_config(topology_from=1_500, topology_until=3_000)
    lifecycle = _surface_lifecycle(1_600, overlapping)
    assert lifecycle.window == "standard"
    assert lifecycle.topology_budget is None
