from __future__ import annotations

import pytest
import torch

from densification import DensityController, TopologyBudget
from semantic.region_membership import SparseRegionMembership


class DummyGaussians:
    def __init__(self, count: int = 4) -> None:
        self.xyz = torch.stack((torch.arange(count), torch.zeros(count), torch.ones(count)), dim=-1).float()
        self.scaling = torch.full((count, 3), 0.005)
        self.scaling[-1] = 0.1
        self.rotation = torch.zeros(count, 4)
        self.rotation[:, 0] = 1
        self.opacity = torch.full((count, 1), 0.5)
        self.confidence = torch.zeros(count, 1)
        self.boundary = torch.zeros(count, 1)
        self.posterior = torch.full((count, 5), 0.2)
        self.observation_count = torch.zeros(count, 1)
        self.last_mutation = None

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scaling(self):
        return self.scaling

    @property
    def get_rotation(self):
        return self.rotation

    @property
    def get_opacity(self):
        return self.opacity

    @property
    def get_semantic_confidence(self):
        return self.confidence

    @property
    def get_boundary_score(self):
        return self.boundary

    @property
    def get_geometry_posterior(self):
        return self.posterior

    def update_evidence(self, **kwargs):
        indices = kwargs.get("indices")
        if indices is None:
            indices = torch.arange(self.xyz.shape[0])
        self.confidence[indices] = (
            0.9 * self.confidence[indices] + 0.1 * kwargs["semantic_confidence"]
        )
        self.boundary[indices] = (
            0.9 * self.boundary[indices] + 0.1 * kwargs["boundary_score"]
        )
        self.observation_count[indices] += 1

    def mutate_topology(self, clone, split, prune, *, children, offsets, scale_factor):
        self.last_mutation = (clone.clone(), split.clone(), prune.clone(), offsets.clone())
        survivor = ~(split | prune)
        pieces = [self.xyz[survivor], self.xyz[clone], (self.xyz[split, None] + offsets).reshape(-1, 3)]
        self.xyz = torch.cat(pieces)
        new_count = self.xyz.shape[0]
        self.scaling = torch.full((new_count, 3), 0.005)
        self.rotation = torch.zeros(new_count, 4)
        self.rotation[:, 0] = 1
        self.opacity = torch.full((new_count, 1), 0.5)
        self.confidence = torch.zeros(new_count, 1)
        self.boundary = torch.zeros(new_count, 1)
        self.posterior = torch.full((new_count, 5), 0.2)
        self.observation_count = torch.zeros(new_count, 1)


def _config(**overrides):
    values = dict(
        gradient_threshold=2e-4,
        rgb_weight=1.0,
        semantic_weight=0.5,
        boundary_weight=0.75,
        geometry_weight=0.75,
        mesh_coverage_weight=0.5,
        min_opacity=0.005,
        max_screen_radius=20.0,
        max_gaussians=100,
        capacity_replacement_enabled=True,
        replace_near_cap_ratio=0.98,
        max_replacements_per_step=100,
        split_children=2,
        region_budget_temperature=0.5,
        thin_protection=0.8,
    )
    values.update(overrides)
    return values


def _membership(
    ids,
    *,
    weights=None,
    confidence=None,
    background=None,
    tail=None,
) -> SparseRegionMembership:
    ids = torch.as_tensor(ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids[:, None]
    rows, width = ids.shape
    if weights is None:
        weights = torch.ones(rows, width)
    weights = torch.as_tensor(weights, dtype=torch.float32).reshape(rows, width)
    if confidence is None:
        confidence = torch.ones(rows, 1)
    confidence = torch.as_tensor(confidence, dtype=torch.float32).reshape(rows, 1)
    if background is None:
        background = torch.zeros(rows, 1)
    background = torch.as_tensor(background, dtype=torch.float32).reshape(rows, 1)
    if tail is None:
        tail = 1.0 - background - weights.sum(dim=1, keepdim=True)
    tail = torch.as_tensor(tail, dtype=torch.float32).reshape(rows, 1)
    return SparseRegionMembership(
        ids=ids,
        weights=weights,
        background=background,
        tail=tail,
        confidence=confidence,
    )


def test_zero_confidence_gates_semantic_score() -> None:
    gaussians = DummyGaussians()
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.grad_accum[:] = torch.tensor([0.0, 1e-3, 2e-3, 3e-3])
    controller.grad_denom[:] = 1
    controller.observation_count[:] = 1
    baseline = controller.scores(gaussians)
    controller.semantic_accum[:] = 100
    controller.boundary_accum[:] = 100
    controller.geometry_accum[:] = 100
    assert torch.equal(controller.scores(gaussians), baseline)


def test_zero_guidance_weights_make_density_a_true_rgb_only_ablation() -> None:
    class ForbiddenPolicy:
        def from_gaussians(self, _gaussians):
            raise AssertionError("geometry policy must not affect RGB-only density")

    gaussians = DummyGaussians(6)
    gaussians.confidence[:] = 1.0
    controller = DensityController(
        _config(
            semantic_weight=0.0,
            boundary_weight=0.0,
            geometry_weight=0.0,
        ),
        scene_extent=1.0,
        policy_bank=ForbiddenPolicy(),
    )
    controller._ensure(gaussians)
    controller.observation_count[:] = 1.0
    controller.confidence_accum[:] = 1.0
    controller.semantic_accum[:] = 100.0
    controller.boundary_accum[:] = 100.0
    controller.geometry_accum[:] = 100.0
    controller.geometry_denom[:] = 1.0

    decision = controller.decide(
        gaussians,
        region_membership_resolver=lambda _indices: (_ for _ in ()).throw(
            AssertionError("semantic region routing must be disabled")
        ),
    )

    assert controller.semantic_guidance_enabled is False
    assert not decision.clone.any()
    assert not decision.split.any()


def test_rgb_only_split_offsets_do_not_read_geometry_experts() -> None:
    class RgbOnlyGaussians(DummyGaussians):
        @property
        def get_geometry_posterior(self):
            raise AssertionError("RGB-only split offsets must not read experts")

    gaussians = RgbOnlyGaussians(3)
    controller = DensityController(
        _config(
            semantic_weight=0.0,
            boundary_weight=0.0,
            geometry_weight=0.0,
        ),
        scene_extent=1.0,
    )
    split = torch.tensor([False, True, False])
    offsets = controller._policy_offsets(gaussians, split, children=2)
    assert offsets.shape == (1, 2, 3)


def test_atomic_update_uses_source_topology_masks() -> None:
    torch.manual_seed(7)
    gaussians = DummyGaussians()
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1e-3
    controller.grad_denom[:] = 1
    controller.observation_count[:] = 1
    controller.rgb_accum[:] = torch.arange(4).float()
    report = controller.step(gaussians, percent_dense=0.01)
    clone, split, prune, offsets = gaussians.last_mutation
    assert clone.shape == split.shape == prune.shape == (4,)
    assert offsets.shape == (int(split.sum()), 2, 3)
    assert report.before == 4
    assert report.after == gaussians.get_xyz.shape[0]


def test_soft_region_budget_never_exceeds_capacity() -> None:
    gaussians = DummyGaussians(6)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    candidate = torch.ones(6, dtype=torch.bool)
    score = torch.arange(6).float()
    membership = _membership(
        [[1, 2], [1, 2], [1, 2], [2, 1], [2, 1], [2, 1]],
        weights=[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ],
    )
    selected = controller._balanced_candidates(
        score,
        candidate,
        lambda indices: membership.index_select(indices),
        capacity=3,
    )
    assert int(selected.sum()) == 3


def test_region_membership_resolver_receives_only_eligible_candidates() -> None:
    gaussians = DummyGaussians(8)
    controller = DensityController(
        _config(max_growth_fraction=1.0, max_new_per_step=100),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = torch.tensor(
        [0.0, 1e-3, 0.0, 2e-3, 0.0, 3e-3, 0.0, 4e-3]
    )
    controller.grad_denom[:] = 1
    seen = []
    dense_membership = _membership([[9], [1], [9], [2], [9], [2], [9], [3]])

    def resolve(indices):
        seen.append(indices.clone())
        return dense_membership.index_select(indices)

    controller.decide(gaussians, region_membership_resolver=resolve)

    assert len(seen) == 1
    assert seen[0].tolist() == [1, 3, 5, 7]


def test_region_membership_resolver_is_lazy_for_empty_candidates_or_capacity() -> None:
    gaussians = DummyGaussians(8)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)

    def forbidden(_indices):
        raise AssertionError("region resolver must not be called")

    controller.decide(gaussians, region_membership_resolver=forbidden)

    full = DensityController(_config(max_gaussians=8), scene_extent=1.0)
    full._ensure(gaussians)
    full.grad_accum[:] = 1.0
    full.grad_denom[:] = 1.0
    full.decide(gaussians, region_membership_resolver=forbidden)


def test_region_membership_resolver_must_return_sparse_membership() -> None:
    gaussians = DummyGaussians(4)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    with pytest.raises(TypeError, match="SparseRegionMembership"):
        controller.decide(
            gaussians,
            region_membership_resolver=lambda indices: torch.zeros(
                indices.numel(), 1, dtype=torch.long
            ),
        )


def test_region_membership_resolver_must_align_with_candidate_indices() -> None:
    gaussians = DummyGaussians(4)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    with pytest.raises(ValueError, match="align with density candidates"):
        controller.decide(
            gaussians,
            region_membership_resolver=lambda _indices: _membership([[1], [2]]),
        )


def test_region_mass_and_soft_priority_control_allocation() -> None:
    controller = DensityController(
        _config(region_budget_temperature=1.0),
        scene_extent=1.0,
    )
    candidate = torch.ones(5, dtype=torch.bool)
    score = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
    # Region 1 owns 2.1 mass and region 2 owns 1.0, yielding budgets 2 and 1.
    # Within region 1, candidate 1 outranks candidate 0 after confidence and
    # membership weight are applied despite its lower global score.
    membership = _membership(
        [[1], [1], [1], [2], [2]],
        weights=[[0.1], [1.0], [1.0], [0.5], [0.5]],
        confidence=[[1.0], [1.0], [1.0], [1.0], [1.0]],
    )

    selected = controller._balanced_candidates(
        score,
        candidate,
        lambda indices: membership.index_select(indices),
        capacity=3,
    )

    assert selected.tolist() == [False, True, True, True, False]


def test_cross_region_duplicates_are_stable_and_global_score_fills_capacity() -> None:
    controller = DensityController(
        _config(region_budget_temperature=0.0),
        scene_extent=1.0,
    )
    candidate = torch.ones(4, dtype=torch.bool)
    score = torch.tensor([10.0, 8.0, 9.0, 7.0])
    membership = _membership(
        [[1, 2], [1, 2], [2, 1], [1, 2]],
        weights=[[0.5, 0.5], [0.4, 0.1], [0.4, 0.1], [0.2, 0.2]],
    )

    selected = controller._balanced_candidates(
        score,
        candidate,
        lambda indices: membership.index_select(indices),
        capacity=2,
    )

    # Both one-slot regional budgets choose candidate 0. Stable de-duplication
    # selects it once, then the global score fills the remaining slot with 2.
    assert selected.tolist() == [True, False, True, False]


def test_all_background_or_zero_evidence_uses_global_topk() -> None:
    controller = DensityController(_config(), scene_extent=1.0)
    candidate = torch.ones(4, dtype=torch.bool)
    score = torch.tensor([1.0, 4.0, 3.0, 2.0])
    membership = _membership(
        [[1], [1], [2], [2]],
        weights=torch.zeros(4, 1),
        confidence=torch.zeros(4, 1),
        background=torch.ones(4, 1),
    )

    selected = controller._balanced_candidates(
        score,
        candidate,
        lambda indices: membership.index_select(indices),
        capacity=2,
    )

    assert selected.tolist() == [False, True, True, False]


def test_cached_region_membership_resolver_uses_index_select() -> None:
    candidate = torch.tensor([False, True, False, True, False, True])
    source = _membership(
        [[1, 2], [3, 4], [5, 6]],
        weights=[[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]],
    )
    calls = []

    def resolve(indices):
        calls.append(indices.clone())
        return source

    cached = DensityController._cached_region_membership_resolver(
        candidate,
        resolve,
    )
    assert cached is not None
    selected = cached(torch.tensor([1, 5]))

    assert len(calls) == 1
    assert calls[0].tolist() == [1, 3, 5]
    assert selected.ids.tolist() == [[1, 2], [5, 6]]
    assert torch.allclose(selected.weights, torch.tensor([[0.7, 0.3], [0.9, 0.1]]))


def test_uniform_rgb_residual_does_not_bypass_absolute_gradient_gate() -> None:
    gaussians = DummyGaussians(20)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.observation_count[:] = 1
    controller.rgb_accum[:] = 0.5
    decision = controller.decide(gaussians)
    assert not decision.clone.any()
    assert not decision.split.any()


def test_density_step_has_a_bounded_growth_budget() -> None:
    gaussians = DummyGaussians(100)
    controller = DensityController(
        _config(max_growth_fraction=0.05, max_new_per_step=100),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0
    controller.observation_count[:] = 1.0
    decision = controller.decide(gaussians)
    assert int(decision.clone.sum() + decision.split.sum()) <= 5


def test_size_pruning_is_disabled_by_default_but_opacity_pruning_remains() -> None:
    gaussians = DummyGaussians(3)
    gaussians.scaling[-1] = 0.2
    gaussians.opacity[0] = 0.001
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.max_radii[1] = 100.0

    decision = controller.decide(gaussians)

    assert decision.prune.tolist() == [True, False, False]


def test_size_pruning_requires_explicit_enablement() -> None:
    gaussians = DummyGaussians(3)
    gaussians.scaling[-1] = 0.2
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.max_radii[1] = 100.0

    decision = controller.decide(gaussians, enable_size_pruning=True)

    assert decision.prune.tolist() == [False, True, True]


def test_pre_cap_standard_candidate_keeps_original_refine_before_prune_order() -> None:
    gaussians = DummyGaussians(4)
    gaussians.opacity[0] = 0.001
    controller = DensityController(_config(max_gaussians=100), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.grad_accum[0] = 1.0
    controller.grad_denom[0] = 1.0

    decision = controller.decide(gaussians)

    assert decision.clone[0]
    assert not decision.prune[0]


def test_density_accumulation_window_round_trips() -> None:
    gaussians = DummyGaussians(4)
    original = DensityController(_config(), scene_extent=1.0)
    original._ensure(gaussians)
    for offset, name in enumerate(original._ACCUMULATORS):
        getattr(original, name).copy_(torch.arange(4).float() + offset)

    restored = DensityController(_config(), scene_extent=1.0)
    restored.load_state_dict(original.state_dict(), gaussians)
    for name in original._ACCUMULATORS:
        assert torch.equal(getattr(restored, name), getattr(original, name))


def test_sparse_mesh_coverage_accumulates_duplicate_indices_and_valid_values() -> None:
    gaussians = DummyGaussians(4)
    controller = DensityController(_config(), scene_extent=1.0)

    controller.observe_mesh_coverage(
        gaussians,
        indices=torch.tensor([1, 1, 2, 3]),
        residual=torch.tensor([1.0, 3.0, 9.0, float("nan")]),
        valid=torch.tensor([True, True, False, True]),
    )

    assert controller.mesh_coverage_accum.tolist() == [0.0, 4.0, 0.0, 0.0]
    assert controller.mesh_coverage_denom.tolist() == [0.0, 2.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        (torch.tensor([-1]), IndexError, "outside Gaussian topology"),
        (torch.tensor([4]), IndexError, "outside Gaussian topology"),
        (torch.tensor([1.0]), TypeError, "integer dtype"),
        (torch.tensor([True]), TypeError, "integer dtype"),
        (torch.tensor([[1]]), ValueError, "one-dimensional"),
    ],
)
def test_mesh_coverage_rejects_invalid_indices(indices, error, message) -> None:
    gaussians = DummyGaussians(4)
    controller = DensityController(_config(), scene_extent=1.0)

    with pytest.raises(error, match=message):
        controller.observe_mesh_coverage(
            gaussians,
            indices=indices,
            residual=torch.ones(indices.numel()),
        )


def test_density_checkpoint_requires_region_conditioned_accumulators() -> None:
    gaussians = DummyGaussians(4)
    original = DensityController(_config(), scene_extent=1.0)
    original._ensure(gaussians)
    original.grad_accum[:] = 3.0
    state = original.state_dict()
    state["version"] = 2
    state["accumulators"].pop("mesh_coverage_accum")
    state["accumulators"].pop("mesh_coverage_denom")

    restored = DensityController(_config(), scene_extent=1.0)
    with pytest.raises(ValueError, match="region-conditioned schema"):
        restored.load_state_dict(state, gaussians)


def test_mesh_coverage_score_ranks_topology_candidates_and_clears_after_step() -> None:
    gaussians = DummyGaussians(4)
    gaussians.scaling[:] = 0.005
    controller = DensityController(
        _config(max_growth_fraction=0.25, max_new_per_step=1),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1e-3
    controller.grad_denom[:] = 1
    controller.observation_count[:] = 1
    controller.observe_mesh_coverage(
        gaussians,
        indices=torch.tensor([1]),
        residual=torch.tensor([2.0]),
    )

    score = controller.scores(gaussians)
    assert score[1] > score[[0, 2, 3]].max()
    report = controller.step(gaussians)
    clone, split, _, _ = gaussians.last_mutation
    assert clone.tolist() == [False, True, False, False]
    assert not split.any()
    assert report.cloned == 1
    assert not controller.mesh_coverage_accum.any()
    assert not controller.mesh_coverage_denom.any()


def test_density_rejects_unknown_future_checkpoint_schema() -> None:
    gaussians = DummyGaussians(4)
    controller = DensityController(_config(), scene_extent=1.0)
    state = controller.state_dict(gaussians)
    state["version"] = 4
    with pytest.raises(ValueError, match="newer schema"):
        controller.load_state_dict(state, gaussians)


def test_confidence_normalizes_residual_then_gates_once() -> None:
    gaussians = DummyGaussians(2)
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.observation_count[:] = 4
    controller.confidence_accum[:] = torch.tensor([2.0, 0.4])
    controller.semantic_accum[:] = torch.tensor([6.0, 1.2])
    conditional = controller._confidence_weighted_observation(
        controller.semantic_accum
    )
    assert torch.allclose(conditional, torch.tensor([3.0, 3.0]))
    assert torch.allclose(
        controller._semantic_confidence(gaussians),
        torch.tensor([0.5, 0.1]),
    )


def _surface_budget(**overrides) -> TopologyBudget:
    values = {
        "max_net_growth": 0,
        "replacement_budget": 20,
        "protect_min_confidence": 0.5,
        "protect_boundary": 0.25,
        "protect_thin_probability": 0.5,
    }
    values.update(overrides)
    return TopologyBudget(**values)


def test_full_cap_uses_current_prune_donors_for_clone_replacement() -> None:
    gaussians = DummyGaussians(8)
    gaussians.opacity[:2] = 0.001
    controller = DensityController(
        _config(max_gaussians=8, max_growth_fraction=1.0),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    report = controller.step(gaussians, topology_budget=_surface_budget())

    clone, split, prune, _ = gaussians.last_mutation
    assert int(prune.sum()) == 2
    assert int(clone.sum()) == 2
    assert not split.any()
    assert report.before == report.after == 8


@pytest.mark.parametrize("children", [2, 3, 4])
def test_split_replacement_accounts_for_every_child_slot(children: int) -> None:
    count = 12
    gaussians = DummyGaussians(count)
    gaussians.scaling[:] = 0.1
    donor_slots = 2 * (children - 1)
    gaussians.opacity[:donor_slots] = 0.001
    controller = DensityController(
        _config(
            max_gaussians=count,
            max_growth_fraction=1.0,
            split_children=children,
        ),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    report = controller.step(gaussians, topology_budget=_surface_budget())

    assert report.pruned == donor_slots
    assert report.split_parents == 2
    assert report.after == count
    assert report.after <= controller.cfg["max_gaussians"]


def test_full_cap_without_donor_performs_no_mutation() -> None:
    gaussians = DummyGaussians(8)
    controller = DensityController(
        _config(max_gaussians=8, max_growth_fraction=1.0),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    report = controller.step(gaussians, topology_budget=_surface_budget())

    assert gaussians.last_mutation is None
    assert report.cloned == report.split_parents == report.pruned == 0
    assert report.after == 8


def test_surface_replacement_budget_caps_donor_churn() -> None:
    gaussians = DummyGaussians(10)
    gaussians.opacity[:5] = 0.001
    controller = DensityController(_config(max_gaussians=10), scene_extent=1.0)

    decision = controller.decide(
        gaussians,
        topology_budget=_surface_budget(replacement_budget=2),
    )

    assert int(decision.prune.sum()) == 2


def test_surface_net_growth_budget_uses_only_existing_cap_room() -> None:
    gaussians = DummyGaussians(8)
    controller = DensityController(
        _config(max_gaussians=10, max_growth_fraction=1.0),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    report = controller.step(
        gaussians,
        topology_budget=_surface_budget(max_net_growth=5),
    )

    assert report.after == 10
    assert report.after - report.before == 2


def test_surface_donors_protect_confident_boundaries_and_thin_structures() -> None:
    gaussians = DummyGaussians(5)
    gaussians.opacity[:3] = 0.001
    gaussians.confidence[:2] = 0.9
    gaussians.boundary[0] = 0.8
    gaussians.posterior[1] = torch.tensor([0.02, 0.02, 0.92, 0.02, 0.02])
    controller = DensityController(_config(max_gaussians=5), scene_extent=1.0)

    surface = controller.decide(gaussians, topology_budget=_surface_budget())
    standard = controller.decide(gaussians)

    assert surface.prune.tolist() == [False, False, True, False, False]
    # Protection is intentionally scoped to the new surface topology window;
    # the pre-existing standard lifecycle remains unchanged.
    assert standard.prune.tolist()[:3] == [True, True, True]


def test_near_cap_standard_window_can_prune_and_replace() -> None:
    gaussians = DummyGaussians(10)
    gaussians.opacity[:2] = 0.001
    controller = DensityController(
        _config(max_gaussians=10, max_growth_fraction=1.0),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    decision = controller.decide(gaussians)

    assert int(decision.prune.sum()) == 2
    assert int(decision.clone.sum() + decision.split.sum()) == 2


def test_capacity_replacement_ablation_does_not_spend_donor_slots() -> None:
    gaussians = DummyGaussians(10)
    gaussians.opacity[:2] = 0.001
    controller = DensityController(
        _config(
            max_gaussians=10,
            max_growth_fraction=1.0,
            capacity_replacement_enabled=False,
        ),
        scene_extent=1.0,
    )
    controller._ensure(gaussians)
    controller.grad_accum[:] = 1.0
    controller.grad_denom[:] = 1.0

    decision = controller.decide(gaussians, topology_budget=_surface_budget())

    assert int(decision.prune.sum()) == 2
    assert not decision.clone.any()
    assert not decision.split.any()


def test_evidence_update_preserves_unobserved_gaussian_history() -> None:
    gaussians = DummyGaussians(3)
    gaussians.confidence[:] = torch.tensor([[0.7], [0.6], [0.5]])
    gaussians.observation_count[:] = 4
    controller = DensityController(_config(), scene_extent=1.0)
    controller._ensure(gaussians)
    controller.observation_count[1] = 3
    controller.confidence_accum[1] = 2.4
    controller.boundary_accum[1] = 1.2

    controller.update_evidence(gaussians)

    assert gaussians.confidence[0].item() == pytest.approx(0.7)
    assert gaussians.confidence[2].item() == pytest.approx(0.5)
    assert gaussians.observation_count[:, 0].tolist() == [4.0, 5.0, 4.0]


def test_density_checkpoint_without_window_schema_is_rejected() -> None:
    gaussians = DummyGaussians(3)
    source = DensityController(_config(), scene_extent=1.0)
    source._ensure(gaussians)
    source.grad_accum[:] = 7.0
    legacy_state = source.state_dict()
    legacy_state["version"] = 1
    legacy_state.pop("window")

    restored = DensityController(_config(), scene_extent=1.0)
    with pytest.raises(ValueError, match="region-conditioned schema"):
        restored.load_state_dict(legacy_state, gaussians)
