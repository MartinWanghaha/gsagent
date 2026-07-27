from pathlib import Path

import pytest

from utils.config_utils import apply_overrides, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_inherited_ablation_config() -> None:
    config = load_config(ROOT / "configs" / "rgb_only.yaml", ["optimization.iterations=12000"])
    assert config["model"]["semantic_dim"] == 16
    assert config["loss"]["lambda_semantic"] == 0.0
    assert config["optimization"]["iterations"] == 12000
    # Adjust default phase endpoints because this deliberately short smoke
    # config would otherwise be invalid.
    config["phases"].update(semantic_from=1000, joint_from=2000, surface_from=3000)
    validate_config(config)


def test_factorized_full_ablation_configs() -> None:
    no_mesh = load_config(ROOT / "configs" / "full_no_mesh_feedback.yaml")
    no_topology = load_config(ROOT / "configs" / "full_no_surface_topology.yaml")
    assert no_mesh["loss"]["lambda_surface"] > 0
    assert no_mesh["loss"]["lambda_mesh"] == 0
    assert no_topology["loss"]["lambda_mesh"] > 0
    assert no_topology["surface"]["topology_enabled"] is False
    validate_config(no_mesh)
    validate_config(no_topology)


def test_new_module_ablation_configs() -> None:
    no_propagation = load_config(
        ROOT / "configs" / "full_no_confidence_propagation.yaml"
    )
    no_certainty = load_config(ROOT / "configs" / "full_no_expert_certainty.yaml")
    no_replacement = load_config(ROOT / "configs" / "full_no_prune_replace.yaml")
    assert no_propagation["semantic"]["propagation_enabled"] is False
    assert no_certainty["semantic"]["evidence_entropy_weight"] == 0.0
    assert no_certainty["semantic"]["evidence_balance_weight"] == 0.0
    assert no_certainty["semantic"]["geometry_evidence_temperature"] == 0.25
    assert no_replacement["density"]["capacity_replacement_enabled"] is False
    validate_config(no_propagation)
    validate_config(no_certainty)
    validate_config(no_replacement)


def test_geometry_evidence_default_has_meaningful_two_million_point_coverage() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config["semantic"]["evidence_samples"] == 2_048


def test_mesh_coverage_density_signal_has_an_explicit_default() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")

    assert config["density"]["mesh_coverage_weight"] == 0.5
    validate_config(config)


@pytest.mark.parametrize("value", [True, float("nan"), -0.01])
def test_mesh_coverage_weight_must_be_finite_non_negative(value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["density"]["mesh_coverage_weight"] = value

    with pytest.raises(ValueError, match="mesh_coverage_weight"):
        validate_config(config)


@pytest.mark.parametrize("value", [-1, True, 1.5, "10"])
def test_profile_interval_must_be_a_non_negative_integer(value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["logging"]["profile_interval"] = value
    with pytest.raises(ValueError, match="profile_interval"):
        validate_config(config)


def test_training_profiling_is_disabled_by_default() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config["logging"]["profile_interval"] == 0


def test_compiled_dimensions_are_validated() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["model"]["semantic_dim"] = 8
    with pytest.raises(ValueError, match="semantic_dim"):
        validate_config(config)


def test_unsupported_spherical_harmonic_degree_fails_before_training() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["model"]["sh_degree"] = 4
    with pytest.raises(ValueError, match="sh_degree"):
        validate_config(config)


def test_depth_normal_alpha_threshold_is_validated() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["loss"]["normal_alpha_threshold"] = 1.01
    with pytest.raises(ValueError, match="normal_alpha_threshold"):
        validate_config(config)


def test_size_pruning_switch_must_be_boolean() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["density"]["enable_size_pruning"] = "false"
    with pytest.raises(ValueError, match="enable_size_pruning"):
        validate_config(config)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "32768"])
def test_region_decode_chunk_size_must_be_a_positive_integer(value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["semantic"]["region_decode_chunk_size"] = value
    with pytest.raises(ValueError, match="region_decode_chunk_size"):
        validate_config(config)


def test_region_decode_chunk_is_required_by_the_region_training_schema() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    del config["semantic"]["region_decode_chunk_size"]
    with pytest.raises(ValueError, match="region_decode_chunk_size"):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("support_routing_query_chunk", 0, "support_routing_query_chunk"),
        ("support_routing_query_chunk", True, "support_routing_query_chunk"),
        ("scipy_workers", 0, "scipy_workers"),
        ("scipy_workers", -2, "scipy_workers"),
        ("scipy_workers", True, "scipy_workers"),
        ("mesh_feedback_scipy_workers", 0, "mesh_feedback_scipy_workers"),
        ("mesh_feedback_scipy_workers", -2, "mesh_feedback_scipy_workers"),
        ("mesh_feedback_scipy_workers", True, "mesh_feedback_scipy_workers"),
    ],
)
def test_surface_routing_execution_policy_is_validated(
    key,
    value,
    message,
) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("topology_enabled", "true", "topology_enabled"),
        ("topology_interval", 0, "topology_interval"),
        ("topology_max_net_growth", -1, "topology_max_net_growth"),
        ("topology_replacement_budget", -1, "topology_replacement_budget"),
        ("topology_protect_min_confidence", 1.1, "protect_min_confidence"),
        ("topology_enable_size_pruning", 1, "topology_enable_size_pruning"),
        ("mesh_feedback_async", "yes", "mesh_feedback_async"),
        ("mesh_feedback_snapshot_device", "mps", "snapshot_device"),
        ("mesh_feedback_min_component_faces", -1, "min_component_faces"),
    ],
)
def test_surface_topology_and_feedback_configuration_is_validated(
    key,
    value,
    message,
) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_surface_topology_window_must_stay_inside_surface_phase() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"]["topology_from"] = config["phases"]["surface_from"] - 1
    with pytest.raises(ValueError, match="topology window"):
        validate_config(config)


def test_mesh_v4_feedback_schema_has_quality_first_defaults() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    surface = config["surface"]

    assert surface["mesh_feedback_resolution"] == 96
    assert surface["mesh_feedback_samples"] == 8_192
    assert surface["support_candidate_budget"] == 2_048
    assert {
        key: surface[f"mesh_feedback_{key}"]
        for key in (
            "max_candidate_age",
            "max_topology_events",
            "max_churn_ratio",
            "retry_interval",
            "blend_iterations",
            "gate_probes",
            "gate_min_score",
            "gate_sdf_p90",
            "gate_normal",
            "gate_semantic",
            "min_opacity",
            "min_confidence",
            "min_expert_certainty",
            "match_k",
            "match_radius",
            "match_semantic",
            "robust_delta",
            "min_matches",
        )
    } == {
        "max_candidate_age": 1_000,
        "max_topology_events": 1,
        "max_churn_ratio": 0.015,
        "retry_interval": 100,
        "blend_iterations": 125,
        "gate_probes": 2_048,
        "gate_min_score": 0.70,
        "gate_sdf_p90": 2.5,
        "gate_normal": 0.60,
        "gate_semantic": 0.50,
        "min_opacity": 0.05,
        "min_confidence": 0.35,
        "min_expert_certainty": 0.55,
        "match_k": 16,
        "match_radius": 3.0,
        "match_semantic": 0.50,
        "robust_delta": 1.5,
        "min_matches": 256,
    }
    validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mesh_feedback_max_candidate_age", 0),
        ("mesh_feedback_max_candidate_age", True),
        ("mesh_feedback_max_topology_events", -1),
        ("mesh_feedback_max_churn_ratio", 1.01),
        ("mesh_feedback_max_churn_ratio", float("nan")),
        ("mesh_feedback_retry_interval", 0),
        ("mesh_feedback_blend_iterations", -1),
        ("mesh_feedback_gate_probes", 0),
        ("mesh_feedback_gate_min_score", 1.01),
        ("mesh_feedback_gate_sdf_p90", 0.0),
        ("mesh_feedback_gate_normal", -0.01),
        ("mesh_feedback_gate_semantic", 1.01),
        ("mesh_feedback_min_opacity", -0.01),
        ("mesh_feedback_min_confidence", 1.01),
        ("mesh_feedback_min_expert_certainty", "0.55"),
        ("mesh_feedback_match_k", 0),
        ("mesh_feedback_match_radius", float("inf")),
        ("mesh_feedback_match_semantic", -1.01),
        ("mesh_feedback_robust_delta", 0.0),
        ("mesh_feedback_min_matches", 0),
    ],
)
def test_mesh_v4_feedback_controls_are_strictly_validated(key, value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"][key] = value

    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mesh_feedback_min_matches", 2_049),
        ("mesh_feedback_match_k", 2_049),
    ],
)
def test_mesh_v4_match_counts_cannot_exceed_gate_probes(key, value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"][key] = value

    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.mark.parametrize("value", [-0.01, 1.0, 2.0])
def test_semantic_confidence_floor_must_leave_a_baseline_interval(value) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["semantic"]["confidence_floor"] = value
    with pytest.raises(ValueError, match="confidence_floor"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("density", "split_children", 1, "split_children"),
        ("density", "capacity_replacement_enabled", 1, "capacity_replacement_enabled"),
        ("density", "replace_near_cap_ratio", 0, "replace_near_cap_ratio"),
        ("density", "max_replacements_per_step", -1, "max_replacements"),
        ("density", "semantic_weight", -0.1, "semantic_weight"),
        ("semantic", "evidence_entropy_weight", -0.1, "evidence_entropy_weight"),
        ("semantic", "evidence_balance_weight", -0.1, "evidence_balance_weight"),
        ("semantic", "evidence_interval", 0, "evidence_interval"),
        ("semantic", "evidence_samples", True, "evidence_samples"),
        ("semantic", "geometry_evidence_temperature", 0, "geometry_evidence_temperature"),
        ("semantic", "propagation_enabled", 1, "propagation_enabled"),
        ("semantic", "propagation_samples", 0, "propagation_samples"),
        ("semantic", "propagation_neighbors", True, "propagation_neighbors"),
        ("semantic", "propagation_momentum", 1.0, "propagation_momentum"),
        ("semantic", "propagation_decay", 1.1, "propagation_decay"),
        ("semantic", "propagation_semantic_floor", 1.0, "propagation_semantic_floor"),
        ("semantic", "propagation_support_sigma", 0, "propagation_support_sigma"),
        ("semantic", "propagation_boundary_barrier", -1, "propagation_boundary_barrier"),
    ],
)
def test_new_density_and_semantic_controls_are_validated(
    section,
    key,
    value,
    message,
) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_region_conditioned_schema_rejects_missing_architecture_controls() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["logging"].pop("profile_interval")
    for key in (
        "capacity_replacement_enabled",
        "replace_near_cap_ratio",
        "max_replacements_per_step",
        "mesh_coverage_weight",
    ):
        config["density"].pop(key)
    for key in tuple(config["surface"]):
        if key.startswith("topology_") or key.startswith("mesh_feedback_"):
            config["surface"].pop(key)
    for key in (
        "evidence_entropy_weight",
        "evidence_balance_weight",
        "geometry_evidence_temperature",
        "propagation_enabled",
        "propagation_samples",
        "propagation_neighbors",
        "propagation_min_seed_confidence",
        "propagation_max_confidence",
        "propagation_momentum",
        "propagation_decay",
        "propagation_semantic_floor",
        "propagation_support_sigma",
        "propagation_boundary_barrier",
    ):
        config["semantic"].pop(key)
    with pytest.raises(ValueError, match="region-conditioned configuration"):
        validate_config(config)


def test_checkpoint_override_allowlist_rejects_objective_changes() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    resumed = apply_overrides(
        config,
        [
            "optimization.iterations=40000",
            "semantic.region_decode_chunk_size=16384",
            "logging.log_interval=25",
        ],
        allowed_keys={
            "optimization.iterations",
            "semantic.region_decode_chunk_size",
        },
        allowed_prefixes=("logging.",),
    )
    assert resumed["optimization"]["iterations"] == 40000
    assert resumed["semantic"]["region_decode_chunk_size"] == 16384
    with pytest.raises(ValueError, match="not allowed"):
        apply_overrides(
            config,
            ["loss.lambda_semantic=9"],
            allowed_keys={"optimization.iterations"},
            allowed_prefixes=("logging.",),
        )
