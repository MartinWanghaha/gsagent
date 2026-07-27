"""YAML configuration loading with typed command-line overrides."""

from __future__ import annotations

import copy
from argparse import Namespace
import math
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic configuration inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    parents = config.pop("_base_", None)
    if parents is None:
        return config
    if isinstance(parents, (str, Path)):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_merge(merged, _load_config_tree(parent_path, (*stack, path)))
    return deep_merge(merged, config)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _load_config_tree(Path(path))
    return apply_overrides(config, overrides)


def apply_overrides(
    config: dict[str, Any],
    overrides: list[str] | None = None,
    *,
    allowed_keys: set[str] | None = None,
    allowed_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Apply typed dotted overrides to a copy of a resolved configuration."""

    config = copy.deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Override must be key=value, got {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        allowed = allowed_keys is None or dotted_key in allowed_keys
        allowed |= any(dotted_key.startswith(prefix) for prefix in allowed_prefixes)
        if not allowed:
            raise ValueError(
                f"Override {dotted_key!r} is not allowed for this operation"
            )
        value = yaml.safe_load(raw_value)
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "model",
        "data",
        "renderer",
        "semantic",
        "optimization",
        "phases",
        "loss",
        "density",
        "surface",
        "mesh",
        "logging",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("configuration is missing sections: " + ", ".join(missing))
    schema_controls = {
        "semantic": (
            "region_decode_chunk_size",
            "region_top_k",
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
        ),
        "loss": (
            "lambda_mesh",
            "lambda_region_rgb",
            "region_area_temperature",
            "region_geometry_mix",
        ),
        "density": (
            "capacity_replacement_enabled",
            "replace_near_cap_ratio",
            "max_replacements_per_step",
            "mesh_coverage_weight",
            "region_budget_temperature",
        ),
        "surface": (
            "enabled",
            "field_neighbors",
            "query_chunk",
            "gaussian_chunk",
            "neighbor_query_chunk",
            "neighbor_backend",
            "occupancy_isovalue",
            "max_distance_bytes",
            "support_log_cutoff",
            "support_candidate_budget",
            "support_routing_query_chunk",
            "scipy_workers",
            "region_candidate_neighbors",
            "region_min_membership",
            "region_surface_min_weight",
            "topology_enabled",
            "topology_from",
            "topology_until",
            "topology_interval",
            "topology_max_net_growth",
            "topology_replacement_budget",
            "topology_enable_size_pruning",
            "topology_protect_min_confidence",
            "topology_protect_boundary",
            "topology_protect_thin_probability",
            "mesh_feedback_async",
            "mesh_refresh_interval",
            "mesh_feedback_resolution",
            "mesh_feedback_samples",
            "mesh_feedback_snapshot_device",
            "mesh_feedback_scipy_workers",
            "mesh_feedback_min_component_faces",
            "mesh_feedback_max_candidate_age",
            "mesh_feedback_max_topology_events",
            "mesh_feedback_max_churn_ratio",
            "mesh_feedback_retry_interval",
            "mesh_feedback_blend_iterations",
            "mesh_feedback_gate_probes",
            "mesh_feedback_gate_min_score",
            "mesh_feedback_gate_sdf_p90",
            "mesh_feedback_gate_normal",
            "mesh_feedback_gate_semantic",
            "mesh_feedback_min_opacity",
            "mesh_feedback_min_confidence",
            "mesh_feedback_min_expert_certainty",
            "mesh_feedback_match_k",
            "mesh_feedback_match_radius",
            "mesh_feedback_match_semantic",
            "mesh_feedback_robust_delta",
            "mesh_feedback_min_matches",
        ),
        "mesh": ("method", "padding", "support_sigma", "isovalue"),
        "logging": ("profile_interval",),
    }
    missing_controls = [
        f"{section}.{key}"
        for section, keys in schema_controls.items()
        for key in keys
        if key not in config[section]
    ]
    if missing_controls:
        raise ValueError(
            "region-conditioned configuration is missing: "
            + ", ".join(missing_controls)
        )
    semantic_dim = int(config["model"].get("semantic_dim", 16))
    if semantic_dim != 16:
        raise ValueError("semantic_dim must be 16 for the compiled joint rasterizer")
    experts = int(config["model"].get("geometry_experts", 5))
    if experts != 5:
        raise ValueError("geometry_experts must be 5 (planar/curved/thin/freeform/fuzzy)")
    sh_degree = int(config["model"].get("sh_degree", 3))
    if not 0 <= sh_degree <= 3:
        raise ValueError("model.sh_degree must be in [0,3] for the compiled rasterizer")
    iterations = int(config["optimization"]["iterations"])
    boundaries = (
        int(config["phases"]["semantic_from"]),
        int(config["phases"]["joint_from"]),
        int(config["phases"]["surface_from"]),
        iterations,
    )
    if tuple(sorted(boundaries)) != boundaries:
        raise ValueError(f"phase boundaries must be ordered and <= iterations: {boundaries}")
    if int(config["density"]["max_gaussians"]) < 1:
        raise ValueError("density.max_gaussians must be positive")
    if int(config["density"].get("split_children", 2)) < 2:
        raise ValueError("density.split_children must be at least 2")
    if float(config["density"].get("max_growth_fraction", 0.05)) <= 0:
        raise ValueError("density.max_growth_fraction must be positive")
    if int(config["density"].get("max_new_per_step", 100_000)) < 1:
        raise ValueError("density.max_new_per_step must be positive")
    for key, default in (
        ("rgb_weight", 1.0),
        ("semantic_weight", 0.5),
        ("boundary_weight", 0.75),
        ("geometry_weight", 0.75),
        ("mesh_coverage_weight", 0.5),
    ):
        value = config["density"].get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"density.{key} must be non-negative")
    capacity_replacement = config["density"].get(
        "capacity_replacement_enabled",
        True,
    )
    if not isinstance(capacity_replacement, bool):
        raise ValueError("density.capacity_replacement_enabled must be a boolean")
    replace_near_cap_ratio = float(
        config["density"].get("replace_near_cap_ratio", 0.98)
    )
    if not 0.0 < replace_near_cap_ratio <= 1.0:
        raise ValueError("density.replace_near_cap_ratio must be in (0,1]")
    max_replacements = config["density"].get(
        "max_replacements_per_step",
        config["density"].get("max_new_per_step", 100_000),
    )
    if (
        isinstance(max_replacements, bool)
        or not isinstance(max_replacements, int)
        or max_replacements < 0
    ):
        raise ValueError("density.max_replacements_per_step must be a non-negative integer")
    if int(config["density"].get("interval", 100)) < 1:
        raise ValueError("density.interval must be positive")
    if int(config["density"].get("opacity_reset_interval", 3_000)) < 1:
        raise ValueError("density.opacity_reset_interval must be positive")
    size_pruning = config["density"].get("enable_size_pruning", False)
    if not isinstance(size_pruning, bool):
        raise ValueError("density.enable_size_pruning must be a boolean")
    surface = config["surface"]
    routing_query_chunk = surface.get("support_routing_query_chunk", 8_192)
    if (
        isinstance(routing_query_chunk, bool)
        or not isinstance(routing_query_chunk, int)
        or routing_query_chunk < 1
    ):
        raise ValueError(
            "surface.support_routing_query_chunk must be a positive integer"
        )
    scipy_workers = surface.get("scipy_workers", 4)
    if (
        isinstance(scipy_workers, bool)
        or not isinstance(scipy_workers, int)
        or scipy_workers == 0
        or scipy_workers < -1
    ):
        raise ValueError("surface.scipy_workers must be -1 or a positive integer")
    mesh_feedback_scipy_workers = surface.get("mesh_feedback_scipy_workers", 1)
    if (
        isinstance(mesh_feedback_scipy_workers, bool)
        or not isinstance(mesh_feedback_scipy_workers, int)
        or mesh_feedback_scipy_workers == 0
        or mesh_feedback_scipy_workers < -1
    ):
        raise ValueError(
            "surface.mesh_feedback_scipy_workers must be -1 or a positive integer"
        )
    support_candidate_budget = surface.get("support_candidate_budget", 2_048)
    if (
        isinstance(support_candidate_budget, bool)
        or not isinstance(support_candidate_budget, int)
        or support_candidate_budget < 1
    ):
        raise ValueError("surface.support_candidate_budget must be a positive integer")
    for key, default in (("enabled", True), ("topology_enabled", True)):
        if not isinstance(surface.get(key, default), bool):
            raise ValueError(f"surface.{key} must be a boolean")
    surface_topology_enabled = bool(surface.get("enabled", True)) and bool(
        surface.get("topology_enabled", True)
    )
    topology_from = surface.get("topology_from", config["phases"]["surface_from"])
    topology_until = surface.get("topology_until", iterations)
    topology_interval = surface.get("topology_interval", 500)
    topology_net_growth = surface.get("topology_max_net_growth", 0)
    topology_replacements = surface.get("topology_replacement_budget", 20_000)
    integer_values = {
        "topology_from": topology_from,
        "topology_until": topology_until,
        "topology_interval": topology_interval,
        "topology_max_net_growth": topology_net_growth,
        "topology_replacement_budget": topology_replacements,
    }
    for key, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"surface.{key} must be an integer")
    if int(topology_interval) < 1:
        raise ValueError("surface.topology_interval must be positive")
    if int(topology_net_growth) < 0:
        raise ValueError("surface.topology_max_net_growth must be non-negative")
    if int(topology_replacements) < 0:
        raise ValueError("surface.topology_replacement_budget must be non-negative")
    if surface_topology_enabled:
        topology_bounds = (int(config["phases"]["surface_from"]), int(topology_from), int(topology_until), iterations)
        if tuple(sorted(topology_bounds)) != topology_bounds:
            raise ValueError(
                "surface topology window must lie inside the surface phase: "
                f"{topology_bounds}"
            )
    topology_size_pruning = surface.get("topology_enable_size_pruning", True)
    if not isinstance(topology_size_pruning, bool):
        raise ValueError("surface.topology_enable_size_pruning must be a boolean")
    for key, default in (
        ("topology_protect_min_confidence", 0.5),
        ("topology_protect_boundary", 0.25),
        ("topology_protect_thin_probability", 0.5),
    ):
        value = float(surface.get(key, default))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"surface.{key} must be in [0,1]")
    if not isinstance(surface.get("mesh_feedback_async", True), bool):
        raise ValueError("surface.mesh_feedback_async must be a boolean")
    snapshot_device = surface.get("mesh_feedback_snapshot_device", "auto")
    if snapshot_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "surface.mesh_feedback_snapshot_device must be auto, cpu, or cuda"
        )
    minimum_component_faces = surface.get("mesh_feedback_min_component_faces", 64)
    if (
        isinstance(minimum_component_faces, bool)
        or not isinstance(minimum_component_faces, int)
        or minimum_component_faces < 0
    ):
        raise ValueError(
            "surface.mesh_feedback_min_component_faces must be a non-negative integer"
        )
    # Mesh-v4 refresh candidates are derived state crossing an asynchronous
    # boundary. Keep their freshness, quality and matching policy explicit in
    # the authoritative experiment configuration, even for runtimes that have
    # not yet enabled the v4 consumer.
    feedback_integer_controls = (
        ("mesh_feedback_max_candidate_age", 1_000, 1, "positive"),
        ("mesh_feedback_max_topology_events", 1, 0, "non-negative"),
        ("mesh_feedback_retry_interval", 100, 1, "positive"),
        ("mesh_feedback_blend_iterations", 125, 0, "non-negative"),
        ("mesh_feedback_gate_probes", 2_048, 1, "positive"),
        ("mesh_feedback_match_k", 16, 1, "positive"),
        ("mesh_feedback_min_matches", 256, 1, "positive"),
    )
    feedback_integers: dict[str, int] = {}
    for key, default, minimum, description in feedback_integer_controls:
        value = surface.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"surface.{key} must be a {description} integer")
        feedback_integers[key] = value

    feedback_unit_controls = (
        ("mesh_feedback_max_churn_ratio", 0.015),
        ("mesh_feedback_gate_min_score", 0.70),
        ("mesh_feedback_gate_normal", 0.60),
        ("mesh_feedback_gate_semantic", 0.50),
        ("mesh_feedback_min_opacity", 0.05),
        ("mesh_feedback_min_confidence", 0.35),
        ("mesh_feedback_min_expert_certainty", 0.55),
    )
    for key, default in feedback_unit_controls:
        value = surface.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"surface.{key} must be a finite number in [0,1]")

    match_semantic = surface.get("mesh_feedback_match_semantic", 0.50)
    if (
        isinstance(match_semantic, bool)
        or not isinstance(match_semantic, (int, float))
        or not math.isfinite(float(match_semantic))
        or not -1.0 <= float(match_semantic) <= 1.0
    ):
        raise ValueError(
            "surface.mesh_feedback_match_semantic must be a finite cosine threshold in [-1,1]"
        )

    feedback_positive_controls = (
        ("mesh_feedback_gate_sdf_p90", 2.5),
        ("mesh_feedback_match_radius", 3.0),
        ("mesh_feedback_robust_delta", 1.5),
    )
    for key, default in feedback_positive_controls:
        value = surface.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"surface.{key} must be a finite positive number")

    probes = feedback_integers["mesh_feedback_gate_probes"]
    if feedback_integers["mesh_feedback_min_matches"] > probes:
        raise ValueError(
            "surface.mesh_feedback_min_matches cannot exceed mesh_feedback_gate_probes"
        )
    if feedback_integers["mesh_feedback_match_k"] > probes:
        raise ValueError(
            "surface.mesh_feedback_match_k cannot exceed mesh_feedback_gate_probes"
        )
    profile_interval = config.get("logging", {}).get("profile_interval", 0)
    if (
        isinstance(profile_interval, bool)
        or not isinstance(profile_interval, int)
        or profile_interval < 0
    ):
        raise ValueError("logging.profile_interval must be a non-negative integer")
    semantic = config["semantic"]
    region_decode_chunk_size = semantic.get("region_decode_chunk_size")
    if (
        isinstance(region_decode_chunk_size, bool)
        or not isinstance(region_decode_chunk_size, int)
        or region_decode_chunk_size < 1
    ):
        raise ValueError("semantic.region_decode_chunk_size must be a positive integer")
    region_top_k = semantic.get("region_top_k")
    if (
        isinstance(region_top_k, bool)
        or not isinstance(region_top_k, int)
        or region_top_k < 1
    ):
        raise ValueError("semantic.region_top_k must be a positive integer")
    field_neighbors = surface.get("field_neighbors")
    region_candidates = surface.get("region_candidate_neighbors")
    if (
        isinstance(field_neighbors, bool)
        or not isinstance(field_neighbors, int)
        or field_neighbors < 1
    ):
        raise ValueError("surface.field_neighbors must be a positive integer")
    if (
        isinstance(region_candidates, bool)
        or not isinstance(region_candidates, int)
        or region_candidates < field_neighbors
    ):
        raise ValueError(
            "surface.region_candidate_neighbors must be an integer no smaller "
            "than surface.field_neighbors"
        )
    for key in ("region_min_membership", "region_surface_min_weight"):
        value = surface.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
        ):
            raise ValueError(f"surface.{key} must be a finite number in [0,1)")
    region_rgb = config["loss"].get("lambda_region_rgb")
    if (
        isinstance(region_rgb, bool)
        or not isinstance(region_rgb, (int, float))
        or not math.isfinite(float(region_rgb))
        or float(region_rgb) < 0.0
    ):
        raise ValueError("loss.lambda_region_rgb must be a finite non-negative number")
    for key in ("region_area_temperature", "region_geometry_mix"):
        value = config["loss"].get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"loss.{key} must be a finite number in [0,1]")
    confidence_floor = float(semantic.get("confidence_floor", 0.05))
    if not 0.0 <= confidence_floor < 1.0:
        raise ValueError("semantic.confidence_floor must be in [0,1)")
    for key, default in (("evidence_interval", 100), ("evidence_samples", 2_048)):
        value = semantic.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"semantic.{key} must be a positive integer")
    for key, default in (
        ("evidence_entropy_weight", 0.05),
        ("evidence_balance_weight", 0.10),
    ):
        if float(semantic.get(key, default)) < 0:
            raise ValueError(f"semantic.{key} must be non-negative")
    if float(semantic.get("geometry_evidence_temperature", 0.20)) <= 0:
        raise ValueError("semantic.geometry_evidence_temperature must be positive")
    if not isinstance(semantic.get("propagation_enabled", True), bool):
        raise ValueError("semantic.propagation_enabled must be a boolean")
    for key, default in (("propagation_samples", 65_536), ("propagation_neighbors", 12)):
        value = semantic.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"semantic.{key} must be a positive integer")
    seed_confidence = float(semantic.get("propagation_min_seed_confidence", 0.35))
    max_confidence = float(semantic.get("propagation_max_confidence", 0.85))
    if not 0.0 <= seed_confidence <= max_confidence <= 1.0:
        raise ValueError("semantic propagation confidence must satisfy 0 <= seed <= max <= 1")
    propagation_momentum = float(semantic.get("propagation_momentum", 0.5))
    if not 0.0 <= propagation_momentum < 1.0:
        raise ValueError("semantic.propagation_momentum must be in [0,1)")
    propagation_decay = float(semantic.get("propagation_decay", 0.995))
    if not 0.0 <= propagation_decay <= 1.0:
        raise ValueError("semantic.propagation_decay must be in [0,1]")
    propagation_floor = float(semantic.get("propagation_semantic_floor", 0.25))
    if not -1.0 < propagation_floor < 1.0:
        raise ValueError("semantic.propagation_semantic_floor must be in (-1,1)")
    if float(semantic.get("propagation_support_sigma", 3.0)) <= 0:
        raise ValueError("semantic.propagation_support_sigma must be positive")
    if float(semantic.get("propagation_boundary_barrier", 2.0)) < 0:
        raise ValueError("semantic.propagation_boundary_barrier must be non-negative")
    normal_alpha_threshold = float(config["loss"].get("normal_alpha_threshold", 0.5))
    if not 0.0 <= normal_alpha_threshold <= 1.0:
        raise ValueError("loss.normal_alpha_threshold must be in [0,1]")


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return Namespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def save_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
