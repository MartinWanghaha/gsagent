"""CPU planning contracts. No project utils, Open3D, Torch or CUDA imports."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from virtual_render_io import camera_contract, identity, read_json, sha256

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "virtual_view_planning.yaml"


def load_config(path=None):
    import yaml
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(defaults)
    if path is not None:
        custom = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        def merge(base, override, label):
            if not isinstance(override, dict):
                raise ValueError(f"{label} must be a mapping")
            for key, value in override.items():
                if key not in base:
                    raise ValueError(f"unknown planning option: {label}.{key}")
                if isinstance(base[key], dict):
                    merge(base[key], value, label + "." + key)
                else:
                    base[key] = value
        merge(config, custom, "config")
    if config["algorithm"] != "collision_aware_v1" or config["collision"]["unknown_space"] != "reject":
        raise ValueError("only collision_aware_v1 with unknown_space=reject is supported")
    def check_values(value, base, label):
        for key, item in value.items():
            template = base[key]
            name = label + "." + key
            if isinstance(template, dict):
                check_values(item, template, name)
            elif isinstance(template, (int, float)):
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                    raise ValueError(f"{name} must be finite")
                if type(template) is int and type(item) is not int:
                    raise ValueError(f"{name} must be an integer")
                if item < 0 or (key != "seed" and item == 0):
                    raise ValueError(f"{name} must be positive (seed may be zero)")
            elif isinstance(template, list):
                if not isinstance(item, list) or not item or any(
                    isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in item):
                    raise ValueError(f"{name} must be a nonempty positive numeric list")
    check_values(config, defaults, "config")
    q = config["view_quality"]
    if (len(q["target_area_fraction"]) != 2 or
            not 0 < q["target_area_fraction"][0] < q["target_area_fraction"][1] < 1):
        raise ValueError("target_area_fraction must be [min,max] inside (0,1)")
    for section, keys in (("view_quality", ("min_target_visible_ratio", "min_context_valid_ratio",
            "min_known_surface_coverage", "preview_alpha_min")), ("free_space", ("alpha_min",))):
        if any(not 0 < config[section][key] <= 1 for key in keys):
            raise ValueError(f"{section} ratios must be in (0,1]")
    if not 0 < q["image_border_fraction"] < .5 or not 0 < q["max_elevation_deg"] < 90:
        raise ValueError("invalid framing/elevation range")
    if not 0 < config["selection"]["max_neighbor_rotation_deg"] < 180:
        raise ValueError("invalid neighbor rotation range")
    # Explicit budgets prevent accidental unbounded CPU/GPU allocations.
    for section, key, maximum in ((None, "candidate_directions", 4096), (None, "surface_probes", 32768),
            (None, "max_refinement_rounds", 10), ("free_space", "max_size", 1024),
            ("free_space", "max_training_views", 512), ("selection", "beam_width", 256),
            ("collision", "edge_max_subdivisions", 24)):
        if (config[section] if section else config)[key] > maximum:
            raise ValueError(f"resource limit: {section}.{key} > {maximum}")
    if len(config["radius_multipliers"]) > 8:
        raise ValueError("at most 8 radius multipliers")
    if config["free_space"]["max_size"] < 8:
        raise ValueError("preview max_size must be >= 8")
    return config


def resize_planning_cameras(records, max_size):
    result = []
    for record in records:
        camera = dict(record)
        factor = min(1., max_size/max(camera["image_width"], camera["image_height"]))
        camera["image_width"] = max(8, round(camera["image_width"]*factor))
        camera["image_height"] = max(8, round(camera["image_height"]*factor))
        result.append(camera)
    return result


def file_record(path, *, canonical_json=False):
    path = Path(path).resolve(strict=True)
    if canonical_json:
        return {"path": str(path), "sha256": identity(read_json(path)), "encoding": "canonical_json"}
    return {"path": str(path), "sha256": sha256(path)}


def validate_records(records):
    for label, record in records.items():
        actual = identity(read_json(record["path"])) if record.get("encoding") == "canonical_json" else sha256(record["path"])
        if actual != record["sha256"]:
            raise ValueError(f"planning input changed ({label}); use a new RUN_NAME")


def validate_published_plan(camera_path, *, backend=None, config=None):
    """Read-only provenance preflight also used before reusing tracking masks."""
    camera_path = Path(camera_path)
    camera_contract().load_virtual_camera_manifest(camera_path)
    cameras = read_json(camera_path)
    if cameras["schema_version"] != 3:
        return
    trajectory = cameras["trajectory"]
    root = camera_path.parent / "view_planning"
    inputs = read_json(root / "input_manifest.json")
    resolved = read_json(root / "resolved_config.json")
    evidence = read_json(root / "safety_report.json")
    if (identity(inputs) != trajectory["input_sha256"] or identity(resolved) != trajectory["config_sha256"]
            or identity(evidence) != trajectory["evidence_sha256"]):
        raise ValueError("planning evidence identity mismatch")
    validate_records(inputs["files"])
    validate_records(evidence["files"])
    if backend is not None and inputs["backend"] != backend:
        raise ValueError("planning/render backend changed; use a new RUN_NAME")
    if config is not None and identity(config) != trajectory["config_sha256"]:
        raise ValueError("planning configuration changed; use a new RUN_NAME")
    if (evidence.get("complete") is not True or evidence["unknown_space"] != "reject"
            or len(evidence["positions"]) != cameras["frame_count"]
            or len(evidence["edges"]) != cameras["frame_count"] - 1
            or evidence["selected_candidate_ids"] != trajectory["selected_candidate_ids"]):
        raise ValueError("incomplete collision-aware safety certificate")
    # Bind reports to the exact matrices, not just the selected candidate IDs.
    if evidence["cameras_sha256"] != identity(cameras["cameras"]):
        raise ValueError("safety certificate describes different cameras")
    import numpy as np
    from view_planning_geometry import camera_center
    radius = trajectory["safe_radius_world"]
    margin = evidence.get("numeric_margin_world")
    if (isinstance(margin, bool) or not isinstance(margin, (float, int)) or not math.isfinite(margin)
            or margin <= 0 or evidence.get("safe_radius_world") != radius):
        raise ValueError("invalid safety radius/numeric margin")
    positions = np.asarray(evidence["positions"], dtype=float)
    centers = np.array([camera_center(c) for c in cameras["cameras"]])
    distances = np.asarray(evidence.get("mesh_distances"), dtype=float)
    if (positions.shape != centers.shape or not np.isfinite(positions).all()
            or not np.allclose(positions, centers, atol=margin, rtol=0)
            or distances.shape != (len(centers),) or not np.isfinite(distances).all()
            or (distances < radius+margin).any()):
        raise ValueError("invalid camera safety evidence")
    for i, edge in enumerate(evidence["edges"]):
        if (edge.get("observed_free") is not True or edge.get("swept_sphere") is not True
                or type(edge.get("certified_leaves")) is not int or edge["certified_leaves"] < 1
                or not math.isfinite(edge.get("minimum_mesh_distance_lower_bound", float("nan")))
                or edge["minimum_mesh_distance_lower_bound"] < radius
                or not math.isclose(edge.get("length", -1.), np.linalg.norm(centers[i+1]-centers[i]), abs_tol=margin)):
            raise ValueError("invalid continuous-path safety evidence")
