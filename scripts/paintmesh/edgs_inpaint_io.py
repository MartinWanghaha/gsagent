"""CPU-only contracts for the peer EDGS initialization / PGSR training path."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np

from local_geometry_io import (REPO, atomic_write, identity, read_json, read_receipt,
                               record, seal, sha256, verify_record, write_json)
from virtual_render_io import read_camera_manifest

INIT_KIND = "paintmesh-edgs-initialization"
JOINT_KIND = "paintmesh-edgs-joint"
DEFAULT_CONFIG = Path(__file__).parent / "configs/edgs_inpaint.yaml"


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
        return value.lower() in ("true", "1", "yes", "on")
    raise ValueError(f"expected a boolean, got {value!r}")


def load_config(path=DEFAULT_CONFIG, **overrides):
    import yaml
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text())
    supplied = yaml.safe_load(Path(path).read_text())
    if not isinstance(supplied, dict):
        raise ValueError("EDGS inpaint config must be a mapping")
    cfg = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if key not in cfg:
            raise ValueError(f"unknown EDGS inpaint setting: {key}")
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"unknown EDGS inpaint section: {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    for key, value in overrides.items():
        if value is None:
            continue
        if key.startswith("match_"):
            field = key.removeprefix("match_")
            if field not in ("confidence_min", "cycle_pixels", "reprojection_pixels"):
                raise ValueError(f"unknown EDGS override: {key}")
            cfg["init"][field] = value
        elif key.startswith("init_") or key.startswith("train_"):
            section = "init" if key.startswith("init_") else "supervision"
            cfg[section]["use_" + key.rsplit("_", 1)[1]] = boolean(value)
        elif key in cfg:
            cfg[key] = value
        else:
            raise ValueError(f"unknown EDGS override: {key}")
    for key in ("use_depth", "use_normal"):
        if not isinstance(cfg["init"][key], bool):
            raise ValueError(f"init.{key} must be boolean")
        if cfg["supervision"][key] is None:
            cfg["supervision"][key] = cfg["init"][key]
        if not isinstance(cfg["supervision"][key], bool):
            raise ValueError(f"supervision.{key} must be boolean or null")
    if cfg["init"]["use_normal"] and not cfg["init"]["use_depth"]:
        raise ValueError("normal initialization requires depth initialization")
    for key in ("iterations", "geometry_from_iter", "geometry_ramp_iters", "seed",
                "checkpoint_interval", "diagnostic_interval"):
        value = cfg[key]
        minimum = 1 if key in ("iterations", "checkpoint_interval", "diagnostic_interval") else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {key}")
    for section in ("optimizer", "loss", "validity"):
        for key, value in cfg[section].items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {section}.{key}")
    for key in ("neighbors", "samples_per_pair", "max_points"):
        if type(cfg["init"][key]) is not int or cfg["init"][key] <= 0:
            raise ValueError(f"invalid init.{key}")
    for key in ("reprojection_pixels", "cycle_pixels", "min_angle_deg", "depth_weight", "normal_thickness", "opacity"):
        value = cfg["init"][key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid init.{key}")
    confidence = cfg["init"]["confidence_min"]
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("init.confidence_min must be a finite number in [0,1]")
    if not (cfg["init"]["opacity"] < 1 and cfg["init"]["normal_thickness"] < 1):
        raise ValueError("init opacity and normal_thickness must be in (0,1)")
    matcher = cfg["matcher"]
    if matcher["model"] not in ("indoor", "outdoor") or type(matcher["resolution"]) is not int or matcher["resolution"] <= 0 or matcher["resolution"] % 14:
        raise ValueError("invalid RoMa model/resolution (must be a positive multiple of 14)")
    for key in ("weights", "dinov2_weights"):
        if matcher[key] is not None:
            if not isinstance(matcher[key], str) or not matcher[key].strip():
                raise ValueError(f"matcher.{key} must be a filename or null")
            value = Path(matcher[key]).expanduser()
            matcher[key] = str((Path(path).resolve().parent / value).resolve()) if not value.is_absolute() else str(value.resolve())
    debug = cfg["debug"]
    if type(debug["enabled"]) is not bool or type(debug["interval"]) is not int or debug["interval"] <= 0 or type(debug["view_index"]) is not int or debug["view_index"] < 0:
        raise ValueError("invalid debug configuration")
    validity = cfg["validity"]
    if not (0 < validity["render_alpha_min"] <= validity["coverage_alpha_floor"] <= 1 and validity["depth_jump_relative"] > 0):
        raise ValueError("invalid alpha/depth validity configuration")
    if not all(v > 0 for v in cfg["optimizer"].values()) or cfg["loss"]["rgb_hole"] <= 0:
        raise ValueError("joint training requires positive learning rates and RGB loss")
    for modality, term in (("depth", "depth"), ("normal", "lama_normal")):
        if not cfg["supervision"]["use_" + modality]:
            cfg["loss"][term] = 0.0
    return cfg


def config_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("init-depth", "init-normal", "train-depth", "train-normal"):
        parser.add_argument("--" + name, type=boolean)
    for name in ("iterations", "geometry-from-iter", "geometry-ramp-iters"):
        parser.add_argument("--" + name, type=int)
    for name in ("match-confidence-min", "match-cycle-pixels", "match-reprojection-pixels"):
        parser.add_argument("--" + name, type=float)


def resolve_config(args):
    return load_config(args.config, **{key: getattr(args, key, None) for key in (
        "init_depth", "init_normal", "train_depth", "train_normal", "iterations",
        "geometry_from_iter", "geometry_ramp_iters", "match_confidence_min",
        "match_cycle_pixels", "match_reprojection_pixels")})


def verify_targets(lama_path, camera_path, *, use_depth=False, use_normal=False):
    """Verify identities and ONLY consumed files; public completion is checked upstream."""
    lama = read_json(lama_path)
    if lama.get("kind") != "paintmesh-lama-completion" or not lama.get("complete") or lama.get("status") != "complete":
        raise ValueError("incomplete LaMa completion")
    inputs = read_json(verify_record(lama["input_manifest"]))
    expected_input = identity(dict(kind=inputs["kind"], schema_version=inputs["schema_version"],
        parameters=inputs["parameters"], inputs={s: {k: v["sha256"] for k, v in f["inputs"].items()}
                                               for s, f in inputs["frames"].items()}))
    if inputs["artifact_id"] != lama["input_artifact_id"] or inputs["artifact_id"] != expected_input:
        raise ValueError("LaMa input identity mismatch")
    expected = identity(dict(kind=lama["kind"], schema_version=lama["schema_version"],
        input_artifact_id=lama["input_artifact_id"], model={k: v["sha256"] for k, v in lama["model"].items()},
        parameters=lama["parameters"], outputs={s: {k: v["sha256"] for k, v in f["outputs"].items()}
                                              for s, f in lama["frames"].items()}))
    if expected != lama["artifact_id"]:
        raise ValueError("LaMa completion identity mismatch")
    cameras = read_camera_manifest(camera_path)
    from virtual_render_io import camera_contract
    if identity(camera_contract().load_virtual_camera_manifest(camera_path)) != cameras["artifact_id"]:
        raise ValueError("camera content does not match artifact identity")
    stems = [c["image_name"] for c in cameras["cameras"]]
    if set(stems) != set(lama["frames"]) or set(stems) != set(inputs["frames"]):
        raise ValueError("completion/camera frames differ")
    metadata = inputs["parameters"].get("normal")
    if metadata:
        if verify_record(metadata["camera_manifest"]) != Path(camera_path).resolve() or metadata["camera_artifact_id"] != cameras["artifact_id"]:
            raise ValueError("completion camera mismatch")
    if use_depth or use_normal:
        if not metadata:
            raise ValueError("geometry targets require declared PGSR render/camera provenance")
        render = read_json(verify_record(metadata["render_manifest"]))
        if render.get("artifact_id") != identity({k: v for k, v in render.items() if k != "artifact_id"}) or render["artifact_id"] != metadata["render_artifact_id"]:
            raise ValueError("render identity mismatch")
        if use_depth and (render.get("depth_kind"), render.get("depth_unit")) != ("plane_z", "scene"):
            raise ValueError("completed depth must be PGSR plane_z in scene units")
        if use_normal and (metadata["space"], metadata["orientation"]) != ("camera", "toward_camera"):
            raise ValueError("unsupported normal coordinates")
    required = ["color"] + (["depth"] if use_depth else []) + (["normal", "normal_valid"] if use_normal else [])
    for stem in stems:
        for key in required:
            if key not in lama["frames"][stem]["outputs"]:
                raise ValueError(f"enabled modality missing: {stem}/{key}")
            verify_record(lama["frames"][stem]["outputs"][key])
        verify_record(inputs["frames"][stem]["outputs"]["color_mask"])
    return lama, inputs, cameras


def implementation():
    paths = [Path(__file__), DEFAULT_CONFIG,
             REPO / "scripts/paintmesh/local_geometry_io.py",
             REPO / "scripts/paintmesh/virtual_render_io.py"]
    paths += sorted((REPO / "submodules/EDGS/source").glob("paintmesh_joint*.py"))
    paths += [REPO / "submodules/EDGS" / name for name in (
        "source/paintmesh_edgs_init.py", "source/paintmesh_local_data.py",
        "source/paintmesh_local_losses.py", "source/pgsr_geometry.py",
        "source/renderers/pgsr.py", "source/correspondence/geometry.py", "source/correspondence/contracts.py",
        "source/vendor.py", "submodules/RoMa/romatch/models/matcher.py",
        "submodules/RoMa/romatch/models/model_zoo/__init__.py",
        "tools/initialize_paintmesh_edgs.py", "tools/train_paintmesh_pgsr.py")]
    paths.append(REPO / "submodules/Inpaint360GS/utils/virtual_camera_manifest.py")
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def claim(root, request, validate_only=False):
    root = Path(root)
    path = root / "request.json"
    if path.exists():
        if read_json(path) != request:
            raise ValueError("EDGS request changed; choose a new INPAINT_RUN_NAME")
    elif validate_only:
        raise ValueError("EDGS stage incomplete; run its producing stage first")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("unowned EDGS output directory; choose a new run")
    else:
        write_json(path, request)


def check_preserved(source, result, editable):
    from plyfile import PlyData
    a, b = [PlyData.read(str(p), mmap=True)["vertex"].data for p in (source, result)]
    if a.dtype != b.dtype or len(a) != len(b) or editable.shape != (len(a),) or editable.dtype != np.bool_:
        raise ValueError("joint PLY schema/row order/count changed")
    for key in a.dtype.names:
        mask = slice(None) if key.startswith("obj_") or key in ("nx", "ny", "nz") else ~editable
        if not np.array_equal(a[key][mask], b[key][mask]):
            raise ValueError(f"joint training changed frozen field: {key}")
        if not np.isfinite(b[key]).all():
            raise ValueError(f"nonfinite joint PLY field: {key}")


def _receipt(value, kind):
    """Also validate in-memory receipts BEFORE committing a complete marker."""
    if not isinstance(value, dict):
        return read_receipt(value, kind)
    if value.get("kind") != kind or value.get("complete") is not True or value.get("status") != "complete" or value.get("artifact_id") != identity({k: v for k, v in value.items() if k != "artifact_id"}):
        raise ValueError(f"invalid {kind} payload")
    for section in ("inputs", "outputs"):
        for rec in value[section].values():
            verify_record(rec)
    return value


def validate_init(path, request=None):
    from plyfile import PlyData
    receipt = _receipt(path, INIT_KIND)
    if request is not None and request != receipt["request"]:
        raise ValueError("EDGS initialization request differs")
    if receipt["inputs"] != receipt["request"]["inputs"]:
        raise ValueError("EDGS initialization input identity mismatch")
    verify_targets(receipt["inputs"]["lama"]["path"], receipt["inputs"]["camera"]["path"],
                   use_depth=receipt["request"]["config"]["init"]["use_depth"],
                   use_normal=receipt["request"]["config"]["init"]["use_normal"])
    for value in receipt["request"].get("matcher_weights", {}).values():
        verify_record(value)
    vertex = PlyData.read(receipt["outputs"]["ply"]["path"], mmap=True)["vertex"].data
    background = PlyData.read(receipt["inputs"]["background"]["path"], mmap=True)["vertex"].data
    editable = np.load(receipt["outputs"]["editable_mask"]["path"], allow_pickle=False)
    if len(vertex) <= len(background) or editable.dtype != np.bool_ or editable.shape != (len(vertex),):
        raise ValueError("invalid initialization row ownership")
    if editable[:len(background)].any() or not editable[len(background):].all() or vertex.dtype != background.dtype:
        raise ValueError("initialization background ownership mismatch")
    for key in background.dtype.names:
        if not np.array_equal(vertex[key][:len(background)], background[key]) or not np.isfinite(vertex[key]).all():
            raise ValueError(f"initialization modified background/nonfinite {key}")
    if receipt["point_count"] != len(vertex):
        raise ValueError("initialization point count mismatch")
    return receipt


def validate_joint(path, *, selected_ply=None, lama_path=None, camera_path=None, request=None):
    receipt = _receipt(path, JOINT_KIND)
    if request is not None and request != receipt["request"]:
        raise ValueError("EDGS joint request differs")
    initial = validate_init(receipt["inputs"]["initialization"]["path"])
    if receipt["initialization_id"] != initial["artifact_id"] or receipt["inputs"] != receipt["request"]["inputs"]:
        raise ValueError("EDGS joint initialization identity mismatch")
    for key, expected in (("lama", lama_path), ("camera", camera_path)):
        if receipt["inputs"][key] != initial["inputs"][key] or (expected is not None and record(expected) != receipt["inputs"][key]):
            raise ValueError(f"EDGS joint {key} provenance mismatch")
    cfg = receipt["request"]["config"]
    if any(cfg[k] != initial["request"]["config"][k] for k in ("init", "matcher", "seed")):
        raise ValueError("joint/initialization config mismatch")
    if receipt["point_count"] != initial["point_count"]:
        raise ValueError("joint point count differs from initialization")
    verify_targets(receipt["inputs"]["lama"]["path"], receipt["inputs"]["camera"]["path"], **cfg["supervision"])
    if receipt["parameters"]["iterations"] != cfg["iterations"]:
        raise ValueError("EDGS joint iteration mismatch")
    result = receipt["outputs"]["ply"]["path"]
    if selected_ply is not None and record(selected_ply)["sha256"] != record(result)["sha256"]:
        raise ValueError("selected PLY differs from EDGS joint result")
    editable = np.load(initial["outputs"]["editable_mask"]["path"], allow_pickle=False)
    check_preserved(initial["outputs"]["ply"]["path"], result, editable)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    config_arguments(sub.add_parser("config"))
    policy = sub.add_parser("select")
    policy.add_argument("--root", type=Path, required=True)
    policy.add_argument("--pipeline", choices=("inpaint360gs", "edgs-pgsr"), required=True)
    check = sub.add_parser("check-published")
    check.add_argument("--model-manifest", type=Path, required=True)
    check.add_argument("--joint-manifest", type=Path, required=True)
    check.add_argument("--selected-ply", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "config":
        print(resolve_config(args)["iterations"])
    elif args.command == "select":
        path = args.root / "manifests/pipeline.json"
        value = {"pipeline": args.pipeline}
        if path.exists() and read_json(path) != value:
            raise ValueError("INPAINT_PIPELINE changed; choose a new INPAINT_RUN_NAME")
        if not path.exists() and args.pipeline == "edgs-pgsr" and any((args.root / p).exists() for p in (
                "fused", "local_geometry", "manifests/fusion_manifest.json", "inpainted_3dgs")):
            raise ValueError("existing old-path artifacts; choose a new EDGS inpaint run")
        if not path.exists() and args.pipeline == "inpaint360gs" and any((args.root / p).exists() for p in (
                "edgs_init", "edgs_joint", "manifests/edgs_joint_manifest.json")):
            raise ValueError("existing EDGS path artifacts; choose a new Inpaint360GS run")
        write_json(path, value)
    else:
        model = read_json(args.model_manifest)
        joint = validate_joint(args.joint_manifest, selected_ply=args.selected_ply)
        if (model["parameters"].get("pipeline") != "edgs-pgsr" or
                model["upstream_artifact_ids"].get("edgs_joint") != joint["artifact_id"] or
                model["inputs"]["inpainted_gaussian_ply"]["sha256"] != record(args.selected_ply)["sha256"]):
            raise ValueError("published model differs from selected EDGS pipeline")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"error: {exc}")
