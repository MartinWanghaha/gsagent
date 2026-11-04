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
SELECTION_KIND = "paintmesh-edgs-local-selection"
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
        if key == "surface_init" and isinstance(value, dict) and value.get("mode", "none") == "none":
            # Historical disabled-plane configs remain readable; never execute a plane fallback.
            value = {k: v for k, v in value.items() if k not in (
                "boundary_pixels", "fit_distance_factor", "min_inlier_fraction")}
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
        if key == "surface_init_mode":
            cfg["surface_init"]["mode"] = value
        elif key.startswith("match_"):
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
    for key, minimum in (("lpips_padding", 0), ("lpips_max_side", 32)):
        value = cfg["appearance"][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid appearance.{key}")
    appearance_mode = cfg["appearance"]["mode"]
    if appearance_mode not in ("masked", "inpaint360gs"):
        raise ValueError("appearance.mode must be masked or inpaint360gs")
    if cfg["appearance"]["rgb_weight_mode"] not in ("continuity", "uniform"):
        raise ValueError("appearance.rgb_weight_mode must be continuity or uniform")
    for section in ("optimizer", "loss", "validity"):
        for key, value in cfg[section].items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {section}.{key}")
    for key in ("neighbors", "samples_per_pair"):
        if type(cfg["init"][key]) is not int or cfg["init"][key] <= 0:
            raise ValueError(f"invalid init.{key}")
    limit = cfg["init"]["max_points"]
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("init.max_points must be a positive integer or null (unlimited)")
    for key in ("reprojection_pixels", "cycle_pixels", "min_angle_deg", "depth_weight", "normal_thickness", "opacity"):
        value = cfg["init"][key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid init.{key}")
    confidence = cfg["init"]["confidence_min"]
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("init.confidence_min must be a finite number in [0,1]")
    if not (cfg["init"]["opacity"] < 1 and cfg["init"]["normal_thickness"] < 1):
        raise ValueError("init opacity and normal_thickness must be in (0,1)")
    surface = cfg["surface_init"]
    if surface["mode"] not in ("none", "depth_anchor"):
        raise ValueError("surface_init.mode must be none or depth_anchor; plane was removed")
    for key in ("min_anchor_observations", "grid_pixels", "anchor_radius_pixels", "candidate_pixel_step", "min_cross_depth_views"):
        if type(surface[key]) is not int or surface[key] < 1:
            raise ValueError(f"surface_init.{key} must be a positive integer")
    for key in ("track_join_pixels", "anchor_min_angle_deg", "anchor_max_sigma_spacing", "depth_edge_relative",
                "smooth_weight", "ridge_weight", "cross_depth_spacing", "sampling_distance_factor", "min_cross_depth_fraction"):
        value = surface[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid surface_init.{key}")
    if surface["min_cross_depth_fraction"] > 1 or surface["depth_edge_relative"] >= 1:
        raise ValueError("invalid surface_init fraction/relative depth edge")
    if surface["min_anchor_observations"] < 3 or surface["min_cross_depth_views"] < 2 or surface["anchor_min_angle_deg"] >= 90:
        raise ValueError("surface_init requires >=3 anchor observations, >=2 other views, and an angle below 90 degrees")
    if surface["mode"] == "depth_anchor" and not cfg["init"]["use_depth"]:
        raise ValueError("surface_init depth_anchor requires init.use_depth=true")
    if surface["mode"] != "none" and cfg["init"]["use_normal"]:
        raise ValueError("center-only surface initialization requires init.use_normal=false")
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
    if not all(v > 0 for v in cfg["optimizer"].values()) or (
            appearance_mode == "masked" and cfg["loss"]["rgb_hole"] <= 0):
        raise ValueError("joint training requires positive learning rates and RGB loss")
    if appearance_mode == "inpaint360gs":
        if cfg["loss"]["rgb_hole"] != 0:
            raise ValueError("inpaint360gs appearance requires loss.rgb_hole=0")
        if any(cfg["loss"][key] <= 0 for key in ("rgb_known_preserve", "ssim", "lpips")):
            raise ValueError("inpaint360gs appearance requires positive outside L1, SSIM and LPIPS weights")
    if type(cfg["freeze_background"]) is not bool:
        raise ValueError("freeze_background must be boolean")
    local = cfg["local_background"]
    if (type(local["enabled"]) is not bool or type(local["mask_dilation"]) is not int
            or local["mask_dilation"] < 0 or type(local["min_views"]) is not int or local["min_views"] < 1):
        raise ValueError("invalid local_background configuration")
    for key in ("support_distance_factor", "gaussian_radius_cap_factor"):
        value = local[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid local_background.{key}")
    if local["enabled"] and not cfg["freeze_background"]:
        raise ValueError("local_background requires freeze_background=true to protect far rows")
    writeback = cfg["background_writeback"]
    if type(writeback["enabled"]) is not bool:
        raise ValueError("background_writeback.enabled must be boolean")
    for key, minimum in (("mask_area_factor", 1.), ("seed_std_factor", 0.)):
        value = writeback[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= minimum):
            raise ValueError(f"invalid background_writeback.{key}")
    if writeback["enabled"]:
        if cfg["freeze_background"] or local["enabled"]:
            raise ValueError("background_writeback requires all old/new rows trainable, without local_background")
        if cfg["adaptive_density"]["enabled"]:
            raise ValueError("background_writeback currently supports prune-only topology")
    pruning = cfg["pruning"]
    if type(pruning["enabled"]) is not bool or type(pruning["until_iter"]) is not int or pruning["until_iter"] < 0:
        raise ValueError("invalid pruning enabled/until_iter")
    value = pruning["min_opacity"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("pruning.min_opacity must be a finite number in [0,1)")
    density = cfg["adaptive_density"]
    if type(density["enabled"]) is not bool:
        raise ValueError("invalid adaptive_density.enabled")
    for key, minimum in (("from_iter", 0), ("until_iter", 1), ("interval", 1), ("max_new_points", 1)):
        if type(density[key]) is not int or density[key] < minimum:
            raise ValueError(f"invalid adaptive_density.{key}")
    if density["until_iter"] <= density["from_iter"]:
        raise ValueError("adaptive_density.until_iter must exceed from_iter")
    for key in ("grad_threshold", "percent_dense"):
        value = density[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid adaptive_density.{key}")
    for modality, term in (("depth", "depth"), ("normal", "lama_normal")):
        if not cfg["supervision"]["use_" + modality]:
            cfg["loss"][term] = 0.0
    return cfg


def config_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("init-depth", "init-normal", "train-depth", "train-normal"):
        parser.add_argument("--" + name, type=boolean)
    parser.add_argument("--freeze-background", type=boolean)
    parser.add_argument("--surface-init-mode", choices=("none", "depth_anchor"))
    for name in ("iterations", "geometry-from-iter", "geometry-ramp-iters"):
        parser.add_argument("--" + name, type=int)
    for name in ("match-confidence-min", "match-cycle-pixels", "match-reprojection-pixels"):
        parser.add_argument("--" + name, type=float)


def resolve_config(args):
    cfg = load_config(args.config, **{key: getattr(args, key, None) for key in (
        "init_depth", "init_normal", "train_depth", "train_normal", "iterations",
        "geometry_from_iter", "geometry_ramp_iters", "match_confidence_min",
        "match_cycle_pixels", "match_reprojection_pixels", "freeze_background", "surface_init_mode")})
    return cfg


def initialization_config(cfg):
    """Keep legacy initializer identities unchanged while recording opt-in modes."""
    result = {key: cfg[key] for key in ("init", "matcher", "seed")}
    if cfg.get("surface_init", {}).get("mode", "none") != "none":
        result["surface_init"] = cfg["surface_init"]
    return result


def check_initialization_config(initial, cfg):
    expected = initialization_config(cfg)
    actual = initial["request"]["config"]
    if any(actual.get(key) != value for key, value in expected.items()) or (
            actual.get("surface_init", {}).get("mode", "none") !=
            cfg.get("surface_init", {}).get("mode", "none")):
        raise ValueError("initialization config differs from joint request")


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
        "source/paintmesh_surface_init.py",
        "source/paintmesh_local_losses.py", "source/pgsr_geometry.py",
        "source/renderers/pgsr.py", "source/correspondence/geometry.py", "source/correspondence/contracts.py",
        "source/vendor.py", "submodules/RoMa/romatch/models/matcher.py",
        "submodules/RoMa/romatch/models/model_zoo/__init__.py",
        "tools/initialize_paintmesh_edgs.py", "tools/train_paintmesh_pgsr.py")]
    paths.append(REPO / "submodules/EDGS/tools/select_paintmesh_local_background.py")
    paths.append(REPO / "submodules/EDGS/source/losses.py")
    paths.append(REPO / "submodules/Inpaint360GS/utils/virtual_camera_manifest.py")
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def appearance_weights(cfg):
    """Record the exact cached assets consumed by optional training LPIPS."""
    if not cfg["loss"].get("lpips", 0):
        return {}
    import importlib.util
    import os
    spec = importlib.util.find_spec("lpips")
    if spec is None or not spec.origin:
        raise ValueError("LPIPS training requires the lpips package")
    package = Path(spec.origin).parent
    cache = Path(os.environ.get("TORCH_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "torch")))
    paths = dict(vgg_backbone=cache / "hub/checkpoints/vgg16-397923af.pth",
        lpips_calibration=package / "weights/v0.1/vgg.pth",
        lpips_implementation=package / "lpips.py", vgg_implementation=package / "pretrained_networks.py")
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"required cached LPIPS asset is missing: {path}")
    return {k: record(path) for k, path in paths.items()}


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


def check_preserved(source, result, editable, *, retained_rows=None, freeze_background=True,
                    trainable_mask=None, new_descendants=False):
    from plyfile import PlyData
    a, b = [PlyData.read(str(p), mmap=True)["vertex"].data for p in (source, result)]
    if a.dtype != b.dtype or editable.shape != (len(a),) or editable.dtype != np.bool_:
        raise ValueError("joint PLY schema/row order/count changed")
    if retained_rows is None:
        retained_rows = np.arange(len(a), dtype=np.int64)
    rows = np.asarray(retained_rows)
    if (rows.dtype != np.int64 or rows.ndim != 1 or len(rows) != len(b)
            or (rows < 0).any() or (rows >= len(a)).any()
            or (not new_descendants and (np.diff(rows) <= 0).any())):
        raise ValueError("invalid joint retained row mapping")
    if new_descendants and len(np.unique(rows[~editable[rows]])) != int((~editable[rows]).sum()):
        raise ValueError("adaptive density duplicated an old Gaussian")
    trainable = (editable if freeze_background else np.ones_like(editable)) if trainable_mask is None else np.asarray(trainable_mask)
    if trainable.dtype != np.bool_ or trainable.shape != editable.shape or np.any(editable & ~trainable):
        raise ValueError("invalid joint trainable ownership mask")
    if not np.array_equal(rows[~trainable[rows]], np.flatnonzero(~trainable)):
        raise ValueError("joint pruning removed frozen background rows")
    a, frozen = a[rows], ~trainable[rows]
    for key in a.dtype.names:
        mask = (slice(None) if key.startswith("obj_") or key in ("nx", "ny", "nz")
                else frozen)
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
    surface = receipt["request"]["config"].get("surface_init", {"mode": "none"})
    if surface["mode"] == "plane":
        # Read-only validation of historical experiments; new plane runs are rejected in load_config.
        model = read_json(receipt["outputs"]["surface_model"]["path"])
        normal = np.asarray(model["normal"], dtype=float)
        if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6):
            raise ValueError("invalid initialization surface normal")
        xyz = np.column_stack([vertex[key][editable] for key in "xyz"])
        residual = np.abs(xyz @ normal + model["offset"])
        if not np.isfinite(residual).all() or residual.max() > model["spacing"] * 1e-3:
            raise ValueError("initialization centers do not lie on the declared surface")
    elif surface["mode"] == "depth_anchor":
        model = read_json(receipt["outputs"]["surface_model"]["path"])
        verify_record(receipt["outputs"]["surface_evidence"])
        if model.get("mode") != "depth_anchor" or model.get("status") != "complete" or model.get("output_points") != int(editable.sum()):
            raise ValueError("invalid depth-anchor initialization model")
        with np.load(receipt["outputs"]["support"]["path"], allow_pickle=False) as data:
            xyz = np.column_stack([vertex[key][editable] for key in "xyz"])
            if not np.array_equal(data["xyz"], xyz) or len(data["frame"]) != len(xyz):
                raise ValueError("depth-anchor support/PLY differs")
            if ((data["source_kind"] != 2).any() or (data["other_frame"] != -1).any()
                    or data["normal_valid"].any() or not np.isfinite(data["corrected_depth"]).all()
                    or (data["corrected_depth"] <= 0).any()):
                raise ValueError("invalid depth-anchor support ownership/depth")
            if ((data["depth_votes"] < surface["min_cross_depth_views"]).any()
                    or (data["depth_votes"] > data["depth_checks"]).any()
                    or (data["depth_votes"] < surface["min_cross_depth_fraction"] * data["depth_checks"]).any()):
                raise ValueError("depth-anchor support lacks cross-view evidence")
            for frame in np.unique(data["frame"]):
                cam = model["camera_models"][str(frame)]
                rows = data["frame"] == frame
                uv = data["pixel_uv"][rows]; d = data["corrected_depth"][rows]
                rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(cam["k"]).T
                transform = np.asarray(cam["c2w"])
                expected = (rays * d[:, None]) @ transform[:3, :3].T + transform[:3, 3]
                if not np.allclose(xyz[rows], expected, rtol=1e-6, atol=model["spacing"]*1e-4):
                    raise ValueError("depth-anchor centers do not follow corrected camera rays")
    elif surface["mode"] != "none":
        raise ValueError("unsupported initialization surface mode")
    return receipt


def validate_selection(path, *, initial=None, config=None):
    receipt = _receipt(path, SELECTION_KIND)
    initial = initial or validate_init(receipt["inputs"]["initialization"]["path"])
    if receipt["inputs"] != receipt["request"]["inputs"] or receipt["initialization_id"] != initial["artifact_id"]:
        raise ValueError("local selection initialization identity mismatch")
    for key in ("lama", "camera"):
        if receipt["inputs"][key] != initial["inputs"][key]:
            raise ValueError(f"local selection {key} differs from initialization")
    if config is not None and receipt["request"]["config"] != config:
        raise ValueError("local selection configuration mismatch")
    editable = np.load(initial["outputs"]["editable_mask"]["path"], allow_pickle=False)
    trainable = np.load(receipt["outputs"]["trainable_mask"]["path"], allow_pickle=False)
    if (trainable.dtype != np.bool_ or trainable.shape != editable.shape
            or np.any(editable & ~trainable) or not np.any(trainable & ~editable)
            or not np.any(~trainable)):
        raise ValueError("local selection must include all new, some old, and freeze far rows")
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
    if cfg["loss"].get("lpips", 0):
        assets = receipt["request"].get("appearance_weights", {})
        if not {"vgg_backbone", "lpips_calibration", "lpips_implementation", "vgg_implementation"} <= assets.keys():
            raise ValueError("joint LPIPS weights provenance missing")
        for value in assets.values():
            verify_record(value)
    check_initialization_config(initial, cfg)
    verify_targets(receipt["inputs"]["lama"]["path"], receipt["inputs"]["camera"]["path"], **cfg["supervision"])
    if receipt["parameters"]["iterations"] != cfg["iterations"]:
        raise ValueError("EDGS joint iteration mismatch")
    result = receipt["outputs"]["ply"]["path"]
    if selected_ply is not None and record(selected_ply)["sha256"] != record(result)["sha256"]:
        raise ValueError("selected PLY differs from EDGS joint result")
    editable = np.load(initial["outputs"]["editable_mask"]["path"], allow_pickle=False)
    retained = receipt["outputs"].get("retained_rows")
    rows = np.load(retained["path"], allow_pickle=False) if retained else None
    count = len(rows) if rows is not None else initial["point_count"]
    pruning = cfg.get("pruning", {})
    density = cfg.get("adaptive_density", {"enabled": False})
    growing = density["enabled"]
    if growing and (receipt["parameters"].get("adaptive_density") != density or rows is None):
        raise ValueError("adaptive density configuration/ancestor mapping missing")
    if growing:
        diagnostics = read_json(receipt["outputs"]["diagnostics"]["path"])
        actual = diagnostics["adaptive_density"]
        if any(actual.get(k) != v for k, v in density.items()):
            raise ValueError("adaptive density diagnostics configuration mismatch")
        counts = actual["counters"]
        required = {"events", "cloned", "split_parents", "split_children", "pruned_new", "pruned_old",
                    "budget_limited", "observed"}
        if (set(counts) != required or any(type(v) is not int or v < 0 for v in counts.values())
                or counts["split_children"] != 2 * counts["split_parents"]):
            raise ValueError("invalid adaptive density population counters")
        remaining_new = int(editable[rows].sum())
        expected_new = int(editable.sum()) + counts["cloned"] + counts["split_children"] - counts["split_parents"] - counts["pruned_new"]
        if (remaining_new != expected_new or remaining_new > density["max_new_points"]
                or len(rows) - remaining_new != int((~editable).sum()) - counts["pruned_old"]):
            raise ValueError("adaptive density population accounting mismatch")
    if "pruning" in cfg and receipt["parameters"].get("pruning") != pruning:
        raise ValueError("joint pruning configuration mismatch")
    if receipt["point_count"] != count:
        raise ValueError("joint point count differs from retained row mapping")
    if pruning.get("enabled") and rows is None:
        raise ValueError("pruned joint output requires retained row mapping")
    if not growing and not pruning.get("enabled", False) and count != initial["point_count"]:
        raise ValueError("joint point count changed with pruning disabled")
    freeze_background = cfg.get("freeze_background", True)
    if type(freeze_background) is not bool or receipt["parameters"].get("freeze_background", True) != freeze_background:
        raise ValueError("joint background freezing configuration mismatch")
    trainable = None
    local = cfg.get("local_background", {"enabled": False})
    if local.get("enabled"):
        if not freeze_background or receipt["parameters"].get("local_background") != local:
            raise ValueError("joint local background configuration mismatch")
        selection = validate_selection(receipt["inputs"]["local_selection"]["path"], initial=initial, config=local)
        if selection["inputs"]["initialization"] != receipt["inputs"]["initialization"]:
            raise ValueError("joint local selection initialization differs")
        trainable = np.load(selection["outputs"]["trainable_mask"]["path"], allow_pickle=False)
    elif "local_selection" in receipt["inputs"]:
        raise ValueError("local selection provided while local background disabled")
    writeback = cfg.get("background_writeback", {"enabled": False})
    if writeback["enabled"]:
        if (freeze_background or local.get("enabled") or growing
                or receipt["parameters"].get("background_writeback") != writeback or rows is None):
            raise ValueError("invalid joint background writeback configuration")
        trainable = np.load(receipt["outputs"]["writeback_mask"]["path"], allow_pickle=False)
        if (trainable.dtype != np.bool_ or trainable.shape != editable.shape
                or np.any(editable & ~trainable)):
            raise ValueError("invalid joint writeback mask")
        if not np.array_equal(rows[~editable[rows]], np.flatnonzero(~editable)):
            raise ValueError("background writeback must retain every original old row")
        summary = read_json(receipt["outputs"]["writeback_summary"]["path"])
        if (summary["config"] != writeback or summary["selected_old"] != int((trainable & ~editable).sum())
                or summary["restored_old"] != int((~trainable).sum())):
            raise ValueError("joint writeback population mismatch")
    check_preserved(initial["outputs"]["ply"]["path"], result, editable, retained_rows=rows,
                    freeze_background=freeze_background, trainable_mask=trainable, new_descendants=growing)
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
