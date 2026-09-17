"""CPU-only, content-addressed Stage 5a/5b contracts shared by both projects.

No renderer, torch, scene or global training configuration is imported here.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np

from virtual_render_io import atomic_write, identity, read_json, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
RGB_KIND = "paintmesh-rgb-finetune"
LOCAL_KIND = "paintmesh-local-geometry"
EDIT_FIELDS = ("x", "y", "z", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3")


def record(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def verify_record(value):
    path = Path(value["path"])
    actual = record(path)
    if any(actual[key] != value[key] for key in ("sha256", "size_bytes")):
        raise ValueError(f"local geometry input/output changed: {path}")
    return path.resolve()


def seal(kind, **values):
    payload = dict(schema_version=1, kind=kind, complete=True, status="complete", **values)
    payload["artifact_id"] = identity(payload)
    return payload


def read_receipt(path, kind):
    payload = read_json(path)
    if (payload.get("kind") != kind or payload.get("schema_version") != 1 or
            payload.get("complete") is not True or payload.get("status") != "complete" or
            payload.get("artifact_id") != identity(
                {k: v for k, v in payload.items() if k != "artifact_id"})):
        raise ValueError(f"invalid/incomplete {kind} manifest: {path}")
    for section in ("inputs", "outputs"):
        for value in payload.get(section, {}).values():
            verify_record(value)
    return payload


def load_config(path, **overrides):
    import yaml
    defaults = yaml.safe_load((Path(__file__).parent / "configs/local_geometry.yaml").read_text())
    supplied = yaml.safe_load(Path(path).read_text())
    if not isinstance(supplied, dict):
        raise ValueError("local geometry config must be a mapping")
    cfg = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if key not in cfg:
            raise ValueError(f"unknown local geometry configuration: {key}")
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"unknown/invalid local geometry section: {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("iterations", "geometry_from_iter", "geometry_ramp_iters", "seed",
                "checkpoint_interval", "diagnostic_interval"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    total, start, ramp = (cfg[k] for k in (
        "iterations", "geometry_from_iter", "geometry_ramp_iters"))
    if not (0 <= start < total - 1 and ramp > 0 and start + ramp <= total - 1):
        raise ValueError("local schedule requires 0 <= start < T-1, ramp > 0, start+ramp <= T-1")
    if min(cfg["checkpoint_interval"], cfg["diagnostic_interval"]) < 1:
        raise ValueError("checkpoint/diagnostic intervals must be positive")
    for section in ("loss", "optimizer", "validity"):
        for key, value in cfg[section].items():
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid local {section}.{key}")
    valid = cfg["validity"]
    debug = cfg["debug"]
    if not isinstance(debug["enabled"], bool):
        raise ValueError("debug.enabled must be a boolean")
    for key, minimum in (("interval", 1), ("from_step", 0), ("view_index", 0), ("jpeg_quality", 1)):
        value = debug[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"debug.{key} must be an integer >= {minimum}")
    if debug["view_index"] >= 30 or debug["jpeg_quality"] > 100:
        raise ValueError("debug.view_index must be < 30 and jpeg_quality <= 100")
    if not (0 < valid["render_alpha_min"] <= valid["coverage_alpha_floor"] <= 1
            and 0 < valid["min_coverage_ratio"] <= 1 and valid["depth_jump_relative"] > 0
            and isinstance(valid["max_empty_steps"], int) and valid["max_empty_steps"] > 0):
        raise ValueError("invalid local geometry validity thresholds")
    if not any(cfg["optimizer"].values()) or not any(cfg["loss"][k] for k in (
            "depth", "lama_normal", "depth_normal_consistency")):
        raise ValueError("local refinement requires a learning rate and a geometry loss")
    return cfg


def verify_targets(lama_path, camera_path):
    """Verify content of all targets, masks and their render/camera provenance."""
    lama = read_json(lama_path)
    if (lama.get("kind") != "paintmesh-lama-completion" or
            lama.get("complete") is not True or lama.get("status") != "complete" or
            "normal" not in lama.get("parameters", {}).get("required_modalities", [])):
        raise ValueError("Stage 5b requires completed LaMa normal; finish PGSR Stage 2/3 first")
    inputs = read_json(verify_record(lama["input_manifest"]))
    if (inputs.get("artifact_id") != lama["input_artifact_id"] or
            inputs.get("kind") != "paintmesh-lama-inputs" or
            inputs.get("complete") is not True or inputs.get("status") != "complete"):
        raise ValueError("LaMa input identity mismatch")
    input_identity = dict(kind=inputs["kind"], schema_version=inputs["schema_version"],
        parameters=inputs["parameters"], inputs={stem: {key: value["sha256"]
            for key, value in frame["inputs"].items()} for stem, frame in inputs["frames"].items()})
    if identity(input_identity) != inputs["artifact_id"]:
        raise ValueError("LaMa input manifest content changed")
    metadata = inputs["parameters"]["normal"]
    if (metadata["space"], metadata["orientation"]) != ("camera", "toward_camera"):
        raise ValueError("unsupported LaMa normal coordinates")
    if verify_record(metadata["camera_manifest"]) != Path(camera_path).resolve():
        raise ValueError("LaMa targets use different virtual cameras")
    render = read_json(verify_record(metadata["render_manifest"]))
    if (render.get("depth_kind"), render.get("depth_unit")) != ("plane_z", "scene"):
        raise ValueError("Stage 5b requires PGSR plane z-depth in the same scene scale")
    if (render.get("artifact_id") != metadata["render_artifact_id"] or
            render.get("complete") is not True or render.get("status") != "complete" or
            render.get("artifact_id") != identity({k: v for k, v in render.items() if k != "artifact_id"})):
        raise ValueError("removed render identity mismatch")
    camera = read_json(camera_path)
    if camera.get("artifact_id") != metadata["camera_artifact_id"]:
        raise ValueError("camera identity mismatch")
    stems = [f"{index:05d}" for index in range(30)]
    if set(lama["frames"]) != set(stems) or set(inputs["frames"]) != set(stems):
        raise ValueError("local geometry requires exactly 30 matched target frames")
    output_hashes = {}
    for stem in stems:
        outputs = lama["frames"][stem]["outputs"]
        for name in ("color", "depth", "normal", "normal_valid", "normal_vis"):
            verify_record(outputs[name])
        masks = inputs["frames"][stem]["outputs"]
        for name in ("color_mask", "depth_mask", "normal_mask"):
            verify_record(masks[name])
        if len({masks[name]["sha256"] for name in ("color_mask", "depth_mask", "normal_mask")}) != 1:
            raise ValueError("RGB/depth/normal hole masks differ")
        output_hashes[stem] = {key: value["sha256"] for key, value in outputs.items()}
    expected = identity(dict(kind=lama["kind"], schema_version=lama["schema_version"],
        input_artifact_id=lama["input_artifact_id"],
        model={key: value["sha256"] for key, value in lama["model"].items()},
        parameters=lama["parameters"], outputs=output_hashes))
    if expected != lama["artifact_id"]:
        raise ValueError("LaMa completion identity mismatch")
    return lama, inputs


def prepare_rgb(args):
    verify_targets(args.lama, args.camera)
    inputs = {name: record(getattr(args, name)) for name in (
        "source_ply", "classifier", "inpaint_config", "camera", "lama", "fusion", "support")}
    context = seal("paintmesh-rgb-finetune-context", inputs=inputs,
        parameters=dict(iterations=args.rgb_iterations, seed_frame=args.seed_frame),
        rgb_ply=str(args.rgb_ply.resolve()), manifest=str(args.manifest.resolve()),
        implementation=sha256(REPO / "submodules/Inpaint360GS/edit_object_inpaint.py"))
    if args.context.exists() and read_json(args.context) != context:
        raise ValueError("RGB finetune inputs changed; choose a new INPAINT_RUN_NAME")
    if args.manifest.exists():
        receipt = validate_rgb(args.manifest)
        if receipt["context_artifact_id"] != context["artifact_id"]:
            raise ValueError("RGB finetune receipt belongs to different inputs")
        print("reuse")
        return
    if args.rgb_ply.exists():
        raise ValueError("existing Stage 5a PLY has no editable receipt; rebuild Stage 5 in a new inpaint run")
    write_json(args.context, context)
    print("train")


def save_rgb_receipt(context_path, ply_path, editable, gate):
    context = read_receipt(context_path, "paintmesh-rgb-finetune-context")
    ply_path = Path(ply_path).resolve()
    if str(ply_path) != context["rgb_ply"]:
        raise ValueError("Stage 5a output differs from requested PLY")
    root = Path(context_path).parent
    editable = np.asarray(editable, dtype=bool)
    if not editable.any():
        raise ValueError("Stage 5a spatial gate has no editable Gaussians")
    mask_path, gate_path = root / "editable_mask.npy", root / "gate.npz"
    atomic_write(mask_path, lambda stream: np.save(stream, editable, allow_pickle=False))
    atomic_write(gate_path, lambda stream: np.savez(stream, **gate))
    receipt = seal(RGB_KIND,
        inputs={"context": record(context_path), **context["inputs"]},
        outputs={"ply": record(ply_path), "editable_mask": record(mask_path), "gate": record(gate_path)},
        parameters=context["parameters"], context_artifact_id=context["artifact_id"],
        point_count=len(editable), editable_count=int(editable.sum()))
    write_json(Path(context["manifest"]), receipt)


def validate_rgb(path):
    from plyfile import PlyData
    receipt = read_receipt(path, RGB_KIND)
    context = read_receipt(receipt["inputs"]["context"]["path"], "paintmesh-rgb-finetune-context")
    if (receipt["context_artifact_id"] != context["artifact_id"] or
            receipt["parameters"] != context["parameters"] or
            any(receipt["inputs"].get(key) != value for key, value in context["inputs"].items()) or
            Path(receipt["outputs"]["ply"]["path"]).resolve() != Path(context["rgb_ply"]).resolve()):
        raise ValueError("RGB finetune context identity mismatch")
    ply = PlyData.read(receipt["outputs"]["ply"]["path"], mmap=True)["vertex"].data
    editable = np.load(receipt["outputs"]["editable_mask"]["path"], allow_pickle=False)
    if (editable.dtype != np.bool_ or editable.shape != (len(ply),) or
            len(ply) != receipt["point_count"] or int(editable.sum()) != receipt["editable_count"]
            or not editable.any()):
        raise ValueError("editable mask does not match final Stage 5a PLY rows")
    return receipt


def gate_membership(xyz, gate_path, support_path):
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    with np.load(gate_path, allow_pickle=False) as gate:
        projection, mask = gate["projection"], gate["mask"]
        threshold = float(gate["distance_threshold"])
    hom = np.concatenate((xyz, np.ones((len(xyz), 1), dtype=xyz.dtype)), axis=1) @ projection
    ndc = hom[:, :2] / (hom[:, 3:4] + 1e-8)
    height, width = mask.shape
    coords = np.rint(((ndc + 1) * np.array([width, height]) - 1) * .5)
    valid = np.isfinite(coords).all(axis=1) & (hom[:, 2] > 0)
    valid &= (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    candidates = np.flatnonzero(valid)
    pixels = coords[candidates].astype(np.int64)
    valid[candidates] &= mask[pixels[:, 1], pixels[:, 0]].astype(bool)
    if threshold > 0:
        support = PlyData.read(str(support_path), mmap=True)["vertex"].data
        points = np.stack([support[axis] for axis in "xyz"], axis=1)
        candidates = np.flatnonzero(valid)
        distances, _ = cKDTree(points).query(xyz[candidates])
        valid[candidates] &= distances < threshold
    return valid


def check_preserved(source_path, result_path, editable):
    from plyfile import PlyData
    source = PlyData.read(str(source_path), mmap=True)["vertex"].data
    result = PlyData.read(str(result_path), mmap=True)["vertex"].data
    if source.dtype != result.dtype or len(source) != len(result) or editable.shape != (len(source),):
        raise ValueError("local refinement changed PLY schema, row count or order")
    for name in source.dtype.names:
        frozen = ~editable if name in EDIT_FIELDS else slice(None)
        if not np.array_equal(source[name][frozen], result[name][frozen]):
            raise ValueError(f"local refinement changed frozen PLY field: {name}")
        if name in EDIT_FIELDS and not np.isfinite(result[name]).all():
            raise ValueError(f"nonfinite refined geometry: {name}")


def validate_local(path, *, selected_ply=None, lama_path=None, camera_path=None, request=None):
    receipt = read_receipt(path, LOCAL_KIND)
    if request is not None and receipt["request"] != request:
        raise ValueError("local geometry settings/inputs changed; use a new INPAINT_RUN_NAME")
    rgb = validate_rgb(receipt["inputs"]["rgb_manifest"]["path"])
    if (receipt["rgb_artifact_id"] != rgb["artifact_id"] or
            receipt["point_count"] != rgb["point_count"] or
            receipt["parameters"]["rgb_iterations"] != rgb["parameters"]["iterations"] or
            receipt["parameters"]["local_geometry_iterations"] != receipt["request"]["config"]["iterations"] or
            receipt["parameters"]["config"] != receipt["request"]["config"] or
            receipt["inputs"] != receipt["request"]["inputs"]):
        raise ValueError("local geometry receipt identity/parameters mismatch")
    for key in ("lama", "camera"):
        if receipt["inputs"][key] != rgb["inputs"][key]:
            raise ValueError(f"local geometry {key} differs from Stage 5a")
    for key, expected in (("lama", lama_path), ("camera", camera_path)):
        if expected is not None and Path(receipt["inputs"][key]["path"]).resolve() != Path(expected).resolve():
            raise ValueError(f"local geometry {key} source differs from published chain")
    verify_targets(receipt["inputs"]["lama"]["path"], receipt["inputs"]["camera"]["path"])
    ply = Path(receipt["outputs"]["ply"]["path"])
    if selected_ply is not None and record(selected_ply)["sha256"] != record(ply)["sha256"]:
        raise ValueError("selected PLY is not the completed local geometry output")
    editable = np.load(rgb["outputs"]["editable_mask"]["path"], allow_pickle=False)
    check_preserved(rgb["outputs"]["ply"]["path"], ply, editable)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare-rgb")
    for name in ("source-ply", "classifier", "inpaint-config", "camera", "lama", "fusion", "support",
                 "rgb-ply", "context", "manifest"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--rgb-iterations", type=int, required=True)
    prepare.add_argument("--seed-frame", type=int, required=True)
    config = subs.add_parser("config")
    config.add_argument("--config", type=Path, required=True)
    for name in ("iterations", "geometry-from-iter", "geometry-ramp-iters"):
        config.add_argument("--" + name, type=int)
    published = subs.add_parser("check-published")
    published.add_argument("--model-manifest", type=Path, required=True)
    published.add_argument("--selected-ply", type=Path, required=True)
    published.add_argument("--local-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare-rgb":
            prepare_rgb(args)
        elif args.command == "config":
            cfg = load_config(args.config, iterations=args.iterations,
                geometry_from_iter=args.geometry_from_iter, geometry_ramp_iters=args.geometry_ramp_iters)
            print(cfg["iterations"])
        else:
            model = read_json(args.model_manifest)
            if (model.get("complete") is not True or
                    bool(model.get("parameters", {}).get("local_geometry_refine", False)) != bool(args.local_manifest)):
                raise ValueError("published local geometry selection differs from this run; choose a new run")
            if record(args.selected_ply)["sha256"] != model["inputs"]["inpainted_gaussian_ply"]["sha256"]:
                raise ValueError("published model differs from selected Stage 5a/5b result")
            if args.local_manifest:
                local = validate_local(args.local_manifest, selected_ply=args.selected_ply)
                if model["upstream_artifact_ids"].get("local_geometry") != local["artifact_id"]:
                    raise ValueError("published model refers to a different local geometry result")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
