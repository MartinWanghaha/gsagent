#!/usr/bin/env python3
"""Stage 4 of the peer EDGS-PGSR inpaint pipeline."""
import argparse
from pathlib import Path
import sys

EDGS = Path(__file__).resolve().parents[1]
REPO = EDGS.parents[1]
sys.path.insert(0, str(EDGS))
sys.path.insert(0, str(REPO / "scripts/paintmesh"))

from edgs_inpaint_io import (INIT_KIND, atomic_write, claim, config_arguments, implementation,
    read_json, record, resolve_config, seal, validate_init, verify_record, verify_targets, write_json,
    initialization_config)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("removed-model", "inpaint-config", "lama", "camera", "output-root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--proposal-initialization", type=Path,
        help="Reuse verified mixed proposals for the opt-in center/sampling ablation")
    config_arguments(p)
    return p


def run(args):
    import numpy as np
    import torch
    from plyfile import PlyData
    from source.paintmesh_joint_data import make_views, read_targets
    from source.paintmesh_edgs_init import RomaMatcher, matcher_weights, initialize_candidates, compose
    cfg = resolve_config(args)
    surface_enabled = cfg["surface_init"]["mode"] != "none"
    proposal_path = getattr(args, "proposal_initialization", None)
    if proposal_path is not None and not surface_enabled:
        raise ValueError("--proposal-initialization requires surface initialization mode")
    if surface_enabled and cfg["init"]["use_normal"]:
        raise ValueError("center-only surface initialization requires init.use_normal=false")
    lama, inputs, _ = verify_targets(args.lama, args.camera, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    removed = read_json(args.removed_model)
    if removed.get("kind") != "paintmesh-removed-edgs-model" or removed.get("complete") is not True or removed.get("status") != "complete":
        raise ValueError("requires a complete removed EDGS model")
    selection = read_json(args.inpaint_config)
    if selection.get("target_id") != removed["parameters"]["target_ids"] or selection.get("surrounding_ids") != removed["parameters"]["surrounding_ids"]:
        raise ValueError("initialization target/surrounding selection differs from removal")
    background = verify_record(removed["inputs"]["removed_gaussian_ply"])
    classifier = verify_record(removed["inputs"]["classifier"])
    weights = matcher_weights(cfg["matcher"], download=not args.validate_only and proposal_path is None)
    records = {key: record(getattr(args, key)) for key in ("removed_model", "inpaint_config", "lama", "camera")}
    records.update(background=record(background), classifier=record(classifier))
    proposal = None
    if proposal_path is not None:
        proposal = validate_init(proposal_path)
        if (any(proposal["inputs"][key] != value for key, value in records.items())
                or any(proposal["request"]["config"][key] != cfg[key] for key in ("init", "matcher", "seed"))
                or proposal["request"]["config"].get("surface_init", {}).get("mode", "none") != "none"
                or proposal["request"]["matcher_weights"] != weights):
            raise ValueError("cached mixed proposals differ from requested inputs/config/weights")
        records["proposal_initialization"] = record(proposal_path)
    request = dict(pipeline="edgs-pgsr", inputs=records, implementation=implementation(),
        config=initialization_config(cfg), matcher_weights=weights,
        output_root=str(args.output_root.resolve()), manifest=str(args.manifest.resolve()),
        removal_variant="published_target_removed", selection=removed["parameters"])
    output = args.output_root / "point_cloud.ply"
    if args.manifest.exists():
        validate_init(args.manifest, request)
        print(f"Reused EDGS initialization: {output}")
        return
    claim(args.output_root, request, args.validate_only)
    if args.validate_only:
        raise ValueError("EDGS initialization incomplete; run Stage 4")
    if proposal is None and not torch.cuda.is_available():
        raise RuntimeError("RoMa initialization requires CUDA")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    views = make_views(args.camera, device="cpu")
    targets = read_targets(lama, inputs, views, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    original_masks = []
    if surface_enabled:
        from PIL import Image
        for view, target in zip(views, targets):
            mask_record = inputs["frames"][view.image_name]["inputs"]["mask"]
            with Image.open(verify_record(mask_record)) as im:
                mask = np.array(im.convert("L")) > 0
            if mask.shape != tuple(target["mask"].shape):
                raise ValueError("original mask/camera shape mismatch")
            target["original_mask"] = mask
            original_masks.append(mask_record)
    vertex = PlyData.read(str(background), mmap=True)["vertex"].data
    xyz = np.column_stack([vertex[k] for k in "xyz"])
    rgb_pairs = [] if surface_enabled and proposal is None else None
    if proposal is None:
        matcher = RomaMatcher(cfg["matcher"], weights)
        support, report = initialize_candidates(views, targets, xyz, cfg, matcher, rgb_pairs=rgb_pairs)
    else:
        with np.load(verify_record(proposal["outputs"]["support"]), allow_pickle=False) as data:
            support = {key: data[key].copy() for key in data.files}
        report = dict(proposal["report"])
        print(f"Reusing verified mixed proposals: {len(support['xyz'])}", flush=True)
    surface_model, surface_outputs = None, {}
    if surface_enabled:
        from source.paintmesh_surface_init import correct_surface_support
        def save_surface(model, evidence):
            write_json(args.output_root / "surface_model.json", model)
            atomic_write(args.output_root / "surface_evidence.npz", lambda f: np.savez_compressed(f, **evidence))
        # Diagnostics are saved even when there is insufficient support to publish a PLY.
        support, stats, surface_model = correct_surface_support(support, xyz, views, targets,
            cfg["surface_init"], seed=cfg["seed"], max_points=cfg["init"]["max_points"],
            init_config=cfg["init"], rgb_pairs=rgb_pairs, artifact_callback=save_surface)
        report.update(surface=stats, initialized_points=len(support["xyz"]),
            rgb_points=stats["parent_rgb_points"], depth_points=stats["parent_depth_points"],
            surface_points=len(support["xyz"]), normal_oriented_points=0)
        write_json(args.output_root / "surface_model.json", surface_model)
        surface_outputs["surface_model"] = record(args.output_root / "surface_model.json")
        surface_outputs["surface_evidence"] = record(args.output_root / "surface_evidence.npz")
        print(f"Surface initialization: {stats}", flush=True)
    if implementation() != request["implementation"]:
        raise ValueError("initialization implementation changed during matching; rerun in a new run")
    editable = compose(background, support, cfg["init"], output)
    atomic_write(args.output_root / "support.npz", lambda f: np.savez_compressed(f, **support))
    atomic_write(args.output_root / "editable_mask.npy", lambda f: np.save(f, editable, allow_pickle=False))
    write_json(args.output_root / "diagnostics.json", report)
    write_json(args.output_root / "config.resolved.json", request["config"])
    for rec in [*records.values(), *weights.values(), *original_masks]:
        verify_record(rec)
    verify_targets(args.lama, args.camera, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    receipt = seal(INIT_KIND, request=request, inputs=records, point_count=len(editable),
        parameters=request["config"], report=report, outputs={**surface_outputs, **{key: record(args.output_root / path) for key, path in (
            ("ply", "point_cloud.ply"), ("support", "support.npz"), ("editable_mask", "editable_mask.npy"),
            ("diagnostics", "diagnostics.json"), ("config", "config.resolved.json"))}})
    validate_init(receipt)
    write_json(args.manifest, receipt)
    print(f"EDGS initialization complete: {output}", flush=True)


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        if not args.validate_only and (args.output_root / "request.json").exists():
            write_json(args.output_root / "failure.json", dict(status="failed", error=str(exc)))
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
