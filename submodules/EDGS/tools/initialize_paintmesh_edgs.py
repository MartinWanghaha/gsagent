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
    read_json, record, resolve_config, seal, validate_init, verify_record, verify_targets, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("removed-model", "inpaint-config", "lama", "camera", "output-root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    config_arguments(p)
    return p


def run(args):
    import numpy as np
    import torch
    from plyfile import PlyData
    from source.paintmesh_joint_data import make_views, read_targets
    from source.paintmesh_edgs_init import RomaMatcher, matcher_weights, initialize_candidates, compose
    cfg = resolve_config(args)
    lama, inputs, _ = verify_targets(args.lama, args.camera, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    removed = read_json(args.removed_model)
    if removed.get("kind") != "paintmesh-removed-edgs-model" or removed.get("complete") is not True or removed.get("status") != "complete":
        raise ValueError("requires a complete removed EDGS model")
    selection = read_json(args.inpaint_config)
    if selection.get("target_id") != removed["parameters"]["target_ids"] or selection.get("surrounding_ids") != removed["parameters"]["surrounding_ids"]:
        raise ValueError("initialization target/surrounding selection differs from removal")
    background = verify_record(removed["inputs"]["removed_gaussian_ply"])
    classifier = verify_record(removed["inputs"]["classifier"])
    weights = matcher_weights(cfg["matcher"], download=not args.validate_only)
    records = {key: record(getattr(args, key)) for key in ("removed_model", "inpaint_config", "lama", "camera")}
    records.update(background=record(background), classifier=record(classifier))
    request = dict(pipeline="edgs-pgsr", inputs=records, implementation=implementation(),
        config={k: cfg[k] for k in ("init", "matcher", "seed")}, matcher_weights=weights,
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
    if not torch.cuda.is_available():
        raise RuntimeError("RoMa initialization requires CUDA")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    views = make_views(args.camera, device="cpu")
    targets = read_targets(lama, inputs, views, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    vertex = PlyData.read(str(background), mmap=True)["vertex"].data
    xyz = np.column_stack([vertex[k] for k in "xyz"])
    matcher = RomaMatcher(cfg["matcher"], weights)
    support, report = initialize_candidates(views, targets, xyz, cfg, matcher)
    if implementation() != request["implementation"]:
        raise ValueError("initialization implementation changed during matching; rerun in a new run")
    editable = compose(background, support, cfg["init"], output)
    atomic_write(args.output_root / "support.npz", lambda f: np.savez_compressed(f, **support))
    atomic_write(args.output_root / "editable_mask.npy", lambda f: np.save(f, editable, allow_pickle=False))
    write_json(args.output_root / "diagnostics.json", report)
    write_json(args.output_root / "config.resolved.json", request["config"])
    for rec in [*records.values(), *weights.values()]:
        verify_record(rec)
    verify_targets(args.lama, args.camera, use_depth=cfg["init"]["use_depth"], use_normal=cfg["init"]["use_normal"])
    receipt = seal(INIT_KIND, request=request, inputs=records, point_count=len(editable),
        parameters=request["config"], report=report, outputs={key: record(args.output_root / path) for key, path in (
            ("ply", "point_cloud.ply"), ("support", "support.npz"), ("editable_mask", "editable_mask.npy"),
            ("diagnostics", "diagnostics.json"), ("config", "config.resolved.json"))})
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
