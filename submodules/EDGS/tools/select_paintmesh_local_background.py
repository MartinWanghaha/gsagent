#!/usr/bin/env python3
"""Produce immutable local edit ownership from training inputs only."""
import argparse
from pathlib import Path
import sys

EDGS = Path(__file__).resolve().parents[1]
REPO = EDGS.parents[1]
sys.path.insert(0, str(EDGS))
sys.path.insert(0, str(REPO / "scripts/paintmesh"))

from edgs_inpaint_io import (SELECTION_KIND, config_arguments, resolve_config,
    validate_init, validate_selection, verify_targets, record, seal, write_json,
    atomic_write, implementation, claim, verify_record)


def main():
    import numpy as np
    from PIL import Image
    from plyfile import PlyData
    from source.paintmesh_joint_data import make_views
    from source.paintmesh_edgs_init import camera
    from source.paintmesh_joint_region import select_local_rows
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("initialization", "lama", "camera", "output-root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target-ply", type=Path, action="append", required=True)
    config_arguments(p)
    args = p.parse_args()
    cfg = resolve_config(args)
    if not cfg["local_background"]["enabled"]:
        raise ValueError("enable local_background in the experiment configuration")
    initial = validate_init(args.initialization)
    _, inputs, _ = verify_targets(args.lama, args.camera)
    records = {k: record(getattr(args, k)) for k in ("initialization", "lama", "camera")}
    for i, path in enumerate(args.target_ply):
        records[f"target_{i}"] = record(path)
    for key in ("lama", "camera"):
        if records[key] != initial["inputs"][key]:
            raise ValueError("local selection differs from initialization inputs")
    request = dict(inputs=records, config=cfg["local_background"], implementation=implementation(),
                   output_root=str(args.output_root.resolve()))
    if args.manifest.exists():
        receipt = validate_selection(args.manifest, initial=initial, config=cfg["local_background"])
        if receipt["request"] != request:
            raise ValueError("local selection request changed")
        print("Reused local selection", flush=True)
        return
    claim(args.output_root, request)
    vertex = PlyData.read(initial["outputs"]["ply"]["path"], mmap=True)["vertex"].data
    new = np.load(initial["outputs"]["editable_mask"]["path"], allow_pickle=False)
    support = [np.column_stack([vertex[k][new] for k in "xyz"])]
    for path in args.target_ply:
        target = PlyData.read(str(path), mmap=True)["vertex"].data
        support.append(np.column_stack([target[k] for k in "xyz"]))
    with np.load(initial["outputs"]["support"]["path"], allow_pickle=False) as data:
        spacing = float(data["spacing"])
    views = make_views(args.camera)
    masks = []
    for v in views:
        with Image.open(inputs["frames"][v.image_name]["outputs"]["color_mask"]["path"]) as im:
            masks.append(np.asarray(im.convert("L")) > 0)
    trainable, votes, stats = select_local_rows(vertex, new, [camera(v) for v in views], masks,
                                               np.concatenate(support), spacing, cfg["local_background"])
    root = args.output_root
    for name, array in (("trainable_mask", trainable), ("overlap_votes", votes)):
        atomic_write(root / (name + ".npy"), lambda f, a=array: np.save(f, a, allow_pickle=False))
    write_json(root / "selection_summary.json", stats)
    for value in records.values():
        verify_record(value)
    if implementation() != request["implementation"]:
        raise ValueError("selection implementation changed during generation")
    receipt = seal(SELECTION_KIND, request=request, inputs=records, initialization_id=initial["artifact_id"],
        outputs={k: record(root / name) for k, name in (
            ("trainable_mask", "trainable_mask.npy"), ("overlap_votes", "overlap_votes.npy"),
            ("summary", "selection_summary.json"))})
    validate_selection(receipt, initial=initial, config=cfg["local_background"])
    write_json(args.manifest, receipt)
    print(f"Local selection: trainable_old={stats['trainable_old']} frozen_old={stats['frozen_old']} "
          f"new={stats['initial_new']} spatial_margin={stats['spatial_margin']:.6g}", flush=True)


if __name__ == "__main__":
    main()
