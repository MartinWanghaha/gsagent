#!/usr/bin/env python3
"""Certify scene-specific revision-3 adaptive paths using training geometry only.

Search only inclination and target framing. Visibility, collision, context,
tracking and stereo gates stay fixed. Every accepted path has all 180 views.
This planner writes proposals only; run_remove produces the real render chain.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
from utils.pose_utils import generate_virtual_path
from utils.virtual_camera_manifest import adaptive_policy
from run_geometry_ablation import read, write, sha, now


def training_views(path):
    result = []
    for row in read(path):
        if Path(row["img_name"]).name.lower().startswith("test_"):
            continue
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = row["rotation"], row["position"]
        w2c = np.linalg.inv(c2w)
        result.append(SimpleNamespace(R=w2c[:3, :3].T, T=w2c[:3, 3],
            FoVx=2*np.arctan(row["width"] / (2*row["fx"])),
            FoVy=2*np.arctan(row["height"] / (2*row["fy"])),
            image_name=Path(row["img_name"]).stem))
    return sorted(result, key=lambda view: view.image_name)


def candidates(base):
    for area in (base["target_roi_fraction"], .06, .04):
        for angles in ([10.,30.], [5.,25.], [15.,35.], [20.,40.], [5.,20.]):
            cfg = copy.deepcopy(base)
            cfg.update(target_roi_fraction=area, inclination_range_deg=angles)
            yield cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()
    job, output = args.job.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    req = read(job / "request.json")
    policy = adaptive_policy()
    base = policy.load_options(REPO / "scripts/paintmesh/configs/inpaint360_adaptive180.json")
    summary = dict(started_at=now(), state="running", scenes={}, test_data_used=False,
                   planner_sha256=sha(Path(policy.__file__)))
    for name in args.scenes:
        scene = next(s for s in req["scenes"] if s["name"] == name)
        directory = Path(scene["removal_root"])
        root = directory.parents[1]
        view_file = root / "edgs/cameras.json"
        views = training_views(view_file)
        print(f"{name}: loading retained geometry and mesh provenance", flush=True)
        geometry = policy.load_geometry(directory / "work_model", 2000, scene["target_ids"], base)
        selection = read(directory / "config/object_removal/Inpaint360GS/inpaint360" / f"{name}.json")
        geometry["target_object_radius"] = selection["target_object_radius"]
        entry = dict(state="searching", trials=[], target_ids=scene["target_ids"],
                     training_camera_file=str(view_file), training_camera_file_sha256=sha(view_file),
                     geometry_inputs=geometry["inputs"])
        summary["scenes"][name] = entry
        for cfg in candidates(base):
            trial = dict(inclination_range_deg=cfg["inclination_range_deg"],
                         target_roi_fraction=cfg["target_roi_fraction"])
            started = time.monotonic()
            print(f"{name}: testing {trial}", flush=True)
            try:
                poses, trajectory = generate_virtual_path(views, 180, "adaptive",
                    adaptive_options=cfg, adaptive_geometry=geometry)
                if len(poses) != 180 or len(trajectory["tracking_edges"]) != 179:
                    raise ValueError("certification requires exactly 180 views and 179 sequential edges")
                # The production plan() checks original raycast, collision,
                # context and motion gates before returning any accepted path.
                config_path = output / f"{name}.json"
                trajectory_path = output / f"{name}.trajectory.json"
                write(config_path, cfg); write(trajectory_path, trajectory)
                trial.update(state="passed", seconds=time.monotonic()-started)
                entry.update(state="certified", config=str(config_path), config_sha256=sha(config_path),
                    trajectory=str(trajectory_path), trajectory_sha256=sha(trajectory_path),
                    frame_count=180, source_test_views_used=False,
                    min_visible_fraction=min(f["visible_fraction"] for f in trajectory["frames"]),
                    min_context_support=min(f["context_support"] for f in trajectory["frames"]),
                    max_rotation_deg=max(e["rotation_deg"] for e in trajectory["tracking_edges"]),
                    max_motion_fraction=max(e["motion_fraction"] for e in trajectory["tracking_edges"]))
                print(f"{name}: CERTIFIED {config_path}; visible={entry['min_visible_fraction']:.3f} "
                      f"context={entry['min_context_support']:.3f}", flush=True)
                entry["trials"].append(trial)
                break
            except ValueError as exc:
                trial.update(state="rejected", reason=str(exc), seconds=time.monotonic()-started)
                print(f"{name}: rejected: {exc}", flush=True)
                entry["trials"].append(trial)
            write(output / "certification.json", summary)
        if entry["state"] != "certified":
            entry["state"] = "no_valid_path"
        write(output / "certification.json", summary)
    summary.update(state="complete" if all(s["state"] == "certified" for s in summary["scenes"].values())
                   else "partial_failure", finished_at=now())
    write(output / "certification.json", summary)


if __name__ == "__main__":
    main()
