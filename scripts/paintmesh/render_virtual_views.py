#!/usr/bin/env python3
"""Dispatch virtual rendering to independent, peer native/PGSR processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from virtual_render_io import (
    BACKENDS,
    atomic_write,
    check_backend,
    read_json,
    sha256,
    validate_render,
    write_json,
)

REPO = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=BACKENDS, default="inpaint360gs")
    p.add_argument(
        "--model-path", type=Path, required=True, help="Inpaint360GS removal work_model"
    )
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--edgs-model-path", type=Path)
    p.add_argument("--source-path", type=Path)
    p.add_argument("--camera-manifest", type=Path)
    p.add_argument("--tracker-archive", type=Path)
    p.add_argument("--resolution", type=int, default=-1)
    p.add_argument("--images", default="images")
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument(
        "--check-backend",
        action="store_true",
        help="Read-only preflight, also supports legacy native runs",
    )
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--inpaint-python", default=sys.executable)
    p.add_argument("--edgs-python", default=sys.executable)
    return p


def render_roots(args):
    root = args.model_path.resolve() / "virtual"
    return (
        root,
        root / f"ours_{args.iteration}",
        root / "ours_object_removal" / f"iteration_{args.iteration}",
    )


def validate_pair(args):
    root, full, removed = render_roots(args)
    check_backend(root, args.backend)
    pair_path = root / "virtual_render_manifest.json"
    if not pair_path.exists():
        # Preserve existing native tracker runs created before this contract.
        if (
            args.backend == "inpaint360gs"
            and not (root / "render_backend.json").exists()
        ):
            return None
        raise ValueError("virtual rendering is incomplete; run removal Stage 4")
    pair = read_json(pair_path)
    if pair.get("complete") is not True or pair.get("backend") != args.backend:
        raise ValueError("virtual render pair is incomplete or has another backend")
    cameras = read_json(args.camera_manifest)
    names = [camera["image_name"] for camera in cameras["cameras"]]
    if names != [f"{index:05d}" for index in range(30)]:
        raise ValueError("virtual cameras must contain exactly 00000..00029")
    for label, directory in (("full", full), ("removed", removed)):
        value = validate_render(directory, args.backend)
        if sorted(value["frames"]) != names:
            raise ValueError("virtual render frame set differs from cameras")
        if args.backend == "edgs-pgsr" and getattr(args, "alpha_min", None) is not None:
            if value["alpha_min"] != args.alpha_min:
                raise ValueError(
                    "normal alpha threshold changed; rerun removal Stage 4"
                )
        if (
            value["artifact_id"] != pair[label]
            or value["camera_artifact_id"] != cameras["artifact_id"]
        ):
            raise ValueError("virtual pair/camera identity mismatch")
        if value["inputs"]["camera_sha256"] != sha256(args.camera_manifest):
            raise ValueError("virtual camera file changed")
        for record in value["inputs"]["files"].values():
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"virtual rendering input changed: {record['path']}")
    if sha256(args.tracker_archive) != pair["archive_sha256"]:
        raise ValueError("virtual tracker archive changed")
    return pair


def main(argv=None):
    args = parser().parse_args(argv)
    if args.iteration <= 0:
        raise ValueError("iteration must be positive")
    if not 0 <= args.alpha_min <= 1:
        raise ValueError("alpha-min must be finite in [0,1]")
    root, full, removed = render_roots(args)
    check_backend(root, args.backend)
    if args.check_backend:
        return 0
    if args.camera_manifest is None or args.tracker_archive is None:
        raise ValueError("camera-manifest and tracker-archive are required")
    if args.validate_only:
        validate_pair(args)
        return 0
    if args.backend == "edgs-pgsr" and args.edgs_model_path is None:
        raise ValueError("edgs-pgsr requires --edgs-model-path for config.yaml")
    if args.tracker_archive.suffix != ".zip":
        raise ValueError("tracker-archive must end in .zip")
    if args.source_path is None:
        raise ValueError("source-path is required")
    # Resolve paths before changing cwd in the worker process.
    worker_args = [
        "--backend",
        args.backend,
        "--iteration",
        str(args.iteration),
        "--resolution",
        str(args.resolution),
        "--images",
        args.images,
        "--alpha-min",
        str(args.alpha_min),
    ]
    for name in ("model_path", "source_path", "camera_manifest", "edgs_model_path"):
        value = getattr(args, name)
        if value is not None:
            worker_args += ["--" + name.replace("_", "-"), str(value.resolve())]
    project = (
        REPO
        / "submodules"
        / ("Inpaint360GS" if args.backend == "inpaint360gs" else "EDGS")
    )
    executable = (
        args.inpaint_python if args.backend == "inpaint360gs" else args.edgs_python
    )
    environment = dict(os.environ, PYTHONPATH=str(project))
    write_json(
        root / "render_backend.json", {"backend": args.backend, "schema_version": 1}
    )
    write_json(
        root / "virtual_render_manifest.json",
        {"backend": args.backend, "complete": False},
    )
    subprocess.run(
        [
            executable,
            str(Path(__file__).with_name("render_virtual_worker.py")),
            *worker_args,
        ],
        cwd=project,
        env=environment,
        check=True,
    )
    full_record = validate_render(full, args.backend)
    removed_record = validate_render(removed, args.backend)
    if full_record["camera_artifact_id"] != removed_record["camera_artifact_id"]:
        raise ValueError("full and removed cameras differ")

    def write_archive(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(removed_record["frames"]):
                archive.write(removed / "renders" / f"{name}.png", f"{name}.png")

    atomic_write(args.tracker_archive, write_archive)
    write_json(
        root / "virtual_render_manifest.json",
        {
            "schema_version": 1,
            "backend": args.backend,
            "complete": True,
            "full": full_record["artifact_id"],
            "removed": removed_record["artifact_id"],
            "archive_sha256": sha256(args.tracker_archive),
        },
    )
    validate_pair(args)
    print(f"Virtual renderer {args.backend}: 30 full + 30 removed views -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
