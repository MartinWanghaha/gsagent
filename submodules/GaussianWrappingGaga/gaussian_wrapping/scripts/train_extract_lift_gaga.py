"""End-to-end Gaussian Wrapping geometry, mesh and Gaga semantic pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "gaussian_wrapping"


def run(command: list[str], stage: str) -> None:
    print(f"[INFO] {stage}")
    result = subprocess.run(command, cwd=PROJECT)
    if result.returncode:
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")


def discover_mesh(model_path: Path) -> Path:
    patterns = (
        "mesh*_texture_refined_*.ply",
        "mesh*_post.ply",
        "mesh*.ply",
    )
    for pattern in patterns:
        candidates = [
            path
            for path in model_path.glob(pattern)
            if "semantic" not in path.stem
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        f"No Gaussian Wrapping mesh found under {model_path}; pass --mesh explicitly"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GW, extract/texture mesh, lift Gaga semantics and render all outputs"
    )
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--semantic_masks", required=True)
    parser.add_argument("--semantic_output", required=True)
    parser.add_argument("--rasterizer", choices=("radegs", "ours"), default="radegs")
    parser.add_argument("-r", "--resolution", default=None)
    parser.add_argument("--semantic_iterations", type=int, default=10_000)
    parser.add_argument("--mesh")
    parser.add_argument("--skip_geometry", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_semantic_mesh", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    arguments, passthrough = parser.parse_known_args()

    shared = [
        "-s",
        arguments.source_path,
        "-m",
        arguments.model_path,
    ]
    if arguments.resolution is not None:
        shared += ["-r", str(arguments.resolution)]

    if not arguments.skip_geometry:
        geometry_script = (
            "train_and_extract_gw_radegs.py"
            if arguments.rasterizer == "radegs"
            else "train_and_extract_gw_ours.py"
        )
        run(
            [
                sys.executable,
                str(PACKAGE / "scripts" / geometry_script),
                *shared,
                *passthrough,
            ],
            "Stage 1/5: Gaussian Wrapping training, mesh extraction and texture",
        )

    run(
        [
            sys.executable,
            str(PACKAGE / "semantic_lift.py"),
            *shared,
            "--semantic_masks",
            arguments.semantic_masks,
            "--semantic_output",
            arguments.semantic_output,
            "--semantic_iterations",
            str(arguments.semantic_iterations),
            "--rasterizer",
            arguments.rasterizer,
        ],
        "Stage 2/5: Gaga semantic lifting",
    )

    checkpoint = (
        Path(arguments.semantic_output)
        / "semantic"
        / f"semantic_chkpnt{arguments.semantic_iterations}.pth"
    )
    semantic_ply_model = Path(arguments.semantic_output)
    if not arguments.skip_render:
        for split in ("train", "test"):
            run(
                [
                    sys.executable,
                    str(PACKAGE / "semantic_render.py"),
                    "-s",
                    arguments.source_path,
                    "-m",
                    str(semantic_ply_model),
                    "--semantic_checkpoint",
                    str(checkpoint),
                    "--output",
                    str(Path(arguments.semantic_output) / "renders"),
                    "--load_iteration",
                    str(arguments.semantic_iterations),
                    "--rasterizer",
                    arguments.rasterizer,
                    "--split",
                    split,
                    *(["-r", str(arguments.resolution)] if arguments.resolution else []),
                ],
                f"Stage 3/5: Render {split} RGB/geometry/semantic outputs",
            )

    semantic_ply = (
        Path(arguments.semantic_output)
        / "point_cloud"
        / f"iteration_{arguments.semantic_iterations}"
        / "point_cloud.ply"
    )
    if not arguments.skip_semantic_mesh:
        mesh = (
            Path(arguments.mesh)
            if arguments.mesh
            else discover_mesh(Path(arguments.model_path))
        )
        run(
            [
                sys.executable,
                str(PACKAGE / "semantic_mesh.py"),
                "--mesh",
                str(mesh),
                "--semantic_ply",
                str(semantic_ply),
                "--semantic_checkpoint",
                str(checkpoint),
                "--output",
                str(Path(arguments.semantic_output) / "mesh" / "semantic_mesh.ply"),
            ],
            "Stage 4/5: Transfer Gaga labels to the Gaussian Wrapping mesh",
        )

    if not arguments.skip_render and not arguments.skip_evaluation:
        run(
            [
                sys.executable,
                str(PACKAGE / "semantic_eval.py"),
                "--render_dir",
                str(Path(arguments.semantic_output) / "renders" / "train"),
                "--semantic_masks",
                arguments.semantic_masks,
                "--semantic_checkpoint",
                str(checkpoint),
            ],
            "Stage 5/5: Evaluate rendered Gaga labels",
        )

    print("[INFO] Pipeline complete")
    print(f"[INFO] Geometry and mesh outputs: {arguments.model_path}")
    print(f"[INFO] Gaga semantic and full render outputs: {arguments.semantic_output}")


if __name__ == "__main__":
    main()
