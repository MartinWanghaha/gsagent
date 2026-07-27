"""Create a standard undistorted COLMAP dataset for 3DGS training."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

from PIL import Image


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def _resize_images(source: Path, destination: Path, factor: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                size = (max(image.width // factor, 1), max(image.height // factor, 1))
                image.resize(size, Image.Resampling.LANCZOS).save(destination / path.name)
        except Image.UnidentifiedImageError:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--camera", default="OPENCV")
    parser.add_argument("--colmap_executable", default="colmap")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--resize", action="store_true")
    args = parser.parse_args()

    root = Path(args.source_path).resolve()
    raw = root / "input"
    distorted = root / "distorted"
    sparse = distorted / "sparse"
    database = distorted / "database.db"
    if not raw.is_dir():
        raise FileNotFoundError(f"put source images in {raw}")
    sparse.mkdir(parents=True, exist_ok=True)
    use_gpu = "0" if args.no_gpu else "1"

    if not args.skip_matching:
        _run(
            [
                args.colmap_executable,
                "feature_extractor",
                "--database_path",
                str(database),
                "--image_path",
                str(raw),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                args.camera,
                "--SiftExtraction.use_gpu",
                use_gpu,
            ]
        )
        _run(
            [
                args.colmap_executable,
                "exhaustive_matcher",
                "--database_path",
                str(database),
                "--SiftMatching.use_gpu",
                use_gpu,
            ]
        )
        _run(
            [
                args.colmap_executable,
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(raw),
                "--output_path",
                str(sparse),
            ]
        )

    model = sparse / "0"
    if not model.is_dir():
        candidates = sorted(path for path in sparse.iterdir() if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"COLMAP mapper did not create a sparse model in {sparse}")
        model = candidates[0]
    _run(
        [
            args.colmap_executable,
            "image_undistorter",
            "--image_path",
            str(raw),
            "--input_path",
            str(model),
            "--output_path",
            str(root),
            "--output_type",
            "COLMAP",
        ]
    )
    generated = root / "sparse"
    if generated.is_dir() and not (generated / "0").is_dir():
        target = generated / "0"
        target.mkdir()
        for path in list(generated.iterdir()):
            if path != target:
                shutil.move(str(path), str(target / path.name))

    if args.resize:
        for factor in (2, 4, 8):
            _resize_images(root / "images", root / f"images_{factor}", factor)


if __name__ == "__main__":
    main()
