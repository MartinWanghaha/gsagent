"""Create missing segmentation image sizes without changing source images."""

import argparse
import os
from pathlib import Path
import tempfile

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def prepare_images(source, resolution, images="images", exclude_test_images=False):
    source = Path(source)
    original = source / images
    selected = sorted(
        path for path in original.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
        and not (exclude_test_images and path.name.lower().startswith("test_"))
    )
    if not selected:
        raise ValueError(f"No reconstruction images in {original}")
    if resolution == 1:
        return len(selected), 0
    destination = source / f"images_{resolution}"
    destination.mkdir(exist_ok=True)
    generated = 0
    for path in selected:
        target = destination / path.name
        if target.is_file():
            continue
        with Image.open(path) as image:
            # Match the camera loader's round(width / resolution) convention.
            size = tuple(max(1, round(length / resolution)) for length in image.size)
            reduced = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".resize-", suffix=path.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            reduced.save(temporary, **({"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        generated += 1
    return len(selected), generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--exclude_test_images", action="store_true")
    args = parser.parse_args()
    selected, generated = prepare_images(args.source, args.resolution, args.images, args.exclude_test_images)
    print(f"Segmentation images: {selected} eligible, {generated} resized files generated")
