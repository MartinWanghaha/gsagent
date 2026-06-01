"""
Generate ReferSplat JSON annotations for a scene using LocateAnything.

Usage:
    python generate_annotations.py <scene_folder> [--device cuda:0]

Output:
    <scene_folder>/train_json/<image_name>.json
    <scene_folder>/test_json/<image_name>.json  (every 8th image)
"""
import sys
import os
import re
import json
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "submodules/Eagle/Embodied"))
from locateanything_worker import LocateAnythingWorker

CKPT = str(Path(__file__).parent.parent / "checkpoints/LocateAnything-3B")


# COCO 80 categories — broad enough for indoor/outdoor scenes
COCO_CATEGORIES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
    "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]

MAX_SIZE = 1024

def detect_objects(worker, image: Image.Image) -> list[dict]:
    """Return list of {label, x1, y1, x2, y2} in pixel coords."""
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = image.size

    result = worker.detect(image, COCO_CATEGORIES)
    answer = result["answer"]

    # Parse only boxes with valid coords (skip <box>None</box>)
    objects = []
    for m in re.finditer(r"<ref>([^<]+)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        label = m.group(1).strip()
        x1, y1, x2, y2 = [int(g) / 1000 for g in m.groups()[1:]]
        objects.append({
            "label": label,
            "x1": x1 * w, "y1": y1 * h,
            "x2": x2 * w, "y2": y2 * h,
        })
    return objects


def build_json(objects: list[dict]) -> dict:
    """Convert detected objects to ReferSplat JSON format."""
    seen = {}
    for obj in objects:
        label = obj["label"]
        if label not in seen:
            seen[label] = {"category": label, "sentence": [label], "segmentation": f"{label}.png"}
        else:
            # deduplicate: add index suffix for multiple instances
            idx = sum(1 for k in seen if k.startswith(label))
            key = f"{label}_{idx}"
            seen[key] = {"category": label, "sentence": [f"the {label} number {idx+1}", label],
                         "segmentation": f"{key}.png"}
    return {"object": list(seen.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_folder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    scene = Path(args.scene_folder)
    images_dir = scene / "images"
    assert images_dir.exists(), f"No images/ dir in {scene}"

    image_files = sorted(images_dir.glob("*.[jJpP][pPnN][gG]"))
    assert image_files, f"No images found in {images_dir}"

    # Split: every 8th → test, rest → train (matches ReferSplat convention)
    test_set = set(image_files[i] for i in range(0, len(image_files), 8))

    train_json_dir = scene / "train_json"
    test_json_dir = scene / "test_json"
    train_json_dir.mkdir(exist_ok=True)
    test_json_dir.mkdir(exist_ok=True)

    print(f"Loading LocateAnything from {CKPT} ...")
    worker = LocateAnythingWorker(CKPT, device=args.device)

    for i, img_path in enumerate(image_files):
        out_dir = test_json_dir if img_path in test_set else train_json_dir
        out_file = out_dir / f"{img_path.name}.json"
        if out_file.exists():
            print(f"[{i+1}/{len(image_files)}] skip {img_path.name} (exists)")
            continue

        print(f"[{i+1}/{len(image_files)}] {img_path.name} ...", end=" ", flush=True)
        image = Image.open(img_path).convert("RGB")
        objects = detect_objects(worker, image)
        data = build_json(objects)
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"{len(objects)} objects")

    print(f"\nDone. train_json: {train_json_dir}, test_json: {test_json_dir}")


if __name__ == "__main__":
    main()
