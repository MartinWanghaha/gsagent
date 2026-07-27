"""Evaluate rendered semantic labels against associated Gaga masks."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from semantic.metrics import (
    confusion_matrix,
    hungarian_permutation,
    metrics_from_confusion,
)
from semantic.observations import GagaObservationStore


def load_prediction(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path, allow_pickle=False)).long()
    return torch.from_numpy(np.asarray(Image.open(path)).copy()).long()


def evaluate(args) -> dict:
    num_classes = args.num_classes
    if num_classes is None:
        if not args.semantic_checkpoint:
            raise ValueError("--num_classes or --semantic_checkpoint is required")
        payload = torch.load(args.semantic_checkpoint, map_location="cpu")
        num_classes = int(payload["num_classes"])
    render_dir = Path(args.render_dir)
    with (render_dir / "manifest.json").open("r", encoding="utf8") as stream:
        manifest = json.load(stream)
    masks = GagaObservationStore(args.semantic_masks, require_all=True)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    label_dir = render_dir / "semantic_labels"
    for item in manifest:
        png = label_dir / f"{item['stem']}.png"
        path = png if png.is_file() else label_dir / f"{item['stem']}.npy"
        prediction = load_prediction(path)
        observation = masks.load(
            item["image_name"],
            prediction.shape[0],
            prediction.shape[1],
        )
        matrix += confusion_matrix(
            prediction,
            observation.labels,
            num_classes,
            masks.ignore_label,
        )
    direct = metrics_from_confusion(matrix)
    permutation = hungarian_permutation(matrix)
    matched_matrix = matrix[:, torch.from_numpy(np.argsort(permutation))]
    matched = metrics_from_confusion(matched_matrix)
    result = {
        "direct": direct,
        "hungarian": matched,
        "confusion_matrix": matrix.tolist(),
        "prediction_to_target": permutation.tolist(),
    }
    output = Path(args.output or render_dir / "semantic_metrics.json")
    with output.open("w", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate Gaga semantic rendering")
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--semantic_masks", required=True)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--semantic_checkpoint")
    parser.add_argument("--output")
    evaluate(parser.parse_args())
