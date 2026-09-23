#!/usr/bin/env python3
"""Complete raw camera-space normals with LaMa, preserving the known region."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

INPAINT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(INPAINT_ROOT))
sys.path.insert(0, str(INPAINT_ROOT / "LaMa"))

import numpy as np

from tools import paintmesh_normal as normals
from tools.prepare_paintmesh_lama_data import (
    INPUT_KIND, _artifact, _atomic_json, _atomic_npy, _atomic_png,
    _load_complete_manifest, _sha256, _validate_partial_destination,
    _validate_run_local_outputs, verify_normal_inputs,
)


def predict(args):
    # Contract checks happen before loading torch or allocating GPU memory.
    input_manifest_path = args.input_manifest.resolve(strict=True)
    input_manifest = _load_complete_manifest(input_manifest_path, INPUT_KIND)
    if "normal" not in input_manifest["parameters"].get("required_modalities", []):
        raise ValueError("normal predictor requires manifested normal inputs")
    root, cameras = verify_normal_inputs(input_manifest)
    if args.input_dir.resolve(strict=True) != root.resolve():
        raise ValueError("normal input directory differs from manifest")
    output = args.output_dir.absolute()
    model_path = args.model_path.resolve(strict=True)
    sources = [Path(p) for p in input_manifest["roots"].values()] + [model_path, input_manifest_path]
    _validate_run_local_outputs((output,), sources)
    if any(output.resolve() == p.resolve() or output.resolve() in p.resolve().parents for p in sources):
        raise ValueError("normal output cannot contain input/model directories")
    stems = input_manifest["parameters"]["frame_names"]
    _validate_partial_destination(output, {f"{s}.npy" for s in stems} | {"valid", "vis", "prediction.json"}, "normal-output")
    for directory in ("valid", "vis"):
        _validate_partial_destination(output / directory, {f"{s}.png" for s in stems}, f"normal {directory}")
    prediction_config = INPAINT_ROOT / "LaMa/configs/prediction/default.yaml"
    model = {
        "config": _sha256(model_path / "config.yaml"),
        "checkpoint": _sha256(model_path / "models/best.ckpt"),
    }
    receipt = {
        "kind": "paintmesh-normal-prediction", "schema_version": 1,
        "complete": False, "status": "in_progress",
        "input_artifact_id": input_manifest["artifact_id"],
        "input_sha256": _sha256(input_manifest_path),
        "model": model, "method": normals.METHOD, "encoding": normals.ENCODING,
        "prediction_config_sha256": _sha256(prediction_config),
        "refine": True, "frames": {},
    }
    _atomic_json(output / "prediction.json", receipt)

    import torch
    import yaml
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.training.trainers import load_checkpoint

    config = OmegaConf.load(prediction_config)
    with (model_path / "config.yaml").open() as stream:
        train_config = OmegaConf.create(yaml.safe_load(stream))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"
    network = load_checkpoint(train_config, str(model_path / "models/best.ckpt"), strict=False, map_location="cpu")
    network.freeze()
    dataset = normals.NormalInpaintingDataset(root, stems, config.dataset.pad_out_to_modulo)
    # Match the depth predictor's refinement policy. Refinement needs gradients
    # with respect to intermediate features; do not wrap it in no_grad().
    for index, stem in enumerate(stems):
        batch = default_collate([dataset[index]])
        result = refine_predict(batch, network, **config.refiner)
        prediction = result[0].permute(1, 2, 0).detach().cpu().numpy()
        source, valid = normals.read_normal(root / f"{stem}.npy", root / "valid" / f"{stem}.png")
        hole = normals.read_mask(root / f"{stem}_mask.png")
        completed, completed_valid = normals.compose_normal(source, valid, hole, prediction, cameras[stem])
        _atomic_npy(output / f"{stem}.npy", completed)
        _atomic_png(output / "valid" / f"{stem}.png", completed_valid.astype(np.uint8) * 255, "L")
        _atomic_png(output / "vis" / f"{stem}.png", normals.normal_preview(completed, completed_valid), "RGB")
        receipt["frames"][stem] = {
            "normal": _artifact(output / f"{stem}.npy"),
            "normal_valid": _artifact(output / "valid" / f"{stem}.png"),
            "normal_vis": _artifact(output / "vis" / f"{stem}.png"),
        }
        print(f"normal {index + 1}/{len(stems)} {stem}: hole valid={completed_valid[hole].mean():.4f}", flush=True)
    verify_normal_inputs(input_manifest)
    if receipt["input_sha256"] != _sha256(input_manifest_path):
        raise ValueError("normal input manifest changed during inference")
    if model != {"config": _sha256(model_path / "config.yaml"), "checkpoint": _sha256(model_path / "models/best.ckpt")}:
        raise ValueError("LaMa checkpoint/config changed during normal inference")
    receipt.update(complete=True, status="complete")
    receipt["artifact_id"] = normals.identity(receipt)
    _atomic_json(output / "prediction.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
