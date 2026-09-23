"""Evaluate PGSR-compatible renders produced from an EDGS training run.

Images are evaluated one pair at a time, keeping peak GPU memory independent
of the number of test views.  Output is a compatible superset of PGSR's
``results.json`` and ``per_view.json`` schema.  ``LPIPS_3dgs`` is retained as
an EDGS compatibility alias for the same pinned 3DGS/PGSR VGG LPIPS value.
"""

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the applicable repository license.
#
# For inquiries contact george.drettakis@inria.fr
#
# Adapted from the Gaussian Splatting/PGSR evaluation scripts.

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from source.losses import psnr, ssim
from source.vendor import bootstrap_gaussian_splatting

_METHOD_PATTERN = re.compile(r"^ours_(\d+)$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
_METRIC_NAMES = ("SSIM", "PSNR", "LPIPS", "LPIPS_3dgs")


def _method_sort_key(path: Path) -> tuple[int, str]:
    match = _METHOD_PATTERN.fullmatch(path.name)
    return (int(match.group(1)), path.name) if match else (2**63 - 1, path.name)


def discover_methods(model_path: Path) -> list[Path]:
    """Discover valid ``test/ours_<N>`` directories deterministically."""

    test_path = model_path / "test"
    if not test_path.is_dir():
        raise FileNotFoundError(
            f"missing {test_path}; run render.py without --skip-test first"
        )

    methods: list[Path] = []
    incomplete: list[str] = []
    for candidate in test_path.iterdir():
        if not candidate.is_dir() or not _METHOD_PATTERN.fullmatch(candidate.name):
            continue
        missing = [
            name for name in ("renders", "gt") if not (candidate / name).is_dir()
        ]
        if missing:
            incomplete.append(f"{candidate.name} (missing {', '.join(missing)})")
        else:
            methods.append(candidate)

    if incomplete:
        raise FileNotFoundError(
            f"incomplete render methods in {test_path}: " + "; ".join(incomplete)
        )
    if not methods:
        raise FileNotFoundError(
            f"no test/ours_<N>/{{renders,gt}} directories found in {model_path}"
        )
    return sorted(methods, key=_method_sort_key)


def _image_files(directory: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def paired_images(method_path: Path) -> list[tuple[str, Path, Path]]:
    """Return sorted render/GT pairs, rejecting silent filename mismatches."""

    renders = _image_files(method_path / "renders")
    ground_truth = _image_files(method_path / "gt")
    render_names = set(renders)
    gt_names = set(ground_truth)
    manifest_path = method_path / "render_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_views = manifest["views"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid render manifest: {manifest_path}") from error
        if not isinstance(manifest_views, dict) or not all(
            isinstance(name, str) for name in manifest_views.values()
        ):
            raise ValueError(f"invalid view mapping in {manifest_path}")
        if manifest.get("complete") is not True:
            raise RuntimeError(
                f"render method is incomplete according to {manifest_path}; "
                "rerun render.py for this iteration"
            )
        expected_names = set(manifest_views.values())
        if len(expected_names) != len(manifest_views):
            raise ValueError(f"duplicate output filenames in {manifest_path}")
        if manifest.get("num_views") != len(expected_names):
            raise ValueError(
                f"num_views does not match view mapping in {manifest_path}"
            )
        if render_names != expected_names or gt_names != expected_names:
            details: list[str] = []
            for label, actual in (("render", render_names), ("GT", gt_names)):
                missing = sorted(expected_names - actual)
                unexpected = sorted(actual - expected_names)
                if missing:
                    details.append(f"missing {label}: " + ", ".join(missing[:10]))
                if unexpected:
                    details.append(f"unexpected {label}: " + ", ".join(unexpected[:10]))
            raise FileNotFoundError(
                f"images do not match {manifest_path}: " + "; ".join(details)
            )

    if render_names != gt_names:
        missing_gt = sorted(render_names - gt_names)
        missing_render = sorted(gt_names - render_names)
        details: list[str] = []
        if missing_gt:
            details.append("missing GT: " + ", ".join(missing_gt[:10]))
        if missing_render:
            details.append("missing render: " + ", ".join(missing_render[:10]))
        raise FileNotFoundError(
            f"image pairing failed for {method_path}: " + "; ".join(details)
        )
    if not render_names:
        raise FileNotFoundError(f"no images found in {method_path / 'renders'}")
    return [(name, renders[name], ground_truth[name]) for name in sorted(render_names)]


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = (
        torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0).unsqueeze(0)
    )
    return tensor.to(device=device, non_blocking=device.type == "cuda")


def build_lpips(device: torch.device) -> torch.nn.Module:
    """Build PGSR's pinned LPIPS implementation once for the entire run."""

    bootstrap_gaussian_splatting()
    from lpipsPyTorch import LPIPS

    return LPIPS("vgg").eval().to(device)


def evaluate_method(
    method_path: Path,
    *,
    device: torch.device,
    lpips_model: torch.nn.Module,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    pairs = paired_images(method_path)
    per_view: dict[str, dict[str, float]] = {name: {} for name in _METRIC_NAMES}

    with torch.inference_mode():
        for name, render_path, gt_path in tqdm(
            pairs,
            desc=f"Metrics {method_path.name}",
        ):
            rendering = _load_rgb(render_path, device)
            ground_truth = _load_rgb(gt_path, device)
            if rendering.shape != ground_truth.shape:
                raise ValueError(
                    f"image size mismatch for {method_path.name}/{name}: "
                    f"render={tuple(rendering.shape[-2:])}, "
                    f"gt={tuple(ground_truth.shape[-2:])}"
                )

            values = {
                "SSIM": float(ssim(rendering, ground_truth).item()),
                "PSNR": float(psnr(rendering, ground_truth).mean().item()),
                "LPIPS": float(lpips_model(rendering, ground_truth).mean().item()),
            }
            values["LPIPS_3dgs"] = values["LPIPS"]
            for metric_name, value in values.items():
                per_view[metric_name][name] = value

    # PGSR reduces a float32 tensor, so keep the same final-rounding behavior.
    averages = {
        metric_name: float(
            torch.tensor(list(values.values()), dtype=torch.float32).mean().item()
        )
        for metric_name, values in per_view.items()
    }
    return averages, per_view


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_scene(
    model_path: Path,
    *,
    device: torch.device,
    lpips_model: torch.nn.Module,
) -> dict[str, dict[str, float]]:
    """Evaluate all rendered iterations in one EDGS output directory."""

    methods = discover_methods(model_path)
    results: dict[str, dict[str, float]] = {}
    per_view_results: dict[str, dict[str, dict[str, float]]] = {}
    print(f"Scene: {model_path}")
    for method_path in methods:
        averages, per_view = evaluate_method(
            method_path,
            device=device,
            lpips_model=lpips_model,
        )
        results[method_path.name] = averages
        per_view_results[method_path.name] = per_view
        print(
            f"  {method_path.name}: "
            f"SSIM={averages['SSIM']:.7f}  "
            f"PSNR={averages['PSNR']:.7f}  "
            f"LPIPS={averages['LPIPS']:.7f}"
        )

    # Write only after every method succeeds, so a failed evaluation cannot
    # replace a previously complete report with partial results.
    _write_json(model_path / "results.json", results)
    _write_json(model_path / "per_view.json", per_view_results)
    return results


def evaluate(
    model_paths: Sequence[str | Path] | str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compatibility-friendly programmatic entry point for one or more scenes."""

    if device is None or str(device) == "auto":
        resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if resolved_device.index is not None:
            torch.cuda.set_device(resolved_device.index)

    if isinstance(model_paths, (str, Path)):
        model_paths = (model_paths,)
    resolved_paths = [Path(value).expanduser().resolve() for value in model_paths]
    if not resolved_paths:
        raise ValueError("at least one model path is required")

    # Fail on missing/incomplete images before loading the large VGG network.
    for model_path in resolved_paths:
        for method_path in discover_methods(model_path):
            paired_images(method_path)

    lpips_model = build_lpips(resolved_device)
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for model_path in resolved_paths:
        all_results[str(model_path)] = evaluate_scene(
            model_path,
            device=resolved_device,
            lpips_model=lpips_model,
        )
    return all_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate renders produced by EDGS render.py"
    )
    parser.add_argument(
        "-m",
        "--model-paths",
        "--model_paths",
        required=True,
        nargs="+",
        help="one or more EDGS output directories",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="evaluation device (default: cuda:0 when available, otherwise cpu)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluate(args.model_paths, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
