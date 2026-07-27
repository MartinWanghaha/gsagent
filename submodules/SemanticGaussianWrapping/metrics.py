"""Evaluate saved render sets in the standard 3DGS directory layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable
import warnings

import numpy as np
import torch
from PIL import Image

from evaluation import ImageMetricAccumulator


METHOD_PATTERN = re.compile(r"^ours_(\d+)$")


def _image(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def _method_iteration(path: Path) -> int | None:
    match = METHOD_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _available_methods(root: Path, split: str) -> list[Path]:
    candidates = [
        path
        for path in (root / split).glob("ours_*")
        if path.is_dir() and _method_iteration(path) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"no {split}/ours_* render directory below {root}")
    return sorted(candidates, key=lambda path: int(_method_iteration(path) or -1))


def _latest_method(root: Path, split: str) -> Path:
    return _available_methods(root, split)[-1]


def select_methods(
    root: Path,
    split: str,
    iterations: Iterable[int] = (),
    *,
    all_iterations: bool = False,
) -> list[Path]:
    """Resolve an ordered, explicit render-method selection."""

    available = _available_methods(root, split)
    requested = list(dict.fromkeys(int(value) for value in iterations))
    if all_iterations:
        if requested:
            raise ValueError("explicit iterations cannot be combined with all_iterations")
        return available
    if not requested or requested == [-1]:
        return [available[-1]]
    if -1 in requested or any(value < 0 for value in requested):
        raise ValueError("iterations must be non-negative, or use -1 alone for latest")
    lookup = {_method_iteration(path): path for path in available}
    missing = [value for value in requested if value not in lookup]
    if missing:
        raise FileNotFoundError(
            f"missing {split} render iteration(s) below {root}: "
            + ", ".join(str(value) for value in missing)
        )
    return [lookup[value] for value in requested]


def _lpips_model(device: torch.device):
    try:
        import lpips

        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as error:  # Optional metric must not invalidate PSNR/SSIM.
        warnings.warn(f"LPIPS is unavailable: {error}", RuntimeWarning, stacklevel=2)
        return None


def evaluate_directory(
    method: Path,
    device: torch.device,
    compute_lpips: bool = True,
    lpips_model=None,
) -> dict[str, object]:
    renders = method / "renders"
    ground_truth = method / "gt"
    metadata_path = method / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf8"))
        if metadata_path.is_file()
        else {}
    )
    names = sorted(path.name for path in renders.glob("*.png") if (ground_truth / path.name).is_file())
    rendered_indices = metadata.get("view_indices")
    if isinstance(rendered_indices, list):
        selected_names = {
            f"{int(index):05d}.png"
            for index in rendered_indices
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0
        }
        names = [name for name in names if name in selected_names]
    if not names:
        raise FileNotFoundError(f"no matching PNG files in {renders} and {ground_truth}")
    metrics = ImageMetricAccumulator()
    lpips_values: list[float] = []
    if compute_lpips and lpips_model is None:
        lpips_model = _lpips_model(device)
    for name in names:
        prediction = _image(renders / name, device)
        target = _image(ground_truth / name, device)
        metrics.update_image(prediction, target)
        if lpips_model is not None:
            with torch.no_grad():
                value = lpips_model(prediction[None] * 2 - 1, target[None] * 2 - 1)
            lpips_values.append(float(value))

    semantic_predictions = method / "semantic_id"
    semantic_targets = method / "gt_semantic_id"
    num_classes = int(metadata.get("num_semantic_classes", 0))
    if num_classes > 0 and semantic_predictions.is_dir() and semantic_targets.is_dir():
        for name in names:
            prediction_path = semantic_predictions / f"{Path(name).stem}.npy"
            target_path = semantic_targets / f"{Path(name).stem}.npy"
            if prediction_path.is_file() and target_path.is_file():
                prediction = torch.from_numpy(np.load(prediction_path, allow_pickle=False))
                target = torch.from_numpy(np.load(target_path, allow_pickle=False))
                metrics.update_semantic(prediction, target, num_classes)
    result = metrics.compute()
    result["lpips"] = sum(lpips_values) / len(lpips_values) if lpips_values else None
    return result


def _write_json_preserving(path: Path, value: Any) -> Path:
    """Write without replacing a previous, different evaluation record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        try:
            if path.read_text(encoding="utf8") == serialized:
                return path
        except OSError:
            pass
        for index in range(1, 10_000):
            candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
            if not candidate.exists():
                path = candidate
                break
        else:
            raise RuntimeError(f"cannot allocate a non-overwriting result beside {path}")
    path.write_text(serialized, encoding="utf8")
    return path


def result_output_path(root: Path, split: str, iteration: int) -> Path:
    return root / "results" / split / f"ours_{iteration}.json"


def resolve_metric_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_paths", "-m", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument(
        "--iteration",
        action="append",
        type=int,
        default=[],
        help="Evaluate one ours_<iteration> directory; repeat for several. Default: latest.",
    )
    parser.add_argument(
        "--all-iterations",
        action="store_true",
        help="Evaluate every numeric ours_* directory in ascending order.",
    )
    args = parser.parse_args()
    if args.all_iterations and args.iteration:
        parser.error("--all-iterations cannot be combined with --iteration")
    device = resolve_metric_device(args.device)
    shared_lpips = None if args.skip_lpips else _lpips_model(device)
    all_results: dict[str, dict[str, object]] = {}
    for value in args.model_paths:
        root = Path(value).resolve()
        try:
            methods = select_methods(
                root,
                args.split,
                args.iteration,
                all_iterations=args.all_iterations,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        root_results: dict[str, object] = {}
        for method in methods:
            iteration = int(_method_iteration(method) or 0)
            result = evaluate_directory(
                method,
                device,
                compute_lpips=shared_lpips is not None,
                lpips_model=shared_lpips,
            )
            record = {
                "model_path": str(root),
                "split": args.split,
                "iteration": iteration,
                **result,
            }
            result_path = _write_json_preserving(
                result_output_path(root, args.split, iteration),
                record,
            )
            root_results[f"ours_{iteration}"] = record
            lpips_text = "n/a" if result["lpips"] is None else f"{result['lpips']:.4f}"
            print(
                f"{root} ours_{iteration}: PSNR={result['psnr']:.4f}, "
                f"SSIM={result['ssim']:.4f}, LPIPS={lpips_text}, "
                f"L1={result['l1']:.6f} -> {result_path}"
            )
        all_results[str(root)] = root_results
        token = "_".join(str(_method_iteration(method)) for method in methods)
        summary_path = _write_json_preserving(
            root / "results" / args.split / f"summary_{token}.json",
            root_results,
        )
        print(f"{root}: summary -> {summary_path}")
    if len(all_results) > 1:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
