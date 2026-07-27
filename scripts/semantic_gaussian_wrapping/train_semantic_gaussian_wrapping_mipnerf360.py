#!/usr/bin/env python3
"""Train Semantic Gaussian Wrapping on Mip-NeRF 360 COLMAP scenes.

Examples:
    python scripts/semantic_gaussian_wrapping/train_semantic_gaussian_wrapping_mipnerf360.py \
        --scene counter --gpu 0 --eval

    python scripts/semantic_gaussian_wrapping/train_semantic_gaussian_wrapping_mipnerf360.py \
        --scene bicycle,garden --semantic-path sam_mask --dry-run

    python scripts/semantic_gaussian_wrapping/train_semantic_gaussian_wrapping_mipnerf360.py \
        --scene counter --resume --resume-run counter_rerun_002 -- --quiet

The semantic observations are produced by the Gaga preprocessing and
association scripts. This launcher validates them but deliberately does not
run those expensive stages implicitly.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SGW_ROOT = GSAGENT_ROOT / "submodules" / "SemanticGaussianWrapping"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "semantic_gaussian_wrapping_mipnerf360"

BENCHMARK_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OBSERVATION_SUFFIXES = {".png", ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth"}
CHECKPOINT_PATTERN = re.compile(r"^chkpnt(\d+)\.pth$")
CONTROLLED_NATIVE_OPTIONS = {
    "-s",
    "--source_path",
    "--source-path",
    "-m",
    "--model_path",
    "--model-path",
    "--config",
    "--checkpoint",
    "--device",
    "--set",
}
CONTROLLED_CONFIG_KEYS = {
    "optimization.iterations": "--iterations",
    "optimization.position_lr_max_steps": "--iterations",
    "data.images": "--images",
    "data.resolution": "--resolution",
    "data.data_device": "--data-device",
    "data.semantic_path": "--semantic-path",
    "data.semantic_confidence": "--semantic-confidence",
    "data.semantic_boundary": "--semantic-boundary",
    "data.eval": "--eval",
    "data.holdout": "--holdout with --eval",
    "density.max_gaussians": "--max-gaussians",
}
FRESH_ONLY_OPTIONS = {
    "--config",
    "-r",
    "--resolution",
    "--images",
    "--semantic-path",
    "--semantic-confidence",
    "--semantic-boundary",
    "--data-device",
    "--max-gaussians",
    "--eval",
    "--holdout",
}
OBSERVATION_KEYS = {
    "semantic": ("ids", "labels", "mask", "objects"),
    "confidence": ("confidence", "score", "probability"),
    "boundary": ("boundary", "edge", "seam"),
}
RESUME_SAFE_CONFIG_KEYS = {
    "optimization.iterations",
    "semantic.region_decode_chunk_size",
    "surface.support_routing_query_chunk",
    "surface.scipy_workers",
    "surface.mesh_feedback_scipy_workers",
}


class ObservationValidationError(FileNotFoundError):
    """Associated semantic evidence is absent or incomplete for a scene."""


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train Semantic Gaussian Wrapping on Mip-NeRF 360 scenes."
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name. Repeat or pass comma-separated names. Default: the seven "
            "benchmark scenes; use 'all' for every discovered COLMAP scene."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mip-NeRF 360 data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--semantic-gaussian-wrapping-root",
        "--sgw-root",
        dest="sgw_root",
        type=Path,
        default=DEFAULT_SGW_ROOT,
        help=f"SemanticGaussianWrapping checkout root. Default: {DEFAULT_SGW_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Training output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config. Default: <sgw-root>/configs/full.yaml.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Total training iterations. Default: 30000.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help=(
            "Image downsample factor/target width. Default: -1, which caps original "
            "Mip-NeRF 360 images at approximately 1600 pixels wide."
        ),
    )
    parser.add_argument(
        "--images",
        default="images",
        help="Image directory inside each scene. Default: images.",
    )
    parser.add_argument(
        "--semantic-path",
        default="sam_mask",
        help=(
            "Associated Gaga mask directory inside each scene. An absolute root may "
            "contain one subdirectory per scene. Default: sam_mask."
        ),
    )
    parser.add_argument(
        "--semantic-confidence",
        default="",
        help="Optional confidence-map directory, with the same path rules as --semantic-path.",
    )
    parser.add_argument(
        "--semantic-boundary",
        default="",
        help="Optional boundary-map directory, with the same path rules as --semantic-path.",
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used to store source images. Default: cpu.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device passed to train.py. Default: cuda.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="Override density.max_gaussians. Default: use the selected config (2,000,000).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value assigned to CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Hold out every --holdout image for novel-view evaluation.",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=8,
        help="Evaluation holdout stride used with --eval. Default: 8.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Additional non-wrapper config override. Repeat as needed; use the dedicated "
            "launcher flags for data, iteration, evaluation, and density-cap settings."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each incomplete scene from its latest checkpoint not newer than --iterations.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Explicit run directory to resume. A relative value is resolved below "
            "--output-root; an absolute directory is used as-is. Requires --resume "
            "and exactly one concrete --scene."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Start a fresh run even when outputs exist. Existing state is preserved and "
            "the new run uses an isolated <scene>_rerun_NNN directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without launching training.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after a validation or training failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )

    argv = sys.argv[1:] if argv is None else argv
    args, extra_args = parser.parse_known_args(argv)
    args.iterations_explicit = any(
        value == "--iterations" or value.startswith("--iterations=") for value in argv
    )
    if extra_args:
        if extra_args[0] != "--":
            parser.error(
                "unexpected arguments: "
                + " ".join(extra_args)
                + ". Put native train.py flags after a standalone '--'."
            )
        extra_args = extra_args[1:]

    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive factor/target width.")
    if args.max_gaussians is not None and args.max_gaussians <= 0:
        parser.error("--max-gaussians must be positive when supplied.")
    if args.holdout <= 0:
        parser.error("--holdout must be positive.")
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")
    if args.resume_run is not None:
        if not args.resume:
            parser.error("--resume-run requires --resume.")
        requested_scenes = [
            part.strip()
            for item in (args.scenes or [])
            for part in item.split(",")
            if part.strip()
        ]
        if len(requested_scenes) != 1 or requested_scenes[0].lower() == "all":
            parser.error(
                "--resume-run requires exactly one concrete --scene because one run "
                "directory cannot be shared across scenes."
            )
    wrapper_argv = argv[: argv.index("--")] if "--" in argv else argv
    supplied_options = set()
    for value in wrapper_argv:
        if not value.startswith("-"):
            continue
        option = value.split("=", 1)[0]
        if option.startswith("-r") and not option.startswith("--"):
            option = "-r"
        supplied_options.add(option)
    resume_conflicts = sorted(supplied_options.intersection(FRESH_ONLY_OPTIONS))
    if args.resume and resume_conflicts:
        parser.error(
            "fresh-run options cannot be combined with --resume because the checkpoint "
            "owns its data and objective: " + ", ".join(resume_conflicts)
        )
    if not args.semantic_path:
        parser.error("--semantic-path cannot be empty")
    for expression in args.overrides:
        if "=" not in expression:
            parser.error(f"--set must be KEY=VALUE, got {expression!r}.")
        key = expression.split("=", 1)[0]
        if not key or key != key.strip():
            parser.error(f"--set has an invalid key: {key!r}.")
        controlled = next(
            (
                name
                for name in CONTROLLED_CONFIG_KEYS
                if key == name
                or key.startswith(f"{name}.")
                or name.startswith(f"{key}.")
            ),
            None,
        )
        if controlled is not None:
            parser.error(
                f"use {CONTROLLED_CONFIG_KEYS[controlled]} instead of --set {key}=...; "
                "the launcher validates the effective data and schedule it constructs"
            )

    native_names = {
        value.split("=", 1)[0] for value in extra_args if value.startswith("-")
    }
    conflicts = sorted(native_names.intersection(CONTROLLED_NATIVE_OPTIONS))
    if conflicts:
        parser.error(
            "wrapper-controlled native options cannot be repeated after '--': "
            + ", ".join(conflicts)
        )
    return args, extra_args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_resume_run(run: Path, output_root: Path) -> Path:
    """Resolve and validate one explicitly selected resume directory."""

    output_root = output_root.expanduser().resolve()
    requested = run.expanduser()
    if requested.is_absolute():
        result = requested.resolve()
    else:
        result = (output_root / requested).resolve()
        try:
            result.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"relative --resume-run must remain below --output-root: {run}"
            ) from error
    if not result.is_dir():
        raise FileNotFoundError(f"--resume-run directory does not exist: {result}")
    return result


def sparse_directory(scene_dir: Path) -> Path:
    sparse_zero = scene_dir / "sparse" / "0"
    return sparse_zero if sparse_zero.is_dir() else scene_dir / "sparse"


def is_colmap_scene(scene_dir: Path, images: str | None) -> bool:
    sparse = sparse_directory(scene_dir)
    has_cameras = (sparse / "cameras.bin").is_file() or (
        sparse / "cameras.txt"
    ).is_file()
    has_images = (sparse / "images.bin").is_file() or (sparse / "images.txt").is_file()
    has_points = (sparse / "points3D.bin").is_file() or (
        sparse / "points3D.txt"
    ).is_file()
    has_image_directory = images is None or (scene_dir / images).is_dir()
    return has_image_directory and has_cameras and has_images and has_points


def discover_scenes(data_root: Path, images: str | None) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    return [
        scene_dir.name
        for scene_dir in sorted(data_root.iterdir())
        if scene_dir.is_dir() and is_colmap_scene(scene_dir, images)
    ]


def expand_requested_scenes(
    requested: list[str] | None, available: list[str]
) -> list[str]:
    if not requested:
        missing = [scene for scene in BENCHMARK_SCENES if scene not in available]
        if missing:
            raise FileNotFoundError(
                "Missing default benchmark scene(s): " + ", ".join(missing)
            )
        return list(BENCHMARK_SCENES)

    scenes = [
        part.strip() for item in requested for part in item.split(",") if part.strip()
    ]
    if any(scene.lower() == "all" for scene in scenes):
        return available
    missing = sorted(set(scenes) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )
    requested_set = set(scenes)
    return [scene for scene in available if scene in requested_set]


def validate_sgw_root(root: Path) -> Path:
    required = [
        root / "train.py",
        root / "configs" / "default.yaml",
        root / "configs" / "full.yaml",
        root / "utils" / "config_utils.py",
        root / "submodules" / "diff-semantic-gaussian-rasterization" / "setup.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "SemanticGaussianWrapping root is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )
    return root


def resolve_config(path: Path | None, workdir: Path) -> Path:
    if path is None:
        result = workdir / "configs" / "full.yaml"
    elif path.expanduser().is_absolute():
        result = path.expanduser()
    else:
        candidates = [
            Path.cwd() / path,
            workdir / path,
            workdir / "configs" / path,
        ]
        result = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Training config does not exist: {result}")
    return result


def load_checkpoint_configuration(checkpoint: Path) -> tuple[dict[str, Any], int]:
    """Load authoritative resume metadata without faulting all tensor pages."""

    import torch

    try:
        load_options = {"map_location": "cpu", "weights_only": True}
        try:
            state = torch.load(checkpoint, mmap=True, **load_options)
        except TypeError:
            # PyTorch before mmap checkpoint support uses the eager fallback.
            state = torch.load(checkpoint, **load_options)
    except Exception as error:
        raise ValueError(
            f"cannot load resume checkpoint {checkpoint}: {error}"
        ) from error
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a training state mapping: {checkpoint}")
    version = state.get("version")
    if type(version) is not int:
        raise ValueError(f"checkpoint has an invalid training schema: {checkpoint}")
    if version > 3:
        raise ValueError(f"checkpoint uses a newer training schema: {checkpoint}")
    if version != 3:
        raise ValueError(
            "checkpoint does not contain the region-conditioned training schema; "
            f"start a fresh run: {checkpoint}"
        )
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint has no resolved training config: {checkpoint}")
    saved_iteration = state.get("iteration")
    if type(saved_iteration) is not int or saved_iteration < 0:
        raise ValueError(f"checkpoint has an invalid iteration: {checkpoint}")
    return config, saved_iteration


def resume_data_options(
    config: Mapping[str, Any], checkpoint: Path
) -> tuple[str, str, str, str]:
    """Read data paths from a selected checkpoint's authoritative config."""

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint config has no data mapping: {checkpoint}")
    required = ("images", "semantic_path", "semantic_confidence", "semantic_boundary")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            f"checkpoint config is missing data keys {', '.join(missing)}: {checkpoint}"
        )
    return tuple(str(data[key] or "") for key in required)


def resume_target_iteration(
    args: argparse.Namespace, config: Mapping[str, Any], checkpoint: Path
) -> int:
    if args.iterations_explicit:
        return int(args.iterations)
    try:
        return int(config["optimization"]["iterations"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"checkpoint config has no valid optimization.iterations: {checkpoint}"
        ) from error


def image_manifest(image_dir: Path) -> dict[str, tuple[int, int]]:
    """Return image stem -> (height, width), rejecting ambiguous/corrupt inputs."""

    from PIL import Image

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    manifest: dict[str, tuple[int, int]] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in manifest:
            raise ValueError(f"Duplicate image stem {path.stem!r} in {image_dir}")
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode source image {path}: {error}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"Source image has an invalid size: {path}")
        manifest[path.stem] = (height, width)
    if not manifest:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return manifest


def image_stems(image_dir: Path) -> set[str]:
    """Compatibility helper used by callers that only need image names."""

    return set(image_manifest(image_dir))


def resolve_observation_directory(
    scene_dir: Path, scene: str, value: str
) -> tuple[Path | None, str]:
    if not value:
        return None, ""
    requested = Path(value).expanduser()
    if requested.is_absolute():
        per_scene = requested / scene
        resolved = per_scene if per_scene.is_dir() else requested
        return resolved.resolve(), str(resolved.resolve())
    return scene_dir / requested, value


def _load_gaga_instance_count(info_path: Path) -> int:
    try:
        metadata = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationValidationError(
            f"cannot read Gaga association metadata {info_path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ObservationValidationError(
            f"Gaga association metadata must be a JSON object: {info_path}"
        )
    count = metadata.get("num_mask", metadata.get("num_instances"))
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ObservationValidationError(
            f"Gaga association metadata has no positive integer num_mask/num_instances: "
            f"{info_path}"
        )
    return count


def _inspect_observation(
    path: Path, label: str
) -> tuple[tuple[int, int], int | None, bool | None]:
    """Decode one observation and report shape plus semantic ID evidence."""

    import numpy as np

    suffix = path.suffix.lower()
    if suffix in {".png", ".tif", ".tiff"}:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            channels = len(image.getbands())
            if label != "semantic":
                image.verify()
                return (height, width), None, None
            if channels == 2:
                raise ValueError(
                    "semantic observations cannot have exactly two channels"
                )
            if channels == 1:
                _minimum, maximum = image.getextrema()
                maximum = int(bool(maximum)) if image.mode == "1" else int(maximum)
                return (height, width), maximum, maximum > 0
            value: Any = np.asarray(image)
    elif suffix == ".npy":
        value: Any = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = OBSERVATION_KEYS[label]
            selected = next((key for key in keys if key in archive), None)
            if selected is None:
                if len(archive.files) != 1:
                    raise KeyError(f"none of {keys} found in {path}")
                selected = archive.files[0]
            value = archive[selected]
    elif suffix in {".pt", ".pth"}:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            keys = OBSERVATION_KEYS[label]
            selected = next((key for key in keys if key in value), None)
            if selected is None:
                if len(value) != 1:
                    raise KeyError(f"none of {keys} found in {path}")
                value = next(iter(value.values()))
            else:
                value = value[selected]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    else:
        raise ValueError(f"unsupported observation format: {path}")

    shape = np.asarray(value).shape
    if len(shape) not in (2, 3):
        raise ValueError(f"observation must be [H,W] or [H,W,C], got {shape}")
    if len(shape) == 3:
        channels = int(shape[2])
        if channels <= 0:
            raise ValueError(f"observation has no channels: {shape}")
        if label == "semantic" and channels == 2:
            raise ValueError("semantic observations cannot have exactly two channels")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"observation has an invalid spatial shape: {shape}")
    if label != "semantic":
        return (height, width), None, None

    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"semantic IDs must use an integer dtype, got {array.dtype}")
    if array.ndim == 2:
        ids = array.astype(np.int64, copy=False)
    elif array.shape[2] == 1:
        ids = array[..., 0].astype(np.int64, copy=False)
    else:
        rgb = array[..., :3].astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    valid = ids >= 0
    if not bool(valid.any()):
        return (height, width), None, False
    maximum = int(ids[valid].max())
    return (height, width), maximum, bool((ids[valid] > 0).any())


def _aspect_ratios_compatible(
    first: tuple[int, int], second: tuple[int, int], tolerance: float = 0.01
) -> bool:
    first_height, first_width = first
    second_height, second_width = second
    lhs = first_width * second_height
    rhs = second_width * first_height
    return abs(lhs - rhs) / max(lhs, rhs, 1) <= tolerance


def validate_observations(
    directory: Path | None,
    expected_stems: set[str],
    *,
    label: str,
    require_info: bool,
    expected_shapes: Mapping[str, tuple[int, int]] | None = None,
    decoded_shapes: dict[str, tuple[int, int]] | None = None,
    native_reference_shapes: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[int, int]:
    if directory is None or not directory.is_dir():
        raise ObservationValidationError(
            f"{label} directory does not exist: {directory}"
        )
    declared_instances = None
    if require_info:
        info_path = directory / "info.json"
        if not info_path.is_file():
            raise ObservationValidationError(
                f"Gaga association metadata is missing: {info_path}"
            )
        declared_instances = _load_gaga_instance_count(info_path)

    available_files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in OBSERVATION_SUFFIXES:
            continue
        previous = available_files.get(path.stem)
        if previous is not None:
            raise ObservationValidationError(
                f"ambiguous {label} observations for stem {path.stem!r}: "
                f"{previous.name}, {path.name}"
            )
        available_files[path.stem] = path
    available = set(available_files)
    missing = sorted(expected_stems - available)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ObservationValidationError(
            f"{label} observations are incomplete in {directory}: missing "
            f"{len(missing)}/{len(expected_stems)} image stems ({preview}{suffix})"
        )
    semantic_maximum: int | None = None
    semantic_has_foreground = False
    if expected_shapes is not None:
        for stem in sorted(expected_stems & available):
            path = available_files[stem]
            try:
                actual_shape, maximum, has_foreground = _inspect_observation(
                    path, label
                )
            except Exception as error:
                raise ObservationValidationError(
                    f"cannot decode {label} observation {path}: {error}"
                ) from error
            if maximum is not None:
                semantic_maximum = (
                    maximum
                    if semantic_maximum is None
                    else max(semantic_maximum, maximum)
                )
            semantic_has_foreground |= bool(has_foreground)
            if decoded_shapes is not None:
                decoded_shapes[stem] = actual_shape
            if (
                native_reference_shapes is not None
                and stem in native_reference_shapes
                and actual_shape != native_reference_shapes[stem]
            ):
                raise ObservationValidationError(
                    f"unaligned {label} observation {path}: native shape={actual_shape}, "
                    f"semantic native shape={native_reference_shapes[stem]}"
                )
            expected_shape = expected_shapes[stem]
            if not _aspect_ratios_compatible(actual_shape, expected_shape):
                raise ObservationValidationError(
                    f"unaligned {label} observation {path}: incompatible aspect ratio; "
                    f"shape={actual_shape}, source image shape={expected_shape}"
                )
    if (
        label == "semantic"
        and expected_shapes is not None
        and available & expected_stems
    ):
        if not semantic_has_foreground:
            raise ObservationValidationError(
                f"semantic observations contain no foreground IDs in {directory}"
            )
        if declared_instances is not None and semantic_maximum is not None:
            if semantic_maximum > declared_instances:
                raise ObservationValidationError(
                    f"semantic ID {semantic_maximum} exceeds info.json num_mask "
                    f"{declared_instances} in {directory}"
                )
            if semantic_maximum < declared_instances:
                raise ObservationValidationError(
                    f"semantic ID domain is incomplete in {directory}: maximum observed ID "
                    f"{semantic_maximum}, info.json num_mask {declared_instances}"
                )
    return len(expected_stems) - len(missing), len(expected_stems)


def latest_checkpoint(
    model_path: Path, target_iteration: int
) -> tuple[Path | None, int | None, int | None]:
    checkpoints: list[tuple[int, Path]] = []
    for path in model_path.glob("chkpnt*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None, None, None
    latest_any = max(iteration for iteration, _ in checkpoints)
    eligible = [
        (iteration, path)
        for iteration, path in checkpoints
        if iteration <= target_iteration
    ]
    if not eligible:
        return None, None, latest_any
    iteration, path = max(eligible, key=lambda item: item[0])
    return path, iteration, latest_any


def _encoded(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append_override(command: list[str], key: str, value: object) -> None:
    command.extend(["--set", f"{key}={_encoded(value)}"])


def fresh_overrides(
    args: argparse.Namespace,
    semantic_path: str,
    confidence_path: str,
    boundary_path: str,
) -> list[str]:
    command: list[str] = []
    _append_override(command, "optimization.iterations", args.iterations)
    _append_override(command, "optimization.position_lr_max_steps", args.iterations)
    _append_override(command, "data.images", args.images)
    _append_override(command, "data.resolution", args.resolution)
    _append_override(command, "data.data_device", args.data_device)
    _append_override(command, "data.semantic_path", semantic_path)
    _append_override(command, "data.semantic_confidence", confidence_path)
    _append_override(command, "data.semantic_boundary", boundary_path)
    _append_override(command, "data.eval", args.eval)
    _append_override(command, "data.holdout", args.holdout)
    if args.max_gaussians is not None:
        _append_override(command, "density.max_gaussians", args.max_gaussians)
    for expression in args.overrides:
        command.extend(["--set", expression])
    return command


def _load_config_utils(workdir: Path) -> ModuleType:
    module_path = workdir / "utils" / "config_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_sgw_mip_launcher_config_utils", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load SemanticGaussianWrapping config utilities: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _override_expressions(override_tokens: list[str]) -> list[str]:
    expressions: list[str] = []
    index = 0
    while index < len(override_tokens):
        if override_tokens[index] != "--set" or index + 1 >= len(override_tokens):
            raise ValueError(
                "internal launcher error: malformed config override tokens"
            )
        expressions.append(override_tokens[index + 1])
        index += 2
    return expressions


def validate_fresh_configuration(
    workdir: Path,
    config_path: Path,
    override_tokens: list[str],
) -> dict[str, Any]:
    """Resolve exactly the fresh-run config passed to train.py and fail early."""

    config_utils = _load_config_utils(workdir)
    resolved = config_utils.load_config(
        config_path, _override_expressions(override_tokens)
    )
    config_utils.validate_config(resolved)
    return resolved


def validate_resume_configuration(
    workdir: Path,
    config: Mapping[str, Any],
    saved_iteration: int,
    override_tokens: list[str],
) -> dict[str, Any]:
    """Apply the native resume allowlist and validate the resulting curriculum."""

    config_utils = _load_config_utils(workdir)
    resolved = config_utils.apply_overrides(
        dict(config),
        _override_expressions(override_tokens),
        allowed_keys=RESUME_SAFE_CONFIG_KEYS,
        allowed_prefixes=("logging.",),
    )
    config_utils.validate_config(resolved)
    target_iteration = int(resolved["optimization"]["iterations"])
    if target_iteration < saved_iteration:
        raise ValueError(
            "optimization.iterations cannot precede checkpoint iteration "
            f"{saved_iteration}"
        )
    return resolved


def resume_overrides(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    _append_override(command, "optimization.iterations", args.iterations)
    for expression in args.overrides:
        key = expression.split("=", 1)[0]
        if key not in RESUME_SAFE_CONFIG_KEYS and not key.startswith("logging."):
            raise ValueError(
                f"resume cannot change {key!r}; checkpoints only allow "
                "optimization.iterations, semantic.region_decode_chunk_size, "
                "surface.support_routing_query_chunk, surface.scipy_workers, "
                "surface.mesh_feedback_scipy_workers, "
                "and logging.* overrides"
            )
        command.extend(["--set", expression])
    return command


def build_command(
    source_path: Path,
    model_path: Path,
    config: Path,
    args: argparse.Namespace,
    semantic_path: str,
    confidence_path: str,
    boundary_path: str,
    extra_args: list[str],
    checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(source_path),
        "-m",
        str(model_path),
    ]
    if checkpoint is None:
        command.extend(["--config", str(config)])
        command.extend(
            fresh_overrides(args, semantic_path, confidence_path, boundary_path)
        )
    else:
        command.extend(["--checkpoint", str(checkpoint)])
        command.extend(resume_overrides(args))
    command.extend(["--device", args.device])
    command.extend(extra_args)
    return command


def final_outputs(model_path: Path, iteration: int) -> tuple[Path, Path, Path]:
    return (
        model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_path / f"chkpnt{iteration}.pth",
        model_path / "config.yaml",
    )


def _path_has_entries(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def select_model_path(base_path: Path, force: bool) -> Path:
    """Keep old runs immutable by assigning forced runs an unused directory."""

    if not force or not base_path.exists():
        return base_path
    for index in range(1, 10_000):
        candidate = base_path.with_name(f"{base_path.name}_rerun_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"Cannot allocate an isolated rerun directory beside {base_path}"
    )


def main() -> int:
    args, extra_args = parse_args()
    data_root = resolve_path(args.data_root)
    workdir = validate_sgw_root(resolve_path(args.sgw_root))
    output_root = resolve_path(args.output_root)

    discovery_images = None if args.resume else args.images
    available_scenes = discover_scenes(data_root, discovery_images)
    if args.list_scenes:
        print("\n".join(available_scenes))
        return 0
    if not available_scenes:
        raise RuntimeError(f"No Mip-NeRF 360 COLMAP scenes found in {data_root}")
    scenes = expand_requested_scenes(args.scenes, available_scenes)
    explicit_resume_path = None
    if args.resume_run is not None:
        if len(scenes) != 1:
            raise ValueError("--resume-run requires exactly one resolved scene")
        explicit_resume_path = resolve_resume_run(args.resume_run, output_root)
    config = resolve_config(args.config, workdir)

    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, str]] = []
    semantic_failure = False
    for scene in scenes:
        source_path = data_root / scene
        base_model_path = output_root / scene
        try:
            model_path = (
                explicit_resume_path
                if explicit_resume_path is not None
                else select_model_path(base_model_path, args.force)
            )
            if args.force and model_path != base_model_path:
                print(
                    f"[fresh] {scene}: preserving existing output; using {model_path}"
                )
            elif explicit_resume_path is not None:
                print(f"[resume-run] {scene}: using {model_path}")
            scene_args = args
            if not args.resume:
                outputs = final_outputs(model_path, args.iterations)
                if all(path.is_file() for path in outputs):
                    print(
                        f"[skip] {scene}: complete iteration {args.iterations} already exists"
                        "; existing run settings are authoritative, use --force for a new "
                        "configuration"
                    )
                    continue

            checkpoint = None
            checkpoint_iteration = None
            selection_target = (
                args.iterations
                if not args.resume or args.iterations_explicit
                else sys.maxsize
            )
            _candidate, _candidate_iteration, latest_any = latest_checkpoint(
                model_path, selection_target
            )
            if latest_any is not None and latest_any > selection_target:
                raise FileExistsError(
                    f"checkpoint iteration {latest_any} is newer than requested iteration "
                    f"{selection_target}: {model_path}"
                )
            if args.resume:
                checkpoint, checkpoint_iteration, _ = latest_checkpoint(
                    model_path, selection_target
                )
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"--resume found no checkpoint at or before iteration "
                        f"{selection_target}: {model_path}"
                    )
            elif latest_any is not None:
                raise FileExistsError(
                    f"incomplete output contains checkpoint iteration {latest_any}: {model_path}. "
                    "Use --resume to continue it or --force to create an isolated fresh run."
                )
            elif _path_has_entries(model_path):
                raise FileExistsError(
                    f"non-empty output has no resumable checkpoint: {model_path}. "
                    "Use --force to preserve it and create an isolated fresh run."
                )

            if checkpoint is None:
                images_value = args.images
                requested_semantic = args.semantic_path
                requested_confidence = args.semantic_confidence
                requested_boundary = args.semantic_boundary
            else:
                resume_config, saved_iteration = load_checkpoint_configuration(
                    checkpoint
                )
                scene_args = copy.copy(args)
                scene_args.iterations = resume_target_iteration(
                    args, resume_config, checkpoint
                )
                outputs = final_outputs(model_path, scene_args.iterations)
                if all(path.is_file() for path in outputs):
                    print(
                        f"[skip] {scene}: complete iteration "
                        f"{scene_args.iterations} already exists; checkpoint settings "
                        "are authoritative"
                    )
                    continue
                if checkpoint_iteration == scene_args.iterations:
                    previous, previous_iteration, _ = latest_checkpoint(
                        model_path, scene_args.iterations - 1
                    )
                    if previous is None:
                        raise FileExistsError(
                            f"checkpoint {checkpoint.name} reached the target but final outputs "
                            "are incomplete, and no earlier checkpoint can replay the final "
                            "interval"
                        )
                    checkpoint = previous
                    checkpoint_iteration = previous_iteration
                    previous_config, saved_iteration = load_checkpoint_configuration(
                        checkpoint
                    )
                    previous_target = resume_target_iteration(
                        args, previous_config, checkpoint
                    )
                    if previous_target != scene_args.iterations:
                        raise ValueError(
                            "resume checkpoints disagree on optimization.iterations: "
                            f"{previous_target} != {scene_args.iterations}"
                        )
                    resume_config = previous_config
                (
                    images_value,
                    requested_semantic,
                    requested_confidence,
                    requested_boundary,
                ) = resume_data_options(resume_config, checkpoint)
                validate_resume_configuration(
                    workdir,
                    resume_config,
                    saved_iteration,
                    resume_overrides(scene_args),
                )

            semantic_dir, semantic_value = resolve_observation_directory(
                source_path, scene, requested_semantic
            )
            confidence_dir, confidence_value = resolve_observation_directory(
                source_path, scene, requested_confidence
            )
            boundary_dir, boundary_value = resolve_observation_directory(
                source_path, scene, requested_boundary
            )
            if checkpoint is None:
                validate_fresh_configuration(
                    workdir,
                    config,
                    fresh_overrides(
                        scene_args,
                        semantic_value,
                        confidence_value,
                        boundary_value,
                    ),
                )

            images = image_manifest(source_path / images_value)
            expected_stems = set(images)
            semantic_native_shapes: dict[str, tuple[int, int]] = {}
            present, total = validate_observations(
                semantic_dir,
                expected_stems,
                label="semantic",
                require_info=True,
                expected_shapes=images,
                decoded_shapes=semantic_native_shapes,
            )
            if confidence_dir is not None:
                validate_observations(
                    confidence_dir,
                    expected_stems,
                    label="confidence",
                    require_info=False,
                    expected_shapes=images,
                    native_reference_shapes=semantic_native_shapes,
                )
            if boundary_dir is not None:
                validate_observations(
                    boundary_dir,
                    expected_stems,
                    label="boundary",
                    require_info=False,
                    expected_shapes=images,
                    native_reference_shapes=semantic_native_shapes,
                )

            command = build_command(
                source_path,
                model_path,
                config,
                scene_args,
                semantic_value,
                confidence_value,
                boundary_value,
                extra_args,
                checkpoint,
            )
            if checkpoint is None:
                print(f"[train] {scene}: semantic masks {present}/{total}")
            else:
                print(
                    f"[resume] {scene}: iteration {checkpoint_iteration} -> "
                    f"{scene_args.iterations}; "
                    "checkpoint configuration is authoritative"
                )
            print(shlex.join(command))
            if args.dry_run:
                continue

            if args.force:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.mkdir(exist_ok=False)
            else:
                model_path.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                command, cwd=workdir, env=environment, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"training returned exit code {completed.returncode}"
                )
        except (
            FileNotFoundError,
            FileExistsError,
            ImportError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            failures.append((scene, str(error)))
            semantic_failure |= isinstance(error, ObservationValidationError)
            print(f"[failed] {scene}: {error}", file=sys.stderr)
            if not args.keep_going:
                break

    if failures:
        if len(failures) > 1:
            print("[summary] failed scenes:", file=sys.stderr)
            for scene, message in failures:
                print(f"  - {scene}: {message}", file=sys.stderr)
        if semantic_failure:
            print(
                "Generate associated Gaga masks first with "
                "scripts/gaga/preprocess_gaga_masks_mipnerf360.py and "
                "scripts/gaga/associate_gaga_masks_mipnerf360.py (use its --force flag "
                "to replace stale outputs); corrupt or inconsistent ID maps must be "
                "regenerated.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
