"""Shared Mip-NeRF 360 launcher utilities for GaussianWrappingGaga."""

from __future__ import annotations

import json
import importlib
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = GSAGENT_ROOT / "submodules" / "GaussianWrappingGaga"
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_OUTPUT_ROOT = GSAGENT_ROOT / "outputs" / "gaussian_wrapping_gaga_mipnerf360"

BENCHMARK_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
)
MASK_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_colmap_scene(scene_dir: Path, images: str = "images") -> bool:
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = scene_dir / "sparse"
    return (
        (scene_dir / images).is_dir()
        and any((sparse_dir / name).is_file() for name in ("cameras.bin", "cameras.txt"))
        and any((sparse_dir / name).is_file() for name in ("images.bin", "images.txt"))
        and any(
            (sparse_dir / name).is_file()
            for name in ("points3D.bin", "points3D.txt")
        )
    )


def discover_scenes(data_root: Path, images: str = "images") -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Mip-NeRF 360 data root does not exist: {data_root}")
    return [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir() and is_colmap_scene(path, images)
    ]


def select_scenes(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    if not requested:
        missing = [scene for scene in BENCHMARK_SCENES if scene not in available]
        if missing:
            raise FileNotFoundError(
                "Missing default benchmark scene(s): " + ", ".join(missing)
            )
        return list(BENCHMARK_SCENES)
    expanded = [
        part.strip()
        for value in requested
        for part in value.split(",")
        if part.strip()
    ]
    if any(scene.lower() == "all" for scene in expanded):
        return available
    missing = sorted(set(expanded) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. "
            f"Available: {', '.join(available)}"
        )
    requested_set = set(expanded)
    return [scene for scene in available if scene in requested_set]


def validate_project(project_root: Path) -> Path:
    workdir = project_root / "gaussian_wrapping"
    required = (
        workdir / "train.py",
        workdir / "semantic_lift.py",
        workdir / "semantic_mesh.py",
        workdir / "pivot_based_mesh_extraction.py",
        project_root / "submodules" / "diff-gaussian-rasterization-semantic",
        project_root / "submodules" / "diff-gaussian-rasterization_ours-semantic",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "GaussianWrappingGaga is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    return workdir


def validate_semantic_extension(project_root: Path, rasterizer: str) -> None:
    packages = {
        "radegs": (
            "diff_gaussian_rasterization_gw_semantic",
            project_root / "submodules" / "diff-gaussian-rasterization-semantic",
        ),
        "ours": (
            "diff_gaussian_rasterization_gw_ours_semantic",
            project_root
            / "submodules"
            / "diff-gaussian-rasterization_ours-semantic",
        ),
    }
    package, extension_root = packages[rasterizer]
    installed_error: Exception | None = None
    if importlib.util.find_spec(package) is not None:
        try:
            importlib.import_module(package)
            return
        except (ImportError, ModuleNotFoundError) as error:
            installed_error = error
    local_package = extension_root / package
    if local_package.is_dir():
        extension_root_string = str(extension_root)
        if extension_root_string not in sys.path:
            sys.path.insert(0, extension_root_string)
        sys.modules.pop(package, None)
        try:
            importlib.import_module(package)
            return
        except (ImportError, ModuleNotFoundError) as error:
            installed_error = error
    detail = f"\nImport error: {installed_error}" if installed_error else ""
    raise RuntimeError(
        f"Native {rasterizer} semantic rasterizer is not built. Install it "
        f"with:\n  {sys.executable} -m pip install --no-build-isolation "
        f"{extension_root}{detail}"
    )


def validate_masks(
    scene_dir: Path,
    mask_method: str,
    images: str,
    *,
    allow_missing: bool,
) -> tuple[Path, int]:
    image_dir = scene_dir / images
    mask_dir = scene_dir / f"{mask_method}_mask"
    info_path = mask_dir / "info.json"
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Associated Gaga masks do not exist: {mask_dir}. "
            "Run scripts/gaga/associate_gaga_masks_mipnerf360.py first."
        )
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing Gaga mask metadata: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON metadata: {info_path}") from error
    num_masks = info.get("num_mask", info.get("num_instances"))
    if not isinstance(num_masks, int) or num_masks < 1:
        raise ValueError(f"Invalid num_mask in {info_path}: {num_masks!r}")

    image_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    mask_stems = {
        path.stem
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES
    }
    missing = sorted(image_stems - mask_stems)
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{mask_dir} is missing {len(missing)} camera mask(s): "
            f"{preview}{suffix}"
        )
    return mask_dir, num_masks + 1


def run_directory(
    output_root: Path,
    scene: str,
    mode: str,
    mask_method: str,
) -> Path:
    return output_root / scene / mode / mask_method


def model_directories(run_dir: Path, mode: str) -> tuple[Path, Path]:
    if mode == "two-stage":
        return run_dir / "geometry", run_dir / "semantic"
    model_dir = run_dir / "model"
    return model_dir, model_dir


def point_cloud_path(model_dir: Path, iteration: int) -> Path:
    return (
        model_dir
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )


def semantic_checkpoint_path(model_dir: Path, iteration: int) -> Path:
    return model_dir / "semantic" / f"semantic_chkpnt{iteration}.pth"


def available_point_cloud_iterations(model_dir: Path) -> list[int]:
    root = model_dir / "point_cloud"
    if not root.is_dir():
        return []
    result = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.removeprefix("iteration_"))
        except ValueError:
            continue
        if point_cloud_path(model_dir, iteration).is_file():
            result.append(iteration)
    return sorted(result)


def resolve_iteration(model_dir: Path, requested: int) -> int:
    available = available_point_cloud_iterations(model_dir)
    if not available:
        raise FileNotFoundError(f"No complete point cloud found in {model_dir}")
    if requested == -1:
        return available[-1]
    if requested not in available:
        raise FileNotFoundError(
            f"Iteration {requested} is unavailable in {model_dir}; "
            f"available: {available}"
        )
    return requested


def available_semantic_iterations(model_dir: Path) -> list[int]:
    return [
        iteration
        for iteration in available_point_cloud_iterations(model_dir)
        if semantic_checkpoint_path(model_dir, iteration).is_file()
    ]


def resolve_semantic_iteration(model_dir: Path, requested: int) -> int:
    available = available_semantic_iterations(model_dir)
    if not available:
        raise FileNotFoundError(
            f"No matching semantic PLY/checkpoint pair found in {model_dir}"
        )
    if requested == -1:
        return available[-1]
    if requested not in available:
        raise FileNotFoundError(
            f"Semantic iteration {requested} is unavailable in {model_dir}; "
            f"available: {available}"
        )
    return requested


def print_command(stage: str, command: Iterable[str]) -> None:
    print(f"[{stage}] {shlex.join(list(command))}")


def execute(
    command: list[str],
    *,
    stage: str,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    print_command(stage, command)
    if dry_run:
        return 0
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
    ).returncode


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
