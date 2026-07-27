"""Checkpoint-only runtime context for region-conditioned Gaussian wrapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import torch

from .cameras import MeshCamera


_CHECKPOINT_PATTERN = re.compile(r"^chkpnt(\d+)\.pth$")


def checkpoint_iterations(model_path: str | Path) -> list[int]:
    """Return iterations backed by complete native checkpoints."""

    root = Path(model_path)
    return sorted(
        int(match.group(1)) for path in root.glob("chkpnt*.pth") if (match := _CHECKPOINT_PATTERN.fullmatch(path.name))
    )


def resolve_checkpoint_iteration(model_path: str | Path, iteration: int) -> int:
    """Resolve one mesh iteration without accepting PLY-only snapshots."""

    available = checkpoint_iterations(model_path)
    if not available:
        raise FileNotFoundError(f"mesh extraction requires chkpnt*.pth under {Path(model_path).resolve()}")
    if int(iteration) == -1:
        return available[-1]
    if int(iteration) not in available:
        raise FileNotFoundError(f"checkpoint iteration {iteration} is unavailable; available iterations: {available}")
    return int(iteration)


@dataclass(frozen=True)
class MeshExtractionContext:
    """Minimal strong ownership required by the wrapping pipeline."""

    model_path: Path
    checkpoint_path: Path
    iteration: int
    device: torch.device
    gaussians: Any
    cameras: tuple[MeshCamera, ...]
    pipeline: Any
    semantic_decoder: Any
    experiment_config: dict[str, Any]
    scene_extent: float
    background: torch.Tensor

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        iteration: int = -1,
        device: str | torch.device = "cuda",
    ) -> "MeshExtractionContext":
        """Load inference tensors and calibrated training cameras exactly once."""

        from model_io import load_trained_scene

        root = Path(model_path).expanduser().resolve()
        selected = resolve_checkpoint_iteration(root, int(iteration))
        bundle = load_trained_scene(
            root,
            iteration=selected,
            device=device,
            with_surface_field=False,
            inference_scope="surface",
        )
        checkpoint_path = root / f"chkpnt{selected}.pth"
        if bundle.get("checkpoint_path") is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"iteration {selected} has no self-contained checkpoint")
        scene = bundle["scene"]
        cameras = tuple(
            MeshCamera.from_camera(camera)
            for camera in sorted(
                scene.getTrainCameras(),
                key=lambda value: (int(value.uid), str(value.image_name)),
            )
        )
        if not cameras:
            raise ValueError("mesh extraction requires at least one training camera")
        semantic_decoder = bundle["semantic_decoder"]
        if semantic_decoder is None:
            raise RuntimeError("mesh extraction requires semantic decoder weights from a checkpoint")
        scene_extent = float(scene.extent)
        if not math.isfinite(scene_extent) or scene_extent <= 0.0:
            raise ValueError("scene extent must be finite and positive")
        target_device = torch.device(bundle["device"])
        white = bool(bundle["config"]["model"].get("white_background", False))
        background = torch.full(
            (3,),
            1.0 if white else 0.0,
            dtype=torch.float32,
            device=target_device,
        )
        return cls(
            model_path=root,
            checkpoint_path=checkpoint_path,
            iteration=selected,
            device=target_device,
            gaussians=bundle["gaussians"],
            cameras=cameras,
            pipeline=bundle["pipeline"],
            semantic_decoder=semantic_decoder,
            experiment_config=bundle["config"],
            scene_extent=scene_extent,
            background=background,
        )


__all__ = [
    "MeshExtractionContext",
    "checkpoint_iterations",
    "resolve_checkpoint_iteration",
]
