"""Versioned semantic sidecar checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

FORMAT_VERSION = 1


def save_semantic_checkpoint(
    output_dir,
    *,
    head,
    gaussian_model,
    iteration: int,
    num_classes: int,
    renderer: str,
    optimizer=None,
    metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"semantic_chkpnt{iteration}.pth"
    payload = {
        "format_version": FORMAT_VERSION,
        "iteration": int(iteration),
        "semantic_dim": int(gaussian_model.semantic_dim),
        "num_classes": int(num_classes),
        "renderer": renderer,
        "compositing": "premultiplied_alpha",
        "gradient_policy": "embedding_only",
        "semantic_features": gaussian_model.get_semantic_features.detach().cpu(),
        "head": head.state_dict(),
        "optimizer": (
            None
            if optimizer is None
            else optimizer.state_dict()
            if hasattr(optimizer, "state_dict")
            else optimizer
        ),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)
    with (output_dir / "semantic_metadata.json").open("w", encoding="utf8") as stream:
        json.dump(
            {
                key: value
                for key, value in payload.items()
                if key not in {"semantic_features", "head", "optimizer"}
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return checkpoint_path


def load_semantic_checkpoint(path, *, head, gaussian_model, optimizer=None) -> dict:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported semantic checkpoint version: {payload.get('format_version')}")
    features = payload["semantic_features"].to(
        device=gaussian_model.get_xyz.device,
        dtype=torch.float32,
    )
    if features.shape != (gaussian_model.get_xyz.shape[0], 16):
        raise ValueError(
            "Semantic checkpoint Gaussian count does not match the loaded PLY: "
            f"{tuple(features.shape)} vs {(gaussian_model.get_xyz.shape[0], 16)}"
        )
    gaussian_model.semantic_dim = 16
    gaussian_model.use_semantic_features = True
    gaussian_model._semantic_features = torch.nn.Parameter(features.requires_grad_(True))
    head.load_state_dict(payload["head"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
