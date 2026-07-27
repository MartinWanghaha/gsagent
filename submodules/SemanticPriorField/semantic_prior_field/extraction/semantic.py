"""Semantic state for mesh extraction.

Loads the semantic sidecar checkpoint written during training, derives
per-Gaussian instance labels with the same classifier used at train time,
and expands per-Gaussian quantities to per-pivot arrays matching the
``pivots.view(-1, 3)`` layout of the pivot-based extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from semantic import SemanticHead
from semantic.prior_field import head_logits_per_gaussian


def resolve_semantic_checkpoint(model_path: str, iteration: int) -> Optional[Path]:
    """Locate the semantic sidecar for a given training iteration.

    Falls back to the latest available sidecar if the exact iteration is
    missing (e.g. mesh extraction after a resumed run).
    """
    semantic_dir = Path(model_path) / "semantic"
    exact = semantic_dir / f"semantic_chkpnt{iteration}.pth"
    if exact.is_file():
        return exact
    if not semantic_dir.is_dir():
        return None
    candidates = sorted(
        semantic_dir.glob("semantic_chkpnt*.pth"),
        key=lambda p: int(p.stem.replace("semantic_chkpnt", "") or -1),
    )
    return candidates[-1] if candidates else None


@torch.no_grad()
def load_semantic_state(
    checkpoint_path,
    gaussians,
    device: str = "cuda",
    min_label_confidence: float = 0.0,
) -> dict:
    """Load head + embeddings and derive per-Gaussian labels.

    Embeddings stored in the loaded PLY (joint training) take precedence;
    otherwise the sidecar copy is used. Either way the Gaussian count must
    match the loaded model.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    num_classes = int(payload["num_classes"])
    head = SemanticHead(16, num_classes).to(device)
    head.load_state_dict(payload["head"])

    n_gaussians = gaussians.get_xyz.shape[0]
    if (
        getattr(gaussians, "use_semantic_features", False)
        and gaussians.get_semantic_features.numel() > 0
        and gaussians.get_semantic_features.shape[0] == n_gaussians
    ):
        features = gaussians.get_semantic_features.detach().to(device)
    else:
        features = payload["semantic_features"].to(device=device, dtype=torch.float32)
        if features.shape[0] != n_gaussians:
            raise ValueError(
                "Semantic checkpoint Gaussian count does not match the loaded model: "
                f"{features.shape[0]} vs {n_gaussians}"
            )

    logits = head_logits_per_gaussian(head, features)
    probabilities = torch.softmax(logits, dim=-1)
    top2 = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1)
    labels = top2.indices[:, 0].long()
    if top2.values.shape[-1] > 1:
        confidence = top2.values[:, 0] - top2.values[:, 1]
    else:
        confidence = top2.values[:, 0]
    if min_label_confidence > 0:
        labels = torch.where(
            confidence >= min_label_confidence, labels, torch.full_like(labels, -1)
        )

    return {
        "head": head,
        "features": features,
        "labels": labels,
        "confidence": confidence,
        "num_classes": num_classes,
    }


def expand_per_gaussian_to_pivots(values: torch.Tensor, n_pivots: int) -> torch.Tensor:
    """(N, ...) per-Gaussian tensor -> (N * n_pivots, ...) per-pivot tensor.

    Matches the Gaussian-major ``pivots.view(-1, 3)`` layout used by the
    structured pivot strategies (normals-based, tetra points, searched).
    """
    return values.repeat_interleave(n_pivots, dim=0)
