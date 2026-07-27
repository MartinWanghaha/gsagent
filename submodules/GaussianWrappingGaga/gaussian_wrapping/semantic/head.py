"""Scene-level decoder for rendered Gaga embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SemanticHead(nn.Module):
    def __init__(self, semantic_dim: int, num_classes: int) -> None:
        super().__init__()
        if semantic_dim != 16:
            raise ValueError("The native CUDA compositor currently supports 16 channels")
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one object")
        self.semantic_dim = int(semantic_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Conv2d(
            self.semantic_dim,
            self.num_classes,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squeeze = features.ndim == 3
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 4 or features.shape[1] != self.semantic_dim:
            raise ValueError(
                f"Expected [B,{self.semantic_dim},H,W], got {tuple(features.shape)}"
            )
        logits = self.classifier(features)
        return logits[0] if squeeze else logits
