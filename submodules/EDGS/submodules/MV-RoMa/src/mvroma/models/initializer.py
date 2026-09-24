from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse RoMA utilities you already have
# If your local path differs, change this import accordingly.
from ..romatch.utils.utils import cls_to_flow_refine


def _norm_to_cells_align_corners_false(flow_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert normalized grid coords (align_corners=False) to *absolute cell coords*.
    flow_norm: (N, 2, H, W) in [-1, 1], sampling *B at A*.
    Returns:   (N, 2, H, W) cell coords in [0, H-1]/[0, W-1].
    Mapping (grid_sample, ac=False): x_pix = ((x_norm + 1) * W - 1) / 2, y_pix analogously.
    """
    x_norm, y_norm = flow_norm[:, 0], flow_norm[:, 1]
    x_cell = ((x_norm + 1.0) * W - 1.0) * 0.5
    y_cell = ((y_norm + 1.0) * H - 1.0) * 0.5
    return torch.stack([x_cell, y_cell], dim=1)


def _coords_grid_cells(H: int, W: int, device, dtype) -> torch.Tensor:
    """(1,2,H,W) integer cell centers (x,y) in [0..W-1]/[0..H-1]."""
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid = torch.stack([xx, yy], dim=0).to(dtype=dtype).unsqueeze(0)
    return grid


class PairwiseGPInitializer(nn.Module):
    """
    Produce the *first* flow & covis logits at the coarse grid using RoMA's GP + embedding decoder.
    - Pairwise over R refs (weights shared).
    - Minimal assumptions: your embedding decoder follows RoMA's call signature:
        (gp_feats, f1, old_stuff, scale_tag) -> (flow_or_cls, certainty_logits, next_state)
      and has .hidden_dim and .is_classifier attributes.
    - Outputs:
        flow_c_init  : (B, R, 2, Hc, Wc)  in *cell-delta* units (for your refiner)
        covis_c_init : (B, R, 1, Hc, Wc)  logits
    """

    def __init__(
        self,
        gp_module: nn.Module,               # Reuse RoMA's GP (CosKernel -> GP)
        embedding_decoder: nn.Module,       # Reuse your RoMA embedding decoder
        scale_tag: str = "1",               # what the decoder expects as "scale"
        detach_outputs: bool = True,        # RoMA-style: stop-grad into the refiner
    ):
        super().__init__()
        self.gp = gp_module
        self.embedding_decoder = embedding_decoder
        self.scale_tag = scale_tag
        self.detach_outputs = detach_outputs

    # @torch.no_grad()  # keep the initializer “frozen” by default; remove if you want it trainable
    def forward(
        self,
        Fq_c: torch.Tensor,               # (B, Cc, Hc, Wc)
        Fr_c: torch.Tensor,               # (B, R, Cc, Hc, Wc)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R, Cc, Hc, Wc = Fr_c.shape
        device, dtype = Fq_c.device, Fq_c.dtype

        # Flatten pairwise across refs -> (B*R, Cc, Hc, Wc)
        f1 = Fq_c.unsqueeze(1).expand(B, R, Cc, Hc, Wc).contiguous().view(B * R, Cc, Hc, Wc)
        f2 = Fr_c.contiguous().view(B * R, Cc, Hc, Wc)

        # 1) GP posterior features (A-anchored)
        #    Reuse RoMA GP exactly; it returns channels (dim may depend on config/basis/cov K)
        gp_feats = self.gp(f1, f2)  # (B*R, Dg, Hc, Wc)

        # 2) Transformer embedding decoder (RoMA)
        #    Prepare old_stuff (state) per decoder’s contract
        hidden_dim = getattr(self.embedding_decoder, "hidden_dim", gp_feats.shape[1])
        old_stuff = torch.zeros(B * R, hidden_dim, Hc, Wc, device=device, dtype=gp_feats.dtype)

        flow_or_cls, certainty_logits, _ = self.embedding_decoder(gp_feats, f1, old_stuff, self.scale_tag)

        # 3) Convert to normalized flow grid (if classifier) -> (B*R, 2, Hc, Wc)
        if getattr(self.embedding_decoder, "is_classifier", False):
            # RoMA utility converts class grid to normalized flow (B, H, W, 2), then permute
            flow_norm = cls_to_flow_refine(flow_or_cls).permute(0, 3, 1, 2)  # (B*R, 2, Hc, Wc)
        else:
            # Assume decoder already returns normalized grid as (B*R, 2, Hc, Wc) or (B*R, Hc, Wc, 2)
            if flow_or_cls.dim() == 4 and flow_or_cls.shape[1] == 2:
                flow_norm = flow_or_cls
            else:
                flow_norm = flow_or_cls.permute(0, 3, 1, 2)

        # 4) Convert normalized flow → absolute cell coords → *cell-delta* (for your refiner)
        # target_cells = _norm_to_cells_align_corners_false(flow_norm, Hc, Wc)  # (B*R,2,Hc,Wc)
        # base_cells = _coords_grid_cells(Hc, Wc, device, target_cells.dtype).expand(B * R, -1, -1, -1)
        # flow_cells_delta = target_cells - base_cells                              # (B*R,2,Hc,Wc)

        # 5) Reshape back to (B,R,...) and detach if requested
        flow_cls = flow_or_cls.view(B, R, 4096, Hc, Wc) # TODO: 4096 is hardcoded here for now.
        flow_c_init  = flow_norm.view(B, R, 2, Hc, Wc) # Flow itself instead of delta flow
        covis_c_init = certainty_logits.view(B, R, 1, Hc, Wc)  # logits already (per your choice)

        # if self.detach_outputs:
        flow_c_init  = flow_c_init.detach()
        covis_c_init_detached = covis_c_init.clone().detach()

        return flow_cls.to(dtype), flow_c_init.to(dtype), covis_c_init.to(dtype), covis_c_init_detached.to(dtype)
