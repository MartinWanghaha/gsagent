# mvtracktention/models/embedding_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse RoMA utils exactly
from ..romatch.utils.utils import get_grid, get_autocast_params

class TransformerDecoder(nn.Module):
    """
    Minimal lift from RoMA:
      - blocks: a nn.Module taking (B, H*W, C) tokens -> (B, H*W, C)
      - hidden_dim: channel dim of the tokens
      - out_dim: number of output channels per pixel
      - is_classifier: just stored as an attribute (Decoder/Initializer decide how to use it)
    """
    def __init__(
        self,
        blocks: nn.Module,
        hidden_dim: int,
        out_dim: int,
        is_classifier: bool = False,
        *args,
        amp: bool = False,
        pos_enc: bool = True,
        learned_embeddings: bool = False,
        embedding_dim: int = None,
        amp_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]            # kept for API parity; you can ignore/use as you wish
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            assert embedding_dim is not None, "embedding_dim required when learned_embeddings=True"
            pe = torch.empty((1, hidden_dim, embedding_dim, embedding_dim))
            self.learned_pos_embeddings = nn.Parameter(nn.init.kaiming_normal_(pe))

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior: torch.Tensor, features: torch.Tensor, old_stuff: torch.Tensor, new_scale):
        """
        gp_posterior: (B, Dg, H, W)
        features:     (B, Cq, H, W)   (query features at same scale)
        old_stuff:    (B, hidden_dim, H, W) (unused in this minimal lift, kept for API)
        returns:
          out[..., :-1] = either flow logits (cls) or regressed flow (2ch)
          out[..., -1:] = certainty logits (1ch)
        """
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(
            gp_posterior.device, enabled=self.amp, dtype=self.amp_dtype
        )
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            B, C, H, W = gp_posterior.shape
            x = torch.cat((gp_posterior, features), dim=1)   # (B, Dg+Cq, H, W)
            _, C, H, W = x.shape

            # (Optional) position encoding: RoMA passes zeros when pos_enc=False
            _ = get_grid(B, H, W, x.device)  # kept to mimic RoMA API; returned grid not used unless you want it
            if self.learned_embeddings:
                pos_enc = F.interpolate(
                    self.learned_pos_embeddings, size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, H * W, C)
            else:
                pos_enc = 0

            tokens = x.reshape(B, C, H * W).permute(0, 2, 1) + pos_enc   # (B, HW, C)
            z = self.blocks(tokens)                                      # (B, HW, C)
            out = self.to_out(z)                                         # (B, HW, out_dim)
            out = out.permute(0, 2, 1).reshape(B, self.out_dim, H, W)    # (B, out_dim, H, W)

            warp_or_cls, certainty_logits = out[:, :-1], out[:, -1:]
            return warp_or_cls, certainty_logits, None
