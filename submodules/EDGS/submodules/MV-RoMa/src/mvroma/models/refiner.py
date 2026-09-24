# mvtracktention/models/refiner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialConvNeXtBlock(nn.Module):
    """ConvNeXt-like 2D block with layer scale, residual."""
    def __init__(self, c_in: int, c_out: int, mlp_ratio: float = 4.0, init_scale: float = 1e-5):
        super().__init__()
        hidden = int(mlp_ratio * c_in)
        self.dw = nn.Conv2d(c_in, c_in, 7, padding=3, groups=c_in, bias=True)
        self.pw1 = nn.Conv2d(c_in, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, c_out, 1, bias=True)
        self.short = (nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act(self.pw1(self.dw(x))))
        return y


class ViewAttentionBlock(nn.Module):
    """
    MHSA + MLP over the view axis N (= R+1) at each pixel; permutation-equivariant.
    **No key-bias** (removed).
    """
    def __init__(self, dim: int, num_heads: int, init_scale: float = 1e-5):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C, Hc, Wc)
        returns: (B, N, C, Hc, Wc)
        """
        B, N, C, Hc, Wc = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * Hc * Wc, N, C)  # (BHW, N, C)

        y = self.norm1(x_flat)
        qkv = self.qkv(y).view(y.shape[0], y.shape[1], 3, self.num_heads, self.head_dim) # BHW, N, 3, head, head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3, BHW, head, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BHW, heads, N, N)

        attn = attn.softmax(dim=-1)
        z = attn @ v  # (BHW, heads, N, d)
        z = z.transpose(1, 2).reshape(y.shape[0], N, C)
        z = self.proj(z)

        y = self.norm2(z)
        y = self.mlp(y)
        x_flat = y 

        return x_flat.view(B, Hc, Wc, N, C).permute(0, 3, 4, 1, 2).contiguous()


