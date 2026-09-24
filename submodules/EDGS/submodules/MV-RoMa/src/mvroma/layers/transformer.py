import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DinoV2Attention(nn.Module):
    """
    Multi-head self-attention with optional masks:
      - key_padding_mask: (B, N) 1=keep, 0=mask  -> post-softmax masking + renorm
      - query_mask:       (B, N) 1=keep, 0=mask  -> output gating
      - attn_mask:        broadcastable to (B, H, N, N) -> additive to logits
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None, query_mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B, H, N, Dh)

        # scaled dot-product logits
        logits = (q @ k.transpose(-2, -1)) * self.scale      # (B, H, N, N)

        # optional additive mask (e.g., causal or bias), broadcastable to (B, H, N, N)
        if attn_mask is not None:
            logits = logits + attn_mask

        # softmax over keys
        attn = logits.softmax(dim=-1)                        # (B, H, N, N)

        # key padding mask: post-softmax masking + renormalization
        if key_padding_mask is not None:
            km = key_padding_mask.to(dtype=attn.dtype).unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            attn = attn * km
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn = attn / denom

        # aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)    # (B, N, C)

        # query gating
        if query_mask is not None:
            qm = query_mask.to(dtype=out.dtype).unsqueeze(-1)   # (B, N, 1)
            out = out * qm

        return self.proj(out)


class SelfAttentionBlock(nn.Module):
    """
    DinoV2-style Transformer block with LayerScale & DropPath.
    Supports masks passed through to attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, init_values=1e-5, drop_path=0.0):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = DinoV2Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=init_values)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, key_padding_mask=None, query_mask=None, attn_mask=None):
        # Attention (with masks)
        x = x + self.drop_path1(self.ls1(
            self.attn(self.norm1(x),
                      key_padding_mask=key_padding_mask,
                      query_mask=query_mask,
                      attn_mask=attn_mask)
        ))
        # MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
