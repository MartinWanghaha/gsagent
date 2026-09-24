import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.transformer import SelfAttentionBlock


class DistanceBiasedCrossAttention(nn.Module):
    """
    Cross attention with distance bias.
    """
    def __init__(self, cfg, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query,             # (B, Nq, C)
        key,               # (B, Nk, C)
        value,             # (B, Nk, C)
        attn_bias,         # (B, Nq, Nk) additive (distance) bias
        kv_positions,      # unused, kept for interface compatibility
        key_padding_mask=None,   # Optional (B, Nk) with 1=keep, 0=mask
        query_mask=None,         # Optional (B, Nq) with 1=keep, 0=mask
        **kwargs,
    ):
        B, Nq, C = query.shape
        Nk = key.shape[1]

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        logits = (q * self.scale) @ k.transpose(-2, -1)
        logits = logits + attn_bias.unsqueeze(1)

        attn = logits.softmax(dim=-1)

        if key_padding_mask is not None:
            km = key_padding_mask.to(dtype=attn.dtype).unsqueeze(1).unsqueeze(2)
            attn = attn * km
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn = attn / denom

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)

        if query_mask is not None:
            qm = query_mask.to(dtype=x.dtype).unsqueeze(-1)
            x = x * qm

        return self.proj(x)


class TrackTokenSampler(nn.Module):
    """
    Samples multi-view features into track tokens with distance-biased cross-attn.
    """
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.v_proj = nn.Identity()
        self.attn = DistanceBiasedCrossAttention(cfg, self.dim, self.num_heads)
        self.register_buffer('sigma_sq', torch.tensor((0.5) ** 2), persistent=False)

    def _calculate_bias(self, track_positions, feature_positions):
        diff = track_positions[:, :, None, :] - feature_positions[:, None, :, :]
        dist_sq = (diff * diff).sum(-1)
        return -dist_sq / (2 * self.sigma_sq)

    def forward(
        self,
        multi_view_features,     # (B*T, NumTokens, D)
        track_tokens,            # (B*T, M, D)
        point_tracks,            # (B*T, M, 2)
        feature_grid_coords,     # (B*T, NumTokens, 2)
        H, W,
        track_mask_bt_m=None,    # (B*T, M)  1=visible, 0=occluded
    ):
        attn_bias = self._calculate_bias(point_tracks, feature_grid_coords)
        return self.attn(
            query=track_tokens,
            key=multi_view_features,
            value=self.v_proj(multi_view_features),
            attn_bias=attn_bias,
            kv_positions=feature_grid_coords,
            key_padding_mask=None,
            query_mask=track_mask_bt_m,
        )


class TrackTokenEncoder(nn.Module):
    """
    Temporal self-attention over T for each track independently.
    """
    def __init__(self, cfg, num_layers=2):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.blocks = nn.ModuleList([SelfAttentionBlock(self.dim, self.num_heads) for _ in range(num_layers)])

    def forward(self, track_tokens, track_mask_btm=None):
        B, T, M, D = track_tokens.shape
        x = track_tokens.permute(0, 2, 1, 3).reshape(B * M, T, D)

        key_mask = None
        if track_mask_btm is not None:
            key_mask = track_mask_btm.permute(0, 2, 1).reshape(B * M, T)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_mask)

        return x.reshape(B, M, T, D).permute(0, 2, 1, 3)


class TrackFeatureSplat(nn.Module):
    """
    Writes updated track tokens back to the feature grid with distance-biased cross-attn.
    """
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.attn = DistanceBiasedCrossAttention(cfg, self.dim, self.num_heads)
        self.output_proj = nn.Linear(self.dim, self.dim)
        self.query_proj = nn.Linear(2, self.dim)

        nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-6)
        if self.output_proj.bias is not None:
            nn.init.normal_(self.output_proj.bias, mean=0.0, std=1e-6)

        self.register_buffer('sigma_sq', torch.tensor((0.5) ** 2), persistent=False)

    def _calculate_bias(self, track_positions, feature_positions):
        diff = feature_positions[:, :, None, :] - track_positions[:, None, :, :]
        dist_sq = (diff * diff).sum(-1)
        return -dist_sq / (2 * self.sigma_sq)

    def forward(
        self,
        updated_track_tokens,    # (B*T, M, D)
        point_tracks,            # (B*T, M, 2)
        feature_grid_coords,     # (B*T, NumTokens, 2)
        H, W,
        track_mask_bt_m=None,    # (B*T, M) 1=visible, 0=occluded
    ):
        attn_bias = self._calculate_bias(point_tracks, feature_grid_coords)
        query_tokens = self.query_proj(feature_grid_coords)
        feature_updates = self.attn(
            query=query_tokens,
            key=updated_track_tokens,
            value=updated_track_tokens,
            attn_bias=attn_bias,
            kv_positions=point_tracks,
            key_padding_mask=track_mask_bt_m,
            query_mask=None,
        )
        return self.output_proj(feature_updates)


class TracktentionBlock(nn.Module):
    """
    End-to-end block with mask support.
    """
    def __init__(self, cfg):
        super().__init__()
        self.sampler = TrackTokenSampler(cfg)
        self.transformer = TrackTokenEncoder(cfg)
        self.splatter = TrackFeatureSplat(cfg)

    def forward(self, multi_view_features, track_tokens, point_tracks, feature_grid_coords, H, W, track_mask=None):
        B, T, NumTokens, D = multi_view_features.shape
        mvf = multi_view_features.view(B * T, NumTokens, D)
        tkn = track_tokens.view(B * T, -1, D)
        pts = point_tracks.view(B * T, -1, 2)
        grid = feature_grid_coords.view(B * T, NumTokens, 2)

        mask_bt_m = None
        if track_mask is not None:
            mask_bt_m = track_mask.reshape(B * T, -1)

        sampled = self.sampler(mvf, tkn, pts, grid, H, W, track_mask_bt_m=mask_bt_m).view(B, T, -1, D)
        updated_track = self.transformer(sampled, track_mask_btm=track_mask)
        upd_flat = updated_track.reshape(B * T, -1, D)
        feat_upd = self.splatter(upd_flat, pts, grid, H, W, track_mask_bt_m=mask_bt_m).view(B, T, NumTokens, D)

        return multi_view_features + feat_upd
