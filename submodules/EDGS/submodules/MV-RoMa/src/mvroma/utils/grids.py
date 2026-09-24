import torch
from uniflowmatch.utils.geometry import get_meshgrid_torch

def build_pixel_grid(H: int, W: int, device) -> torch.Tensor:
    """Return pixel grid (2,H,W) with x/y coordinates."""
    grid = get_meshgrid_torch(W=W, H=H, device=device).permute(2, 0, 1)  # (2,H,W)
    return grid

def build_patch_token_centers_px(H: int, W: int, patch: int, device) -> torch.Tensor:
    """Tensor of patch-center coordinates in pixels: (N,2)."""
    ys = (torch.arange(H // patch, device=device) * patch + patch / 2.0)
    xs = (torch.arange(W // patch, device=device) * patch + patch / 2.0)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (N,2)
    return coords

def build_token_center_grid(B: int, T: int, H: int, W: int, patch: int, device) -> torch.Tensor:
    """Replicate patch token centers into (B,T,N,2)."""
    grid = build_patch_token_centers_px(H, W, patch, device)
    return grid.view(1, 1, -1, 2).repeat(B, T, 1, 1)
