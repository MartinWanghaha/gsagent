from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

@dataclass
class ModelConfig:
    # --- Backbone / tokenizer ---
    img_size: int = 512
    patch_size: int = 16
    dim: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    tracktention_start_block: int = 12
    dpt_hooks: List[int] = field(default_factory=lambda: [11, 15, 19, 23])
    backbone_name: Literal["v2", "v3"] = "v2"
    backbone_norm: bool = True

    #jm add:
    backbone_apply_tracktention: bool = True
    num_cluster: int = 512
    
    # --- Coarse (s=16) head / stride ---
    coarse_stride_px: int = 16
    coarse_feature_dim: int = 1024

    # --- Refiner core knobs (already externalized) ---
    refiner_attn_dim: Dict[str, int] = field(
        # default_factory=lambda: {"16": 512, "8": 256, "4": 256, "2": 64, "1": 16}
        default_factory=lambda: {"16": 512, "8": 512, "4": 256, "2": 64, "1": 16} 
    )
    
    refiner_iters: int = 4
    refiner_use_covis: bool = True

    # --- Refiner per-scale structure ---
    # Order of scales processed coarse->fine:
    refiner_scales: Tuple[str, ...] = ("16", "8", "4", "2", "1")

    # Channels seen by MVRefinerBlock at each scale (post-projector)
    refiner_in_map: Dict[str, int] = field(
        # default_factory=lambda: {"16": 512, "8": 512, "4": 256, "2": 64, "1": 16}
       default_factory=lambda: {"16": 512, "8": 512, "4": 256, "2": 64, "1": 9}
    )

    # Local correlation radius by scale
    refiner_local_corr_radii: Dict[str, int] = field(
        default_factory=lambda: {"16": 7, "8": 3, "4": 2, "2": 0, "1": 0}
    )

    # Displacement embedding channels by scale
    refiner_disp_emb_dim: Dict[str, int] = field(
        default_factory=lambda: {"16": 128, "8": 64, "4": 32, "2": 16, "1": 6}
    )

    # Whether to use AllTracker-style blocks for each stage
    refiner_use_mv_interact: Dict[str, bool] = field(
        default_factory=lambda: {"16": False, "8": True, "4": False, "2": False, "1": True}
    )

    # View-axis attention heads per refiner block & init scale
    refiner_view_heads: int = 4
    refiner_init_scale: float = 1e-5
