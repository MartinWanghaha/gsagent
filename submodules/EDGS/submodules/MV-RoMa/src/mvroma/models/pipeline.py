# mvtracktention/models/pipeline.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..tracktention import TracktentionBlock
from .initializer import PairwiseGPInitializer
from .embedding_decoder import TransformerDecoder
from .refiner import ViewAttentionBlock
from .roma_convrefiner import ConvRefiner

# RoMA components
from ..romatch.models.matcher import GP, CosKernel
from ..romatch.models.transformer import Block, MemEffAttention
from ..romatch.models.encoders import VGG19
from ..romatch.utils.local_correlation import local_correlation


# ---------------------------
# Helpers
# ---------------------------

def _assert_int(x: int, name: str):
    assert isinstance(x, int), f"{name} must be int"

def unpack_query_ref_point_pairs(
    paired_tracks: torch.Tensor,
) -> torch.Tensor:
    """
    Convert paired coords (B, T-1, M, 4) -> per-frame tracks (B, T, M, 2).

    Input (paired_tracks):
        [:, t-1, m] = (u_q, v_q, u_r, v_r)   # query is frame-0, refs are frames 1..T-1

    Args:
        paired_tracks: (B, T-1, M, 4)

    Returns:
        point_tracks: (B, T, M, 2) where
            point_tracks[:, 0]   = (u_q, v_q)
            point_tracks[:, 1:]  = (u_r, v_r) for each ref frame
    """
    if paired_tracks.ndim != 4 or paired_tracks.shape[-1] != 4:
        raise ValueError(f"paired_tracks must be (B,T-1,M,4); got {tuple(paired_tracks.shape)}")

    B, R, M, _ = paired_tracks.shape

    # Query coords (assumed duplicated across refs)
    q0 = paired_tracks[:, 0, :, :2]                            # (B, M, 2)

    # Reference coords (per ref frame)
    refs = paired_tracks[:, :, :, 2:]                          # (B, R, M, 2)

    # Concatenate: frame-0 is query, then each ref
    point_tracks = torch.cat([q0.unsqueeze(1), refs], dim=1)   # (B, R+1, M, 2) = (B, T, M, 2)
    return point_tracks.contiguous()

def normalize_xy(xy, H, W, align_corners=False):
    x, y = xy[..., 0], xy[..., 1]
    if align_corners:
        x = x * (2.0 / (W - 1)) - 1.0
        y = y * (2.0 / (H - 1)) - 1.0
    else:
        x = (x + 0.5) * (2.0 / W) - 1.0
        y = (y + 0.5) * (2.0 / H) - 1.0
    return torch.stack((x, y), dim=-1)

def denormalize_xy(xy_norm, H, W, align_corners=False):
    x_n, y_n = xy_norm[..., 0], xy_norm[..., 1]
    if align_corners:
        x = (x_n + 1.0) * (W - 1) / 2.0
        y = (y_n + 1.0) * (H - 1) / 2.0
    else:
        x = (x_n + 1.0) * (W / 2.0) - 0.5
        y = (y_n + 1.0) * (H / 2.0) - 0.5
    return torch.stack((x, y), dim=-1)



# ---------------------------
# RoMA-style multi-view refiner (with x_hat tokens, no motion encoder)
# ---------------------------

class MVRefinerBlock(nn.Module):
    """
    Multi-view refinement with N_iter * (ViewAttn → Spatial) updates.

    Inputs:
      f_q:   (B, C_in, H, W)         per-scale query features (after per-scale proj)
      f_r:   (B, R, C_in, H, W)      per-scale ref features (after per-scale proj)
      flow:  (B, R, 2, H, W)         normalized coordinates [-1,1] (A→B_r)
      covis: (B, R, 1, H, W)         covisibility logits (required)
      scale_factor: float            scale used in displacement embedding (e.g., upsampling factor)

    Returns:
      flow':  (B, R, 2, H, W)
      covis': (B, R, 1, H, W)
    """
    def __init__(
        self,
        *,
        scale_tag: str,                  # "16","8","4","2","1"
        in_ch: int,                      # per-scale feature channels (after external proj)
        attn_dim: int = 256,             # channels for h/tokens
        n_iter: int = 2,
        num_heads: int = 4,
        init_scale: float = 1e-5,
        # Head ingredients (per scale):
        disp_emb_dim: int = 128,         # e.g., s16:128, s8:64, s4:32, s2:16, s1:6
        local_corr_radius: int = 0,      # e.g., s16:7, s8:3, s4:2, else 0
        disp_scale_coeff: float = 40.0/32.0,
        # RoMA head tower hyperparams:
        head_kernel_size: int = 5,
        head_hidden_blocks: int = 8,
        head_dw: bool = True,
        head_hidden_ch: int | None = None,   # default mirrors head_in_ch
        bn_momentum: float = 0.01,
        use_mv_interact: bool = True,     # whether to use AllTracker-style blocks
    ) -> None:
        super().__init__()
        assert n_iter >= 1, "n_iter must be >= 1"
        self.scale_val = int(scale_tag)
        self.in_ch = in_ch
        self.attn_dim = attn_dim
        self.n_iter = n_iter
        self.local_corr_radius = int(local_corr_radius) if local_corr_radius else 0
        self.disp_scale_coeff = float(disp_scale_coeff)
        self.bn_momentum = bn_momentum
        self.use_mv_interact = use_mv_interact

        # ----- shared init proj: concat([feature, covis]) → attn_dim
        self.init_proj = nn.Conv2d(in_ch + 1, attn_dim, kernel_size=1, stride=1, padding=0)

        # ----- head pieces (RoMA-style tower on h-space)
        self.disp_emb = nn.Conv2d(2, disp_emb_dim, kernel_size=1, stride=1, padding=0)
        self._head_lc_ch = (2 * self.local_corr_radius + 1) ** 2 if self.local_corr_radius > 0 else 0
        # head uses attn_dim features from h-space
        self.head_in_ch = (attn_dim + attn_dim) + disp_emb_dim + self._head_lc_ch
        self.head_hidden_ch = head_hidden_ch if head_hidden_ch is not None else self.head_in_ch

        self.head_block1 = self._make_block(
            in_dim=self.head_in_ch, out_dim=self.head_hidden_ch,
            dw=head_dw, kernel_size=head_kernel_size, bias=True, norm_type=nn.BatchNorm2d,
        )
        self.head_hidden_blocks = nn.Sequential(*[
            self._make_block(
                in_dim=self.head_hidden_ch, out_dim=self.head_hidden_ch,
                dw=head_dw, kernel_size=head_kernel_size, bias=True, norm_type=nn.BatchNorm2d,
            ) for _ in range(head_hidden_blocks)
        ])
        self.head_out = nn.Conv2d(self.head_hidden_ch, 3, kernel_size=1, stride=1, padding=0)

    # ---------- helpers ----------
    def _make_block(
        self, *, in_dim: int, out_dim: int, dw: bool, kernel_size: int, bias: bool, norm_type=nn.BatchNorm2d
    ) -> nn.Sequential:
        groups = in_dim if dw else 1
        if dw:
            assert out_dim % in_dim == 0, "out_dim must be divisible by in_dim for depthwise"
        conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2, groups=groups, bias=bias)
        norm = norm_type(out_dim, momentum=self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels=out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        return nn.Sequential(conv1, norm, relu, conv2)

    @staticmethod
    def _warp(flow: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Warp feature map with normalized flow.
        flow: (B,R,2,H,W)
        feat: (B,R,C,H,W)  (or (B,1,C,H,W) broadcastable along R)
        returns warped feat: (B,R,C,H,W)
        """
        B, R, _, H, W = flow.shape
        if feat.dim() == 5 and feat.shape[1] == 1:
            feat = feat.expand(B, R, feat.shape[2], H, W)
        C = feat.shape[2]
        feat_flat = feat.reshape(B * R, C, H, W)
        grid = flow.reshape(B * R, 2, H, W).permute(0, 2, 3, 1)
        out = F.grid_sample(feat_flat, grid, align_corners=False, mode="bilinear")
        return out.reshape(B, R, C, H, W)

    @staticmethod
    def _normalized_disp(delta_flow: torch.Tensor, size_hw: Tuple[int, int], scale_val: int) -> torch.Tensor:
        Hs, Ws = size_hw
        disp_x = delta_flow[:, :, 0] / 400
        disp_y = delta_flow[:, :, 1] / 400
        return scale_val * torch.stack((disp_x, disp_y), dim=2)

    def _build_head_features(
        self,
        h_all: torch.Tensor,        # (B,1+R,A,H,W)  A=attn_dim
        flow: torch.Tensor,         # (B,R,2,H,W)
        scale_factor: float,
    ) -> torch.Tensor:
        """
        Head input on REF views from h-space:
          x     = h_q  (first map of h_all) replicated across refs
          y_ref = h_refs (remaining maps of h_all; in ref frame)
          x_hat = warp(y_ref, flow)
          emb   = disp_emb(coeff * scale_factor * (flow - coords))
          lc    = local_correlation(x, y_ref, flow, radius)   # if radius > 0
        Returns: (B,R, head_in_ch, H, W)
        """
        B, V, A, H, W = h_all.shape
        assert V >= 2, "h_all must contain query + at least one ref"
        R = V - 1
        h_q   = h_all[:, 0, :, :, :].contiguous()                 # (B,A,H,W)
        h_ref = h_all[:, 1:, :, :, :].contiguous()                # (B,R,A,H,W)

        x     = h_q.unsqueeze(1).expand(B, R, A, H, W)            # (B,R,A,H,W)
        x_hat = h_ref # self._warp(flow, h_ref)                           # (B,R,A,H,W)

        # displacement embedding
        xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=h_all.device)
        ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=h_all.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack((gx, gy), dim=0)[None, None].expand(B, R, 2, H, W)  # (B,R,2,H,W)
        in_disp = flow - coords
        emb = self.disp_emb(self.disp_scale_coeff * scale_factor * in_disp.reshape(B * R, 2, H, W)) \
                  .reshape(B, R, -1, H, W)                             # (B,R,disp_emb_dim,H,W)

        # local correlation in the *other* image frame (h-space)
        if self.local_corr_radius > 0:
            lc = local_correlation(
                x.reshape(B * R, A, H, W),
                h_ref.reshape(B * R, A, H, W),
                local_radius=self.local_corr_radius,
                flow=None, #flow.reshape(B * R, 2, H, W),
                sample_mode="bilinear",
            ).reshape(B, R, -1, H, W)                               # (B,R,K^2,H,W)
            head_in = torch.cat([x, x_hat, emb, lc.detach()], dim=2)
        else:
            head_in = torch.cat([x, x_hat, emb], dim=2)

        return head_in  # (B,R, head_in_ch, H, W)
        
    # ---------- forward ----------
    def forward(
        self,
        f_q: torch.Tensor,            # (B,in_ch,H,W)
        f_r: torch.Tensor,            # (B,R,in_ch,H,W)
        flow: torch.Tensor,           # (B,R,2,H,W)
        covis: torch.Tensor,          # (B,R,1,H,W)   (required)
        scale_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R, C, H, W = f_r.shape
        assert f_q.shape == (B, C, H, W)
        assert flow.shape == (B, R, 2, H, W)

        # h_0 with shared init_proj on concat([feature, covis])
        cov_q = torch.ones(B, 1, H, W, device=f_q.device, dtype=f_q.dtype)        
        h_q0 = self.init_proj(torch.cat([f_q, cov_q], dim=1))            # (B,A,H,W)

        cov_r0 = torch.ones(B, R, H, W, device=f_q.device, dtype=f_q.dtype)

        # start from warped refs (feature space) for the initial ref encoding
        x_hat0 = self._warp(flow, f_r)                                   # (B,R,C,H,W)
        h_r0 = self.init_proj(torch.cat([x_hat0.reshape(B * R, C, H, W),
                                          cov_r0.reshape(B * R, 1, H, W)], dim=1)) \
                          .reshape(B, R, self.attn_dim, H, W)            # (B,R,A,H,W)

        tokens = torch.cat([h_q0.unsqueeze(1), h_r0], dim=1)              # (B,1+R,A,H,W)

        # Head (once) on REF views; inputs come from h_all
        head_in = self._build_head_features(tokens, flow, scale_factor)   # (B,R,head_in_ch,H,W)
        z = self.head_block1(head_in.reshape(B * R, self.head_in_ch, H, W))
        z = self.head_hidden_blocks(z)
        out = self.head_out(z).reshape(B, R, 3, H, W)                    # (B,R,3,H,W)

        dflow = out[:, :, 0:2, :, :]
        dcov  = out[:, :, 2:3, :, :]

        # Updates
        disp = self._normalized_disp(dflow, (H, W), self.scale_val)
        flow = flow + disp
        covis = covis + dcov

        return flow, covis




# pipeline.py (replace the class __init__ with this simplified form)
class RomaMultiViewRefinerMV(nn.Module):
    """Multi-scale refiner. All hyperparams/scales come from cfg."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Simple unpack from cfg
        self.scales = tuple(cfg.refiner_scales)[:1]
        self.attn_dim = cfg.refiner_attn_dim
        self.n_iter = cfg.refiner_iters
        self.use_covis = cfg.refiner_use_covis

        # Per-scale maps straight from config
        self.in_map = cfg.refiner_in_map
        self.local_corr_radii = cfg.refiner_local_corr_radii
        self.disp_emb_dim = cfg.refiner_disp_emb_dim
        
        self.amp_dtype = torch.float32

        # Build per-scale blocks using the maps above
        s='16'
        self.blocks = nn.ModuleDict({
            s: MVRefinerBlock(
                scale_tag=s,
                in_ch=self.in_map[s],
                local_corr_radius=self.local_corr_radii[s],
                attn_dim=self.attn_dim[s],
                n_iter=self.n_iter,
                disp_emb_dim=self.disp_emb_dim[s],
                num_heads=cfg.refiner_view_heads,
                init_scale=cfg.refiner_init_scale,
                use_mv_interact=cfg.refiner_use_mv_interact[s],
            )
        })
        
    # forward() stays the same as your current implementation


    def forward(
        self,
        *,
        Fq_pyr: Dict[str, torch.Tensor],
        Fr_pyr: Dict[str, torch.Tensor],
        flow16: torch.Tensor,
        covis16: torch.Tensor,
        gm_cls: Optional[torch.Tensor] = None,
        gm_certainty: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        flow_ref: Dict[str, torch.Tensor] = {}
        covis_ref: Dict[str, torch.Tensor] = {}

        for i, s in enumerate(self.scales):
            f_q = Fq_pyr[s]
            f_r = Fr_pyr[s]
            B, R, _, Hs, Ws = f_r.shape

            if i == 0:
                fl_in = flow16.detach()
                cv_in = covis16.detach()
            else:
                prev = self.scales[i - 1]
                fl_in = F.interpolate(
                    flow_ref[prev].reshape(-1, 2, *flow_ref[prev].shape[-2:]),
                    size=(Hs, Ws), mode="bilinear", align_corners=False
                ).reshape(B, R, 2, Hs, Ws).detach()
                cv_in = F.interpolate(
                    covis_ref[prev].reshape(-1, 1, *covis_ref[prev].shape[-2:]),
                    size=(Hs, Ws), mode="bilinear", align_corners=False
                ).reshape(B, R, 1, Hs, Ws).detach()

            fl_out, cv_out = self.blocks[s](f_q, f_r, fl_in, cv_in)
            flow_ref[s]  = fl_out
            covis_ref[s] = cv_out

        corresps: Dict[int, Dict[str, torch.Tensor]] = {}
        for s in self.scales:
            scale_dict: Dict[str, torch.Tensor] = {
                "flow":      flow_ref[s],
                "certainty": covis_ref[s],
            }
            if s == "16":
                if gm_cls is not None:
                    scale_dict["gm_cls"] = gm_cls
                if gm_certainty is not None:
                    scale_dict["gm_certainty"] = gm_certainty
            corresps[int(s)] = scale_dict

        return corresps


# ---------------------------
# Per-scale projection (shared weights for query & refs, RoMA targets)
# ---------------------------

class SharedPerScaleProjector(nn.Module):
    """Projects raw per-scale features to RoMA target channels.
    Targets: 16→512, 8→512, 4→256, 2→64, 1→9
    Weights are shared across views and between query/ref.
    """
    def __init__(self, Cc_coarse: int) -> None:
        super().__init__()
        self.targets = {"16": 512, "8": 512, "4": 256, "2": 64, "1": 16}
        # Define per-scale conv+bn. Input channels per scale follow VGG19 BN: 16 uses coarse fuser (Cc_coarse),
        # 8→512, 4→256, 2→128, 1→64
        self.proj = nn.ModuleDict({
            "16": nn.Sequential(nn.Conv2d(Cc_coarse, 512, 1, 1), nn.BatchNorm2d(512)),
        })

    def forward_query(self, Fq_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for s, x in Fq_raw.items():
            out[s] = self.proj[s](x)
        return out

    def forward_refs(self, Fr_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for s, x in Fr_raw.items():
            B, R, C, H, W = x.shape
            xr = x.reshape(B * R, C, H, W)
            yr = self.proj[s](xr)
            out[s] = yr.reshape(B, R, -1, H, W)
        return out



class UpsampleRomaMultiViewRefinerMV(nn.Module):
    """Multi-scale refiner. All hyperparams/scales come from cfg."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Simple unpack from cfg
        self.scales = tuple(cfg.refiner_scales)
        self.attn_dim = cfg.refiner_attn_dim
        self.n_iter = cfg.refiner_iters
        self.use_covis = cfg.refiner_use_covis

        # Per-scale maps straight from config
        self.in_map = cfg.refiner_in_map
        self.local_corr_radii = cfg.refiner_local_corr_radii
        self.disp_emb_dim = cfg.refiner_disp_emb_dim
        
        self.amp_dtype = torch.float32

        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True

        proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
        
        self.proj = nn.ModuleDict(
            {
                "8": proj8,
                "4": proj4,
                "2": proj2,
                "1": proj1,
            }
        )

        self.conv_refiner = nn.ModuleDict(
            {
                str(s): ConvRefiner(
                    2 * self.in_map[s] + self.disp_emb_dim[s] if self.local_corr_radii[s]==0 else 2 * self.in_map[s] + self.disp_emb_dim[s] + (2* self.local_corr_radii[s]+1)**2, # in_dim
                    2 * self.in_map[s] + self.disp_emb_dim[s] if self.local_corr_radii[s]==0 else 2 * self.in_map[s] + self.disp_emb_dim[s] + (2* self.local_corr_radii[s]+1)**2, # hidden_dim
                    2+1, # out_dim
                    kernel_size=kernel_size, # 5
                    dw=True,
                    hidden_blocks=hidden_blocks, # 8
                    displacement_emb=displacement_emb, # True
                    displacement_emb_dim=self.disp_emb_dim[s],
                    local_corr_radius=self.local_corr_radii[s],
                    corr_in_other=(self.local_corr_radii[s]!=0),
                    amp=True,
                    disable_local_corr_grad=disable_local_corr_grad,
                    bn_momentum = 0.01,
                    use_mv_interact=cfg.refiner_use_mv_interact[s],
                    attn_dim=cfg.refiner_attn_dim[s],
                    refine_iters=cfg.refiner_iters if str(s)=='8' else 2,
                    in_channel=self.in_map[s]              
                ) for s in self.scales
            }
        )

        
    # forward() stays the same as your current implementation


    def forward(
        self,
        *,
        Fq_pyr: Dict[str, torch.Tensor],
        Fr_pyr: Dict[str, torch.Tensor],
        inference: bool = False,
        input_flow16=None, input_covis16=None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Multi-scale refinement from s=16→8→4→2→1.

        Returns:
            corresps: {
              16: {'flow': (B,R,2,H16,W16), 'certainty': (B,R,1,H16,W16)},
               8: {'flow': (B,R,2,H8 ,W8 ), 'certainty': (B,R,1,H8 ,W8 )},
               4: {'flow': (B,R,2,H4 ,W4 ), 'certainty': (B,R,1,H4 ,W4 )},
               2: {'flow': (B,R,2,H2 ,W2 ), 'certainty': (B,R,1,H2 ,W2 )},
               1: {'flow': (B,R,2,H1 ,W1 ), 'certainty': (B,R,1,H1 ,W1 )},
            }
        """
        # Store refined tensors here (never mutate the inputs)
        flow_ref: Dict[str, torch.Tensor] = {}
        covis_ref: Dict[str, torch.Tensor] = {}

        for i, s in enumerate(self.scales):
            if i == 0:
                flow_ref['16'] = input_flow16
                covis_ref['16'] = input_covis16

            else:
                f_q = Fq_pyr.pop(s)          # free after use
                f_r = Fr_pyr.pop(s)
                B, R, _, Hs, Ws = f_r.shape

                f_q = self.proj[s](f_q)
                f_r = self.proj[s](f_r.view(B*R, -1, Hs, Ws)).view(B, R, -1, Hs, Ws)

                prev = self.scales[i - 1]
                fl_up = F.interpolate(
                    flow_ref[prev].reshape(-1, 2, *flow_ref[prev].shape[-2:]),
                    size=(Hs, Ws), mode="bilinear", align_corners=False
                )
                cv_up = F.interpolate(
                    covis_ref[prev].reshape(-1, 1, *covis_ref[prev].shape[-2:]),
                    size=(Hs, Ws), mode="bilinear", align_corners=False
                )
                fl_in = fl_up.reshape(B, R, 2, Hs, Ws).detach()
                cv_in = cv_up.reshape(B, R, 1, Hs, Ws).detach()

                channel = f_q.shape[1]

                delta_flow, delta_certainty = self.conv_refiner[s](
                    f_q.unsqueeze(1).repeat(1, R, 1, 1, 1).view(B * R, channel, Hs, Ws),
                    f_r.reshape(B * R, channel, Hs, Ws),
                    fl_in.reshape(B * R, 2, Hs, Ws),
                    logits=cv_in.reshape(B * R, 1, Hs, Ws),
                    scale_factor=1,
                    num_ref_view=R,
                )

                displacement = int(s) * torch.stack(
                    (delta_flow[:, 0].float() / (4 * 512),
                     delta_flow[:, 1].float() / (4 * 512)),
                    dim=1,
                ).view(B, R, 2, Hs, Ws)

                delta_certainty = delta_certainty.view(B, R, 1, Hs, Ws)

                flow_ref[s]  = fl_in + displacement
                covis_ref[s] = cv_in + delta_certainty


        # Build your requested structure with integer keys
        corresps: Dict[int, Dict[str, torch.Tensor]] = {}
        for s in self.scales:
            k = int(s)
            scale_dict: Dict[str, torch.Tensor] = {
                "flow":      flow_ref[s],
                "certainty": covis_ref[s],
            }
            corresps[k] = scale_dict


        return corresps



# ---------------------------
# Main Tracktention pipeline
# ---------------------------

class MVRoMa(nn.Module):
    """
    Backbone + interleaved Tracktention → hook tokens → fuse to coarse (s=16)
    → GP+Transformer initializer (pairwise) → RoMA-style multi-view refiner (16→8→4→2→1).
    """
    def __init__(
        self,
        cfg: ModelConfig,
        backbone: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg


        self.patch_size = cfg.patch_size
        self.backbone_name = cfg.backbone_name

        self.backbone = backbone 

        # Interleave Tracktention after chosen start block
        num_tracktention_blocks = self.cfg.num_blocks - self.cfg.tracktention_start_block
        self.tracktention_modules = nn.ModuleList([
            TracktentionBlock(cfg)
            for _ in range(num_tracktention_blocks)
        ])
        self.track_token_proj = nn.Linear(2, cfg.dim)

        # ---- Coarse features at s=16
        self._coarse_from_last = nn.Linear(self.backbone.embed_dim, self.cfg.coarse_feature_dim)

        # --- RoMA-like decoder stack (classifier head)
        gp_dim   = 512
        feat_dim = self.cfg.coarse_feature_dim
        decoder_dim = gp_dim + feat_dim
        blocks = nn.Sequential(*[
            Block(decoder_dim, self.cfg.num_heads, attn_class=MemEffAttention) for _ in range(5)
        ])
        cls_res  = 64
        out_dim  = cls_res * cls_res + 1

        embedding_decoder = TransformerDecoder(
            blocks=blocks,
            hidden_dim=decoder_dim,
            out_dim=out_dim,
            is_classifier=True,
            amp=True,
            pos_enc=False,
        )

        # --- GP module at s=16 (RoMA)
        gp16 = GP(
            kernel=CosKernel,
            T=0.2,
            learn_temperature=False,
            only_attention=False,
            gp_dim=gp_dim,
            basis="fourier",
            no_cov=True,
        )

        # --- Pairwise initializer (kept)
        assert self.cfg.coarse_stride_px == 16, "This implementation assumes s_coarse=16."
        self.flow_initializer = PairwiseGPInitializer(
            gp_module=gp16,
            embedding_decoder=embedding_decoder,
            scale_tag="16",
            detach_outputs=True,
        )


        # --- Shared per-scale projector
        self.projector = SharedPerScaleProjector(Cc_coarse=self.cfg.coarse_feature_dim)

        # --- VGG pyramid for s ∈ {8,4,2,1}
        self.vgg = VGG19(pretrained=False)

        # --- RoMA-style multi-view refiner
        self.refiner = RomaMultiViewRefinerMV(cfg)
        self.upsample_refiner = UpsampleRomaMultiViewRefinerMV(cfg)


    def _build_vgg_pyramids(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        frames: (B, T, 3, H, W)
        returns dict of per-scale tensors for all views (query+refs):
          "8": (B,T,512,H/8,W/8), "4": (B,T,256,H/4,W/4), "2": (B,T,128,H/2,W/2), "1": (B,T,64,H,W)
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        x = normalize_image_tensor(x)
        feats = self.vgg(x) # keys: {1:64, 2:128, 4:256, 8:512}
        out = {
            "8": feats[8].reshape(B, T, 512, H // 8, W // 8).float(),
            "4": feats[4].reshape(B, T, 256, H // 4, W // 4).float(),
            "2": feats[2].reshape(B, T, 128, H // 2, W // 2).float(),
            "1": feats[1].reshape(B, T,  64, H,      W).float(),
        }
        return out


    def prepare_tokens_with_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int]]:
        x = self.backbone.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        cls_token = self.backbone.cls_token
        if self.backbone.n_storage_tokens > 0:
            storage_tokens = self.backbone.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (H, W)


    def _apply_tracktention(self, multi_view_frames: torch.Tensor, point_tracks: torch.Tensor, feature_grid_coords: torch.Tensor, H:int, W:int, track_mask: Optional[torch.Tensor] = None
                           ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        B, T, C, H, W = multi_view_frames.shape
        x = multi_view_frames.view(B * T, C, H, W)
        x = normalize_image_tensor(x)
        
        prefix = 1 # + self.backbone.num_register_tokens
        x = self.backbone.prepare_tokens_with_masks(x)   # x: (B, 1+n_storage+HW, C)
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0            
        rope = None

        # 1) Before Tracktention
        end = self.cfg.tracktention_start_block
        for block in self.backbone.blocks[:end]:
            x = block(x)

        # 2) After Tracktention
        track_tokens = self.track_token_proj(point_tracks)   
        start = self.cfg.tracktention_start_block
        for i, block in enumerate(self.backbone.blocks[start:], start=start):
            x = x.view(B * T, prefix + Hp * Wp, -1)
            x = block(x)
            pre_tokens = x[:, :prefix, :].contiguous().view(B, T, prefix, -1)
            img_tokens = x[:, prefix:, :].contiguous().view(B, T, Hp * Wp, -1)
            
            if self.cfg.backbone_apply_tracktention:
                img_tokens = self.tracktention_modules[i - start](
                    img_tokens, track_tokens, point_tracks, feature_grid_coords, H, W, track_mask=track_mask
                )
                
            x = torch.cat([pre_tokens, img_tokens], dim=2)

        if self.cfg.backbone_norm:
            x = self.backbone.norm(x) 
            
        body_final = x[:, :, prefix :].contiguous()  # (B, T, Hp * Wp, -1)

        return body_final


    def _build_coarse_flow(
        self,
        body_final: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        body_final, x, fused, Fq_c, Fr_c are all local → freed on return.
        Returns: flow_cls, flow_c_init, covis_c_init, covis_c_init_detached, Fq_pyr, Fr_pyr
        """
        x = self._coarse_from_last(body_final)  # (B,T,N,Cc)
        Hp = H // self.cfg.patch_size
        Wp = W // self.cfg.patch_size
        B_, T_, N_, Cc = x.shape
        fused = x.permute(0, 1, 3, 2).contiguous().view(B_, T_, Cc, Hp, Wp)

        Fq_c = fused[:, 0]   # (B,Cc,Hp,Wp)
        Fr_c = fused[:, 1:]  # (B,R,Cc,Hp,Wp)

        flow_cls, flow_c_init, covis_c_init, covis_c_init_detached = self.flow_initializer(Fq_c, Fr_c)

        Fq_pyr = self.projector.forward_query({"16": Fq_c})
        Fr_pyr = self.projector.forward_refs({"16": Fr_c})

        return flow_cls, flow_c_init, covis_c_init, covis_c_init_detached, Fq_pyr, Fr_pyr

    def _run_upsample_refiner(
        self,
        frames: torch.Tensor,
        flow16: torch.Tensor,
        covis16: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        vgg_pyrs is local → freed on return after upsample_refiner finishes.
        """
        vgg_pyrs = self._build_vgg_pyramids(frames)
        Fq_pyr = {s: vgg_pyrs[s][:, 0]  for s in ("8", "4", "2", "1")}
        Fr_pyr = {s: vgg_pyrs[s][:, 1:] for s in ("8", "4", "2", "1")}
        return self.upsample_refiner(
            Fq_pyr=Fq_pyr,
            Fr_pyr=Fr_pyr,
            input_flow16=flow16,
            input_covis16=covis16,
        )

    def match(
            self,
            multi_view_frames: torch.Tensor,
            multi_view_frames_org: torch.Tensor,
            point_tracks: torch.Tensor,
            feature_grid_coords: torch.Tensor,
            upsample_preds: bool = False,
    ) -> Dict[int, Dict[str, torch.Tensor]]:

        B, T, _, H, W = multi_view_frames.shape
        _assert_int(H, "H"); _assert_int(W, "W")
        assert T >= 2, "Require at least 1 query + 1 ref"

        last_dim = point_tracks.shape[-1]
        if last_dim == 4:
            point_tracks = unpack_query_ref_point_pairs(point_tracks)
        elif last_dim == 2:
            if point_tracks.shape[1] != T:
                raise ValueError(f"point_tracks has shape {tuple(point_tracks.shape)} but T={T} from frames.")

        track_mask = (point_tracks.mean(dim=-1) > -100).int()
        point_tracks = normalize_xy(point_tracks, H, W, align_corners=False)
        feature_grid_coords = normalize_xy(feature_grid_coords, H, W, align_corners=False)

        # body_final freed after _build_coarse_flow returns
        body_final = self._apply_tracktention(
            multi_view_frames, point_tracks, feature_grid_coords, H, W, track_mask=track_mask
        )
        flow_cls, flow_c_init, covis_c_init, covis_c_init_detached, Fq_pyr, Fr_pyr = \
            self._build_coarse_flow(body_final, H, W)

        results = self.refiner(
            Fq_pyr=Fq_pyr,
            Fr_pyr=Fr_pyr,
            flow16=flow_c_init,
            covis16=covis_c_init_detached,
            gm_cls=flow_cls,
            gm_certainty=covis_c_init,
        )

        # vgg_pyrs freed after _run_upsample_refiner returns
        update_results = self._run_upsample_refiner(
            multi_view_frames,
            flow16=results[16]['flow'],
            covis16=results[16]['certainty'],
        )
        results[8] = update_results[8]
        results[4] = update_results[4]
        results[2] = update_results[2]
        results[1] = update_results[1]

        if upsample_preds:
            update_results = self._run_upsample_refiner(
                multi_view_frames_org,
                flow16=results[1]['flow'],
                covis16=results[1]['certainty'].detach(),
            )
            results[8] = update_results[8]
            results[4] = update_results[4]
            results[2] = update_results[2]
            results[1] = update_results[1]

        return results



def normalize_image_tensor(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    ImageNet standardize
    
    Args:
        x: [B, 3, H, W] tensor (0-1 )
        mean: RGB 
        std: RGB 
    
    Returns:
        normalized tensor
    """
    device = x.device
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    return (x - mean) / std
