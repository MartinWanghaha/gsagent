import torch
import torch.nn as nn
import torch.nn.functional as F
from ..romatch.utils.local_correlation import local_correlation
from .refiner import ViewAttentionBlock, SpatialConvNeXtBlock


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
        use_mv_interact=False,
        attn_dim=512,
        refine_iters=2,
        in_channel=None
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype

        self.use_mv_interact = use_mv_interact
        self.in_channel = in_channel

        if self.use_mv_interact:
            self.attn_dim = attn_dim
            self.refine_iters = refine_iters
            self.compressor = nn.Conv2d(self.in_channel, attn_dim, 1, bias=True)
            self.expander = nn.Conv2d(attn_dim, self.in_channel, 1, bias=True)        
        
            num_heads = 4 if ((attn_dim > 100) and (attn_dim % 4 ==0)) else 1
            
            self.view_blocks = nn.ModuleList(
                [ViewAttentionBlock(attn_dim, num_heads=num_heads) for _ in range(refine_iters)]
            )
            self.space_blocks = nn.ModuleList(
                [SpatialConvNeXtBlock(attn_dim, attn_dim) for _ in range(refine_iters)]
            )



    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, flow, scale_factor = 1, logits = None, num_ref_view=None):
        b,c,hs,ws = x.shape
        x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)


        if self.use_mv_interact:
            hidden_dim, h, w = x_hat.shape[1:]  
            
            x_hat_vis = torch.cat([x[0:1], x_hat], dim=0)
            tokens = self.compressor(x_hat_vis)
            tokens = tokens.view(-1, num_ref_view+1, self.attn_dim, h, w)
            for it in range(self.refine_iters):                                           # (B,R,A,H,W)
                mv_out = self.view_blocks[it](tokens) #, logits)                       # (B,R,A,H,W)
                spatial = self.space_blocks[it](mv_out.reshape(-1, self.attn_dim, h, w)) \
                                    .reshape(-1, num_ref_view+1, self.attn_dim, h, w)     # (B,R,A,H,W)
                tokens = spatial                                     # residual → next h_all
            tokens = self.expander(tokens.view(-1,self.attn_dim,h,w))
            x_hat = tokens[1:] + x_hat
            x = tokens[:1] + x 


        if self.has_displacement_emb:
            im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
            ), indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-im_A_coords
            emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
            if self.local_corr_radius:
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow, 
                                                    sample_mode = self.sample_mode)
                else:
                    raise NotImplementedError("Local corr in own frame should not be used.")
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:    
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)


        else:
            if self.no_im_B_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        if self.concat_logits:
            d = torch.cat((d, logits), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)

        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty
