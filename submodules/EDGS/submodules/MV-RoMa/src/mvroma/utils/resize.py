import torch.nn.functional as F

def resize_flow_and_covis(flow, covis, out_hw, scale_flow: bool = False):
    """
    flow:  (B,R,2,H,W)  covis: (B,R,1,H,W)  out_hw: (H_out, W_out)
    'Just interpolate' by default (no vector magnitude scaling).
    """
    H_out, W_out = int(out_hw[0]), int(out_hw[1])
    flow_resized = F.interpolate(flow, size=(H_out, W_out), mode="bilinear", align_corners=True)
    covis_resized = F.interpolate(covis, size=(H_out, W_out), mode="bilinear", align_corners=True)

    if scale_flow:
        H_in, W_in = flow.shape[-2:]
        sx, sy = W_out / W_in, H_out / H_in
        flow_resized[:, :, 0].mul_(sx)
        flow_resized[:, :, 1].mul_(sy)

    return flow_resized, covis_resized
