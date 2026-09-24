import argparse
import torch
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
from PIL import Image
from torchvision import transforms

from src.run_model import run_model_test
from src.build_model import build_our_model
from src.matchers import build_prematch_model
from src.mvroma import ModelConfig


def build_model_matcher(device, weight_path):
    args = Namespace(
        use_dinov2=True,
        train_until_16x=False,
        train_refiner=False,
        train_all_model=False
    )

    cfg = ModelConfig()
    num_cluster = 512
    args.num_cluster = num_cluster
    cfg.num_cluster = num_cluster

    dense_comatcher_model, cfg = build_our_model(args, cfg, use_dinov2=True)

    weight = torch.load(weight_path, map_location='cpu')
    dense_comatcher_model.load_state_dict(weight, strict=False)
    dense_comatcher_model.eval()
    dense_comatcher_model.to(device)

    prematch_model_name = "ufm"
    prematch_model = build_prematch_model(model_name=prematch_model_name, device=device)

    return prematch_model_name, prematch_model, dense_comatcher_model


def visualize_warp(src_path, tgt_path, flow_hw, certainty_hw, save_path, device):
    """
    flow_hw:      (2, H, W) — normalized coords in tgt space, for each src pixel
    certainty_hw: (H, W)    — confidence [0, 1] after sigmoid
    Shows: src | tgt | src pixels forward-scattered into tgt space
    """
    H, W = flow_hw.shape[1], flow_hw.shape[2]

    src_img = Image.open(src_path).resize((W, H))
    tgt_img = Image.open(tgt_path).resize((W, H))

    x_src = (torch.tensor(np.array(src_img), dtype=torch.float32) / 255).to(device).permute(2, 0, 1)  # (3,H,W)
    x_tgt = (torch.tensor(np.array(tgt_img), dtype=torch.float32) / 255).to(device).permute(2, 0, 1)  # (3,H,W)

    cert = certainty_hw  # (H,W)

    # flow_hw: normalized [-1,1] tgt coords for each src pixel
    # convert to pixel coords in tgt
    tgt_x = ((flow_hw[0] + 1) / 2 * (W - 1)).long()  # (H,W)
    tgt_y = ((flow_hw[1] + 1) / 2 * (H - 1)).long()  # (H,W)

    # valid: in-bounds + certainty threshold
    valid = (tgt_x >= 0) & (tgt_x < W) & (tgt_y >= 0) & (tgt_y < H) & (cert > 0.5)

    src_flat = x_src.permute(1, 2, 0).reshape(-1, 3)   # (H*W, 3)
    tgt_idx = (tgt_y * W + tgt_x).reshape(-1)           # (H*W,)
    valid_flat = valid.reshape(-1)

    canvas = torch.ones(H * W, 3, device=device)
    canvas[tgt_idx[valid_flat]] = src_flat[valid_flat]
    warped = canvas.reshape(H, W, 3).permute(2, 0, 1)   # (3,H,W)

    # side-by-side: src | tgt | src scattered into tgt
    vis_combined = torch.cat([x_src, x_tgt, warped], dim=2)  # (3, H, 3W)
    vis_np = (vis_combined.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(vis_np).save(save_path)
    print(f"Saved visualization: {save_path}")


"""
python demo.py \
    --weight_path /path/to/model.pth \
    --src assets/DSC_0341.jpg \
    --tgts assets/DSC_0338.jpg assets/DSC_0345.jpg assets/DSC_0352.jpg \
    --viz
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--src", type=str, required=True, help="Path to source image")
    parser.add_argument("--tgts", type=str, nargs="+", required=True, help="Paths to target images")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--viz", action="store_true", help="Save warp visualizations to assets/")
    args = parser.parse_args()

    device = args.device
    prematch_model_name, prematch_model, model = build_model_matcher(device=device, weight_path=args.weight_path)

    image_dict = {
        "query_img_path": args.src,
        "ref_img_paths": args.tgts,
    }

    corresps = run_model_test(
        model, image_dict,
        coarse_res_hw=(560, 560),
        target_res_hw=(560, 840),
        prematch_model=prematch_model,
        prematch_model_name=prematch_model_name,
        upsample_preds=True,
        num_cluster=512,
        device=device,
    )

    print("corresps keys:", list(corresps.keys()))
    for scale, d in corresps.items():
        flow_shape = d['flow'].shape
        cert_shape = d['certainty'].shape
        print(f"  scale={scale}: flow={tuple(flow_shape)}, certainty={tuple(cert_shape)}")

    if args.viz:
        # use the finest scale available
        finest_scale = min(corresps.keys())
        flow = corresps[finest_scale]['flow']          # (B, T, 2, H, W)
        certainty = corresps[finest_scale]['certainty'].sigmoid()  # (B, T, 1, H, W)

        for t, tgt_path in enumerate(args.tgts):
            flow_t = flow[0, t]          # (2, H, W)
            cert_t = certainty[0, t, 0]  # (H, W)

            tgt_name = tgt_path.split("/")[-1].rsplit(".", 1)[0]
            save_path = f"assets/viz_{tgt_name}.jpg"
            visualize_warp(args.src, tgt_path, flow_t, cert_t, save_path, device)
