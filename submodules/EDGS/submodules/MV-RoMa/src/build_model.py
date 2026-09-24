from src.mvroma import ModelConfig, MVRoMa
from src.mvroma.romatch.models.transformer import vit_large
import torch

def build_our_model(args, cfg, use_dinov2=False):
    backbone = vit_large(patch_size=14, init_values=1.0, ffn_layer='mlp', block_chunks=0, img_size=518)

    cfg.img_size = 518
    cfg.patch_size = 14
    cfg.backbone_name = "v2"
    cfg.coarse_feature_dim = 512

    return MVRoMa(cfg, backbone=backbone, train_until_16x=args.train_until_16x, train_refiner=args.train_refiner, train_all_model=args.train_all_model), cfg
