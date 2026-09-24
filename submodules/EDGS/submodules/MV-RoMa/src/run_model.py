import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.matchers import *
from src.track_cluster import extract_tracks_match
from src.mvroma.utils.grids import build_token_center_grid



class SquarePad:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)

        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        pad_right = max_dim - width - pad_left
        pad_bottom = max_dim - height - pad_top

        padded_img = TF.pad(img,
                            padding=[pad_left, pad_top, pad_right, pad_bottom],
                            fill=self.fill)

        padded_img.padding_info = {
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'original_size': (width, height),
            'square_size': max_dim
        }

        return padded_img


def match_and_cluster_path(image_dict, match_W, match_H, img_W, img_H, prematch_model, prematch_model_name, covisibility_threshold, num_cluster, device, batched=False):
    assert prematch_model_name in ['ufm', 'roma_indoor', 'roma_outdoor']

    query_coord, ref_coord, ref_covis = run_match_multi_path(
        image_dict, match_W=match_W, match_H=match_H,
        matcher_model=prematch_model, matcher_name=prematch_model_name, device=device, batched=batched,
    )

    if img_H / match_H != 1. or img_W / match_W != 1.:
        query_coord[..., 1] = query_coord[..., 1] * (img_H / match_H)
        query_coord[..., 0] = query_coord[..., 0] * (img_W / match_W)
        ref_coord[..., 1] = ref_coord[..., 1] * (img_H / match_H)
        ref_coord[..., 0] = ref_coord[..., 0] * (img_W / match_W)

    tracks = extract_tracks_match(
        query_coord, ref_coord, ref_covis,
        N=num_cluster, downsample_stride=4,
        covisibility_threshold=covisibility_threshold, device=device
    )
    return tracks


def run_model_test(model, img_path_dict, coarse_res_hw=(512, 512), target_res_hw=(1024, 1024),
                   prematch_model=None, prematch_model_name="ufm", num_cluster=512,
                   upsample_preds=False, apply_square=False,
                   device='cuda:0'):
    """
    img_path_dict: {'query_img_path': str, 'ref_img_paths': List[str]}
    """
    batch_size = 1
    assert coarse_res_hw[0] % model.patch_size == 0
    assert coarse_res_hw[1] % model.patch_size == 0

    if "roma" in prematch_model_name:
        match_H, match_W = 672, 672
    elif prematch_model_name == "ufm":
        match_H, match_W = 420, 560
    else:
        match_H, match_W = 480, 640

    n_view = len(img_path_dict['ref_img_paths'])
    covisibility_threshold = 0.3 

    with torch.inference_mode():
        batch_track = match_and_cluster_path(
            img_path_dict, match_W=match_W, match_H=match_H,
            img_W=coarse_res_hw[1], img_H=coarse_res_hw[0],
            prematch_model=prematch_model, prematch_model_name=prematch_model_name,
            num_cluster=num_cluster, covisibility_threshold=covisibility_threshold,
            device=device
        )
        model_track_input = batch_track.unsqueeze(0)

    feature_grid_coords = build_token_center_grid(
        B=batch_size, T=n_view + 1,
        H=coarse_res_hw[0], W=coarse_res_hw[1],
        patch=model.patch_size, device=device
    )

    coarse_transform = transforms.Compose([
        transforms.Resize(coarse_res_hw),
        transforms.ToTensor(),
    ])

    square_padder = SquarePad(fill=0) if apply_square else None

    def apply_coarse_transform(img):
        if square_padder is not None:
            padded = square_padder(img)
            padding_info = padded.padding_info
            tensor = coarse_transform(padded)
        else:
            padded = img
            padding_info = None
            tensor = coarse_transform(img)
        return tensor, padding_info

    query_img = Image.open(img_path_dict['query_img_path'])
    query_tensor, query_padding_info = apply_coarse_transform(query_img)

    coarse_img_tensors = [query_tensor]
    padding_infos = [query_padding_info]

    for i in range(n_view):
        ref_img = Image.open(img_path_dict['ref_img_paths'][i])
        ref_tensor, ref_padding_info = apply_coarse_transform(ref_img)
        coarse_img_tensors.append(ref_tensor)
        padding_infos.append(ref_padding_info)

    coarse_padding_infos = list(padding_infos)

    if apply_square and coarse_padding_infos[0] is not None:
        coarse_H_val, coarse_W_val = coarse_res_hw
        tracks_adj = model_track_input.clone()

        q_pinfo = coarse_padding_infos[0]
        q_orig_W, q_orig_H = q_pinfo['original_size']
        q_max_dim = q_pinfo['square_size']
        q_pad_left = q_pinfo['pad_left']
        q_pad_top = q_pinfo['pad_top']

        valid_q = tracks_adj[0, :, :, 0] >= 0
        tracks_adj[0, :, :, 0][valid_q] = (
            tracks_adj[0, :, :, 0][valid_q] * (q_orig_W / q_max_dim)
            + q_pad_left * (coarse_W_val / q_max_dim)
        )
        tracks_adj[0, :, :, 1][valid_q] = (
            tracks_adj[0, :, :, 1][valid_q] * (q_orig_H / q_max_dim)
            + q_pad_top * (coarse_H_val / q_max_dim)
        )

        for i in range(n_view):
            r_pinfo = coarse_padding_infos[i + 1]
            r_orig_W, r_orig_H = r_pinfo['original_size']
            r_max_dim = r_pinfo['square_size']
            r_pad_left = r_pinfo['pad_left']
            r_pad_top = r_pinfo['pad_top']

            valid_r = tracks_adj[0, i, :, 2] >= 0
            tracks_adj[0, i, :, 2][valid_r] = (
                tracks_adj[0, i, :, 2][valid_r] * (r_orig_W / r_max_dim)
                + r_pad_left * (coarse_W_val / r_max_dim)
            )
            tracks_adj[0, i, :, 3][valid_r] = (
                tracks_adj[0, i, :, 3][valid_r] * (r_orig_H / r_max_dim)
                + r_pad_top * (coarse_H_val / r_max_dim)
            )

        model_track_input = tracks_adj

    coarse_img_tensors = torch.stack(coarse_img_tensors).unsqueeze(0).to(device)

    upsample_img_tensors = None
    if upsample_preds:
        upsample_transform = transforms.Compose([
            transforms.Resize(target_res_hw),
            transforms.ToTensor(),
        ])

        def apply_upsample_transform(img):
            if square_padder is not None:
                padded = square_padder(img)
                padding_info = padded.padding_info
                tensor = upsample_transform(padded)
            else:
                padding_info = None
                tensor = upsample_transform(img)
            return tensor, padding_info

        all_img_paths = [img_path_dict['query_img_path']] + img_path_dict['ref_img_paths']
        upsample_img_tensors = []
        padding_infos = []
        for path in all_img_paths:
            img = Image.open(path)
            tensor, padding_info = apply_upsample_transform(img)
            upsample_img_tensors.append(tensor)
            padding_infos.append(padding_info)

        upsample_img_tensors = torch.stack(upsample_img_tensors).unsqueeze(0).to(device)

    with torch.inference_mode():
        corresps = model.match(
            multi_view_frames=coarse_img_tensors,
            multi_view_frames_org=upsample_img_tensors,
            point_tracks=model_track_input,
            feature_grid_coords=feature_grid_coords,
            upsample_preds=upsample_preds
        )

    if apply_square:
        return corresps, padding_infos

    return corresps
