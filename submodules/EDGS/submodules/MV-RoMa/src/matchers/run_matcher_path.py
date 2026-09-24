import cv2
import numpy as np
import torch

from .uniflowmatch.models.ufm import (
    UniFlowMatchClassificationRefinement,
)


def build_prematch_model(model_name="ufm", device=None):
    if model_name == "ufm":
        model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        model.eval()

        if device is not None:
            model = model.to(device)

        return [None, model]

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")



def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def sample_match(
        matches,
        certainty,
        num=10000,
        sample_thresh=0.5,
    ):

    upper_thresh = sample_thresh
    certainty = certainty.clone()
    certainty[certainty > upper_thresh] = 1
    # certainty[certainty < upper_thresh] = 0

    matches, certainty = (
        matches.reshape(-1, 4),
        certainty.reshape(-1),
    )

    # return matches[certainty>sample_thresh], certainty[certainty>sample_thresh]

    sample_mode = "threshold_balanced"
    expansion_factor = 4 if "balanced" in sample_mode else 1

    good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)

    good_matches, good_certainty = matches[good_samples], certainty[good_samples]

    if "balanced" not in sample_mode:
        return good_matches, good_certainty

    density = kde(good_matches, std=0.1)
    p = 1 / (density+1)
    p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
    balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
    return good_matches[balanced_samples], good_certainty[balanced_samples]



def run_match_multi_path(image_dict, match_W, match_H, matcher_model, matcher_name="ufm", device='cuda:0', batched=False):
    query_img_path = image_dict['query_img_path']
    ref_img_paths = image_dict['ref_img_paths']

    if matcher_name == "ufm":
        matcher = matcher_model[1]
        query_coord, ref_coord, ref_vis = run_ufm_multi(matcher, query_img_path, ref_img_paths, match_W=match_W, match_H=match_H, device=device, batched=batched)
        return query_coord, ref_coord, ref_vis

    else:
        raise ValueError(f"Unsupported matcher_name: {matcher_name}")


def run_match_single_path(image_dict, match_W, match_H, matcher_model, matcher_name="ufm", device='cuda:0'):
    query_img_path = image_dict['query_img_path']
    ref_img_path = image_dict['ref_img_path']

    if matcher_name == "ufm":
        matcher = matcher_model[1]
        query_coord_list, ref_coord_list = run_ufm_single(matcher, query_img_path, ref_img_path, match_W=match_W, match_H=match_H, device=device)
        return query_coord_list, ref_coord_list

    else:
        raise ValueError(f"Unsupported matcher_name: {matcher_name}")


def run_ufm_single(matcher_model, query_img_path, ref_img_path, match_W, match_H, certainty_thres=0.5, device='cuda:0'):
    query_img = cv2.resize(cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB), (match_W, match_H))
    query_img = torch.from_numpy(query_img).permute(2,0,1).unsqueeze(0)

    ref_img = cv2.resize(cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB), (match_W, match_H))
    ref_img = torch.from_numpy(ref_img).permute(2,0,1).unsqueeze(0)

    with torch.inference_mode():
        result = matcher_model.predict_correspondences_batched(
                    source_image=query_img.to(device),
                    target_image=ref_img.to(device),
                )

    flow_output = result.flow.flow_output.cpu()[0]
    covisibility = result.covisibility.mask.cpu()[0]

    del result
    torch.cuda.empty_cache()

    H, W = flow_output.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device='cpu'),
                        torch.arange(W, device='cpu'), indexing='ij')
    coords_q = torch.stack((xx, yy), dim=-1).float()
    coords_ref = (coords_q + flow_output.permute(1, 2, 0))

    covis_filtered_mask = (covisibility > certainty_thres)
    coords_q = coords_q[covis_filtered_mask]
    coords_ref = coords_ref[covis_filtered_mask]

    return coords_q.view(-1,2), coords_ref.view(-1,2)


def run_ufm_multi(matcher_model, query_img_path, ref_img_paths, match_W, match_H, certainty_thres=0.5, device='cuda:0', batched=False):
    query_img_np = cv2.resize(cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB), (match_W, match_H))
    query_img = torch.from_numpy(query_img_np).permute(2, 0, 1).unsqueeze(0)

    if batched:
        num_refview = len(ref_img_paths)
        ref_imgs = torch.cat([
            torch.from_numpy(
                cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (match_W, match_H))
            ).permute(2, 0, 1).unsqueeze(0)
            for p in ref_img_paths
        ], dim=0)
        with torch.inference_mode():
            result = matcher_model.predict_correspondences_batched(
                source_image=query_img.repeat(num_refview, 1, 1, 1).to(device),
                target_image=ref_imgs.to(device),
            )
        flow_output = result.flow.flow_output
        covisibility = result.covisibility.mask

    else:
        query_img = query_img.to(device)
        flow_outputs = []
        covis_outputs = []
        for ref_path in ref_img_paths:
            ref_img = torch.from_numpy(
                cv2.resize(cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB), (match_W, match_H))
            ).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                result = matcher_model.predict_correspondences_batched(
                    source_image=query_img,
                    target_image=ref_img,
                )
            flow_outputs.append(result.flow.flow_output)
            covis_outputs.append(result.covisibility.mask)

        flow_output = torch.cat(flow_outputs, dim=0)
        covisibility = torch.cat(covis_outputs, dim=0)

    H, W = flow_output.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords_q = torch.stack((xx, yy), dim=-1)
    coords_refs = coords_q.unsqueeze(0) + flow_output.permute(0, 2, 3, 1)

    return coords_q, coords_refs, covisibility
