from collections import defaultdict
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from argparse import Namespace
import argparse

from hpatches_dataset.homography_eval import *
from hpatches_dataset.hpatches_mv import HPatches

from src.run_model import run_model_test
from src.build_model import build_our_model
from src.matchers import sample_match, build_prematch_model
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



def to_pixel_coordinates(coords, H, W):
    kpts = torch.stack(
        (W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), axis=-1
    )
    return kpts

def convert_coordinates(im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
    offset = 0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0] (I think)
    im_A_coords = (
        torch.stack(
            (
                wq * (im_A_coords[..., 0] + 1) / 2,
                hq * (im_A_coords[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
        - offset
    )
    im_A_to_im_B = (
        torch.stack(
            (
                wsup * (im_A_to_im_B[..., 0] + 1) / 2,
                hsup * (im_A_to_im_B[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
        - offset
    )
    return im_A_coords, im_A_to_im_B

default_conf = {
        "data": {
            "model_data": {
                "name": "multiview_hpatches",
                "batch_size": 1,
                "num_workers": 1,
                "preprocessing": {
                    "resize": 480,
                    "side": "short",
                },
            },
            "eval_data": {
                "name": "hpatches",
                "batch_size": 1,
                "num_workers": 1,
                "preprocessing": {
                    "resize": 480,
                    "side": "short",
                },
            }

        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": 0.5,
        },
}


def plot_cumulative(
    errors: dict,
    thresholds: list,
    colors=None,
    title="",
    unit="-",
    logx=False,
):
    thresholds = np.linspace(min(thresholds), max(thresholds), 100)

    plt.figure(figsize=[5, 8])
    for method in errors:
        recall = []
        errs = np.array(errors[method])
        for th in thresholds:
            recall.append(np.mean(errs <= th))
        plt.plot(
            thresholds,
            np.array(recall) * 100,
            label=method,
            c=colors[method] if colors else None,
            linewidth=3,
        )

    plt.grid()
    plt.xlabel(unit, fontsize=25)
    if logx:
        plt.semilogx()
    plt.ylim([0, 100])
    plt.yticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.ylabel(title + "Recall [%]", rotation=0, fontsize=25)
    plt.gca().yaxis.set_label_coords(x=0.45, y=1.02)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.yticks(rotation=0)

    plt.legend(
        bbox_to_anchor=(0.45, -0.12),
        ncol=2,
        loc="upper center",
        fontsize=20,
        handlelength=3,
    )
    plt.tight_layout()

    return plt.gcf()

def run_eval(dataset, weight_path, match_W=672, match_H=672, save_fig=False):
    results = defaultdict(list)

    conf = OmegaConf.create(default_conf).eval

    test_thresholds = (
        ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        if not isinstance(conf.ransac_th, Iterable)
        else conf.ransac_th
    )
    pose_results = defaultdict(lambda: defaultdict(list))

    num_set = len(dataset)
    fig_dict = {}
    cnt = 0
    device = 'cuda:0'

    prematch_model_name, prematch_model, dense_comatcher_model = build_model_matcher(device=device, weight_path=weight_path)

    for set_idx in tqdm(list(range(num_set))):
        cnt += 1
        data = dataset[set_idx]

        image_dict = {}
        image_dict['query_img_path'] = data['path0']
        image_dict['ref_img_paths'] = data['path_queries']

        num_cluster = 512
        corresps = run_model_test(dense_comatcher_model, image_dict, coarse_res_hw=(match_H, match_W), target_res_hw=(1344, 1344), prematch_model=prematch_model, prematch_model_name=prematch_model_name, \
            upsample_preds=True, num_cluster=num_cluster, device=device)

        dense_matches = corresps[1]['flow'] # B, T, 2, H, W
    
        certainty = corresps[1]['certainty'].sigmoid() # B, T, 1, H, W
        xs = torch.linspace(-1 + 1 / 1344, 1 - 1 / 1344, 1344, device=device)
        ys = torch.linspace(-1 + 1 / 1344, 1 - 1 / 1344, 1344, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack((gx, gy), dim=0)
        query_img_coord = coords.permute(1,2,0).reshape(1344, 1344, 2)

        for h_idx in range(len(data['H_set'])):
            # add custom evaluations here
            one_data = {}
            one_data["H_0to1"] = data['H_set'][h_idx]

            one_data['view0'] = {}
            img1_h, img1_w = data['orig_sizes'][0]
            img2_h, img2_w = data['orig_sizes'][h_idx + 1]

            pred = {}

            target_warp = dense_matches[0, h_idx].permute(1,2,0)
            target_certainty = certainty[0, h_idx, 0]

            one_data['view0']['image_size'] = (img1_w, img1_h)

            good_matches, good_certainty = sample_match(torch.cat([query_img_coord, target_warp], dim=-1), target_certainty)
            pos_a, pos_b = convert_coordinates(
                    good_matches[:, :2], good_matches[:, 2:], img1_w, img1_h, img2_w, img2_h
                )
            keypoint0 = pos_a
            keypoint1 = pos_b

            pred["keypoints0"] = keypoint0
            pred["keypoints1"] = keypoint1


            if "keypoints0" in pred:
                results_i = {**eval_homography_dlt(one_data, pred, scale=min(img2_h, img2_w))}

            else:
                results_i = {}

            for th in test_thresholds:
                pose_results_i = eval_homography_robust(
                    one_data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                    scale=min(img2_h, img2_w)
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)


    # summarize results as a dict[str, float]
    # you can also add your custom evaluations here
    summaries = {}
    for k, v in results.items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)

    auc_ths = [1, 3, 5]
    best_pose_results, best_th = eval_poses(
        pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
    )
    if "H_error_dlt" in results.keys():
        dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
        for i, ath in enumerate(auc_ths):
            summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]

    results = {**results, **pose_results[best_th]}
    summaries = {
        **summaries,
        **best_pose_results,
    }

    print(summaries)

    figures = {
        "homography_recall": plot_cumulative(
            {
                "DLT": results["H_error_dlt"],
                "RANSAC": results["H_error_ransac"],
            },
            [0, 10],
            unit="px",
            title="Homography ",
        ),
        **fig_dict
    }

    return summaries, figures, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--data_root", type=str, default="/cephfs/jongmin/cvpr2025_matcher/hpatches/hpatches-sequences-release", help="Path to hpatches-sequences-release directory")
    args = parser.parse_args()

    dataset = HPatches(data_root=args.data_root)
    run_eval(dataset, weight_path=args.weight_path, match_H=672, match_W=672)
