import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt
from .robust_estimators import load_estimator
from .homography_utils import *
from pdb import set_trace as bb

def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs


class AUCMetric:
    def __init__(self, thresholds, elements=None):
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)


def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        # print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th

def eval_homography_robust(data, pred, conf, scale=None):
    conf = {
            "estimator": "opencv",
            "ransac_th": 3,  # -1 runs a bunch of thresholds and selects the best (was 3)
        }

    H_gt = data["H_0to1"]
    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {}
    if "keypoints0" in pred:
        pts0, pts1 = pred["keypoints0"], pred["keypoints1"]
        data_["m_kpts0"] = pts0
        data_["m_kpts1"] = pts1

    est = estimator(data_, scale)
    if est["success"]:
        M = est["M_0to1"]
        error_r = homography_corner_error(M, H_gt, data["view0"]["image_size"]).item()
        if scale is not None:
            error_r = error_r  / (scale / 480.0)
    else:
        error_r = float("inf")
    # print('ransac:', error_r)
    results = {}
    results["H_error_ransac"] = error_r
    if "inliers" in est:
        inl = est["inliers"]
        results["ransac_inl"] = inl.float().sum().item()
        results["ransac_inl%"] = inl.float().sum().item() / max(len(inl), 1)

    return results


def eval_homography_dlt(data, pred, scale=None):
    H_gt = data["H_0to1"]
    H_inf = torch.ones_like(torch.from_numpy(H_gt)) * float("inf")

    pts0, pts1 = pred["keypoints0"], pred["keypoints1"]
    # m0, scores0 = pred["matches0"], pred["matching_scores0"]
    # pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    if 'scores' in pred.keys():
        scores = pred['scores'] # torch.ones(pts0.shape[0]).to(pts0.device) # scores.to(pts0)
    else:
        scores = torch.ones(pts0.shape[0]).to(pts0.device)
    results = {}
    try:
        if H_gt.ndim == 2:
            pts0, pts1, scores = pts0[None], pts1[None], scores[None]
        h_dlt = find_homography_dlt(pts0, pts1, scores)
        if H_gt.ndim == 2:
            h_dlt = h_dlt[0]
    except AssertionError:
        h_dlt = H_inf

    error_dlt = homography_corner_error(h_dlt, H_gt, data["view0"]["image_size"])
    if scale is not None:
        error_dlt = error_dlt / (scale / 480)
    # print('dlt:', error_dlt)
    results["H_error_dlt"] = error_dlt.item()
    return results

