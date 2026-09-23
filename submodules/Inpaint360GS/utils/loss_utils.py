# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def masked_ssim(img1, img2, mask, window_size=11, size_average=True):
    """
    Compute SSIM over a masked region.

    Args:
        img1: [B, C, H, W] - predicted image
        img2: [B, C, H, W] - ground truth image
        mask: [B, 1, H, W] - binary mask where 1 indicates the inpaint region to evaluate
        window_size: int
        size_average: bool
    Returns:
        scalar SSIM over masked region
    """
    assert img1.size() == img2.size()
    assert mask.size(0) == img1.size(0)
    assert mask.size(2) == img1.size(2) and mask.size(3) == img1.size(3)

    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Compute SSIM map: [B, C, H, W]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12  = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))  # [B, C, H, W]

    # Expand mask to match image shape: [B, 1, H, W] → [B, C, H, W]
    mask = mask.expand_as(ssim_map).float()

    if size_average:
        masked_ssim_val = (ssim_map * mask).sum() / (mask.sum() + 1e-6)
        return masked_ssim_val
    else:
        return (ssim_map * mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d_kl(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.                                         
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.     
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]    
    sample_preds = predictions[indices]     

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]         
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def loss_cls_3d_cosin(xyz, feature_vec, predictions, k=5, lambda_val=1.0, sim_weight=1.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors,
    KL divergence, and cosine similarity regularization.

    :param xyz: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param feature_vec: Tensor of shape (N, 16)
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the KL divergence loss.
    :param sim_weight: Weighting factor for the similarity (cosine) loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed total loss value (KL divergence + similarity loss).
    """

    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        predictions = predictions[indices]
        feature_vec = feature_vec[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = predictions[indices]
    sample_feature_vec = feature_vec[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions and xyz
    neighbor_preds = predictions[neighbor_indices_tensor]  # For KL divergence
    neighbor_feature_vec = feature_vec[neighbor_indices_tensor]  # For similarity

    # ----------------------------
    # 1. KL Divergence Loss
    # ----------------------------
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    kl_loss = kl.sum(dim=-1).mean()  # Sum over classes, mean over samples

    # Normalize KL loss
    num_classes = predictions.size(1)
    normalized_kl_loss = kl_loss / num_classes

    # ----------------------------
    # 2. Similarity Loss (Cosine Similarity)
    # ----------------------------
    sample_feature_vec_expanded = sample_feature_vec.unsqueeze(1).expand(-1, k, -1)  # Shape: (sample_size, k, D)
    cosine_sim = F.cosine_similarity(sample_feature_vec_expanded, neighbor_feature_vec, dim=-1)  # Shape: (sample_size, k)

    similarity_loss = 1 - cosine_sim.mean()  # Cosine similarity closer to 1 is better, so we minimize (1 - sim)

    # ----------------------------
    # 3. Total Loss: KL Divergence + Similarity Regularization
    # ----------------------------
    w_kl_loss = lambda_val * normalized_kl_loss
    w_similarity_loss =  sim_weight * similarity_loss

    return w_kl_loss, w_similarity_loss
