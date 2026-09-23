#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def convert_instance_mask_to_binary_masks(instance_mask):
    """
    Converts an instance segmentation mask into multiple binary masks.

    Args:
        instance_mask (np.ndarray): A 2D array where each pixel contains an object ID.
                                    Background is assumed to be 0.

    Returns:
        np.ndarray: A 3D binary mask array of shape (num_instances, height, width),
                    where each slice corresponds to a single object's mask.
    """
    num_instances = np.max(instance_mask)  # Get the highest object ID
    if num_instances == 0:
        return np.zeros((0, *instance_mask.shape), dtype=bool)  # Handle case with no objects

    # Generate binary masks using vectorized operations
    binary_masks = (instance_mask[None, :, :] == np.arange(1, num_instances + 1)[:, None, None])

    return binary_masks  # Shape: (num_instances, height, width)