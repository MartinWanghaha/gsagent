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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import torch.nn.functional as F

WARNED = False


def _resize_object_mask(objects, target_height, target_width, object_path):
    """Convert an instance mask to [H, W] int64 without interpolating IDs."""
    mask = np.asarray(objects)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(
            f"Expected a 2D integer object mask at {object_path}, "
            f"got shape={mask.shape}, dtype={mask.dtype}."
        )
    if mask.size and (int(mask.min()) < 0 or int(mask.max()) > np.iinfo(np.uint16).max):
        raise ValueError(
            f"Object mask at {object_path} contains IDs outside the uint16 range: "
            f"[{int(mask.min())}, {int(mask.max())}]."
        )

    # PyTorch versions used by the project cannot construct tensors directly
    # from NumPy uint16 arrays, so promote IDs without changing their values.
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64, copy=False)))
    if mask_tensor.shape != (target_height, target_width):
        mask_tensor = F.interpolate(
            mask_tensor[None, None].float(),
            size=(target_height, target_width),
            mode="nearest",
        )[0, 0].to(dtype=torch.long)

    if mask_tensor.shape != (target_height, target_width):
        raise ValueError(
            f"Object mask at {object_path} has shape {tuple(mask_tensor.shape)} after resizing; "
            f"expected {(target_height, target_width)}."
        )
    return mask_tensor


def loadCam(args, id, cam_info, resolution_scale):
    

    orig_w, orig_h = cam_info.image.size  

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.object_path is None or cam_info.objects is None:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  objects=None)
    else:
        object_mask = _resize_object_mask(
            cam_info.objects,
            target_height=gt_image.shape[1],
            target_width=gt_image.shape[2],
            object_path=cam_info.object_path,
        )
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                    objects=object_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    """
    camera: Rcamera to world, tworld to camera
    
    
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
