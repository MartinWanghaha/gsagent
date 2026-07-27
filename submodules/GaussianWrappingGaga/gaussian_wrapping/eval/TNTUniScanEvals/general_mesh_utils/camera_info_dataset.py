import sys
import math
import torch
import os
import numpy as np
from typing import NamedTuple
from PIL import Image
from general_mesh_utils.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary, 
    read_extrinsics_text, 
    read_intrinsics_text
)

def focal2fov(focal, pixels):
    if not isinstance(focal, torch.Tensor):
        return 2. * math.atan(pixels / ( 2. * focal))
    else:
        return 2. * torch.atan(pixels / (2. * focal))

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8):
    sparse_folder_possibilities = ["sparse", "sparse/0/"]
    for sparse_folder in sparse_folder_possibilities:
        try:
            try:
                cameras_extrinsic_file = os.path.join(path, f"{sparse_folder}", "images.bin")
                cameras_intrinsic_file = os.path.join(path, f"{sparse_folder}", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(path, f"{sparse_folder}", "images.txt")
                cameras_intrinsic_file = os.path.join(path, f"{sparse_folder}", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

            reading_dir = "images" if images == None else images
            cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

            if eval:
                raise ValueError("The Virtual Scan Sampling Evaluation does not need a train/test split")
            else:
                train_cam_infos = cam_infos
                test_cam_infos = []
            return train_cam_infos
        except:
            pass
    raise ValueError(f"Could not find Colmap scene info in {path}")

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}