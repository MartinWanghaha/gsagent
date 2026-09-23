# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

# Modified from codes in Gaussian-Grouping https://github.com/lkeab/gaussian-grouping 
# and Gaga https://github.com/weijielyu/Gaga/tree/main?tab=readme-ov-file 

# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, Any
from tools.vis_obj_color import vis_mask_images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_DTYPE = np.uint16

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["DETECTRON2_DATASETS"] = "none"

torch.backends.cuda.cufft_plan_cache.clear()
torch.backends.cuda.cufft_plan_cache.max_size = 0

# --- Fix Detectron2 dataset double-registration ---
os.environ["DETECTRON2_DATASETS"] = "none"

# --- Reset dataset registries to avoid duplicate registration ---
from detectron2.data import DatasetCatalog, MetadataCatalog
try:
    DatasetCatalog.clear()
    MetadataCatalog.clear()
except AttributeError:
    if hasattr(DatasetCatalog, "_REGISTERED"):
        DatasetCatalog._REGISTERED.clear()
    if hasattr(MetadataCatalog, "_NAME_TO_META"):
        MetadataCatalog._NAME_TO_META.clear()

# ============================================================
# Now safely import detectron2 / CropFormer
# ============================================================
from detectron2.config import get_cfg
from detectron2.projects.CropFormer.mask2former import add_maskformer2_config

# SAM
from segment_anything import sam_model_registry
from automatic_mask_generator import SamAutomaticMaskGenerator

# HQ-SAM
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.projects.CropFormer.mask2former import add_maskformer2_config
from detectron2.projects.CropFormer.demo_cropformer.predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image


def load_segmentation_model(config: Dict[str, Any], method: str, device: str):
    """
    Loads the specified segmentation model based on the chosen method:
    - "hqsam": HQ-SAM (High-Quality Segment Anything Model via Detectron2)
    - "sam": Standard SAM (Segment Anything Model)
    """
    if method == "hqsam":
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config["hqsam_config_file"])
        cfg.merge_from_list(["MODEL.WEIGHTS", config["hqsam_weight_path"]])
        cfg.freeze()
        return VisualizationDemo(cfg)

    elif method == "sam":
        sam = sam_model_registry[config["sam_encoder_version"]](checkpoint=config["sam_weight_path"]).to(device)
        return SamAutomaticMaskGenerator(
            sam,
            points_per_side=config["sam_num_points_per_side"],
            points_per_batch=config["sam_num_points_per_batch"],
            pred_iou_thresh=config["sam_pred_iou_threshold"],
        )


def generate_hqsam_mask(model, image, threshold: float):
    """
    Generates masks using HQ-SAM, filters regions based on confidence scores, and assigns unique object IDs.
    """
    predictions = model.run_on_image(image)
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores

    selected_indexes = (pred_scores >= threshold)
    selected_masks = pred_masks[selected_indexes]
    selected_scores = pred_scores[selected_indexes]

    if selected_masks.shape[0] > np.iinfo(MASK_DTYPE).max:
        raise ValueError(
            f"A single image contains {selected_masks.shape[0]} instances, "
            f"which exceeds the {np.iinfo(MASK_DTYPE).max} IDs supported by uint16 PNG masks."
        )

    mask_id = np.zeros(selected_masks.shape[1:], dtype=MASK_DTYPE)

    sorted_indices = torch.argsort(selected_scores)
    for i, index in enumerate(sorted_indices, start=1):
        mask_id[(selected_masks[index] == 1).cpu().numpy()] = i

    return mask_id


def process_images(model, method: str, image_folder: str, output_folder: str, threshold: float):
    
    os.makedirs(output_folder, exist_ok=True)

    image_names = [
        name
        for name in sorted(os.listdir(image_folder))
        if os.path.isfile(os.path.join(image_folder, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not image_names:
        raise FileNotFoundError(f"No supported input images found in {image_folder}")

    for image_name in tqdm(image_names, desc="Processing Images"):
        image_path = os.path.join(image_folder, image_name)
        
        if method == "hqsam":
            image = read_image(image_path, format="BGR")
            mask = generate_hqsam_mask(model, image, threshold)
        elif method == "sam":
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            mask = model.generate(image)["masks"][0]

        save_mask(mask, image_name, output_folder)
    
    output_folder_color = output_folder + "_color"
    vis_mask_images(output_folder, output_folder_color)


def save_mask(mask, image_name: str, output_folder: str):
   
    mask_path = os.path.join(output_folder, image_name.rsplit(".", 1)[0] + ".png")
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer instance mask, got shape={mask.shape}, dtype={mask.dtype}")
    if not cv2.imwrite(mask_path, mask.astype(MASK_DTYPE, copy=False)):
        raise OSError(f"Failed to write mask: {mask_path}")
    # print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", "-d", default="./data/mipnerf360/", type=str)
    parser.add_argument("--scene_name", "-s", default="kitchen", type=str)
    parser.add_argument("--image_folder", "-i", default="images_2", type=str)
    parser.add_argument("--method", "-m", default="hqsam", type=str, choices=["hqsam", "sam"])
    parser.add_argument("--threshold", "-t", default=0.5, type=float)

    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be in [0, 1], got {args.threshold}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_folder = os.path.join(args.dataset_path, args.scene_name, args.image_folder)
    output_folder = os.path.join(args.dataset_path, args.scene_name, f"raw_{args.method}")

    if not os.path.isdir(image_folder):
        parser.error(f"image folder does not exist: {image_folder}")

    with open(os.path.join(os.path.dirname(__file__), "seg_config.json"), "r") as f:
        config = json.load(f)[args.method]

    print(f"Loading {args.method} model...")
    model = load_segmentation_model(config, args.method, device)

    print("Generating masks...")
    process_images(model, args.method, image_folder, output_folder, args.threshold)
