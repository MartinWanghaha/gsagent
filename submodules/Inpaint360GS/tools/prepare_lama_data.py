import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json

from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

def enlarge(input_dir, output_dir, expand_pixels=10, min_area=50):
    """
    Expand the boundaries of object masks and keep only the largest connected component.

    Args:
        input_dir (str): Directory containing the original input mask images.
        output_dir (str): Directory where the processed and enlarged masks will be saved.
        expand_pixels (int): The number of pixels by which to expand the mask boundaries.
        min_area (int): The number of pixels of mask, less than it which will be recognized outlier
    """

    os.makedirs(output_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for filename in tqdm(file_list, desc="Processing Masks", unit="file"):
        file_path = os.path.join(input_dir, filename)
        
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        mask[mask < 128] = 0
        mask[mask >= 128] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned_mask = np.zeros_like(mask)
        
        found_any = False
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 255
                found_any = True
        
        if not found_any and num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned_mask[labels == largest_label] = 255

        pil_mask = Image.fromarray(cleaned_mask).convert("L")
        expanded_mask = pil_mask.filter(ImageFilter.MaxFilter(size=2 * expand_pixels + 1))

        threshold = 128
        binary_mask = expanded_mask.point(lambda p: 255 if p > threshold else 0)

        output_path = os.path.join(output_dir, filename)
        binary_mask.save(output_path)

    print("✅ All masks have been successfully expanded and saved to the specified directory.")



def copy_inpaint2lama_virtual(args):
   
    source_path = args.source_path
    model_path = args.model_path
    scene_name = Path(source_path).name
    
    images_masks_path = './Segment-and-Track-Anything/tracking_results/images/images_masks'
    
    target_path = source_path + '/inpaint_2d_unseen_mask_virtual'
    os.makedirs(target_path, exist_ok=True)
    enlarge(images_masks_path, target_path)
    
    unseen_mask_path = source_path + "/inpaint_2d_unseen_mask_virtual"
    vanilla_virtual_image = model_path + f"/virtual/ours_{args.iterations}/"
    removed_virtual_image = model_path + f"/virtual/ours_object_removal/iteration_{args.iterations}/"

    lama_color_path = "./LaMa/data/color/360_" + scene_name + "_virtual"
    lama_depth_path = "./LaMa/data/depth/360_" + scene_name + "_virtual"
    os.makedirs(lama_color_path, exist_ok=True)
    os.makedirs(lama_depth_path, exist_ok=True)
    os.makedirs(lama_depth_path + "/depth_original", exist_ok=True)

    print("📁 Copying virtual color images...")
    copy_tree(removed_virtual_image + "renders", lama_color_path)
    
    print("📁 Copying original virtual depth images...")
    copy_tree(vanilla_virtual_image + "depth", lama_depth_path + "/depth_original")

    print("📁 Copying removed virtual depth images...")
    copy_tree(removed_virtual_image + "depth", lama_depth_path)
  
    print("🔁 Renaming depth PNGs to *_mask.png...")
    for filename in os.listdir(unseen_mask_path):
        if filename.endswith('.png'):
            new_name = filename.replace('.png', '_mask.png')  
            src_file = os.path.join(unseen_mask_path, filename)
            dst_file = os.path.join(lama_depth_path, new_name)
            shutil.copy2(src_file, dst_file)

    print("📁 Copying *_mask.png files to lama_color_path...")
    for filename in os.listdir(lama_depth_path):
        if filename.endswith('_mask.png'):
            src_file = os.path.join(lama_depth_path, filename)
            dst_file = os.path.join(lama_color_path, filename)
            shutil.copy2(src_file, dst_file)


def copy_lama2inpaint_virtual(args):

    source_path = args.source_path
    model_path = args.model_path
    scene_name = Path(source_path).name

     
    image_inpaint_unseen_path = os.path.join(source_path, "images_inpaint_unseen_virtual")

    os.makedirs(image_inpaint_unseen_path, exist_ok=True)
    inpainted_color_path = "./LaMa/output/color/360_" + scene_name + "_virtual"
    inpainted_depth_path = "./LaMa/output/depth/360_" + scene_name + "_virtual"

    copy_tree(inpainted_color_path, image_inpaint_unseen_path)

    for filename in os.listdir(image_inpaint_unseen_path):
        if filename.endswith('_mask.png'):
            new_name = filename.replace('_mask.png', '.JPG')  
            old_file_path = os.path.join(image_inpaint_unseen_path, filename)
            new_file_path = os.path.join(image_inpaint_unseen_path, new_name)
            os.rename(old_file_path, new_file_path)
        elif filename.endswith('.png'):
            new_name = filename.replace('.png', '.JPG')  
            old_file_path = os.path.join(image_inpaint_unseen_path, filename)
            new_file_path = os.path.join(image_inpaint_unseen_path, new_name)
            os.rename(old_file_path, new_file_path)

    images_path = f"images_{args.resolution}" if args.resolution != 1 else "images"
    for filename in os.listdir(os.path.join(source_path, images_path)):
        if "test_" in filename:
            old_file_path = os.path.join(os.path.join(source_path, images_path), filename)
            new_file_path = os.path.join(image_inpaint_unseen_path, filename)
            shutil.copy2(old_file_path, new_file_path)

    inpainted_depth_list = []
    for filename in os.listdir(inpainted_depth_path):
        if filename.endswith('.npy'):
            inpainted_depth_list.append(filename)
    inpainted_depth_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    removed_virtual_image = model_path + f"/virtual/ours_object_removal/iteration_{args.iterations}/depth_completed"
    os.makedirs(removed_virtual_image, exist_ok=True)

    for i in range(len(inpainted_depth_list)): 
   
        filename = inpainted_depth_list[i]
        old_file_path = os.path.join(inpainted_depth_path, filename)
        new_file_path = os.path.join(removed_virtual_image, filename)    
        shutil.copy2(old_file_path, new_file_path)

    print("✅ Copying data from LaMa finished！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LaMa inpainting prediction")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="e.g. data/inpaint360/doppelherz")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="e.g. output/inpaint360/doppelherz")
    parser.add_argument("-r", "--resolution", type=int, required=True)
    parser.add_argument("--inpaint2lama", action="store_true", help="copy data into LaMa folder")
    args = parser.parse_args()

    with open("config/object_distill/train_distill.json", 'r') as file:
        config = json.load(file)
    args.iterations = config.get("iterations")

    if args.inpaint2lama:
        copy_inpaint2lama_virtual(args)
    else:
        copy_lama2inpaint_virtual(args)
