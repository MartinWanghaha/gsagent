# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import argparse
import os
import sys
import numpy as np
import shutil

# Ensure COLMAP utility is available
try:
    import tools.read_write_model as colmap
except ImportError:
    print("Error: 'read_write_model.py' not found in the current directory.")
    sys.exit(1)

def filter_training_model(scene_name, data_root, output_dir):
    # Path construction
    sparse_0_path = os.path.join(data_root, scene_name, "train_and_test/sparse/0")
    input_image_path = os.path.join(data_root, scene_name, "train_and_test/images")
    output_path = os.path.join(data_root, scene_name, output_dir)

    if not os.path.exists(input_image_path):
        print(f"Error: Image path {input_image_path} does not exist.")
        return

    # 1. Identify training images (exclude files with 'test' in name)
    all_files = os.listdir(input_image_path)
    train_images_list = [
        f for f in all_files 
        if "test" not in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    train_images_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    train_images_set = set(train_images_list)

    # 2. Load COLMAP model
    print(f"Loading COLMAP model from {sparse_0_path}...")
    cameras, images, points3D = colmap.read_model(path=sparse_0_path, ext=".bin")

    # 3. Filter Image IDs belonging to training set
    train_image_ids = {
        image_id for image_id, image in images.items() 
        if image.name in train_images_set
    }
    
    filtered_images = {
        image_id: image for image_id, image in images.items() 
        if image_id in train_image_ids
    }

    # 4. Filter 3D Points based on visibility in training images
    filtered_points3D = {}
    for point_id, point in points3D.items():
        # Mask to identify indices of valid training image observations
        mask = np.isin(point.image_ids, list(train_image_ids))
        
        if np.any(mask):
            filtered_points3D[point_id] = colmap.Point3D(
                id=point.id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids[mask],
                point2D_idxs=point.point2D_idxs[mask]
            )

    # 5. Save the filtered model
    train_only_ply_path = os.path.join(output_path, "sparse/0")
    os.makedirs(train_only_ply_path, exist_ok=True)
    colmap.write_model(cameras, filtered_images, filtered_points3D, 
                       path=train_only_ply_path, ext=".bin")

    # 6. Copy image folders (camera cameras.bin, images.bin, images, images_2, images_4, images_8) to output_path
    image_folders = ['sparse/0/cameras.bin', 'sparse/0/images.bin', "images", "images_2", "images_4", "images_8"]
    for item_name in image_folders:
        src_item = os.path.join(data_root, scene_name, "train_and_test", item_name)
        dst_item = os.path.join(output_path, item_name)
        
        if os.path.exists(src_item):
            os.makedirs(os.path.dirname(dst_item), exist_ok=True)
            
            if os.path.isdir(src_item):
                print(f"Copying directory: {item_name}")
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                print(f"Copying file: {item_name}")
                shutil.copy2(src_item, dst_item)
    
    print(f"Done! Retained {len(filtered_images)} images and {len(filtered_points3D)} points.")
    print(f"Filtered model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training-only COLMAP sparse model.")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., car)")
    parser.add_argument("--root", type=str, default="./data/inpaint360/", help="Data root path")
    parser.add_argument("--output", type=str, default="", help="Output directory name")

    args = parser.parse_args()
    filter_training_model(args.scene, args.root, args.output)