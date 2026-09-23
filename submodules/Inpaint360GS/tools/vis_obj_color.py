import os
import numpy as np
from PIL import Image
from render import visualize_obj


def vis_mask_images(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saved our visualized image under : {output_folder}")

    for image_name in sorted(os.listdir(input_folder)):
        if not image_name.lower().endswith(".png"):
            continue

        file_path = os.path.join(input_folder, image_name)

        # Preserve uint16 instance IDs. Casting to uint8 here aliases every
        # label above 255 and makes the preview disagree with the training mask.
        with Image.open(file_path) as mask_image:
            pred_obj_mask = np.array(mask_image, copy=True)

        if pred_obj_mask.ndim == 3 and pred_obj_mask.shape[-1] == 3:
            print(f"Warning: {image_name} is RGB, extracting the first channel...")
            pred_obj_mask = pred_obj_mask[:, :, 0]
        pred_obj_mask = pred_obj_mask.squeeze()
        if pred_obj_mask.ndim != 2 or not np.issubdtype(pred_obj_mask.dtype, np.integer):
            raise ValueError(
                f"Expected a 2D integer instance mask at {file_path}, "
                f"got shape={pred_obj_mask.shape}, dtype={pred_obj_mask.dtype}."
            )

        pred_obj_color_mask = visualize_obj(pred_obj_mask)
        save_path = os.path.join(output_folder, image_name)
        Image.fromarray(pred_obj_color_mask).save(save_path)
