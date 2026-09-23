#!/usr/bin/env python3
import cv2
import numpy as np
import os
import re
import argparse


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def main(args):

    vis_folder = "images" if args.resolution == 1 else f"images_{args.resolution}"
    img_dir = os.path.join(args.source_path, vis_folder)
    output_dir = img_dir + "_num"
    os.makedirs(output_dir, exist_ok=True)
    images_path = sorted(os.listdir(img_dir), key=extract_number)

    all_labels = set()
    for image_name in images_path:
        image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Cannot read image {image_path}, skipping...")
            continue

        annotation_path = os.path.join(args.source_path,
                                       f"associated_{args.mask_generator}",
                                       os.path.splitext(image_name)[0] + ".png")
        if not os.path.exists(annotation_path):
            print(f"⚠️ Annotation file {annotation_path} not found, skipping...")
            continue

        annotation_data = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        if annotation_data is None:
            raise ValueError(f"Cannot read annotation {annotation_path}")
        if annotation_data.ndim != 2 or not np.issubdtype(annotation_data.dtype, np.integer):
            raise ValueError(
                f"Expected a 2D integer annotation at {annotation_path}, "
                f"got shape={annotation_data.shape}, dtype={annotation_data.dtype}"
            )
        unique_labels = np.unique(annotation_data)
        all_labels.update(unique_labels)

        for label in unique_labels:
            if label == 0:  # ignore background
                continue
            mask = (annotation_data == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(image, str(label), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)

        output_path = os.path.join(output_dir, image_name)
        if not cv2.imwrite(output_path, image):
            raise OSError(f"Cannot write label preview {output_path}")

    # --- Summary ---
    if 0 in all_labels:
        all_labels.remove(0)

    print("\n===================================")
    print(f"✅ Without background, total number of unique classes: {len(all_labels)}")
    print(f"📌 All unique class labels: {sorted(all_labels)}")
    print("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add label numbers to HQSAM segmentation masks.")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to dataset, e.g., data/inpaint360/doppelherz")
    parser.add_argument("--resolution", type=int, default=2,
                        help="Image resolution level, e.g., 1 or 2")
    parser.add_argument("--mask_generator", type=str, default="hqsam",
                        help="Type of mask generator, e.g., hqsam or sam")
    args = parser.parse_args()

    main(args)
