#!/bin/bash
set -e 

# Configuration
############## "inpaint360" ##############
dataset_name="inpaint360"
scene="doppelherz"
resolution=2
##########################################


################ "others" ################
# dataset_name="others"
# scene="kitchen"
# resolution=4
###########################################

# 1. Train Vanilla 3DGS
python gaussian_splatting/train.py \
    -s data/${dataset_name}/${scene} \
    -m output/${dataset_name}/${scene}/3dgs_output \
    --init_mode "sparse" \
    --eval \
    --resolution ${resolution}

# 2. Generate 2D Segmentation Masks
export PYTHONPATH=$(pwd):$(pwd)/seg/detectron2:$PYTHONPATH
python seg/raw_mask_sam.py \
    --dataset_path data/${dataset_name}/ \
    --scene_name ${scene} \
    --image_folder images_${resolution} \
    --method hqsam

# 3. 3D Mask Association (Linking 2D masks to 3D space)
python seg/mask_associate.py \
    --source_path data/${dataset_name}/${scene} \
    --model_path output/${dataset_name}/${scene}/3dgs_output \
    --resolution ${resolution} \
    --mask_generator hqsam \
    --eval

# 4. Label Processing
python tools/add_label_num_hqsam.py \
    --source_path data/${dataset_name}/${scene} \
    --resolution ${resolution} \
    --mask_generator hqsam

# 5. Semantic Distillation
python seg/distillation.py \
    --source_path data/${dataset_name}/${scene} \
    --model_path output/${dataset_name}/${scene} \
    --vanilla_3dgs_path output/${dataset_name}/${scene}/3dgs_output \
    --resolution ${resolution} \
    --object_path associated_hqsam \
    --eval

# 6. Final Rendering and Video Generation
python render.py -m output/${dataset_name}/${scene} --render_video