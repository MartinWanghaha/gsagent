#!/bin/bash
set -e 

# --- Configuration ---
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

# 1. LaMa Inpainting (Depth and Color)
# Activate the 'lama' environment to perform neural inpainting on missing regions
python tools/prepare_lama_data.py  -s data/${dataset_name}/${scene} -m output/${dataset_name}/${scene}  -r ${resolution} --inpaint2lama
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lama
cd LaMa
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict_color.py --data_name 360_${scene}_virtual
python bin/predict_depth.py --data_name 360_${scene}_virtual
cd ..
python tools/prepare_lama_data.py  -s data/${dataset_name}/${scene} -m output/${dataset_name}/${scene}  -r ${resolution}

# 2. Colorful Point Cloud Fusion
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate inpaint360gs
python edit_object_removal_plyfusion.py -s data/${dataset_name}/${scene} -m output/${dataset_name}/${scene} --config_file config/object_removal/${dataset_name}/${scene}.json                                                         

# 3. Missing Area Optimization (3DGS Inpainting)
python edit_object_inpaint.py  -s data/${dataset_name}/${scene} -m output/${dataset_name}/${scene} --config_file config/object_inpaint/${dataset_name}/${scene}.json --resolution ${resolution} --render_video

# 4. Performance Evaluation
if [ "$dataset_name" == "inpaint360" ]; then
    python tools/metrics_fid_masked.py -m output/${dataset_name}/${scene}
fi