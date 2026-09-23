#!/bin/bash
set -e 

GREEN='\033[0;32m'
NC='\033[0m'

############## "inpaint360" ##############
dataset_name="inpaint360"
scene="doppelherz"
resolution=2

# please correct here to correct one according your segmentation label
target_id="26"
target_surronding_id="24,10"
##########################################


################ "others" ################
# dataset_name="others"
# scene="kitchen"
# resolution=4

# target_id="14"
# target_surronding_id=None
###########################################

# 1. Initialize Configuration
echo -e "${GREEN}==> Step 1: Initializing configuration for Target ID: ${target_id}...${NC}"
python tools/init_configs.py --dataset_name ${dataset_name} --scene ${scene} --target_id ${target_id} --target_surronding_id ${target_surronding_id}

# 2. Object Removal Stage
python edit_object_removal.py --source_path data/${dataset_name}/${scene} -m output/${dataset_name}/${scene} --config_file config/object_removal/${dataset_name}/${scene}.json --render_video

# 3. Virtual Camera Trajectory Generation
python tools/virtual_pose.py -s data/${dataset_name}/${scene} -m output/${dataset_name}/${scene}  --config_file config/object_removal/${dataset_name}/${scene}.json  # --circle_radius 0.5  

# 4. Interactive Mask Refinement
# Launch Segment-and-Track-Anything to refine the target object mask after removal
cd Segment-and-Track-Anything
python app.py
cd ..

# Terminate this script Ctr + C