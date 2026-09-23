#!/bin/bash

# 1. Install PyTorch with specific CUDA support
echo "Install the appropriate version of PyTorch based on your CUDA version."

# 2. Install Gaussian Splatting submodules (compiling CUDA kernels)
echo "Compiling and installing CUDA rasterization operators..."
pip install gaussian_splatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install gaussian_splatting/submodules/simple-knn --no-build-isolation

# 3. Install Inpaint360GS specific submodules
echo "Installing Inpaint360GS diff-rasterizer..."
pip install submodules/diff-gaussian-rasterization --no-build-isolation

# 4. Install the main project and remaining dependencies (via pyproject.toml)
echo "Installing project and remaining dependencies..."
pip install .

# 5. Install segmentation packages (SAM & Detectron2)
echo "Setting up segmentation modules..."
cd seg
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install setuptools==75.8.0
python -m pip install -e detectron2 --no-build-isolation

# Build CropFormer specific components
echo "Building CropFormer PythonAPI and Ops..."
cd detectron2/detectron2/projects/CropFormer/entity_api/PythonAPI
make
cd ../../../..
cd projects/CropFormer/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../../../../../..

# Apply fvcore registry patch to allow redundant imports
python seg/patch_fvcore.py

# Setup weights directory
mkdir -p seg/weight/
# NOTE: Download weights manually from HuggingFace to seg/weight/ if not automated:
# https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x

# 6. Install Segment-and-Track-Anything module
echo "Setting up Segment-and-Track-Anything..."
cd Segment-and-Track-Anything/
bash script/install.sh
bash script/download_ckpt.sh
cd ..

echo "Installation complete!"