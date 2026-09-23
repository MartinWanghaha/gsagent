# 🚀 Installation Guide

This document provides step-by-step instructions for setting up the environments for **Inpaint360GS**. 

To ensure stability and avoid dependency conflicts between 3D reconstruction and 2D inpainting, we use **two isolated Conda environments**.


## 🏗️ 1. Main Project Setup: `inpaint360gs`

This environment handles 3D Gaussian Splatting (3DGS), Segment-and-Track-Anything, and the core editing pipeline.
The following configuration and installation steps have been verified on CUDA 11.8, RTX4090 GPU.

### Step 1: Create Environment
```bash
conda create -n inpaint360gs python=3.10 -y
conda activate inpaint360gs
# Install the appropriate version of PyTorch based on your CUDA version.
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# Install project dependencies
bash install.sh
```
Download Segmentation Model Weights [CropFormer_hornet_3x_03823a.pth](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x)， put it under seg/weight/

## 🎨 2. 2D Inpainting Setup: `lama`
```bash
cd LaMa
conda env create -f conda_env.yml
conda activate lama
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pytorch-lightning==1.2.9
```
Download the [big-lama.zip](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips) weights, extract them under ./LaMa