# Install SAM
cd sam; pip install -e . --no-build-isolation
cd -

# Install Grounding-Dino
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO --no-build-isolation

# Install other lib
pip install numpy==1.26.4 opencv-python pycocotools matplotlib Pillow==9.2.0 scikit-image
pip install gradio==3.39.0 ffmpeg==1.4
pip install gdown gradio_client==0.3.0 websockets==10.4 --force-reinstall
pip install "huggingface-hub<1.0,>=0.34" --force-reinstall
pip install timm==0.4.5
pip install wget
pip install moviepy==1.0.3
pip install transformers==4.30.2

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -

# Install AST
git clone https://github.com/YuanGongND/ast.git ast_master
cp ./prepare.py ./ast_master