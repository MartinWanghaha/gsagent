mkdir -p data
cd data

# https://drive.google.com/drive/folders/1UIOPtSJ638VxqLm4yMEcE9hE5mGBwuHH?usp=sharing   All datasets in this repo
gdown --id 1YLpop12JRbzglJfx0FUFUZ2GLaBfZX_x --output inpaint360.zip
unzip inpaint360.zip

gdown --id 1ev6MFuA_Q49aBW-mNqDdr4IQ7pp4WhZS --output others.zip
unzip others.zip

rm inpaint360.zip others.zip

cd ..