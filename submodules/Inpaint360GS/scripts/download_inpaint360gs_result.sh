# Inpaint360GS result
# https://drive.google.com/drive/folders/1NgqE9SVL8e9BO4ZvIrRQHAhmGIdf9C6g?usp=sharing

gdown --folder 1NgqE9SVL8e9BO4ZvIrRQHAhmGIdf9C6g --output inpaint360gs_result

cd inpaint360gs_result

for f in *.zip; do
    echo "Extracting $f..."
    unzip -o "$f"
done

rm *.zip
cd ..