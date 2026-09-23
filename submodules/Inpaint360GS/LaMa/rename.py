import os

# Define the directory containing the images
directory = './data/depth/bear'

# Get a list of all PNG files in the directory, sorted to ensure correct order
files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])

# Rename each file according to the pattern 00000_mask.png, 00001_mask.png, etc.
for idx, filename in enumerate(files):
    file_prefix = os.path.splitext(filename)[0]
   
    # new_name = f"{idx:05d}_mask.png"
    new_name = f"{file_prefix}_mask.png"
    old_file_path = os.path.join(directory, filename)
    new_file_path = os.path.join(directory, new_name)
    os.rename(old_file_path, new_file_path)

print("Files renamed successfully!")