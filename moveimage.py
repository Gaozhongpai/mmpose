import os
import glob
import shutil


root_dir = 'dataset'  # Replace with the path to your root directory
destination_dir = 'demo/data/'  # Replace with the path to your destination data folder

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# List to store paths of all 'RGB.png' files

rgb_images = glob.glob(os.path.join(root_dir, '**/0/*RGB.png'), recursive=True)
# Copy each file to the destination directory
for image in rgb_images:
    base_name = os.path.basename(image)

    # Create a unique new name by incorporating part of the original path
    # For example, replacing '/' with '_' and removing the root directory part from the path
    new_name = image.replace(root_dir, '').replace(os.sep, '_')[1:]

    # Construct the new path with the renamed file
    new_path = os.path.join(destination_dir, new_name)

    # Copy the file to the new path
    shutil.copy(image, new_path)



