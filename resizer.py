from PIL import Image
import os

# Base directory where the image folders are located
base_dir = './data'  

# Target directory where resized images will be saved
target_dir = 'resized_images_224'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Iterate over the folder names
for i in range(1, 13):  # From images_001 to images_012
    folder_name = f"images_{i:03}"
    # Path to the "images" subfolder within the current images folder
    source_dir = os.path.join(base_dir, folder_name, "images")

    # Check if the source directory exists
    if os.path.exists(source_dir) and os.path.isdir(source_dir):
        # Loop through all the files in the current source directory
        for filename in os.listdir(source_dir):
            if filename.endswith('.png'):  # Check if the file is a PNG image
                # Construct the full file path
                file_path = os.path.join(source_dir, filename)
                # Open the image
                img = Image.open(file_path)
                # Resize the image
                img = img.resize((224, 224), Image.ANTIALIAS)
                # Construct the path for the resized image
                # Saving with original file name
                target_path = os.path.join(target_dir, filename)
                # Save the resized image
                img.save(target_path)

print("All images have been resized and saved in:", target_dir)
