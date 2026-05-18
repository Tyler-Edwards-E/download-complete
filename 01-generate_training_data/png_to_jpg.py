from PIL import Image
import os
import time
import datetime
from tqdm import tqdm

start = time.time()
print(datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'), "||", "Starting...")

data = 'vfx-20260518'

# Filepath to train images folder
train_folder = "D:/~dc/yards_output/" + data + "/images/train"
# Output for jpg versions of train images
output_train_folder = "D:/~dc/yards_output/" + data + "/images/train-jpg"

# Filepath to val images folder
val_folder = "D:/~dc/yards_output/" + data + "/images/val"
# Output for jpg versions of val images
output_val_folder = "D:/~dc/yards_output/" + data + "/images/val-jpg"

print("Converting .png images ---> .jpg images....")

# Create the output folders if they don't exist
if not os.path.exists(output_train_folder):
    os.makedirs(output_train_folder)
if not os.path.exists(output_val_folder):
    os.makedirs(output_val_folder)


def pngtojpg(input_dir, output_dir, name):

    file_list = os.listdir(input_dir)

    # Filter the list once to get the total number of items to process
    png_files = [f for f in file_list if f.endswith(".png")]

    # Initialize tqdm with the total number of files to process
    for filename in tqdm(png_files, desc=f"Converting {name} Images to jpegs"):

        image = Image.open(os.path.join(input_dir, filename))

        # Get the filename without the extension
        basename, _ = os.path.splitext(filename)
        image = image.convert('RGB')
        # Save the image as a JPG with 85% quality
        image.save(os.path.join(output_dir, basename + ".jpg"), quality=85)

pngtojpg(train_folder, output_train_folder, "Train")
pngtojpg(val_folder, output_val_folder, "Validation")

print("All .png images have been converted to .jpg images.")
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))