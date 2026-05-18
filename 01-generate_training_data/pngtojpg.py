from PIL import Image
import os


model_name = 'ken-example'
# Filepath to train images folder
train_folder = "example/" + model_name + "/images/train"
# Output for jpg versions of train images
output_train_folder = "example/" + model_name + "/images/train-jpg"

# Filepath to val images folder
val_folder = "example/" + model_name + "/images/val"
# Output for jpg versions of val images
output_val_folder = "example/" + model_name + "/images/val-jpg"

# Create the output folders if they don't exist
if not os.path.exists(output_train_folder):
    os.makedirs(output_train_folder)
if not os.path.exists(output_val_folder):
    os.makedirs(output_val_folder)

def pngtojpg(input, output):
    c = 1 # Loop count
    # Loop through all pngs in the input folder
    for filename in os.listdir(input):
        print(c)
        print(filename)
        print()
        if filename.endswith(".png"):
            # Open the image
            image = Image.open(os.path.join(input, filename))

            # Get the filename without the extension
            basename, _ = os.path.splitext(filename)
            image = image.convert('RGB')
            # Save the image as a JPG with 80% quality (adjustable)
            image.save(os.path.join(output, basename + ".jpg"), quality=85)
        c = c + 1

pngtojpg(train_folder, output_train_folder)
pngtojpg(val_folder, output_val_folder)

print("All .png images have been converted to .jpg images.")
