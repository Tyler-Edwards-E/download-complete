
from PIL import Image, ImageChops
from hueshift import shift_hue
import numpy as np
import os

# Name of character / sprites
name = '~effects'
# Folder with raw sprites
sprite_dir = "YARDS-SF3rdStrike/~sprites/" + name + "-sprites"
# Folder where you want the edited sprites to go
new_dir = "YARDS-SF3rdStrike/" + name + "-sprites-shift"

trim = True
horizontal_flip = True
vertical_flip = False
resize = False
color_shift = True
negative = True
grayscale = True
opacity = False; opacity_alpha = 77  # (transparent) 0 -> 255 (opaque)

# Create new_dir folder if you didn't already
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# TRIM
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 0.001, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# Getting average sprite size so any outliers can be resized to match
def get_resize_values(dir = sprite_dir):  # == True
    sumw = 0
    sumh = 0
    c = 0
    
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            image_raw = Image.open(os.path.join(subdir, file))
            image_trim = trim(image_raw)
            image_size = image_trim.size
            sumw = sumw + image_size[0]
            sumh = sumh + image_size[1]
            c = c + 1

    w = int(sumw / c)
    h = int(sumh / c)

    print("Image Count", c)
    print("Average Width", w)
    print("Average Height", h)
    print()
    print("==========================================================================================")
    
    return [w, h]

def negative_shift(img):
    for i in range(0, img.size[0] - 1):
        for j in range(0, img.size[1] - 1):
            # Get pixel value at (x,y) position of the image
            pixelColorVals = img.getpixel((i, j))
            # Invert color
            if type(pixelColorVals) is tuple and pixelColorVals != (0, 0, 0, 0):
                redPixel = 255 - pixelColorVals[0]  # Negate red pixel
                greenPixel = 255 - pixelColorVals[1]  # Negate green pixel
                bluePixel = 255 - pixelColorVals[2]  # Negate blue pixel
                opacityPixel = pixelColorVals[3]
                # Modify the image with the inverted pixel values
                img.putpixel((i, j), (redPixel, greenPixel, bluePixel, opacityPixel))
    return (img)

if resize:
    avg_size = get_resize_values()
    avgw = avg_size[0]
    avgh = avg_size[1]

# Main
for subdir, dirs, files in os.walk(sprite_dir):
    print(subdir)
    # print(subdir.split("/")[-1].split("\\"))
    # print("X")
    
    # Parsing over images in each folder/subfolder
    for file in files:
        print("     ", file)
        move_folder = new_dir + "//" + subdir.split("/")[-1].split("\\")[-1] # Create folder for that move's animations
        if not os.path.exists(move_folder):
            os.makedirs(move_folder)

        # RAW IMAGE
        image_raw = Image.open(os.path.join(subdir, file))

        # TRIM
        if trim:  # Trim whitespace out of raw image
            # All other edits are based on "image_trim"
            image_trim = trim(image_raw)

            # RESIZE
            if resize:  # Resize image to match the average size of sprites (Works when uncommented but is usually bad for training)
                if image_trim.size[0] > avgw/2 or image_trim.size[1] > avgh/2: # "Resize the raw image and retrim if the sprite is too large (2x avg)"
                    # Can also use to shrink all sprites if you change some values
                    image_resized = image_raw.resize((int(avgw/2) + 100, int(avgh/2 + 100)))
                    image_trim = trim(image_resized)
                    print("      Resizing ->",subdir.split("//")[-1], ",", file)

            image_trim.save(os.path.join(move_folder, "trim_" + file))  # Save trimmed image as new file in move folder

        if horizontal_flip:
            horz_img = image_trim.transpose(method=Image.FLIP_LEFT_RIGHT)
            horz_img.save(os.path.join(move_folder, "horz_" + file))

        if vertical_flip:
            vert_img = image_trim.transpose(method=Image.FLIP_TOP_BOTTOM)
            vert_img.save(os.path.join(move_folder, "vert_" + file))

        # OPACITY
        ##### (WIP) Attempt at trying to train for moves where characters fade away but doesn't work with YARDS and is still buggy
        if opacity:
            img_opac = image_trim.copy().convert("RGBA")
            img_opac.putalpha(opacity_alpha)
            datas = img_opac.getdata()
            newData = []
            for item in datas:
                # print(item) # Find background rgb values to replace
                if item[0] == 0 and item[1] == 140 and item[2] == 74:  # finding black color by its RGB value
                    # storing a transparent value when we find a black color
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)  # other colors remain unchanged
            img_opac.putdata(newData)
            img_opac.save(os.path.join(move_folder, "opac_" + file), format="PNG")

        # GRAYSCALE
        if grayscale:
            img_grayscale = image_trim.convert('LA')
            img_grayscale.save(os.path.join(move_folder, "gray_" + file))

        # NEGATIVE
        if negative:
            img_negative = negative_shift(image_trim.convert('RGBA')).convert('RGBA')
            img_negative.save(os.path.join(move_folder, "trim_negative_" + file))

            img_negative_gray = negative_shift(img_grayscale.convert('RGBA')).convert('RGBA')
            img_negative_gray.save(os.path.join(move_folder, "gray_negative_" + file))

        # COLOR SHIFT
        if color_shift:
            redshift = (180 - 180) / 360.0
            orangeshift = (90 - 65) / 360.0
            yellowshift = (180 - 360) / 360.0
            greenshift = (300 - 180) / 360.0
            blueshift = (400 - 180) / 360.0
            purpleshift = (360 - 90) / 360.0
            rainbowshift = {"red_": redshift, "orange_": orangeshift, "yellow_": yellowshift, "green_": greenshift,
                            "blue_": blueshift, "purple_": purpleshift}

            img_arr = np.array(image_trim.convert('RGBA'))
            for i in rainbowshift:
                img_rainbow = Image.fromarray(shift_hue(img_arr, rainbowshift[i]), 'RGBA')
                img_rainbow.save(os.path.join(move_folder, i + file))

                if negative:  # Creating color shift versions of the negative image too
                    img_rainbow_negative = negative_shift(img_rainbow.convert('RGBA')).convert('RGBA')
                    img_rainbow_negative.save(os.path.join(move_folder,  i + "negative_" + file))

