"""
sprite_transform.py
Tyler Edwards

Applies various size and color transformations to all images in a folder.
Can technically be used for any folder of images but main use is to
add noise and diversity to computer vision training data for the "Download Complete" project.
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops

# --------------------------- Edit Parameters --------------------------------------------

# Name of character / sprites
name = 'vfx'

# Folder with raw sprites
sprite_dir = "YARDS-SF3rdStrike/~sprites/" + name + "-sprites"
# Folder where the transformed sprites are sent
new_dir = "YARDS-SF3rdStrike/" + name + "-sprites-shift"

# Creates new_dir folder if you didn't already
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Transformations
trim = True  # Removes white space around sprites.
horizontal_flip = True  # Flips sprites horizontally
vertical_flip = True  # Flips sprites vertically
resize = False  # Standardizes the size of all the sprites to the average size

color_shift = True  # Hue shifts sprites
color_shift_n = 6  # The range of how many hue shifts will be created
# List of sprites that you don't want to color shift because they're always the same color (Includes grayscale and negative)
color_shift_blacklist = []

negative = True  # Creates color inverse of all transformations
grayscale = True  # Grayscale transformation
silhouette = True  # Creates black silhouette of sprite
sin_cos = True  # Does a sin and cos transformation on sprites to scramble all the colors (Similar to silhouette)

# (WIP) Attempt at trying to train for moves where characters fade away but doesn't work with YARDS and is still buggy
opacity = False  # Makes sprites more transparent
opacity_alpha = 77  # Transparent 0 -> 255 Opaque

debug = False  # Print out a bunch of messages to help debug

# --------------------------- Functions --------------------------------------------

# Removes white space around sprites.
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

# Creates color inverse of image
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

# Does a sin and cos transformation on sprites to scramble all the colors (Similar to silhouette)
def sin_cos_shift(img, mode="mix"):
    img_arr = np.array(img.convert('RGBA'))
    float_rgb = img_arr[..., :3].astype(float)

    if mode == "sin":
        # All channels use Sine
        float_rgb[..., 0] = np.abs(np.sin(float_rgb[..., 0] / 20.0)) * 255
        float_rgb[..., 1] = np.abs(np.sin(float_rgb[..., 1] / 30.0)) * 255
        float_rgb[..., 2] = np.abs(np.sin(float_rgb[..., 2] / 40.0)) * 255
    elif mode == "cos":
        # All channels use Cosine (Blacks will become Bright)
        float_rgb[..., 0] = np.abs(np.cos(float_rgb[..., 0] / 20.0)) * 255
        float_rgb[..., 1] = np.abs(np.cos(float_rgb[..., 1] / 30.0)) * 255
        float_rgb[..., 2] = np.abs(np.cos(float_rgb[..., 2] / 40.0)) * 255
    else:
        # Mix of both (Highly Chaotic)
        float_rgb[..., 0] = np.abs(np.cos(float_rgb[..., 0] / 25.0)) * 255
        float_rgb[..., 1] = np.abs(np.sin(float_rgb[..., 1] / 15.0)) * 255
        float_rgb[..., 2] = np.abs(np.cos(float_rgb[..., 2] / 35.0)) * 255

    new_arr = img_arr.copy()
    new_arr[..., :3] = float_rgb.astype(np.uint8)
    return Image.fromarray(new_arr, 'RGBA')

# Creates black silhouette of sprite
def silhouette_shift(img):
    img_arr = np.array(img.convert('RGBA'))
    img_arr[..., :3] = 0
    return Image.fromarray(img_arr, 'RGBA')

# Used in "shift_hue"
def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

# Used in "shift_hue"
def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

# Used for color shifts
def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb

# --------------------------- Main --------------------------------------------

if __name__ == '__main__':
    if debug:
        print("raw sprites folder:", sprite_dir)
        print("output folder:", new_dir)
        print("---")

    # Progress bar
    total_files = sum(len(files) for root, dirs, files in os.walk(sprite_dir))
    pbar = tqdm(total=total_files, unit='file', desc="Transforming sprites in {f}".format(f=sprite_dir))

    if resize:
        avg_size = get_resize_values()
        avgw = avg_size[0]
        avgh = avg_size[1]

    # Main loop
    for subdir, dirs, files in os.walk(sprite_dir):
        subdir_split = subdir.split("/")[-1].split("\\")  # Last folder/file in path
        animation = subdir_split[-1]
        animation_split = animation.split("-")

        if debug:
            print("sub folder:", subdir)
            print("sub folder split:",subdir_split)
            print("animation:", animation)
            print("animation split:", animation_split)

        # Parsing over images in each folder/subfolder
        for file in files:
            if debug:
                print("     ", file)
            # Create folder for that move's animations
            move_folder = new_dir + "//" + animation
            if not os.path.exists(move_folder):
                os.makedirs(move_folder)

            # RAW IMAGE
            image_raw = Image.open(os.path.join(subdir, file))

            # TRIM
            if trim:  # Trim whitespace out of raw image
                # All other edits are based on "image_trim"
                image_trim = trim(image_raw)
            else:  # If you don't want to trim white space for some reason
                image_trim = image_raw

            # RESIZE
            if resize:  # Resize image to match the average size of sprites
                if image_trim.size[0] > avgw/2 or image_trim.size[1] > avgh/2: # "Resize the raw image and retrim if the sprite is too large (2x avg)"
                    # Can also use to shrink all sprites if you change some values
                    image_resized = image_raw.resize((int(avgw/2) + 100, int(avgh/2 + 100)))
                    image_trim = trim(image_resized)
                    if debug:
                        print("      Resizing ->",subdir.split("//")[-1], ",", file)

            # NEGATIVE TRIM/RAW
            if negative:
                img_negative = negative_shift(image_trim.convert('RGBA')).convert('RGBA')
                img_negative.save(os.path.join(move_folder, "trim_negative_" + file))

            image_trim.save(os.path.join(move_folder, "trim_" + file))  # Save trimmed image as new file in move folder

            if horizontal_flip:
                horz_img = image_trim.transpose(method=Image.FLIP_LEFT_RIGHT)
                horz_img.save(os.path.join(move_folder, "horz_" + file))

            if vertical_flip:
                vert_img = image_trim.transpose(method=Image.FLIP_TOP_BOTTOM)
                vert_img.save(os.path.join(move_folder, "vert_" + file))

            # OPACITY
            if opacity:
                img_opac = image_trim.copy().convert("RGBA")
                img_opac.putalpha(opacity_alpha)
                datas = img_opac.getdata()
                newData = []
                for item in datas:
                    # Find background rgb values to replace
                    if item[0] == 0 and item[1] == 140 and item[2] == 74:  # finding black color by its RGB value
                        # storing a transparent value when we find a black color
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)  # other colors remain unchanged
                img_opac.putdata(newData)
                img_opac.save(os.path.join(move_folder, "opac_" + file), format="PNG")

            if animation not in color_shift_blacklist:
                # GRAYSCALE
                if grayscale:
                    img_grayscale = image_trim.convert('LA')
                    img_grayscale.save(os.path.join(move_folder, "gray_" + file))

                # NEGATIVE GRAYSCALE
                    if negative:
                        img_negative_gray = negative_shift(img_grayscale.convert('RGBA')).convert('RGBA')
                        img_negative_gray.save(os.path.join(move_folder, "gray_negative_" + file))

                # COLOR SHIFT
                if color_shift:
                    color = {f"shift_{j}": j / color_shift_n for j in range(color_shift_n)}

                    img_arr = np.array(image_trim.convert('RGBA'))
                    for i in color:
                        img_rainbow = Image.fromarray(shift_hue(img_arr, color[i]), 'RGBA')
                        img_rainbow.save(os.path.join(move_folder, i + file))

                        # NEGATIVE COLOR SHIFT
                        if negative:
                            img_rainbow_negative = negative_shift(img_rainbow.convert('RGBA')).convert('RGBA')
                            img_rainbow_negative.save(os.path.join(move_folder,  i + "negative_" + file))

                if silhouette:
                    silhouette_arr = silhouette_shift(image_trim)
                    silhouette_arr.save(os.path.join(move_folder,  "silhouette_" + file))

                if sin_cos:
                    img_math_mix = sin_cos_shift(image_trim, 'mix')
                    img_math_mix.save(os.path.join(move_folder,  "math_mix_" + file))

                    img_math_sin = sin_cos_shift(image_trim, 'sin')
                    img_math_sin.save(os.path.join(move_folder,  "math_sin_" + file))

                    img_math_cos = sin_cos_shift(image_trim, 'cos')
                    img_math_cos.save(os.path.join(move_folder,  "math_cos_" + file))

                    # if negative:  # Creating color shift versions of the negative image too (Might be excessive)
                    #     img_rainbow_negative = negative_shift(img_rainbow.convert('RGBA')).convert('RGBA')
                    #     img_rainbow_negative.save(os.path.join(move_folder, i + "negative_" + file))

            if not debug:  # Don't include progress bar when debugging
                pbar.update(1)
        if debug:  # Include line dividers when debugging
            print("---------------------------------------------------------------------------------")
    pbar.close()
    print("All sprites successfully transformed and placed in {f}!".format(f=new_dir))