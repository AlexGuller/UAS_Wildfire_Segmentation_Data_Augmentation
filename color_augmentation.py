# A) Import Relevant Libraries
import os
import glob
import cv2 
import numpy as np 
import shutil
from PIL import Image

# B) Setup Directories 

script_dir = r'/Users/alexanderguller/Desktop/UAS_Research/Color_Adjustment' # replace with your script directory

# Change the current working directory to the script directory
os.chdir(script_dir)
print("Current working directory:", os.getcwd())
print()

# Function to create directory if it doesn't exist, or delete contents if it does
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        # Delete all contents in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Cleared contents of directory: {path}")

def color_adjustment(image_path, output_path, scaleBrightness, scaleSaturation, scaleHue, scaleContrast):
    # Step 1: Read Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    
    # Step 2: Adjust Contrast
    # Keep Contrast within the range (.8, 1.2)
    adjusted = cv2.convertScaleAbs(image, alpha=scaleContrast, beta=0)

    # Step 3: Convert to HSV
    hsv_image = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Step 4: Enhance Saturation (S) and Brightness/Value (V) and Hue (H)
    hsv_image[:, :, 0] *= scaleHue  # Hue(Default 1)
    hsv_image[:, :, 1] *= scaleSaturation # Saturation(Default 1)
    hsv_image[:, :, 2] *= scaleBrightness # Brightness(Default 1)

    # Step 5: Clip the values to valid 8-bit range
    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)

    # Step 6: Convert back to BGR
    enhanced_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # Step 7: Save the final image
    cv2.imwrite(output_path, enhanced_bgr)
    
    print(f"Enhanced image saved: {output_path}")
    
# B) Define Paths 

rgb_original_path = "./color256/train_rgb_mod"

mask_original_path = "./color256/train_mask_mod"

rgb_train_path = "./color256/train_rgb_color"
masked_png_path = "./color256/train_masked_png_color"
concat_path = "./color256/train_concat_color"

# List of paths to manage
paths = [
    masked_png_path, 
    concat_path,
    rgb_train_path,
]

# Apply the function to each path in the list
for path in paths:
    ensure_directory(path)
    
# Define target brightness
user_input_brightness = input("What is the brightness scale(Default 1 => 100%)? ")

# Define target contrast scale
user_input_contrast = input("What is the contrast scale(Default 1 => 100%)? ")

# Define target saturation scale
user_input_saturation = input("What is the saturation scale(Default 1 => 100%)? ")

# Define target hue scale
user_input_hue = input("What is the hue scale(Default 1 => 100%)? ")

# Get the brightness size from the user
BRIGHTNESS_SCALE = float(user_input_brightness.strip())

# Get the saturation scale from the user
SATURATION_SCALE = float(user_input_saturation.strip())

# Get the hue scale from the user
HUE_SCALE = float(user_input_hue.strip())

# Get the contrast scale from the user
CONTRAST_SCALE = float(user_input_contrast.strip())

# C) Define class colors for the colormap
class_colors = ['#000000', '#FF0000', '#404040', '#C0C0C0']  # ash (black), fire (red), veg (dark grey), and background (light grey)

# Convert TIF to PNG with Colormap
def convert_tif_to_png_with_colormap(tif_files, masked_png_path, class_colors):
    
    for tif_file in tif_files:
        
        # Read the mask image
        mask = np.array(Image.open(tif_file))

        # Ensure the mask is in a single-channel format
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Create an output image with 3 channels
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Map each label in the mask to a color
        for label, color in enumerate(class_colors):
            
            # Find pixels with the current label
            indices = mask == label
            # Convert the hex color to an RGB tuple
            rgb_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            # Apply the color to the corresponding pixels
            colored_mask[indices] = rgb_color

        # Save the colored mask as a PNG image
        output_png_file_name = os.path.basename(tif_file).replace('.tif', '.png')
        Image.fromarray(colored_mask).save(os.path.join(masked_png_path, output_png_file_name))

# Concatenate RGB image with Color-labeled Map
def concatenate_images():
    
    rgb_files = sorted(glob.glob(os.path.join(rgb_train_path, '*.png')))
    pred_files = sorted(glob.glob(os.path.join(masked_png_path, '*.png')))
    
    for rgb_file, mask_png_file in zip(rgb_files, pred_files):
        
        rgb_img = Image.open(rgb_file)
        mask_png_img = Image.open(mask_png_file)
        concat_img = Image.new('RGB', (2 * rgb_img.width, rgb_img.height))
        concat_img.paste(rgb_img, (0, 0))
        concat_img.paste(mask_png_img, (rgb_img.width, 0))
        output_file_name = f"concat_{os.path.splitext(os.path.basename(rgb_file))[0]}.png"
        concat_img.save(os.path.join(concat_path, output_file_name))


# Main Function 
def main():
    
    # Clean up previous outputs
    for path in [masked_png_path, concat_path]:
        for file in glob.glob(os.path.join(path, "*.*")):
            os.remove(file)
            
    ### Call Functions ###
        # Adjust all images in the original RGB directory
    for image_file in glob.glob(os.path.join(rgb_original_path, "*.png")): 
        output_file = os.path.join(rgb_train_path, os.path.basename(image_file))
        color_adjustment(image_file, output_file, BRIGHTNESS_SCALE, SATURATION_SCALE, HUE_SCALE, CONTRAST_SCALE)
    
    # Get the list of TIFF files
    tif_files = glob.glob(os.path.join(mask_original_path, '*.tif'))
    
    # Convert TIF to PNG with colormap
    convert_tif_to_png_with_colormap(tif_files, masked_png_path, class_colors)
    
    # Concatenate the images
    concatenate_images()
    
if __name__ == "__main__":
    
    main()
