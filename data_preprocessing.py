import os
import numpy as np
import cv2
from osgeo import gdal
from patchify import patchify

# Directory paths
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
PATCH_SIZE = 256

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_image(image_path):
    """Preprocess a single satellite image."""
    dataset = gdal.Open(image_path)
    image = dataset.ReadAsArray()
    # Normalize image
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = np.uint8(normalized_image)
    return normalized_image

def create_patches(image, patch_size):
    """Create patches from an image."""
    patches = patchify(image, (patch_size, patch_size), step=patch_size)
    return patches.reshape(-1, patch_size, patch_size)

def main():
    """Main preprocessing function."""
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".tif"):
            print(f"Processing {filename}...")
            image_path = os.path.join(RAW_DATA_DIR, filename)
            processed_image = preprocess_image(image_path)
            patches = create_patches(processed_image, PATCH_SIZE)

            # Save patches
            for idx, patch in enumerate(patches):
                patch_filename = os.path.join(PROCESSED_DATA_DIR, f"{os.path.splitext(filename)[0]}_patch_{idx}.npy")
                np.save(patch_filename, patch)

if __name__ == "__main__":
    main()
