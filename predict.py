import os
import numpy as np
import tensorflow as tf
from osgeo import gdal
import argparse
import cv2

# Load model
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

# Preprocess input image
def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    dataset = gdal.Open(image_path)
    image = dataset.ReadAsArray()
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = np.uint8(normalized_image)
    resized_image = cv2.resize(normalized_image, (256, 256))
    reshaped_image = np.expand_dims(resized_image, axis=(0, -1))
    return reshaped_image / 255.0

# Predict mask
def predict_mask(model, input_image):
    print("Making prediction...")
    prediction = model.predict(input_image)
    prediction = (prediction > 0.5).astype(np.uint8)
    return prediction[0, :, :, 0]

# Save predicted mask
def save_mask(mask, output_path):
    print(f"Saving predicted mask to: {output_path}")
    cv2.imwrite(output_path, mask * 255)  # Scale mask to [0, 255]

# Main function
def main():
    parser = argparse.ArgumentParser(description="Predict disaster segmentation masks.")
    parser.add_argument("--input_path", required=True, help="Path to the input satellite image.")
    parser.add_argument("--output_path", required=True, help="Path to save the predicted mask.")
    args = parser.parse_args()

    model_path = "models/unet_model.h5"
    model = load_model(model_path)

    input_image = preprocess_image(args.input_path)
    predicted_mask = predict_mask(model, input_image)

    save_mask(predicted_mask, args.output_path)

if __name__ == "__main__":
    main()
