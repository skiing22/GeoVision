import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import jaccard_score, f1_score

# Directory paths
MODELS_DIR = "models/"
PROCESSED_DATA_DIR = "data/processed/"

# Load model
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

# Load test data
def load_test_data():
    patches = []
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if filename.endswith(".npy"):
            patches.append(np.load(os.path.join(PROCESSED_DATA_DIR, filename)))

    data = np.array(patches)
    data = data / 255.0  # Normalize
    data = np.expand_dims(data, axis=-1)  # Add channel dimension
    return data

# Evaluate model
def evaluate_model(model, test_data):
    print("Evaluating model...")
    predictions = model.predict(test_data)
    predictions = (predictions > 0.5).astype(np.uint8)

    flat_ground_truth = test_data.flatten()
    flat_predictions = predictions.flatten()

    dice_score = f1_score(flat_ground_truth, flat_predictions)
    jaccard_index = jaccard_score(flat_ground_truth, flat_predictions)

    print(f"Dice Coefficient: {dice_score:.4f}")
    print(f"Jaccard Index (IoU): {jaccard_index:.4f}")

# Main function
def main():
    model_path = os.path.join(MODELS_DIR, "unet_model.h5")
    model = load_model(model_path)

    test_data = load_test_data()
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
