import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Directory paths
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
INPUT_SHAPE = (256, 256, 1)
BATCH_SIZE = 16
EPOCHS = 25

# Build U-Net model
def build_unet(input_shape):
    inputs = Input(input_shape)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Load data
def load_data():
    patches = []
    for filename in os.listdir(PROCESSED_DATA_DIR):
        if filename.endswith(".npy"):
            patches.append(np.load(os.path.join(PROCESSED_DATA_DIR, filename)))

    data = np.array(patches)
    data = data / 255.0  # Normalize data
    data = np.expand_dims(data, axis=-1)  # Add channel dimension
    return train_test_split(data, test_size=0.2, random_state=42)

# Main function
def main():
    print("Loading data...")
    X_train, X_test = load_data()

    print("Building model...")
    model = build_unet(INPUT_SHAPE)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    print("Training model...")
    model.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save(os.path.join(MODELS_DIR, "unet_model.h5"))
    print("Model saved!")

if __name__ == "__main__":
    main()
