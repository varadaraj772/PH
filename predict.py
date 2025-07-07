import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# ----------------------------
# Config
MODEL_PATH = "pothole_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Pothole'] 
# ----------------------------

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "Pothole" else 1 - prediction

    print(f"Image: {img_path}")
    print(f"Confidence -------> {predicted_class} ({confidence * 100:.2f}% confidence)")

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py /path/to/image.jpg")
    else:
        predict_image(sys.argv[1])
