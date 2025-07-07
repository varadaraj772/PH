import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
interpreter = tf.lite.Interpreter(model_path="pothole_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Folder path
folder_path = "testImages"

# Supported image extensions
image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

# Loop through all images in folder
for filename in sorted(os.listdir(folder_path)):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)

        # Load and preprocess image
        img = Image.open(image_path).resize((224, 224)).convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Extract confidence
        confidence = float(output[0][0])
        percentage = confidence * 100

        # Print result
        print(f"\n{filename}")
        if confidence > 0.5:
            print(f"Pothole Detected! Confidence -------> {percentage:.2f}%")
        else:
            print(f"No Pothole Detected. Confidence -----> {(100 - percentage):.2f}%")
