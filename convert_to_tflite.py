import tensorflow as tf

# Use saved_model format
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_pothole")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional quantization
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("pothole_model.tflite", "wb") as f:
    f.write(tflite_model)

print("pothole_model.tflite saved")
