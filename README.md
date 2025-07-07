<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pothole Detection with TensorFlow Lite</title>
  <style>
    body {
      font-family: "Segoe UI", sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 2rem auto;
      padding: 1rem;
      color: #333;
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    code, pre {
      background-color: #f4f4f4;
      padding: 0.4em 0.6em;
      border-radius: 5px;
      font-family: monospace;
    }
    pre {
      overflow-x: auto;
    }
    .box {
      background: #fdfdfd;
      border: 1px solid #ddd;
      border-left: 4px solid #007acc;
      padding: 1em;
      margin: 1em 0;
    }
  </style>
</head>
<body>

<h1>Pothole Detection with TensorFlow Lite</h1>

<p>
  This project provides a complete pipeline to <strong>train</strong>, <strong>convert</strong>, and <strong>run inference</strong> on a deep learning model that detects potholes in road images.
  It is built using <strong>TensorFlow/Keras</strong> and supports deployment using <strong>TFLite</strong>.
</p>

<hr>

<h2>Project Structure</h2>

<pre><code>.
├── train.py                  # Train the CNN model
├── convert_to_tflite.py     # Convert trained model to TFLite
├── predict.py               # Predict with Keras model
├── predict_tflite.py        # Predict with TFLite model
├── requirements.txt         # Python dependencies
├── sachin_pothole/          # Dataset: Pothole images
├── viren_pothole_plain/     # Dataset: Plain road images
├── testImages/              # Test images for predictions
└── .gitignore
</code></pre>

<h2>Model Info</h2>
<ul>
  <li><strong>Input:</strong> 224x224 RGB image</li>
  <li><strong>Model:</strong> Custom CNN (TensorFlow/Keras)</li>
  <li><strong>Classes:</strong> pothole / plain road</li>
</ul>

<h2>Setup</h2>
<pre><code>git clone https://github.com/your-username/pothole-detection.git
cd pothole-detection

python3 -m venv ph-venv
source ph-venv/bin/activate

pip install -r requirements.txt
</code></pre>

<h2>Training</h2>
<p>Make sure the image folders are populated:</p>
<pre><code>python train.py</code></pre>
<p>This saves the model to <code>saved_model_pothole/</code> and plot as <code>training_plot.png</code>.</p>

<h2>Convert to TFLite</h2>
<pre><code>python convert_to_tflite.py</code></pre>
<p>This generates the lightweight model file: <code>pothole_model.tflite</code></p>

<h2>Prediction</h2>
<h3>Using TFLite:</h3>
<pre><code>python predict_tflite.py</code></pre>

<p>Example output:</p>
<div class="box">
  Pothole Detected! Confidence -------> (Confidence: 95.24%)<br>
  or<br>
  No Pothole Detected. Confidence -----> (Confidence: 87.36%)
</div>

<h2>License</h2>
<p>This project is provided for educational and research purposes only.</p>

<h2>Acknowledgements</h2>
<ul>
  <li>TensorFlow & Keras</li>
  <li>Dataset curated by <strong>Sachin</strong> & <strong>Viren</strong></li>
</ul>
<h2>Datasets</h2>
<ul>
  <li><a href="https://www.kaggle.com/datasets/virenbr11/pothole-and-plain-rode-images" target="_blank">Viren’s Dataset: Pothole and Plain Road Images</a></li>
  <li><a href="https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset" target="_blank">Sachin’s Dataset: Pothole Image Dataset</a></li>
</ul>
</body>
</html>
