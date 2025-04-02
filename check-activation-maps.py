import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Image path
im_path = "images/generated/gauges/gauge_0-0_var_8.png"


# Function to preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.FIND_EDGES)  # Edge detection
    image = image.resize((28, 28))  # Resize
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize (0 to 1)
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return image

# Load and preprocess the image
im = load_image(im_path)
print("Input Shape:", im.shape)

# Load the trained model
model = tf.keras.models.load_model("models/gauge-classifier.keras")
model.summary()

# Predict class probabilities
pred_probs = model.predict(im)
pred_class = np.argmax(pred_probs, axis=1)[0]  # Get class label

