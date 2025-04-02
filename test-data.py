import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from PIL import Image, ImageFilter

# Image path
im_path = "images/generated/gauges/gauge_0-0_var_3.png"

# Model input dimensions
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1

# Function to preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.FIND_EDGES)  # Edge detection
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize (0 to 1)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Load and preprocess the image
im = load_image(im_path)
im = np.expand_dims(im, axis=0)  # Add batch dimension
print("Input Shape:", im.shape)  # Should print (1, 28, 28, 1)

# Load the trained model
model = tf.keras.models.load_model("models/gauge-classifier.keras")
model.summary()

# Predict class probabilities
pred_probs = model.predict(im)
pred_class = np.argmax(pred_probs, axis=1)[0]  # Extract integer value

# Create GradCAM++ object
gradcam = GradcamPlusPlus(model, model_modifier=None, clone=True)


# Generate cam with GradCAM++
cam = gradcam(CategoricalScore(pred_class), im)

# Convert CAM to colormap
heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

# Display original image with heatmap overlay
plt.imshow(im[0, :, :, 0], cmap="gray")  # Show original grayscale image
plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap
plt.axis("off")
plt.show()
