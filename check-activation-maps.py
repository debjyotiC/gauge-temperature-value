import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import matplotlib as mpl


# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("L").filter(ImageFilter.FIND_EDGES)
    image = image.resize((28, 28))  # Resize to model's input shape
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimension


# Generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(model.inputs, [
        model.get_layer(last_conv_layer_name).output, model.output
    ])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        pred_index = pred_index or tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(conv_output, 0)) / tf.reduce_max(conv_output)
    return heatmap.numpy()


# Overlay heatmap on image
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = np.array(Image.open(img_path).convert("RGB"))
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"](np.arange(256))[:, :3]  # Get colormap
    jet_heatmap = np.uint8(jet[heatmap] * 255)
    jet_heatmap = Image.fromarray(jet_heatmap).resize((img.shape[1], img.shape[0]))
    superimposed_img = Image.blend(Image.fromarray(img), jet_heatmap, alpha)
    superimposed_img.save(cam_path)


# Load model and predict
def main():
    img_path = "images/generated/gauges/gauge_0-0_var_8.png"
    model = tf.keras.models.load_model("models/gauge-classifier.keras")

    img_array = load_image(img_path)
    pred_probs = model.predict(img_array)
    print("Predicted Probabilities:", pred_probs)

    heatmap = make_gradcam_heatmap(img_array, model, "conv2d_1")
    save_and_display_gradcam(img_path, heatmap, cam_path="images/cam.jpg")


if __name__ == "__main__":
    main()