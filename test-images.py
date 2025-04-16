import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths
NEEDLE = 'images/reference/needle-new.png'
GAUGE = 'images/reference/dial-new.png'
NEEDLE_ZERO = 'images/reference/gauge-needle-new-0.png'
OUTPUT_DIR = 'images/generated/gauges'
CSV_PATH = 'images/generated/gauge_labels.csv'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save needle image for 0°C (starting position)
image = Image.open(NEEDLE).convert('RGBA')
image_rot_0 = image.rotate(45, expand=False, resample=Image.BICUBIC)
image_rot_0.save(NEEDLE_ZERO)

# Define a function to compute classification class (5 bins)
def get_class_label(value):
    if value <= 29:
        return 0
    elif value <= 59:
        return 1
    elif value <= 89:
        return 2
    elif value <= 119:
        return 3
    else:
        return 4

# Prepare CSV file
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label_class", "label_regression"])

    for reading in range(0, 151):  # 0 to 150 inclusive
        angle = round(reading * 1.6, 1)  # Regression target (angle)
        label_class = get_class_label(reading)  # Classification target (0–4)

        # Rotate needle
        needle_img = Image.open(NEEDLE_ZERO).convert('RGBA')
        rotated = needle_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
        cropped = rotated.crop((
            rotated.width // 2 - needle_img.width // 2,
            rotated.height // 2 - needle_img.height // 2,
            rotated.width // 2 + needle_img.width // 2,
            rotated.height // 2 + needle_img.height // 2,
        ))

        # Overlay needle on gauge
        gauge_img = Image.open(GAUGE).convert('RGBA')
        gauge_with_needle = gauge_img.copy()
        gauge_with_needle.paste(cropped, (0, 0), cropped)

        # Save image
        filename = f"gauge_{reading}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        gauge_with_needle.save(save_path)

        # Write CSV: filename, class label, regression angle
        writer.writerow([filename, label_class, reading])

        # Optional: Display
        plt.clf()
        plt.axis('off')
        plt.imshow(gauge_with_needle)
        plt.title(f"Value: {reading} °C | Class: {label_class}")
        plt.pause(0.01)

print(f"All images saved to '{OUTPUT_DIR}' and labels to '{CSV_PATH}'")
