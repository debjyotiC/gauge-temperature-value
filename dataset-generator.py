import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

# Paths
NEEDLE = 'images/reference/needle.png'
NEEDLE_2 = 'images/reference/needle-two.png'  # Second needle image
GAUGE = 'images/reference/gauge.png'
NEEDLE_ZERO = 'images/reference/needle_rot_0.png'
OUTPUT_DIR = 'images/generated/gauges'
CSV_PATH = 'images/generated/gauge_labels.csv'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save needle image for 0°C (starting position)
image = Image.open(NEEDLE).convert('RGBA')
image_rot_0 = image.rotate(0, expand=False, resample=Image.BICUBIC)
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

# Generate 500 random readings between 0 and 150 (inclusive)
num_images = 500
readings = sorted(random.choices(range(0, 151), k=num_images))

# Store label data in a list
label_data = []

# Generate images and collect label info
for idx, reading in enumerate(readings):
    angle = round(reading * 1.6, 1)  # Regression target (angle)
    label_class = get_class_label(reading)  # Classification target (0–4)

    # Rotate actual needle
    needle_img = Image.open(NEEDLE_ZERO).convert('RGBA')
    rotated_needle = needle_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
    cropped_needle = rotated_needle.crop((
        rotated_needle.width // 2 - needle_img.width // 2,
        rotated_needle.height // 2 - needle_img.height // 2,
        rotated_needle.width // 2 + needle_img.width // 2,
        rotated_needle.height // 2 + needle_img.height // 2,
    ))

    # Generate random angle for second needle
    rand_angle = round(random.uniform(0, 240), 1)
    needle_2_img = Image.open(NEEDLE_2).convert('RGBA')
    rotated_needle_2 = needle_2_img.rotate(-rand_angle, expand=True, resample=Image.BICUBIC)
    cropped_needle_2 = rotated_needle_2.crop((
        rotated_needle_2.width // 2 - needle_2_img.width // 2,
        rotated_needle_2.height // 2 - needle_2_img.height // 2,
        rotated_needle_2.width // 2 + needle_2_img.width // 2,
        rotated_needle_2.height // 2 + needle_2_img.height // 2,
    ))

    # Overlay both needles on gauge
    gauge_img = Image.open(GAUGE).convert('RGBA')
    gauge_with_needles = gauge_img.copy()
    gauge_with_needles.paste(cropped_needle_2, (0, 0), cropped_needle_2)  # Random needle first (behind)
    gauge_with_needles.paste(cropped_needle, (0, 0), cropped_needle)      # Actual needle on top

    # Save image
    filename = f"gauge_{idx:04d}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    gauge_with_needles.save(save_path)

    # Append to label list
    label_data.append({
        "filename": filename,
        "label_class": label_class,
        "label_regression": reading,
        "needle2_angle": rand_angle
    })

    # Optional: display the image during generation
    plt.clf()
    plt.axis('off')
    plt.imshow(gauge_with_needles)
    plt.title(f"Val: {reading}°C | Class: {label_class} | Needle2: {rand_angle:.1f}°")
    plt.pause(0.001)

# Create DataFrame and save CSV
df = pd.DataFrame(label_data)
df.to_csv(CSV_PATH, index=False)

print(f"{num_images} images saved to '{OUTPUT_DIR}' and labels to '{CSV_PATH}'")