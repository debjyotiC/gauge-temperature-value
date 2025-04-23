import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Paths
NEEDLE = 'images/reference/needle.png'
NEEDLE_2 = 'images/reference/needle-two.png'
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

# Define function to compute class label
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

# Prepare list for labels
label_data = []

# Iterate over all readings for needle one
for reading in range(0, 151):  # 0 to 150 inclusive
    angle = round(reading * 1.6, 1)
    label_class = get_class_label(reading)

    # Rotate actual needle (needle one)
    needle_img = Image.open(NEEDLE_ZERO).convert('RGBA')
    rotated_needle = needle_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
    cropped_needle = rotated_needle.crop((
        rotated_needle.width // 2 - needle_img.width // 2,
        rotated_needle.height // 2 - needle_img.height // 2,
        rotated_needle.width // 2 + needle_img.width // 2,
        rotated_needle.height // 2 + needle_img.height // 2,
    ))

    for needle2_angle in range(271):  # 0 to 270 inclusive
        needle_2_img = Image.open(NEEDLE_2).convert('RGBA')
        rotated_needle_2 = needle_2_img.rotate(-needle2_angle, expand=True, resample=Image.BICUBIC)
        cropped_needle_2 = rotated_needle_2.crop((
            rotated_needle_2.width // 2 - needle_2_img.width // 2,
            rotated_needle_2.height // 2 - needle_2_img.height // 2,
            rotated_needle_2.width // 2 + needle_2_img.width // 2,
            rotated_needle_2.height // 2 + needle_2_img.height // 2,
        ))

        # Overlay both needles on gauge
        gauge_img = Image.open(GAUGE).convert('RGBA')
        gauge_with_needles = gauge_img.copy()
        gauge_with_needles.paste(cropped_needle_2, (0, 0), cropped_needle_2)
        gauge_with_needles.paste(cropped_needle, (0, 0), cropped_needle)

        # Save image
        filename = f"gauge_{reading:03d}_{needle2_angle:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        gauge_with_needles.save(save_path)

        # Save labels
        label_data.append({
            "filename": filename,
            "label_class": label_class,
            "label_regression": reading,
        })

# Save label data
pd.DataFrame(label_data).to_csv(CSV_PATH, index=False)
print(f"{len(label_data)} images saved to '{OUTPUT_DIR}' and labels to '{CSV_PATH}'")
