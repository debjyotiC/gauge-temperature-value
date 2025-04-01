import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import random
import os

NEEDLE = 'images/reference/needle.png'
GAUGE = 'images/reference/gauge.png'
NEEDLES_PATH = 'images/generated/needles/needle_rot_{0}.png'
EDA_GAUGES = 'images/generated/gauges/{0}'
LABELS_CSV = 'images/generated/labels.csv'

# Ensure necessary directories exist
os.makedirs('images/generated/needles', exist_ok=True)
os.makedirs('images/generated/gauges', exist_ok=True)

def format_angle(angle):
    """Format angle to avoid incorrect filename representations."""
    return "{:.1f}".format(angle).replace(".", "-").replace("-", "neg-") if angle < 0 else "{:.1f}".format(angle).replace(".", "-")

def make_needles(angle):
    img = Image.open(NEEDLE)
    rotated_img = img.rotate(-angle, expand=True, resample=Image.BICUBIC)

    # Crop the rotated image to original size
    cropped_img = rotated_img.crop(
        box=(
            rotated_img.size[0] / 2 - img.size[0] / 2,
            rotated_img.size[1] / 2 - img.size[1] / 2,
            rotated_img.size[0] / 2 + img.size[0] / 2,
            rotated_img.size[1] / 2 + img.size[1] / 2
        )
    )

    # Save the image
    filename = NEEDLES_PATH.format(format_angle(angle))
    cropped_img.save(filename)

def save_gauge(item, num, labels, variation=10):
    img_gauge = Image.open(GAUGE)

    for i in range(variation):
        # Small perturbation in angle for augmentation
        angle_variation = random.uniform(-0.5, 0.5)
        perturbed_angle = item + angle_variation

        needle_filename = NEEDLES_PATH.format(format_angle(perturbed_angle))
        if not os.path.exists(needle_filename):
            print(f"Warning: Needle image {needle_filename} not found. Skipping...")
            continue

        img_needle = Image.open(needle_filename)

        # Overlay needle onto gauge
        img_copy = img_gauge.copy()
        img_copy.paste(img_needle.convert('L'), (0, 0), img_needle.convert('RGBA'))

        # Save synthetic image
        eda_name = f"gauge_{format_angle(num)}_var_{i}.png"
        img_copy.save(EDA_GAUGES.format(eda_name))

        # Edge detection augmentation
        # img_edges = img_copy.convert("L").filter(ImageFilter.FIND_EDGES)
        # eda_edge_name = f"gauge_{format_angle(num)}_var_{i}_edges.png"
        # img_edges.save(EDA_GAUGES.format(eda_edge_name))

        # Store label data
        labels.append({'filename': eda_name, 'regression_label': num, 'classification_label': int(num)})

def iterate_gauge(mapping, variation=10):
    labels = []
    for idx, item in enumerate(mapping):
        save_gauge(mapping[item], item, labels, variation)

    # Save labels as CSV
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(LABELS_CSV, index=False)

def generate_all_needles(step=0.1):
    for angle in np.arange(0, 274.5, step):
        make_needles(angle)

# Gauge mapping
gaugeDegreeMap = {
    i / 10: round(i * 4.5 / 10, 1) for i in range(61)
}

generate_all_needles()
iterate_gauge(gaugeDegreeMap, variation=20)  # Generate 10 variations per gauge level
