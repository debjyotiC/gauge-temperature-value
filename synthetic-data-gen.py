import numpy as np
from PIL import Image
import pandas as pd

NEEDLE = 'images/reference/needle.png'
GAUGE = 'images/reference/gauge.png'
NEEDLES_PATH = 'images/generated/needles/needle_rot_{0}.png'
EDA_GAUGES = 'images/generated/gauges/{0}'
LABELS_CSV = 'images/generated/labels.csv'

def make_needles(angle):
    img = Image.open('./images/needle_rot_0.png')
    rotated_img = img.rotate(-angle, expand=True, resample=Image.BICUBIC)

    # Crop the rotated image to the original size
    cropped_img = rotated_img.crop(
        box=(
            rotated_img.size[0] / 2 - img.size[0] / 2,
            rotated_img.size[1] / 2 - img.size[1] / 2,
            rotated_img.size[0] / 2 + img.size[0] / 2,
            rotated_img.size[1] / 2 + img.size[1] / 2
        )
    )

    # Save the image
    filename = NEEDLES_PATH.format(str(angle).replace(".", "-"))
    cropped_img.save(filename)


def save_gauge(item, num, labels):
    img_gauge = Image.open(GAUGE, 'r')

    # Open the corresponding needle image
    filename = NEEDLES_PATH.format(str(item).replace(".", "-"))
    img_needle = Image.open(filename, 'r')

    # Create a copy of the gauge and overlay the needle
    img_copy = img_gauge.copy()
    img_copy.paste(img_needle.convert('L'), (0, 0), img_needle.convert('RGBA'))

    # Save the synthetic image
    eda_name = f"gauge_{str(num).replace('.', '-')}.png"
    img_copy.save(EDA_GAUGES.format(eda_name))

    # Add label information
    labels.append({'filename': eda_name, 'regression_label': num, 'classification_label': int(num)})


def iterate_gauge(mapping):
    labels = []
    for idx, item in enumerate(mapping):
        save_gauge(mapping[item], item, labels)

    # Save labels as CSV
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(LABELS_CSV, index=False)


def generate_all_needles(step=1.0):
    for angle in np.arange(0, 274.5, step):
        make_needles(angle)

# adjust this according to your gauge (value:angle)
gaugeDegreeMap = {
    0.0: 0.0, 0.1: 4.5, 0.2: 9.0, 0.3: 13.5, 0.4: 18.0, 0.5: 22.5, 0.6: 27.0, 0.7: 31.5, 0.8: 36.0, 0.9: 40.5,
    1.0: 45.0, 1.1: 49.5, 1.2: 54.0, 1.3: 58.5, 1.4: 63.0, 1.5: 67.5, 1.6: 72.0, 1.7: 76.5, 1.8: 81.0, 1.9: 85.5,
    2.0: 90.0, 2.1: 94.5, 2.2: 99.0, 2.3: 103.5, 2.4: 108.0, 2.5: 112.5, 2.6: 117.0, 2.7: 121.5, 2.8: 126.0, 2.9: 130.5,
    3.0: 135.0, 3.1: 139.5, 3.2: 144.0, 3.3: 148.5, 3.4: 153.0, 3.5: 157.5, 3.6: 162.0, 3.7: 166.5, 3.8: 171.0, 3.9: 175.5,
    4.0: 180.0, 4.1: 184.5, 4.2: 189.0, 4.3: 193.5, 4.4: 198.0, 4.5: 202.5, 4.6: 207.0, 4.7: 211.5, 4.8: 216.0, 4.9: 220.5,
    5.0: 225.0, 5.1: 229.5, 5.2: 234.0, 5.3: 238.5, 5.4: 243.0, 5.5: 247.5, 5.6: 252.0, 5.7: 256.5, 5.8: 261.0, 5.9: 265.5,
    6.0: 270.0
}

generate_all_needles()  # Generate all needle images
iterate_gauge(gaugeDegreeMap)  # Generate all gauges
