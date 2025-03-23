from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image  # Import PIL to open images

im_path = "images/generated/gauges"

gauge_images = listdir(im_path)

for im in gauge_images:
    full_path = join(im_path, im)
    image = Image.open(full_path)  # Open the image file
    plt.clf()
    plt.imshow(image)
    plt.axis('off')  # Hide axis for better viewing
    plt.pause(1)


