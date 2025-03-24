from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter  # Import PIL to open images

im_path = "images/generated/gauges"

gauge_images = listdir(im_path)

for im in gauge_images:
    full_path = join(im_path, im)
    image = Image.open(full_path)
    image = image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    class_name = im.split("-")

    print(class_name)

    plt.clf()
    plt.title(f"{class_name}")
    plt.imshow(image)
    # plt.axis('off')  # Hide axis for better viewing
    plt.pause(1)


