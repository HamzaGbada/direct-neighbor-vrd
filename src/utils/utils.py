import matplotlib
import matplotlib.pyplot as plt


# Define functions to convert bounding box formats
def convert_format1(box):
    x, y, w, h = box
    x2, y2 = x + w, y + h
    return [x, y, x2, y2]


def convert_format2(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y2]


def convert_xmin_ymin(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return [x_min, y_min, x_max, y_max]


def _getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])

# Crop the image based on the bounding box and plot
def plot_cropped_image(image, box, title):

    cropped_image = image.crop(box)
    plt.imshow(cropped_image)
    plt.title(title)
    plt.axis('off')
    plt.savefig(title+'.png')
    plt.show()
