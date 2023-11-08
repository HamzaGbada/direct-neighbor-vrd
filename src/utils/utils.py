import numpy as np
from matplotlib import pyplot as plt


# Define functions to convert bounding box formats
def convert_format1(box):
    x, y, w, h = box
    x2, y2 = x + w, y + h
    return [x, y, x2, y2]


def convert_format2(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y2]


def convert_xmin_ymin(box):
    if len(box) == 4:
        return box
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return [x_min, y_min, x_max, y_max]


def get_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


# Crop the image based on the bounding box and plot
def plot_cropped_image(image, box, title):
    cropped_image = image.crop(box)
    plt.imshow(cropped_image)
    plt.title(title)
    plt.axis("off")
    plt.savefig(title + ".png")
    plt.show()


def plots(epochs, train_losses, val_losses, type="Loss", name="CORD"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epochs + 1), train_losses, label="Train " + type)
    plt.plot(np.arange(1, epochs + 1), val_losses, label="Validation " + type)
    plt.xlabel("Epochs")
    plt.ylabel(type)
    plt.legend()
    plt.title("Training and Validation " + type)
    plt.savefig(name + "_" + type + "_plot.png")
    plt.show()
