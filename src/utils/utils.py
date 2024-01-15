import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

from src.graph_pack.VRD_graph import VRD2Graph
from src.utils.setup_logger import logger
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


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


def process_labels(dataset):
    # FIXME: THIS is computing the label of all the dataset (if we got an error in metic computing may from this)
    #  convert it to processing label of each doc_index
    num_labels = (
        30
        if type(dataset).__name__ == "CORD"
        else (
            26
            if type(dataset).__name__ == "WILDRECEIPT"
            else (
                5
                if type(dataset).__name__ == "SROIE"
                else (4 if type(dataset).__name__ in ["FUNSD", "XFUND"] else None)
            )
        )
    )
    X = torch.arange(0, num_labels).view(-1, 1)

    if type(dataset).__name__ == "CORD":
        encoded_dic = {
            "menu.sub_cnt": 0,
            "sub_total.othersvc_price": 1,
            "total.total_price": 2,
            "menu.etc": 3,
            "sub_total.discount_price": 4,
            "menu.unitprice": 5,
            "menu.discountprice": 6,
            "void_menu.price": 7,
            "menu.nm": 8,
            "total.menutype_cnt": 9,
            "sub_total.subtotal_price": 10,
            "menu.sub_nm": 11,
            "void_menu.nm": 12,
            "menu.sub_unitprice": 13,
            "menu.sub_etc": 14,
            "menu.cnt": 15,
            "menu.vatyn": 16,
            "total.total_etc": 17,
            "total.menuqty_cnt": 18,
            "total.cashprice": 19,
            "menu.num": 20,
            "total.changeprice": 21,
            "sub_total.tax_price": 22,
            "sub_total.etc": 23,
            "menu.price": 24,
            "total.creditcardprice": 25,
            "total.emoneyprice": 26,
            "sub_total.service_price": 27,
            "menu.itemsubtotal": 28,
            "menu.sub_price": 29,
        }
        labels = (
            encoded_dic[x]
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        )
    elif type(dataset).__name__ == "XFUND":
        encoded_dic = {"question": 0, "answer": 1, "other": 2, "header": 3}
        labels = (
            encoded_dic[x]
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        )
    elif type(dataset).__name__ == "FUNSD":
        encoded_dic = {"question": 0, "answer": 1, "other": 2, "header": 3}
        labels = (
            encoded_dic[x]
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        )
    else:
        labels = (
            x
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        )

    labels = torch.from_numpy(
        OneHotEncoder(sparse=False)
        .fit(X)
        .transform(torch.tensor(list(labels)).reshape(-1, 1))
    )

    name = type(dataset).__name__

    return labels, name


def process_and_save_dataset(dataset, text_model, args, split="train", device="cuda"):
    logger.info(f" Building {type(dataset).__name__} {split} start ")
    labels, name = process_labels(dataset)
    for doc_index in tqdm(range(len(dataset))):
        bbox = dataset.data[doc_index][1]["boxes"]
        text_units = dataset.data[doc_index][1]["text_units"]

        features = [text_model.embed_text(text) for text in text_units]
        graph = VRD2Graph(bbox, labels, features, device=device)
        graph.connect_boxes()
        graph.create_graph()

        graph.save_graph(
            path=f"data/{args.dataset}/{split}",
            graph_name=f"{args.dataset}_{split}_graph{doc_index}",
        )
    logger.info(f" the graph data/{args.dataset}/{split} is saved successfully")


def compute_f1_score(label, pred):
    threshold = 0.5
    predicted_labels = (pred > threshold).float()

    # Convert tensors to numpy arrays for compatibility with scikit-learn
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = label.cpu().numpy()

    # Compute the F1 score
    return f1_score(true_labels, predicted_labels, average="micro")


def plots(epochs, train_losses, val_losses, type="Loss", name="CORD"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epochs + 1), train_losses, label="Train " + type)
    plt.plot(np.arange(1, epochs + 1), val_losses, label="Validation " + type)
    plt.xlabel("Epochs")
    plt.ylabel(type)
    plt.legend()
    plt.title("Training and Validation " + type)
    plt.savefig(name + "_" + type + "_plot.png")
