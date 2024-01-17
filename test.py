import os
import random
import unittest
import warnings
from pathlib import Path

import dgl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from datasets import load_dataset
from dgl import load_graphs
from dgl.data import KarateClubDataset
from dgl.dataloading import DataLoader, GraphDataLoader
from shapely.geometry import Polygon
from sklearn.preprocessing import OneHotEncoder
from torch import nn, tensor, relu
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torchmetrics.functional.classification import multilabel_accuracy
from torchvision import transforms
from transformers import BertTokenizer

from src.cnn_embedding.unet_embedding import (
    UNet,
    EfficientNetV2MultiClass,
    EmbeddingModel,
)
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.funsd_dataloader import FUNSD
from src.dataloader.graph_dataset import GraphDataset
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.dataloader.xfund_dataloader import XFUND
from src.graph_pack.VRD_graph import VRD2Graph
from src.graph_pack.graph_model import WGCN, GCN, GAT
from src.utils.setup_logger import logger
from src.utils.utils import (
    convert_xmin_ymin,
    convert_format1,
    convert_format2,
    plot_cropped_image,
    get_area,
)
from src.word_embedding.BERT_embedding import (
    BertSentenceClassification,
    TextEmbeddingModel,
)
from train_cnn_for_classification import (
    image_dataloader,
    compute_f1_score,
    compute_accuracy,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class TestDataLoader(unittest.TestCase):
    def test_cord(self):
        # train_set = CORD(train=True, download=True)
        # train_set2 = SROIE(train=False)
        # train_set = SROIE(train=True)
        # train_set3 = CORD(train=False, download=True)
        train_set = WILDRECEIPT(train=True, download=True)
        # train_set5 = WILDRECEIPT(train=False, download=True)

        logger.debug(f"the cord dataset {train_set.root}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")

    def test_wildreceipt(self):
        train = True
        if train:
            dataset = load_dataset("Theivaprakasham/wildreceipt")["train"]
            # dataset = load_dataset("jinhybr/WildReceipt")['train']
        else:
            dataset = load_dataset("Theivaprakasham/wildreceipt")["test"]
            # dataset = load_dataset("jinhybr/WildReceipt")['test']
        train_set = WILDRECEIPT(train=train, download=True)
        doc_index = 1
        word_index = 3
        # logger.debug(f"train set data # dataset[1]['words']: {dataset[doc_index]['words']}")
        # logger.debug(f"train set data # dataset[1]['bboxes']: {dataset[doc_index]['bboxes']}")
        # logger.debug(f"train set data # dataset[1]['ner_tags']: {dataset[doc_index]['ner_tags']}")
        # logger.debug(f"train set data # dataset[1]['image_path']: {dataset[doc_index]['image_path']}")
        # logger.debug(f"train set data # dataset[1]['words']: {len(dataset[doc_index]['words'])}")
        # logger.debug(f"train set data # dataset[1]['bboxes']: {len(dataset[doc_index]['bboxes'])}")
        # logger.debug(f"train set data # dataset[1]['ner_tags']: {len(dataset[doc_index]['ner_tags'])}")
        # logger.debug(f"train set data # dataset[1]['image_path']: {len(dataset[doc_index]['image_path'])}")
        # logger.debug(f"train set data # doctr implementation: {train_set.data[doc_index]}")

        filename = train_set.data[doc_index][0]
        image_path = os.path.join(train_set.root, filename)
        image_doctr = Image.open(image_path)
        plt.imshow(image_doctr)
        plt.title("The Current Image, Doctr Implementation")
        plt.show()

        image_hugging = Image.open(dataset[doc_index]["image_path"])
        plt.imshow(image_hugging)
        plt.title("The Current Image, HuggingFace Implementation")
        plt.show()

        bbox_doctr = train_set.data[doc_index][1]["boxes"][word_index]
        text_unit_doctr = train_set.data[doc_index][1]["text_units"][word_index]
        logger.debug(f"Doctr bounding boxes : {bbox_doctr}")

        bbox_hugging_face = dataset[doc_index]["bboxes"][word_index]
        text_unit_face = dataset[doc_index]["words"][word_index]
        logger.debug(f"HuggingFace bounding boxes : {bbox_hugging_face}")

        common_box_doctr = convert_xmin_ymin(bbox_doctr)

        common_box_hugface_1 = convert_format1(bbox_hugging_face)
        common_box_hugface_2 = convert_format2(bbox_hugging_face)

        plot_cropped_image(
            image_doctr,
            common_box_doctr,
            f"Bouding boxes (my implementation) \n its associated text unit: {text_unit_doctr}",
        )
        plot_cropped_image(
            image_hugging,
            common_box_hugface_1,
            f"Hugging Face Bouding boxes (x,y,w,h format) \n its associated text unit: {text_unit_face}",
        )
        plot_cropped_image(
            image_hugging,
            common_box_hugface_2,
            f"Hugging Face Bouding boxes (x1,y1,x2, y2 format) \n its associated text unit: {text_unit_face}",
        )

        # logger.debug(f"the cord dataset {train_set.data}")
        self.assertEqual(
            os.path.basename(filename),
            os.path.basename(dataset[doc_index]["image_path"]),
        )

    def test_sentences_label(self):
        dataset = FUNSD(train=True, download=True)

        sentences = [
            x
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["text_units"]
        ]
        labels = [
            x
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        ]

        self.assertEqual(sentences[2], dataset.data[0][1]["text_units"][2])
        self.assertEqual(labels[2], dataset.data[0][1]["labels"][2])

    def test_sroie(self):
        dataset = FUNSD(train=True)
        for doc_index in range(len(dataset)):
            for x in dataset.data[doc_index][1]["labels"]:
                logger.debug(x)
        labels = [
            x
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        ]
        # nbr_of_node = train_set.data[0][1]['boxes'].shape
        # logger.debug(f"The shape of bbox in the first doc Dataset: \n{nbr_of_node}")
        # logger.debug(f"The shape of bbox in the first doc Dataset: \n{len(train_set.data[0][1]['boxes'])}")
        # logger.debug(f"The bbox in the first doc Dataset: \n{train_set.data[0]}")
        # # logger.debug(f"The bbox in the first doc Dataset: \n{train_set[55]}")
        # plt.imshow(train_set[0][0].permute(1, 2, 0))
        # plt.show()
        # # logger.debug(f"The size text_unit_list for all dataset: {len(train_set.text_units)}")
        # logger.debug(f"text_unit_list for all dataset: {train_set.text_units}")
        # j = 0
        # for i in train_set.data:
        #     j+=1
        logger.debug(f"The FUNSD dataset: {labels}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")

    def test_xfund(self):
        train_set2 = FUNSD(train=True)
        dataset = XFUND(data_folder="data/fr.train.json", train=True)
        logger.debug(f"The XFUND dataset: {dataset.data}")
        logger.debug(f"The sroie dataset: {train_set2.data}")

    def test_cropped_bbox(self):
        # Open the image using PIL
        train_set = FUNSD(train=True)
        path = os.path.join(train_set.root, train_set.data[0][0])
        logger.debug(f"the path is {path}")
        image = Image.open(path).convert("L")

        # Define the bounding box coordinates (left, upper, right, lower)
        bbox = train_set.data[0][1]["boxes"][0]
        text_units = train_set.data[0][1]["text_units"][0]
        logger.debug(f"bbox {bbox}")
        logger.debug(f"text_units {text_units}")

        # Crop the image
        convert_tensor = transforms.ToTensor()
        cropped_image = convert_tensor(image.crop(bbox))
        logger.debug(f"shape before remove channel{cropped_image.shape}")

        # Save or display the cropped image
        plt.imshow(image, cmap="gray")
        # plt.imshow(cropped_image, cmap='gray')
        plt.show()

    def test_image_dataloader(self):
        # Open the image using PIL
        train_set = FUNSD(train=False)
        cropped_images, boxes, labels, text_units = image_dataloader(train_set)
        path = os.path.join(train_set.root, train_set.data[0][0])
        logger.debug(f"the path is {labels}")
        image = Image.open(path)

        logger.debug(f"text_units {text_units[5]}")

        plt.imshow(image, cmap="gray")
        plt.imshow(cropped_images[5], cmap="gray")
        plt.show()

    def test_unet_test(self):
        # Open the image using PIL
        inputs = torch.rand(3, 63, 45).to(device="cuda")
        model = UNet(3, 5).to(device="cuda")
        #
        outputs = model(inputs)
        # logger.debug(f"outputs shape {outputs.shape}")
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.tensor([0]).reshape(-1, 1)
        X = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1)
        enc = OneHotEncoder(sparse=False)
        enc.fit(X)
        logger.debug(f"Labels {labels}")
        labels = torch.from_numpy(enc.transform(labels)).to(device="cuda")

        loss = loss_fn(outputs, labels)

        logger.debug(f"Labels {labels}")

        logger.debug(f"loss_fn shape {loss.shape}")
        logger.debug(f"loss_fn {loss}")

        inputs = torch.rand(3, 15, 52).to(device="cuda")

        outputs = model(inputs)
        logger.debug(f"loss_fn shape {outputs.shape}")

    def test_label(self):
        dataset = CORD(train=True, download=True)
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
        labels = [
            encoded_dic[x]
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["labels"]
        ]
        logger.debug(f"label {labels}")

    def test_bbox_area(self):
        dataset = CORD(train=True, download=True)
        bbox = [
            get_area(x)
            for doc_index in range(len(dataset))
            for x in dataset.data[doc_index][1]["boxes"]
        ]
        bbox.sort()
        logger.debug(f"Sorted area bbox area : {bbox}")

    def test_efficient_test(self):
        # Open the image using PIL
        inputs = torch.rand(3, 63, 45).to(device="cuda")
        inputs = inputs.unsqueeze(0)
        logger.debug(f"inputs shape {inputs.shape}")
        logger.debug(inputs.device)
        model = EfficientNetV2MultiClass(5).to(torch.device("cuda"))
        #
        outputs = model(inputs)
        # logger.debug(f"outputs shape {outputs.shape}")
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.tensor([0]).reshape(-1, 1)
        X = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1)
        enc = OneHotEncoder(sparse=False)
        enc.fit(X)
        logger.debug(f"Labels {labels.shape}")
        logger.debug(f"Labels {labels.shape}")
        logger.debug(f"Labels {labels.shape}")
        logger.debug(f"Labels {labels.shape}")
        logger.debug(f"Labels {labels.shape}")
        labels = torch.from_numpy(enc.transform(labels)).to(device="cuda")

        loss = loss_fn(outputs, labels)

        logger.debug(f"Labels {labels}")

        logger.debug(f"loss_fn shape {loss.shape}")
        logger.debug(f"loss_fn {loss}")

        inputs = torch.rand(3, 15, 52).to(device="cuda")
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        logger.debug(f"loss_fn shape {outputs.shape}")

    def test_efficient_train_test(self):
        X = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1)
        enc = OneHotEncoder(sparse=False)
        enc.fit(X)

        num_classes = 5
        num_epochs = 2
        data_size = 2

        model = EfficientNetV2MultiClass(num_classes).to(torch.device("cuda"))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train_losses = []  # To store training loss for each epoch
        train_f1 = []  # To store validation loss for each epoch

        for epoch in range(num_epochs):
            logger.debug(f"epoch {epoch+1} | {num_epochs} processing")
            model.train()

            total_train_loss = 0
            total_f1_score = 0

            for image_index in range(data_size):
                logger.debug(f"image {image_index + 1} | {data_size} processing")
                inputs, labels = torch.rand(
                    1, 1, random.randint(70, 100), random.randint(70, 100)
                ).to(device="cuda"), torch.randint(0, 4, [1, 1]).to(device="cuda")

                optimizer.zero_grad()

                outputs = model(inputs)
                labels = torch.from_numpy(enc.transform(labels.cpu())).to(device="cuda")
                logger.debug("labels.view(-1)")
                logger.debug(labels)
                logger.debug("outputs.view(-1)")
                logger.debug(outputs)
                logger.debug("labels.view(-1)")
                logger.debug(labels.shape)
                logger.debug("outputs.view(-1)")
                logger.debug(outputs.shape)
                f1_score_train = compute_f1_score(labels.view(-1), outputs.view(-1))
                accuracy = compute_accuracy(labels.view(-1), outputs.view(-1))
                logger.debug(f"f1_score_train {f1_score_train}")
                logger.debug(f"accuracy{accuracy}")

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_f1_score += f1_score_train
                total_train_loss += loss.item()

            avg_f1_score_train = total_f1_score / data_size
            avg_train_loss = total_train_loss / data_size
            train_losses.append(avg_train_loss)
            train_f1.append(avg_f1_score_train)

    def test_accuracy_pytorch(self):
        # TODO: use this accuracy implementation and check this link for more: https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#multilabel-accuracy
        target = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]])
        preds = torch.tensor(
            [[0.12, 0.15, 0.3, 0.22, 0.21], [0.12, 0.05, 0.8, 0.02, 0.01]]
        )
        accuracy = multilabel_accuracy(preds, target, num_labels=5, average="macro")
        accuracy1 = compute_accuracy(target, preds)
        logger.debug(f" the accuracy is {accuracy}")
        logger.debug(f" the accuracy is {accuracy1}")

    def test_pretrained_model_cnn(self):
        model = EfficientNetV2MultiClass(30)
        state_dict = torch.load("Unet_classification.pth")

        model.load_state_dict(state_dict)  # works

        reshaping_layers = nn.Sequential(
            nn.Linear(30, 500),  # Linear layer to reshape from 30 to 500 features
            nn.Tanh(),  # You can add activation functions as needed
        )

        # Transfer the model and reshaping layers to the GPU
        model.to(device="cuda")
        reshaping_layers.to(device="cuda")

        model.eval()
        # model.eval()
        # model = torch.load("Unet_classification.pth")
        # checkpoint = torch.load('Unet_classification.pth')
        # model = checkpoint['model']
        # model.load_state_dict(checkpoint['state_dict'])
        # model.eval()
        inputs = torch.rand(1, 1, 63, 45).to(device="cuda")
        output = model(inputs)

        reshaped_output = reshaping_layers(output)

        logger.debug(f"output shape{reshaped_output.shape}")
        logger.debug(f"output shape{reshaped_output}")

    def test_pretrained_model_word(self):
        model = BertSentenceClassification(30)
        state_dict = torch.load("CORD_word_classification.pth")

        model.load_state_dict(state_dict)  # works

        # x =
        # model.to(device="cuda")
        reshaping_layers = nn.Sequential(
            nn.Linear(30, 500),  # Linear layer to reshape from 30 to 500 features
            nn.Tanh(),  # You can add activation functions as needed
        )

        # Transfer the model and reshaping layers to the GPU
        model.to(device="cuda")
        reshaping_layers.to(device="cuda")

        model.eval()
        sentence = "18.167$"
        label = 2
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        max_len = 128

        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_len,
            return_tensors="pt",
            pad_to_max_length=True,
            truncation=True,
        )

        batch = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.float),
        }
        input_ids = batch["input_ids"].to(device="cuda")
        attention_mask = batch["attention_mask"].to(device="cuda")
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        # labels = batch['label'].to(device="cuda")

        outputs = model(input_ids, attention_mask)

        reshaped_output = reshaping_layers(outputs)

        logger.debug(f"reshaped_output shape{reshaped_output.shape}")
        logger.debug(f"output shape{outputs.shape}")
        # logger.debug(f"output shape{reshaped_output}")

    def test_model_word(self):
        text_model = TextEmbeddingModel(model_path="CORD_word_classification.pth")

        # Embedding a sentence
        sentence_embedding = text_model.embed_text("18.167$")
        logger.debug(f"output shape{sentence_embedding.shape}")
        logger.debug(f"output shape{sentence_embedding}")

    def test_model_cnn(self):
        model = EmbeddingModel(
            num_classes=30, feat_size=500, model_path=Path("Unet_classification.pth")
        )
        model.to_device("cuda")
        model.eval()
        # model.eval()
        # model = torch.load("Unet_classification.pth")
        # checkpoint = torch.load('Unet_classification.pth')
        # model = checkpoint['model']
        # model.load_state_dict(checkpoint['state_dict'])
        # model.eval()
        inputs = torch.rand(1, 1, 63, 45).to(device="cuda")
        output = model(inputs)

        # reshaped_output = reshaping_layers(output)

        logger.debug(f"output shape{output.shape}")
        logger.debug(f"output shape{output}")

    def test_connect_bbox(self):
        bounding_boxes = [
            # (6, 1, 10, 10),
            # (11, 15, 15, 10),  # mid
            # (21, 1, 17, 10),  # Upper
            # (10, 70, 25, 10),  # super low
            # (25, 16, 17, 10),
            (35, 5, 42, 10),
            (20, 32, 5, 5),
            (30, 24, 17, 10),
            (40, 14, 5, 10),
            (60, 34, 17, 10),
            (77, 54, 17, 10),
            (87, 66, 17, 10),
            # (90, 74, 17, 10),
            # (21, 32, 17, 10),  # low
        ]
        # logger.debug(f" debug first {bounding_boxes}")
        # bounding_boxes.sort()
        # logger.debug(f" debug second {bounding_boxes}")
        white_array = np.ones((150, 150), dtype=np.uint8) * 255
        graph = VRD2Graph(bounding_boxes)
        graph.connected_boxes()
        connected_indices = graph.connection_index
        logger.debug(connected_indices)

        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(white_array)
        # Draw bounding boxes on the white array
        for bbox in bounding_boxes:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
            # draw_bounding_box(white_array, bbox)

        for i in range(len(connected_indices)):
            for j in connected_indices[i]:
                logger.debug(i)
                logger.debug(j)
                draw_line_between_bounding_boxes(bounding_boxes[i], bounding_boxes[j])
        # Draw lines from the center of the bounding boxes to the other center
        # for bbox1, bbox2 in zip(bounding_boxes, bounding_boxes[1:]):
        #     draw_line_between_bounding_boxes(bbox1, bbox2)

        # Display the white array with the bounding boxes and lines

        plt.imshow(white_array)
        plt.axis("off")
        plt.show()

    def test_shaply(self):
        rectangle = Polygon([(3, 5), (3, 8), (4, 8), (4, 5)])
        polygon = Polygon([(0, 0), (0, 2), (6, 8), (7, 8), (7, 4), (1, 0)])
        intersection = rectangle.intersection(polygon)
        if intersection.is_empty:
            logger.debug("No part of the rectangle is inside the polygon")
        else:
            logger.debug("A part of the rectangle is inside the polygon")

    def test_dgl(self):
        u, v = tensor([0, 1, 2]), tensor([2, 3, 4])
        logger.debug(torch.cuda.is_available())
        logger.debug(torch.cuda.get_device_name(0))
        g = dgl.graph((u, v))
        g.ndata["x"] = torch.randn(5, 3)
        logger.debug(g.device)
        cuda_g = g.to("cuda:0")
        logger.debug(cuda_g.device)
        logger.debug(cuda_g.ndata["x"].device)
        u, v = u.to("cuda:0"), v.to("cuda:0")
        g = dgl.graph((u, v))
        logger.debug(g.device)
        k = torch.randn(10000, 50000)
        k_g = k.to("cuda:0")
        logger.debug(k_g)

    def test_create_graph(self):
        bounding_boxes = [
            # (6, 1, 10, 10),
            # (11, 15, 15, 10),  # mid
            # (21, 1, 17, 10),  # Upper
            # (10, 70, 25, 10),  # super low
            # (25, 16, 17, 10),
            [77, 54, 17, 10],
            [35, 5, 42, 10],
            [20, 32, 5, 5],
            [30, 24, 17, 10],
            [40, 14, 5, 10],
            [60, 34, 17, 10],
            [87, 66, 17, 10],
            # [90, 74, 17, 10],
            # (21, 32, 17, 10),  # low
        ]
        labels = [
            # (6, 1, 10, 10),
            # (11, 15, 15, 10),  # mid
            # (21, 1, 17, 10),  # Upper
            # (10, 70, 25, 10),  # super low
            # (25, 16, 17, 10),
            77,
            35,
            20,
            30,
            40,
            60,
            87
            # (90, 74, 17, 10),
            # (21, 32, 17, 10),  # low
        ]
        feat = torch.zeros(len(bounding_boxes), dtype=torch.float32)
        graph = VRD2Graph(np.array(bounding_boxes), labels, feat, device="cpu")
        graph.connect_boxes()
        graph.create_graph()
        logger.debug(graph.edges)
        graph.save_graph(path="data/samir", graph_name="bob")
        graph.load_graph(path="data/samir", graph_name="bob")
        graph.plot_dgl_graph()

    def test_plot_fig(self):
        train_set = SROIE(train=True)
        logger.debug(train_set.data[5])
        file_path = os.path.join(train_set.root, train_set.data[5][0])
        # Convert the image to a NumPy array (optional but may be needed)
        img_array = plt.imread(file_path)

        # Plot the image

        bbox = train_set.data[5][1]["boxes"][:2]
        labels = torch.zeros(len(bbox), dtype=torch.float32)
        feat = torch.zeros(len(bbox), dtype=torch.float32)
        # logger.debug(f" debug first {bounding_boxes}")
        # bounding_boxes.sort()
        # logger.debug(f" debug second {bounding_boxes}")

        graph = VRD2Graph(bbox, labels, feat)
        graph.connect_boxes()
        connected_indices = graph.connection_index
        logger.debug(connected_indices)

        fig, ax = plt.subplots()

        # Draw bounding boxes on the white array
        for b in bbox:
            rect = patches.Rectangle(
                (b[0], b[1]),
                b[2],
                b[3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
            # draw_bounding_box(white_array, bbox)

        for i in range(len(connected_indices)):
            for j in connected_indices[i]:
                logger.debug(i)
                logger.debug(j)
                draw_line_between_bounding_boxes(bbox[i], bbox[j])
        # Draw lines from the center of the bounding boxes to the other center
        # for bbox1, bbox2 in zip(bounding_boxes, bounding_boxes[1:]):
        #     draw_line_between_bounding_boxes(bbox1, bbox2)

        # Display the white array with the bounding boxes and lines

        plt.imshow(img_array, cmap="gray")
        plt.axis("off")
        plt.show()

    def test_dataset_graph(self):
        # g = load_graphs("data/CORD/train/CORD_train_graph0.bin")
        # logger.debug(g[0][0].num_edges())
        # logger.debug(g[0][0].edata["weight"].shape)
        dataset = GraphDataset("SROIE")
        
        graph_train = dataset[True].to("cuda")

    def test_train_model(self):
        # Initialize dummy model
        model = DummyModel()

        # Specify the shape of your boolean tensor
        tensor_shape = (100, 30)

        # Generate random values between 0 and 1 for logits
        logits = torch.rand(tensor_shape, requires_grad=True)

        # Generate random labels (0 or 1) for one-hot encoded labels
        labels_one_hot = torch.randint(0, 2, tensor_shape).to(torch.float32)

        # Generate random values between 0 and 1 for train mask
        random_values = torch.rand(100)

        # Set a threshold for converting to boolean values
        threshold = 0.5
        train_mask = random_values > threshold

        # Ensure that both logits and labels require gradients
        logits.requires_grad_()
        labels_one_hot.requires_grad_()

        # Compute the CrossEntropyLoss only for the training samples
        loss_function = CrossEntropyLoss()
        loss = loss_function(
            logits[train_mask], labels_one_hot.argmax(dim=1)[train_mask].long()
        )

        # Perform optimization
        optimizer = Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_dgl_karate_club(self):
        dataset = KarateClubDataset()
        g = dataset[0]
        g.ndata["feat"] = g.in_degrees().view(-1, 1).float()
        g.edata["weight"] = torch.rand(g.num_edges(), dtype=torch.float).view(-1, 1)
        logger.debug(f"number of nodes {g.num_nodes()}")
        logger.debug(f"number of edge {g.num_edges()}")
        logger.debug(f"feat shape of nodes {g.ndata['feat'].shape}")
        logger.debug(f"feat of edge shape {g.edata['weight'].shape}")

        # Split the dataset into training and testing sets
        train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        train_mask[:10] = True  # Let's use the first 10 nodes for training
        test_mask = ~train_mask
        train = TensorDataset(torch.tensor(range(g.num_nodes()))[train_mask])
        # Define the data loader
        # logger.debug(f'the feat {g.ndata["feat"]}')
        logger.debug(f'the feat {g.ndata["feat"].shape}')
        logger.debug(f'the feat {len(g.ndata["feat"])}')
        logger.debug(f'the feat {type(g.ndata["feat"])}')
        model = WGCN(g.ndata["feat"].shape[1], 16, 2, 1, relu)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            inputs = g.ndata["feat"]
            edge_weight = g.edata["weight"]
            # logger.debug(f"edge weight shape {edge_weight.shape}")
            # logger.debug(f"g shape {g.num_edges()}")
            labels = g.ndata["label"]  # Assuming labels are available in the graph
            output = model(g, inputs, edge_weight)
            pred = output.argmax(1)
            acc = (pred == g.ndata["label"]).float().mean()
            logger.debug(f"Train Accuracy: {acc.item()}")
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(g, g.ndata["feat"], g.edata["weight"])
            pred = logits.argmax(1)
            acc = (pred == g.ndata["label"]).float().mean()
            logger.debug(f"Test Accuracy: {acc.item()}")

    def test_Cora_dataset(self):
        dataset = dgl.data.CoraGraphDataset()
        logger.debug(f"Number of categories: {dataset.num_classes}")
        g = dataset[0]
        weight = torch.rand(g.num_edges(), dtype=torch.float, device="cuda").view(-1, 1)
        logger.debug("Node features")
        logger.debug(g.ndata)
        logger.debug("Edge features")
        logger.debug(g.edata)
        g = g.to("cuda")
        model = WGCN(g.ndata["feat"].shape[1], 16, dataset.num_classes, 1, relu).to(
            "cuda"
        )
        train_weight(g, model, weight)

    def test_GAT(self):
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
        features = g.ndata["feat"]
        label = g.ndata["label"]
        mask = g.ndata["train_mask"]
        g = g.to("cuda")
        net = GAT(
            g,
            in_dim=features.size()[1],
            hidden_dim=8,
            out_dim=dataset.num_classes,
            num_heads=2,
        ).to("cuda")
        train(g, model=net)

    def test_publaynet_hugg(self):
        dataset = load_dataset("jordanparker6/publaynet")
        logger.debug(f"dataset data {dataset.data}")
        logger.debug(f"dataset data {dataset.shape}")


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 30)  # Assuming input size is 10, output size is 30

    def forward(self, x):
        return self.fc(x)


def draw_line_between_bounding_boxes(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)

    plt.plot(
        [center1[0], center2[0]], [center1[1], center2[1]], color="blue", linewidth=2
    )


def train_weight(g, model, weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]

    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(500):
        # Forward
        logits = model(g, features, weight)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            logger.debug(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]

    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(500):
        # Forward
        logits = model(features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            logger.debug(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )
