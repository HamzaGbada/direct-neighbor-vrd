import os
import random
import unittest
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch import nn
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from torchmetrics.functional.classification import multilabel_accuracy

from src.cnn_embedding.unet_embedding import UNet, SimpleCNN, EfficientNetV2MultiClass
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.utils.setup_logger import logger
from src.utils.utils import convert_xmin_ymin, convert_format1, convert_format2, plot_cropped_image, get_area
from train_cnn_for_classification import image_dataloader, compute_f1_score, compute_accuracy


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
        # TODO: try with the other vairent of wildreceipt dataset of Huggingface
        #       Here is a similar issue: https://github.com/huggingface/datasets/issues/4775
        train = True
        if train:
            dataset = load_dataset("Theivaprakasham/wildreceipt")['train']
            # dataset = load_dataset("jinhybr/WildReceipt")['train']
        else:
            dataset = load_dataset("Theivaprakasham/wildreceipt")['test']
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

        image_hugging = Image.open(dataset[doc_index]['image_path'])
        plt.imshow(image_hugging)
        plt.title("The Current Image, HuggingFace Implementation")
        plt.show()

        bbox_doctr = train_set.data[doc_index][1]['boxes'][word_index]
        text_unit_doctr = train_set.data[doc_index][1]['text_units'][word_index]
        logger.debug(f"Doctr bounding boxes : {bbox_doctr}")

        bbox_hugging_face = dataset[doc_index]['bboxes'][word_index]
        text_unit_face = dataset[doc_index]['words'][word_index]
        logger.debug(f"HuggingFace bounding boxes : {bbox_hugging_face}")

        common_box_doctr = convert_xmin_ymin(bbox_doctr)

        common_box_hugface_1 = convert_format1(bbox_hugging_face)
        common_box_hugface_2 = convert_format2(bbox_hugging_face)

        plot_cropped_image(image_doctr, common_box_doctr,
                           f'Bouding boxes (my implementation) \n its associated text unit: {text_unit_doctr}')
        plot_cropped_image(image_hugging, common_box_hugface_1,
                           f'Hugging Face Bouding boxes (x,y,w,h format) \n its associated text unit: {text_unit_face}')
        plot_cropped_image(image_hugging, common_box_hugface_2,
                           f'Hugging Face Bouding boxes (x1,y1,x2, y2 format) \n its associated text unit: {text_unit_face}')

        # logger.debug(f"the cord dataset {train_set.data}")
        self.assertEqual(os.path.basename(filename), os.path.basename(dataset[doc_index]['image_path']))

    def test_sentences_label(self):
        dataset = CORD(train=False)

        sentences = [x for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['text_units']]
        labels = [x for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['labels']]

        self.assertEqual(sentences[2], dataset.data[0][1]['text_units'][2])
        self.assertEqual(labels[2], dataset.data[0][1]['labels'][2])

    def test_sroie(self):
        train_set = SROIE(train=True)
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
        logger.debug(f"The sroie dataset: {train_set.data}")
        # self.assertEqual(train_set.data[0][0], "receipt_00425.png")

    def test_cropped_bbox(self):
        # Open the image using PIL
        train_set = CORD(train=True)
        path = os.path.join(train_set.root, train_set.data[0][0])
        logger.debug(f"the path is {path}")
        image = Image.open(path).convert("L")

        # Define the bounding box coordinates (left, upper, right, lower)
        bbox = train_set.data[0][1]['boxes'][0]
        text_units = train_set.data[0][1]['text_units'][0]
        logger.debug(f"bbox {bbox}")
        logger.debug(f"text_units {text_units}")

        # Crop the image
        convert_tensor = transforms.ToTensor()
        cropped_image = convert_tensor(image.crop(bbox))
        logger.debug(f"shape before remove channel{cropped_image.shape}")

        # Save or display the cropped image
        plt.imshow(image, cmap='gray')
        # plt.imshow(cropped_image, cmap='gray')
        plt.show()

    def test_image_dataloader(self):

        # Open the image using PIL
        train_set = CORD(train=False)
        cropped_images, boxes, labels, text_units = image_dataloader(train_set)
        path = os.path.join(train_set.root, train_set.data[0][0])
        logger.debug(f"the path is {labels}")
        image = Image.open(path)

        logger.debug(f"text_units {text_units[5]}")

        plt.imshow(image, cmap='gray')
        plt.imshow(cropped_images[5], cmap='gray')
        plt.show()

    def test_unet_test(self):
        # Open the image using PIL
        inputs = torch.rand(3, 63, 45).to(device="cuda")
        model = UNet(3,5).to(device="cuda")
        #
        outputs = model(inputs)
        # logger.debug(f"outputs shape {outputs.shape}")
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.tensor([0]).reshape(-1, 1)
        X = torch.tensor([0,1,2,3,4]).view(-1,1)
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
            encoded_dic = {'menu.sub_cnt': 0,
                           'sub_total.othersvc_price': 1,
                           'total.total_price': 2,
                           'menu.etc': 3,
                           'sub_total.discount_price': 4,
                           'menu.unitprice': 5,
                           'menu.discountprice': 6,
                           'void_menu.price': 7,
                           'menu.nm': 8,
                           'total.menutype_cnt': 9,
                           'sub_total.subtotal_price': 10,
                           'menu.sub_nm': 11,
                           'void_menu.nm': 12,
                           'menu.sub_unitprice': 13,
                           'menu.sub_etc': 14,
                           'menu.cnt': 15,
                           'menu.vatyn': 16,
                           'total.total_etc': 17,
                           'total.menuqty_cnt': 18,
                           'total.cashprice': 19,
                           'menu.num': 20,
                           'total.changeprice': 21,
                           'sub_total.tax_price': 22,
                           'sub_total.etc': 23,
                           'menu.price': 24,
                           'total.creditcardprice': 25,
                           'total.emoneyprice': 26,
                           'sub_total.service_price': 27,
                           'menu.itemsubtotal': 28,
                           'menu.sub_price': 29
                           }
        labels = [encoded_dic[x] for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['labels']]
        logger.debug(f"label {labels}")
    def test_bbox_area(self):
        dataset = CORD(train=True, download=True)
        bbox = [get_area(x) for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['boxes']]
        bbox.sort()
        logger.debug(f"Sorted area bbox area : {bbox}")

    def test_efficient_test(self):
        # Open the image using PIL
        inputs = torch.rand(3, 63, 45).to(device="cuda")
        inputs = inputs.unsqueeze(0)
        logger.debug(f"inputs shape {inputs.shape}")
        logger.debug(inputs.device)
        model = EfficientNetV2MultiClass(5).to(torch.device('cuda'))
        #
        outputs = model(inputs)
        # logger.debug(f"outputs shape {outputs.shape}")
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.tensor([0]).reshape(-1, 1)
        X = torch.tensor([0,1,2,3,4]).view(-1,1)
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

        X = torch.tensor([0,1,2,3,4]).view(-1,1)
        enc = OneHotEncoder(sparse=False)
        enc.fit(X)

        num_classes = 5
        num_epochs = 2
        data_size = 2

        model = EfficientNetV2MultiClass(num_classes).to(torch.device('cuda'))
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
                inputs, labels = torch.rand(1, 1, random.randint(70,100), random.randint(70,100)).to(device="cuda"), torch.randint(0,4,[1, 1]).to(device="cuda")

                optimizer.zero_grad()

                outputs = model(inputs)
                labels = torch.from_numpy(enc.transform(labels.cpu())).to(device="cuda")
                logger.debug("labels.view(-1)")
                logger.debug(labels.view(-1))
                logger.debug("outputs.view(-1)")
                logger.debug(outputs.view(-1))
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
        target = torch.tensor([[0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.]])
        preds = torch.tensor([[0.12, 0.15, 0.3, 0.22, 0.21], [0.12, 0.05, 0.8, 0.02, 0.01]])
        accuracy = multilabel_accuracy(preds, target, num_labels=5, average='weighted')
        logger.debug(f" the accuracy is {accuracy}")




