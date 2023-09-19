import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch import nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

from src.cnn_embedding.unet_embedding import UNet, SimpleCNN
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.utils.setup_logger import logger
from src.utils.utils import convert_xmin_ymin, convert_format1, convert_format2, plot_cropped_image, get_area
from train_cnn_for_classification import image_dataloader


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
        train_set = CORD(train=True)
        cropped_images, boxes, labels, text_units = image_dataloader(train_set)
        path = os.path.join(train_set.root, train_set.data[0][0])
        logger.debug(f"the path is {path}")
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


    def test_bbox_area(self):
        dataset = SROIE(train=True)
        bbox = [get_area(x) for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['boxes']]
        bbox.sort()
        logger.debug(f"Sorted area bbox area : {bbox}")

