import os
import unittest
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.utils.setup_logger import logger
from src.utils.utils import convert_format3, convert_format1, convert_format2, plot_cropped_image


class TestDataLoader(unittest.TestCase):
    def test_cord(self):
        train_set = CORD(train=False, download=True)
        logger.debug(f"the cord dataset {train_set.data}")
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

        common_box_doctr = convert_format3(bbox_doctr)

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
        # TODO: retrieve sentence and label sperately perform encoding to pass it BERT model

        logger.debug(f'the dataset contains {dataset.data}')


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
