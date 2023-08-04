import json
import os
from typing import Any, List, Tuple, Dict

import numpy as np
from doctr.datasets.datasets import VisionDataset
from doctr.datasets.utils import convert_target_to_relative

from src.utils.setup_logger import logger


class WILDRECEIPT(VisionDataset):
    dataset = ('https://download.openmmlab.com/mmocr/data/wildreceipt.tar', 'wildreceipt.tar')

    def __init__(self, train: bool = True, **kwargs: Any) -> None:
        url, filename = self.dataset
        super().__init__(url, filename, None, True, pre_transforms=convert_target_to_relative, **kwargs)

        tmp_root = os.path.join(self.root, 'wildreceipt/')
        self.train = train

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32
        text_unit_list = []
        self.filename = "train.txt" if self.train else "test.txt"
        file_path = os.path.join(tmp_root, self.filename)
        # logger.debug(f'the file names: {tmp_root}')
        with open(file_path, 'r') as file:
            data = file.read()
        # Split the text file into separate JSON strings
        json_strings = data.strip().split('\n')
        # TODO: Check FUNSD implementation for more details
        for json_string in json_strings:
            json_data = json.loads(json_string)

            # Access the data in the JSON object as needed
            file_name = json_data['file_name']
            height = json_data['height']
            width = json_data['width']
            annotations = json_data['annotations']

            # Process the data or perform any required operations on each JSON separately
            # For example, logger.debug the file name, height, and width
            logger.debug(f"File Name: {file_name}")
            logger.debug(f"Height: {height}")
            logger.debug(f"Width: {width}")
            _targets = [(annotation['box'], annotation['text'], annotation['label']) for annotation in annotations]
            box_targets, text_units, labels = zip(*_targets)
            # Print the annotations for each JSON
            # for annotation in annotations:
            #     logger.debug(f"Box: {annotation['box']}")
            #     logger.debug(f"Text: {annotation['text']}")
            #     logger.debug(f"Label: {annotation['label']}")
            self.data.append((
                file_name,
                dict(boxes=np.asarray(box_targets, dtype=int), labels=list(labels),
                     text_units=text_units),
            ))
        self.root = tmp_root

    #     if train:
    #         dataset = load_dataset("Theivaprakasham/wildreceipt")['train']
    #     else:
    #         dataset = load_dataset("Theivaprakasham/wildreceipt")['test']
    #
    #     CONTEXT_SIZE = 2
    #     EMBEDDING_DIM = 50
    #     self.words = [dataset[doc_index]["words"][bbox_index] for doc_index in range(len(dataset)) for bbox_index in
    #                   range(len(dataset[doc_index]["words"]))]
    #
    #     vocab = set(self.words)
    #     word_to_ix = {word: i for i, word in enumerate(vocab)}
    #
    #     model = torch.load('wild_50_embedding.pt')
    #     global_vectors = GloVe(name='6B', dim=EMBEDDING_DIM)
    #     tokenizer = get_tokenizer("basic_english")
    #
    #     self.images = [np.array(
    #         Image.open(os.path.join(dataset[doc_index]["image_path"])).crop(dataset[doc_index]["bboxes"][bbox_index]))
    #         for doc_index in range(len(dataset)) for bbox_index in range(len(dataset[doc_index]["bboxes"]))]
    #     self.words = [dataset[doc_index]["words"][bbox_index] for doc_index in range(len(dataset)) for bbox_index in
    #                   range(len(dataset[doc_index]["words"]))]
    #     self.img_labels = [dataset[doc_index]["ner_tags"][bbox_index] for doc_index in range(len(dataset)) for
    #                        bbox_index in range(len(dataset[doc_index]["ner_tags"]))]
    #     self.word_tensors = []
    #     for doc_index in range(len(dataset)):
    #         for word in dataset[doc_index]["words"]:
    #             try:
    #                 embedding = model.weight[word_to_ix[word.lower()]]
    #             except KeyError:
    #                 embedding = global_vectors.get_vecs_by_tokens([word.lower()])
    #             self.word_tensors.append(embedding)
    #
    # def __len__(self):
    #     logger.debug(len(self.img_labels), len(self.images), len(self.word_tensors))
    #     return len(self.img_labels)
    #
    # def __getitem__(self, idx):
    #     image = self.images[idx]
    #     transform = transforms.Compose([
    #         ToTensor()
    #     ])
    #     image = transform(image)
    #
    #     word_embed = self.word_tensors[idx]
    #
    #     label = self.img_labels[idx]
    #
    #     return image, word_embed, label
    def extra_repr(self) -> str:
        return f"train={self.train}"
