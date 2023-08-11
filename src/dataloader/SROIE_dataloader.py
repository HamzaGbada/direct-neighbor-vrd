import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import pandas as pd
from doctr.datasets.datasets.pytorch import VisionDataset

from src.utils.SROIE_utils import read_bbox_and_words, read_entities, assign_labels

__all__ = ['SROIE']

from src.utils.setup_logger import logger


class SROIE(VisionDataset):
    """SROIE dataset from `"ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction"
    <https://arxiv.org/pdf/2103.10213.pdf>`_.

    .. image:: https://github.com/mindee/doctr/releases/download/v0.5.0/sroie-grid.png
        :align: center

    >>> from src.DataLoader.sroie_dataloader import SROIE
    >>> train_set = SROIE(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    def __init__(
            self,
            train: bool = True,
            img_transforms: Optional[Callable[[Any], Any]] = None,
            sample_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
            pre_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ) -> None:

        self.train = train
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self.pre_transforms = pre_transforms

        self.data: List[Tuple[str, Dict[str, Any]]] = []

        if not os.path.isdir('data/SROIE_CSV/'):
            train_path = 'data/SROIE2019/train/'
            test_path = 'data/SROIE2019/test/'

            bbox_train_path = train_path + "box/"
            entities_train_path = train_path + "entities/"

            bbox_test_path = test_path + "box/"
            entities_test_path = test_path + "entities/"
            # Train Loop
            for filename in os.listdir(bbox_train_path):
                bbox_file_path = bbox_train_path + filename
                entities_file_path = entities_train_path + filename

                bbox = read_bbox_and_words(path=Path(bbox_file_path))
                entities = read_entities(path=Path(entities_file_path))

                bbox_labeled = assign_labels(bbox, entities)
                # indexAge = bbox_labeled[bbox_labeled['label'] == 'O'].index
                # bbox_labeled.drop(indexAge, inplace=True)
                if not os.path.isdir("data/SROIE_CSV/train/"):
                    os.makedirs("data/SROIE_CSV/train/")
                bbox_labeled.to_csv("data/SROIE_CSV/train/" + filename[:-4] + ".csv")

            # Test Loop
            for filename in os.listdir(bbox_test_path):
                bbox_file_path = bbox_test_path + filename
                entities_file_path = entities_test_path + filename

                bbox = read_bbox_and_words(path=Path(bbox_file_path))
                entities = read_entities(path=Path(entities_file_path))

                bbox_labeled = assign_labels(bbox, entities)
                # indexAge = bbox_labeled[bbox_labeled['label'] == 'O'].index
                # bbox_labeled.drop(indexAge, inplace=True)
                logger.debug(f'the condition {os.path.isdir("data/SROIE_CSV/test/")}')
                if not os.path.isdir("data/SROIE_CSV/test/"):
                    os.makedirs("data/SROIE_CSV/test/")
                bbox_labeled.to_csv("data/SROIE_CSV/test/" + filename[:-4] + ".csv")

        if self.train:
            path = 'data/SROIE_CSV/train/'
            img_path = 'data/SROIE2019/train/img/'
        else:
            path = 'data/SROIE_CSV/test/'
            img_path = 'data/SROIE2019/test/img/'
        self.data_update = []
        encoded_dic = {"TOTAL": 0,
                       "DATE": 1,
                       "ADDRESS": 2,
                       "COMPANY": 3,
                       "O": 4
                       }
        for csv_file in os.listdir(path):
            bbox_and_label = {}
            df = pd.read_csv(path + csv_file)
            bbox_array = df[["x0", "y0", "x2", "y2"]].to_numpy()
            bbox_and_label['boxes'] = bbox_array
            bbox_and_label['labels'] = [encoded_dic[x] for x in df["label"]]
            bbox_and_label['text_units'] = [x for x in df["line"]]
            t_data = (csv_file[:-4] + ".jpg", bbox_and_label)

            self.data.append(t_data)

        self.root = img_path

    def extra_repr(self) -> str:
        return f"train={self.train}"