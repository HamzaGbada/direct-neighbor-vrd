import json
import os
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union

import numpy as np
from doctr.datasets.datasets import VisionDataset
from doctr.datasets.utils import convert_target_to_relative

from src.utils.setup_logger import logger
from src.utils.utils import convert_xmin_ymin, get_area


class XFUND(VisionDataset):
    """
    >>> # NOTE: You need to download the dataset first.
    >>> train_set = XFUND(train=True, data_folder="data/fr.train.json")
    >>> img, target = train_set[0]
    >>> test_set = XFUND(train=False, data_folder="data/fr.val.json")
    :return:
        Bounding boxes are in the Format (xmin, ymin, xmax, ymax) top left, bottom right corners
    """

    def __init__(
        self,
        data_folder: str,
        train: bool = True,

    ) -> None:
        # File existence check
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"unable to locate {data_folder}"
            )

        tmp_root = data_folder
        self.train = train
        np_dtype = np.float32
        self.data: List[
            Tuple[Union[str, Path, np.ndarray], Union[str, Dict[str, Any]]]
        ] = []

        with open(data_folder, "r") as file:
            data = file.read()
        # Split the text file into separate JSON strings
        json_strings = data.strip().split("\n")
        box: Union[List[float], np.ndarray]
        _targets = []
        logger.debug(f"data {data}")
        # for json_string in json_strings:
        #     json_data = json.loads(json_string)
        #     img_path = json_data["file_name"]
        #     annotations = json_data["annotations"]
        #     for annotation in annotations:
        #         coordinates = annotation["box"]
        #         if use_polygons:
        #             # (x, y) coordinates of top left, top right, bottom right, bottom left corners
        #             box = np.array(
        #                 [
        #                     [coordinates[0], coordinates[1]],
        #                     [coordinates[2], coordinates[3]],
        #                     [coordinates[4], coordinates[5]],
        #                     [coordinates[6], coordinates[7]],
        #                 ],
        #                 dtype=np_dtype,
        #             )
        #         else:
        #             x, y = coordinates[::2], coordinates[1::2]
        #             box = [min(x), min(y), max(x), max(y)]
        #         _targets.append((annotation["text"], box))
        #     text_targets, box_targets = zip(*_targets)
        #
        #     if recognition_task:
        #         crops = crop_bboxes_from_image(
        #             img_path=os.path.join(tmp_root, img_path),
        #             geoms=np.asarray(box_targets, dtype=int).clip(min=0),
        #         )
        #         for crop, label in zip(crops, list(text_targets)):
        #             if label and " " not in label:
        #                 self.data.append((crop, label))
        #     else:
        #         self.data.append(
        #             (
        #                 img_path,
        #                 dict(
        #                     boxes=np.asarray(box_targets, dtype=int).clip(min=0),
        #                     labels=list(text_targets),
        #                 ),
        #             )
        #         )
        # self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
