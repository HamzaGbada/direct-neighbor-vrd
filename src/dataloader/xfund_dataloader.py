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
    # FIXME: CONVERT THIS TO X, Y, W, H
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
            raise FileNotFoundError(f"unable to locate {data_folder}")

        tmp_root = data_folder
        self.train = train
        np_dtype = np.float32
        self.data: List[
            Tuple[Union[str, Path, np.ndarray], Union[str, Dict[str, Any]]]
        ] = []

        with open(data_folder, "r") as file:
            data = file.read()
        # Split the text file into separate JSON strings
        box: Union[List[float], np.ndarray]
        _targets = []
        json_data = json.loads(data)
        for document in json_data["documents"]:
            file_name = document["img"]["fname"]
            annotations = document["document"]
            _targets = [
                (
                    convert_xmin_ymin(annotation["box"]),
                    annotation["text"].lower(),
                    annotation["label"],
                )
                for annotation in annotations
                if get_area(convert_xmin_ymin(annotation["box"])) >= 50
            ]
            if _targets:
                box_targets, text_units, labels = zip(*_targets)
                if (
                    len(box_targets) > 1
                ):  # number of bounding boxes in document should be more than one
                    self.data.append(
                        (
                            file_name,
                            dict(
                                boxes=np.asarray(box_targets, dtype=int),
                                text_units=list(text_units),
                                labels=list(labels),
                            ),
                        )
                    )
        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
