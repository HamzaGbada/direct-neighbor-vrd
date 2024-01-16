# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.
# This class is taken from https://mindee.github.io/doctr/v0.1.1/_modules/doctr/datasets/funsd.html
# in the version 0.5.1 (check https://pypi.org/project/python-doctr/0.5.1/)

# We applied some changes in this class for our purpose.

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from doctr.datasets.datasets.pytorch import VisionDataset
from doctr.datasets.utils import convert_target_to_relative
from src.utils.setup_logger import logger

__all__ = ["FUNSD"]

from src.utils.utils import get_area, convert_xmin_ymin


class FUNSD(VisionDataset):
    # FIXME: CONVERT THIS TO X, Y, W, H
    """FUNSD dataset from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents"
    <https://arxiv.org/pdf/1905.13538.pdf>`_.

    .. image:: https://github.com/mindee/doctr/releases/download/v0.5.0/funsd-grid.png
        :align: center

    >>> from src.dataloader.funsd_dataloader import FUNSD
    >>> train_set = FUNSD(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    SHA256 = "c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"
    FILE_NAME = "funsd.zip"

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            self.URL,
            self.FILE_NAME,
            self.SHA256,
            True,
            pre_transforms=convert_target_to_relative,
            **kwargs,
        )
        self.train = train
        np_dtype = np.float32

        # Use the subset
        subfolder = os.path.join(
            "dataset", "training_data" if train else "testing_data"
        )

        # # List images
        tmp_root = os.path.join(self.root, subfolder, "images")

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        for img_path in os.listdir(tmp_root):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(
                    f"unable to locate {os.path.join(tmp_root, img_path)}"
                )

            stem = Path(img_path).stem
            with open(
                os.path.join(self.root, subfolder, "annotations", f"{stem}.json"), "rb"
            ) as f:
                data = json.load(f)
            _targets = [
                (convert_xmin_ymin(block["box"]), block["text"].lower(), block["label"])
                for block in data["form"]
                if get_area(convert_xmin_ymin(block["box"])) >= 50
            ]

            # for each img_path,
            # data is the data of that image
            # data['form'] get the all data which is under 'form' key
            # data['form'][0] get the data under the first bbox
            if _targets:
                box_targets, text_units, labels = zip(*_targets)
                if (
                    len(box_targets) > 1
                ):  # number of bounding boxes in document should be more than one
                    self.data.append(
                        (
                            img_path,
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
