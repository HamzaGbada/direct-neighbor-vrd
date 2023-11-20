import matplotlib.pyplot as plt
import numpy as np
from shapely import Point
from shapely.geometry import Polygon

from src.utils.setup_logger import logger


def is_connected(box1, box2, all_boxes):
    """
    Check if two bounding boxes are connected without any other boxes in between.

    Parameters:
    - box1, box2: Tuple representing the (x, y, width, height) of the bounding boxes.
    - all_boxes: List of bounding boxes.

    Returns:
    - True if connected, False otherwise.
    """
    poly = Polygon(
        [
            (box1[0], box1[1]),
            (box1[0] + box1[2], box1[1]),
            (box2[0] + box2[2], box2[1]),
            (box2[0], box2[1]),
            (box1[0], box1[1]),
        ]
    )

    for other_box in all_boxes:

        if other_box != box1 and other_box != box2 and poly.is_valid:

            point1 = Point(other_box[0], other_box[1])
            point2 = Point(other_box[0] + other_box[2], other_box[1])
            point3 = Point(other_box[0] + other_box[2], other_box[1] + other_box[3])
            point4 = Point(other_box[0], other_box[1] + other_box[3])
            intersection1 = poly.contains(point1)
            intersection2 = poly.contains(point2)
            intersection3 = poly.contains(point3)
            intersection4 = poly.contains(point4)

            # Check if the other box is not box1 or box2
            if (
                not (intersection1)
                and not (intersection2)
                and not (intersection3)
                and not (intersection4)
            ):
                logger.debug(
                    f"No part of the rectangle is inside the polygon"
                )
            else:
                logger.debug(
                    f"A part of the rectangle is inside the polygon"
                )
                return False

    return True


def connected_boxes(bounding_boxes):
    """
    Given a list of bounding boxes, return a list of indices of connected bounding boxes per bounding box.

    Parameters:
    - bounding_boxes: List of bounding boxes represented as tuples (x, y, width, height).

    Returns:
    - List of indices of connected bounding boxes for each bounding box.
    """
    result = []
    for i, box1 in enumerate(bounding_boxes):

        connected_indices = [
            j
            for j, box2 in enumerate(bounding_boxes)
            if i != j and is_connected(box1, box2, bounding_boxes)[0]
        ]

        result.append(connected_indices)
    return result
