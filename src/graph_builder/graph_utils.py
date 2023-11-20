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
    bounding_boxes_json = {
        # (6, 1, 10, 10),
        (11, 15, 20, 10): "mid",  # mid
        (21, 1, 17, 10): "upper",  # Upper
        (10, 70, 25, 10): "super low",  # super low
        # (30, 12, 25, 10),
        # (25, 16, 17, 10),
        # (35, 5, 17, 10),
        # (20, 32, 5, 5),
        # (30, 24, 17, 10),
        # (40, 14, 5, 10),
        # (60, 34, 17, 10),
        # (77, 54, 17, 10),
        # (87, 66, 17, 10),
        # (90, 74, 17, 10),
        (21, 32, 17, 10): "low",  # low
    }
    poly = Polygon(
        [
            (box1[0], box1[1]),
            (box1[0] + box1[2], box1[1]),
            (box2[0] + box2[2], box2[1]),
            (box2[0] + box2[2], box2[1] + box2[3]),
            (box2[0], box2[1] + box2[3]),
            (box1[0], box1[1] + box1[3]),
            (box1[0], box1[1]),
        ]
    )
    # plt.plot(*poly.exterior.xy)
    logger.debug(f" bounding box one and {bounding_boxes_json[box1]} its coord {box1}")
    logger.debug(f" bounding box two and {bounding_boxes_json[box2]} its coord {box2}")
    for other_box in all_boxes:
        logger.debug(
            f"current box is and ***** {bounding_boxes_json[other_box]} ***** its coord ############ {other_box}"
        )

        if other_box != box1 and other_box != box2 and poly.is_valid:
            rectangle = Polygon(
                [
                    (other_box[0], other_box[1]),
                    (other_box[0] + other_box[2], other_box[1]),
                    (other_box[0] + other_box[2], other_box[1] + other_box[3]),
                    (other_box[0], other_box[1] + other_box[3]),
                ]
            )
            point1 = Point(other_box[0], other_box[1])
            point2 = Point(other_box[0] + other_box[2], other_box[1])
            point3 = Point(other_box[0] + other_box[2], other_box[1] + other_box[3])
            point4 = Point(other_box[0], other_box[1] + other_box[3])
            intersection1 = poly.contains(point1)
            intersection2 = poly.contains(point2)
            intersection3 = poly.contains(point3)
            intersection4 = poly.contains(point4)

            intersection = rectangle.intersection(poly)
            logger.debug(
                f"intersction between {bounding_boxes_json[box1]} and {bounding_boxes_json[box2]} throw {bounding_boxes_json[other_box]} is {intersection}"
            )
            # logger.debug(f" intersection of {box1}")
            if not intersection:
                logger.debug(
                    f"No part of the rectangle {bounding_boxes_json[other_box]} is inside the polygon between {bounding_boxes_json[box1]} and {bounding_boxes_json[box2]}"
                )
            else:
                logger.debug(
                    f"A part of the rectangle {bounding_boxes_json[other_box]} is inside the polygon between {bounding_boxes_json[box1]} and {bounding_boxes_json[box2]}"
                )
                return False, poly
            # Check if the other box is not box1 or box2
            # if intersection1 and intersection2 and intersection3 and intersection4:
            #     print("No part of the rectangle is inside the polygon")
            # else:
            #     print("A part of the rectangle is inside the polygon")
            #     return False

    return True, poly


def connected_boxes(bounding_boxes):
    """
    Given a list of bounding boxes, return a list of indices of connected bounding boxes per bounding box.

    Parameters:
    - bounding_boxes: List of bounding boxes represented as tuples (x, y, width, height).

    Returns:
    - List of indices of connected bounding boxes for each bounding box.
    """
    result = []
    pol = []
    for i, box1 in enumerate(bounding_boxes):
        connected_indices = [
            j
            for j, box2 in enumerate(bounding_boxes)
            if i != j and is_connected(box1, box2, bounding_boxes)[0]
        ]
        pols = [is_connected(box1, box2, bounding_boxes)[1] for j, box2 in enumerate(bounding_boxes)]
        result.append(connected_indices)
        pol.append(pols)
    return result, pol
