import matplotlib.pyplot as plt
import numpy as np
from shapely import Point
from shapely.geometry import Polygon

def vrd_2_graph(bbox):
    """Returns a list of indices of the connected bounding boxes to each bounding box.
    create graph from one VRD

    Here we define le notion de direct node, where two bbox are connected iff there is no bbox between them

    if between the X of inital and the x of test there is an x so does not appartient,
    if x
    Args:
      bbox: A list of bounding boxes, where each bounding box is a tuple of four floats representing the coordinates of the bounding box.

    Returns:
      A list of lists of integers, where each sublist contains the indices of the bounding boxes that are connected to the corresponding bounding box.
    """

    connected_bbox_indices = []
    for i in range(len(bbox)):
        connected_bbox_indices.append([])

        # Iterate over the other bounding boxes.
        for j in range(len(bbox)):
            # Check if there is no bounding box between the two bounding boxes.
            if i != j and (bbox[i][2] < bbox[j][0] or bbox[i][0] > bbox[j][2]):
                connected_bbox_indices[i].append(j)

    return connected_bbox_indices





def is_connected(box1, box2, all_boxes):
    """
    Check if two bounding boxes are connected without any other boxes in between.

    Parameters:
    - box1, box2: Tuple representing the (x, y, width, height) of the bounding boxes.
    - all_boxes: List of bounding boxes.

    Returns:
    - True if connected, False otherwise.
    """
    poly1 = Polygon([(box1[0], box1[1]), (box1[0] + box1[2], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]),
                     (box1[0], box1[1] + box1[3])])
    poly2 = Polygon([(box2[0], box2[1]), (box2[0] + box2[2], box2[1]), (box2[0] + box2[2], box2[1] + box2[3]),
                     (box2[0], box2[1] + box2[3])])

    poly = Polygon([(box1[0], box1[1]), (box1[0] + box1[2], box1[1]), (box2[0] + box2[2], box2[1]), (box2[0] + box2[2], box2[1] + box2[3]), (box2[0], box2[1] + box2[3]), (box1[0], box1[1] + box1[3])])

    for other_box in all_boxes:
        rectangle = Polygon([(other_box[0], other_box[1]), (other_box[0] + other_box[2], other_box[1]),
                                  (other_box[0] + other_box[2], other_box[1] + other_box[3]),
                                  (other_box[0], other_box[1] + other_box[3])])
        point = Point(other_box[0], other_box[1])
        point = Point(other_box[0] + other_box[2], other_box[1])
        point = Point(other_box[0] + other_box[2], other_box[1] + other_box[3])
        point = Point(other_box[0], other_box[1] + other_box[3])
        intersection = rectangle.intersection(poly)
        # Check if the other box is not box1 or box2
        if intersection.is_empty:
            print("No part of the rectangle is inside the polygon")
        else:
            print("A part of the rectangle is inside the polygon")
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
        connected_indices = [j for j, box2 in enumerate(bounding_boxes) if
                             i != j and is_connected(box1, box2, bounding_boxes)]
        result.append(connected_indices)
    return result





