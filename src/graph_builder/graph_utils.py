import matplotlib.pyplot as plt
import numpy as np


def vrd_2_graph(bbox):
    """Returns a list of indices of the connected bounding boxes to each bounding box.
    create graph from one VRD

    Here we define le notion de direct node, where two bbox are connected iff there is no bbox between them
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


