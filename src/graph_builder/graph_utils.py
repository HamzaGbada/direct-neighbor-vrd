import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely import Point
from shapely.geometry import Polygon

from src.utils.setup_logger import logger


class VRD2Graph:
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes
        self.connection_index = []

    def __len__(self):
        return len(self.bounding_boxes)
    @classmethod
    def is_connected(self, box1, box2, all_boxes):
        """
        Check if two bounding boxes are connected without any other boxes in between.

        Parameters:
        - box1, box2: Tuple representing the (x, y, width, height) of the bounding boxes.
        - all_boxes: List of bounding boxes.

        Returns:
        - True if connected, False otherwise.
        """
        polygon = Polygon(
            [
                (box1[0], box1[1]),
                (box1[0] + box1[2], box1[1]),
                (box2[0] + box2[2], box2[1]),
                (box2[0], box2[1]),
                (box1[0], box1[1]),
            ]
        )

        for other_box in all_boxes:
            if other_box != box1 and other_box != box2 and polygon.is_valid:
                x, y, width, height = other_box
                points = [
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                ]

                intersections = [polygon.contains(Point(point)) for point in points]

                if all(not intersection for intersection in intersections):
                    logger.debug("No part of the rectangle is inside the polygon")
                    return True
                else:
                    logger.debug("A part of the rectangle is inside the polygon")
                    return False

    def connected_boxes(self):
        """
        Given a list of bounding boxes, return a list of indices of connected bounding boxes per bounding box.

        Parameters:
        - bounding_boxes: List of bounding boxes represented as tuples (x, y, width, height).

        Returns:
        - List of indices of connected bounding boxes for each bounding box.
        """

        for i, box1 in enumerate(self.bounding_boxes):
            connected_indices = [
                j
                for j, box2 in enumerate(self.bounding_boxes)
                if i != j and self.is_connected(box1, box2, self.bounding_boxes)
            ]

            self.connection_index.append(connected_indices)

    def create_graph(self):
        """
        Create a DGL graph from a list of bounding boxes and a list of connection indices.

        Parameters:
        - bounding_boxes: List of bounding boxes.
        - connections: List of indices defining connections between bounding boxes.

        Returns:
        - DGL Graph object.
        """
        num_nodes = self.__len__()
        logger.debug(f"the size of graph {num_nodes}")
        # Create a DGL graph
        graph = dgl.DGLGraph()

        # Add nodes to the graph
        graph.add_nodes(num_nodes)

        # Add edges based on the connection indices
        src, dst = zip(*[(i, j) for i, j in enumerate(self.connection_index)])
        graph.add_edges(src, dst)

        # Node features (initially all zeros)
        node_features = torch.zeros(num_nodes, dtype=torch.float32)

        # Set node features in the graph
        graph.ndata["features"] = node_features

        return None
