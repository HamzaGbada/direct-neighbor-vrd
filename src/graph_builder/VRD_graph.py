import math

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch
from shapely import Point
from shapely.geometry import Polygon

from src.utils.setup_logger import logger
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class VRD2Graph:
    def __init__(self, bounding_boxes, labels):
        self.bounding_boxes, self.node_label = zip(
            *sorted(zip(bounding_boxes, labels), key=lambda x: (x[0][1], x[0][0]))
        )
        self.connection_index = []
        self.edges = []
        self.graph = dgl.DGLGraph()

    def __len__(self):
        return len(self.bounding_boxes)

    @classmethod
    def is_connected(cls, box1, box2, all_boxes):
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

    @classmethod
    def compute_phi(cls, box1, box2):
        # Assume bounding boxes are represented as (x_min, y_min, x_max, y_max)
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

        # Compute relative polar coordinates
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]

        return math.atan2(dy, dx)

    def connect_boxes(self):
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
                if i < j and self.is_connected(box1, box2, self.bounding_boxes)
            ]

            self.connection_index.append(connected_indices)

    def create_graph(self):
        """
        Create a DGL graph from a list of bounding boxes and a list of connection indices.

        Returns:
        - DGL Graph object.
        """
        num_nodes = self.__len__()

        self.edges = [
            (i, j, self.compute_phi(self.bounding_boxes[i], self.bounding_boxes[j]))
            for i, connected_indices in enumerate(self.connection_index)
            for j in connected_indices
        ]

        # Add nodes to the graph
        self.graph.add_nodes(num_nodes)

        # Add edges based on the connection indices
        src, dst, feat = tuple(zip(*self.edges))

        # Node features (initially all zeros)
        node_features = torch.zeros(num_nodes, dtype=torch.float32)

        self.graph.add_edges(src, dst)
        # self.graph = graph((src, dst), num_nodes=len(self.node_label))

        # Set node features in the graph
        self.graph.ndata["features"] = node_features
        self.graph.ndata["label"] = torch.tensor(self.node_label)
        self.graph.edata["weight"] = torch.tensor(feat)

    def plot_dgl_graph(self):
        """
        Plot the DGL graph using NetworkX and Matplotlib.

        """
        # Convert DGL graph to NetworkX graph
        nx_graph = self.graph.to_networkx()

        # Extract node features from DGL graph
        node_features = self.graph.ndata["features"].numpy()

        # Plot the graph
        pos = nx.spring_layout(nx_graph)  # You can choose a different layout algorithm
        nx.draw(
            nx_graph,
            pos,
            with_labels=True,
            node_size=700,
            node_color=node_features,
            cmap="viridis",
        )
        plt.show()
