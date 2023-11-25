import os
import torch
from dgl import load_graphs, batch
from dgl.data import DGLDataset
from torch import zeros
from src.utils.setup_logger import logger


class GraphDataset(DGLDataset):
    def __init__(self, data_name: str, path: str = "data/"):
        super().__init__(name="GraphDataset")
        dataset_paths = {
            "FUNSD": "FUNSD",
            "CORD": "CORD",
            "SROIE": "SROIE",
            "WILDRECEIPT": "WILDRECEIPT",
        }

        if data_name not in dataset_paths:
            logger.debug("Invalid dataset name. Please provide a valid dataset name.")
            return

        dataset_path = os.path.join(path, dataset_paths[data_name])
        self.num_classes = {"FUNSD": 4, "CORD": 30, "SROIE": 5, "WILDRECEIPT": 26}[
            data_name
        ]

        ext = "bin"
        graph_list_train = [
            load_graphs(os.path.join(dataset_path, "train", file))[0][0]
            for file in os.listdir(os.path.join(dataset_path, "train"))
            if file.endswith(ext)
        ]
        graph_list_test = [
            load_graphs(os.path.join(dataset_path, "test", file))[0][0]
            for file in os.listdir(os.path.join(dataset_path, "test"))
            if file.endswith(ext)
        ]

        self.graph_train = batch(graph_list_train)
        self.graph_test = batch(graph_list_test)

        self.graph_train = batch([self.graph_train, self.graph_test])
        logger.debug(self.graph_train.number_of_nodes())
        logger.debug(self.graph_test.number_of_nodes())

    def process(self):
        n_nodes = self.graph_train.number_of_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def __getitem__(self, train: bool):
        return self.graph_train if train else self.graph_test

    def __len__(self):
        return 1
