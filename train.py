# TODO: - Create multiple model for GNN
import argparse

from args import train_subparser
from src.dataloader.graph_dataset import GraphDataset
from src.utils.setup_logger import logger

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    train_subparser(subparsers)
    args = main_parser.parse_args()

    data_name = args.dataset
    path = args.path
    hidden_size = args.hidden_size
    nbr_hidden_layer = args.hidden_layers
    lr = args.learning_rate
    epochs = args.epochs

    dataset = GraphDataset(data_name, path=path)

    graph_train = dataset[True]
    logger.debug(f"graph train {graph_train}")
