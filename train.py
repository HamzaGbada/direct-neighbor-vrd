# TODO: - Create multiple model for GNN
import argparse

from args import train_subparser
from src.utils.setup_logger import logger

if __name__ == "__main__":
    main_parser = parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    train_subparser(subparsers)
    args = main_parser.parse_args()

    logger.debug(args.dataset)
    logger.debug(args.hidden_size)
