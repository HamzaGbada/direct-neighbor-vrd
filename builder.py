import argparse

from args import default_parser, build_subparser
from src.utils.setup_logger import logger

if __name__ == "__main__":
    main_parser = parser = argparse.ArgumentParser(
        description="This command creates a graph-based dataset for node classification for "
        "a specific dataset in order to extract entities from Visually Rich "
        'Documents. The default is: "./data/<DATASET_NAME>/<Train||Test>/"'
    )
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    build_subparser(subparsers)
    args = main_parser.parse_args()

    logger.debug(args.dataset)
    logger.debug(args.hidden_size)
