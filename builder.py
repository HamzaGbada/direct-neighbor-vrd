from args import default_parser, build_subparser
from src.utils.setup_logger import logger

if __name__ == "__main__":
    main_parser = default_parser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    build_subparser(subparsers)
    args = main_parser.parse_args()

    logger.debug(args.dataset)
    logger.debug(args.hidden_size)
