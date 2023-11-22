from args import build_parser
from src.utils.setup_logger import logger

if "__main__" == __name__:
    parser = build_parser()
    args = parser.parse_args()

    logger.debug(args.dataset)
