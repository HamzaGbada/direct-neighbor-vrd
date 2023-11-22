# TODO: - Prepare the utils (helper) to create the graphs
#       - Prepare CNN (UNET and VGG) and Word Embedding model
#       - The graph building process is independent to training
#       - Use pretrained (I train by my self) for wordEmbedding (use BERT)
#       - Graph building process is as follow:
#           * Build Direct neighbor
#           * Assign Feature vector to each node (use Unet CNN model for embedding)
#           * Assign edge angle
#       - Create multiple model for GNN


from args import train_subparser, default_parser
from src.utils.setup_logger import logger

if __name__ == "__main__":
    main_parser = default_parser()
    subparsers = main_parser.add_subparsers(dest='subcommand', help='Choose subcommand')
    train_subparser(subparsers)
    args = main_parser.parse_args()

    logger.debug(args.dataset)
    logger.debug(args.hidden_size)
