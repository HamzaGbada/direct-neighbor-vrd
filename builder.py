import argparse

from args import build_subparser
from src.utils.setup_logger import logger
from src.word_embedding.BERT_embedding import TextEmbeddingModel

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    build_subparser(subparsers)
    args = main_parser.parse_args()

    text_model = TextEmbeddingModel(model_path="CORD_word_classification.pth")

    # Embedding a sentence
    sentence_embedding = text_model.embed_text("18.167$")
    logger.debug(args.dataset)

    # TODO: create the building process
