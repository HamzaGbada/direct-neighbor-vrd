import argparse

from doctr.datasets import FUNSD

from args import build_subparser
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.graph_builder.VRD_graph import VRD2Graph
from src.utils.setup_logger import logger
from src.word_embedding.BERT_embedding import TextEmbeddingModel
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    build_subparser(subparsers)
    args = main_parser.parse_args()

    if args.dataset == "CORD":
        train_set = CORD(train=True, download=True)
        test_set = CORD(train=False, download=True)
    elif args.dataset == "SROIE":
        train_set = SROIE(train=True, download=True)
        test_set = SROIE(train=False, download=True)
    elif args.dataset == "FUNSD":
        train_set = FUNSD(train=True, download=True)
        test_set = FUNSD(train=False, download=True)
    elif args.dataset == "WILDRECEIPT":
        train_set = WILDRECEIPT(train=True, download=True)
        test_set = WILDRECEIPT(train=False, download=True)
    else:
        logger.debug("Dataset not recognized")

    logger.debug("################# START ##################")
    text_model = TextEmbeddingModel(
        model_path=args.dataset + "_word_classification.pth",
        device="cpu"
    )

    i = 0
    for doc_index in range(len(train_set)):
        bbox = train_set.data[doc_index][1]["boxes"]
        text_units = train_set.data[doc_index][1]["text_units"]
        labels = train_set.data[doc_index][1]["labels"]

        features = [text_model.embed_text(text) for text in text_units]
        logger.debug(f"features {features}")
        logger.debug(f"features {type(features[0])}")

        graph = VRD2Graph(bbox, labels, features)
        graph.connect_boxes()
        graph.create_graph()

        graph.save_graph(path="data/"+args.dataset, graph_name=args.dataset+"_train_graph"+str(doc_index))
        i+=1

        if i > 5:
            break

    # Embedding a sentence
    logger.debug(args.dataset)

    # TODO: create the building process
