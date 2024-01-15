import argparse
import warnings

from doctr.datasets import FUNSD

from args import build_subparser
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.dataloader.xfund_dataloader import XFUND
from src.utils.setup_logger import logger
from src.utils.utils import process_and_save_dataset
from src.word_embedding.BERT_embedding import TextEmbeddingModel

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
        num_classes = 30
    elif args.dataset == "SROIE":
        train_set = SROIE(train=True)
        test_set = SROIE(train=False)
        num_classes = 5
    elif args.dataset == "FUNSD":
        train_set = FUNSD(train=True, download=True)
        test_set = FUNSD(train=False, download=True)
        num_classes = 4
    elif args.dataset == "WILDRECEIPT":
        train_set = WILDRECEIPT(train=True, download=True)
        test_set = WILDRECEIPT(train=False, download=True)
        num_classes = 26
    elif args.dataset == "XFUND":
        train_set = XFUND(train=True, data_folder="data/fr.train.json")
        test_set = XFUND(train=False, data_folder="data/fr.val.json")
        num_classes = 3
    else:
        logger.debug("Dataset not recognized")

    device = "cpu"
    logger.info("################# BUILDING GRAPHS START ##################")
    text_model = TextEmbeddingModel(
        model_path=args.dataset + "_word_classification.pth",
        num_classes=num_classes,
        device=device,
    )

    process_and_save_dataset(train_set, text_model, args, split="train", device=device)
    process_and_save_dataset(test_set, text_model, args, split="test", device=device)
