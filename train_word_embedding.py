import argparse

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_accuracy
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from args import train_embedding_subparser
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.sentence_classification_dataloader import SentenceDataset
from src.dataloader.wildreceipt_dataloader import WILDRECEIPT
from src.utils.setup_logger import logger
from src.utils.utils import plots, process_labels
from src.word_embedding.BERT_embedding import BertSentenceClassification
from train_cnn_for_classification import compute_f1_score


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    num_classes,
    loss_fn,
    optimizer,
    device,
    num_epochs,
):
    train_losses = []  # To store training loss for each epoch
    val_losses = []  # To store validation loss for each epoch
    train_f1 = []  # To store validation loss for each epoch
    train_accuracy = []  # To store validation loss for each epoch
    val_f1 = []  # To store validation loss for each epoch
    val_accuracy = []  # To store validation loss for each epoch
    model.eval()
    for epoch in range(num_epochs):
        logger.debug(f"the epoch is {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0
        total_f1_score = 0
        total_accuracy = 0
        # logger.debug(f"THEEEEEE BATCH 11 {train_dataloader}")
        # logger.debug(f"THEEEEEE BATCH 000 {train_dataloader.__iter__()}")
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            f1_score_train = compute_f1_score(labels.view(-1), outputs.view(-1))
            accuracy_train = multilabel_accuracy(
                outputs, labels, num_labels=num_classes, average="macro"
            )
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            total_f1_score += f1_score_train
            total_train_loss += loss.item()
            total_accuracy += accuracy_train

        avg_f1_score_train = total_f1_score / len(train_dataloader)
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_accuracy_loss = total_accuracy / len(train_dataloader)

        train_losses.append(avg_train_loss)
        train_f1.append(avg_f1_score_train)
        train_accuracy.append(avg_accuracy_loss.cpu())

        # Validation loss calculation
        model.eval()
        total_val_loss = 0
        total_f1_score_val = 0
        total_accuracy_val = 0
        logger.debug(f"The validation for the epoch is {epoch + 1} start")
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                val_input_ids = batch["input_ids"].to(device)
                val_attention_mask = batch["attention_mask"].to(device)
                val_labels = batch["label"].to(device)

                val_outputs = model(val_input_ids, val_attention_mask)

                f1_score_val = compute_f1_score(
                    val_labels.view(-1), val_outputs.view(-1)
                )
                accuracy_val = multilabel_accuracy(
                    val_outputs, val_labels, num_labels=num_classes, average="macro"
                )
                val_loss = loss_fn(val_outputs, val_labels)

                total_val_loss += val_loss.item()
                total_f1_score_val += f1_score_val
                total_accuracy_val += accuracy_val

        avg_f1_score_val = total_f1_score_val / len(val_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_accuracy_loss = total_accuracy_val / len(val_dataloader)

        val_losses.append(avg_val_loss)
        val_f1.append(avg_f1_score_val)
        val_accuracy.append(avg_accuracy_loss.cpu())

        # Print and plot the losses
        logger.debug(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Train F1 score: {avg_f1_score_train:.4f} - Train accuracy: {avg_accuracy_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation F1 score: {avg_f1_score_val:.4f} - Validation accuracy: {avg_accuracy_loss:.4f}"
        )

    return (
        model,
        train_losses,
        val_losses,
        train_f1,
        val_f1,
        train_accuracy,
        val_accuracy,
    )


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            predictions = np.argmax(logits.cpu().numpy(), axis=1)

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    # report = classification_report(all_labels, all_predictions,
    #                                target_names=["0", "1", "3", "4", "5"])  # Replace with your class names


def word_embedding_dataloader(dataset, max_len=128, batch_size=16):
    sentences = [
        x
        for doc_index in range(len(dataset))
        for x in dataset.data[doc_index][1]["text_units"]
    ]
    labels, name = process_labels(dataset)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = SentenceDataset(sentences, labels, tokenizer, max_len, name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main(
    train_dataloader,
    val_dataloader,
    num_classes=5,
    num_epochs=10,
    device=torch.device("cpu"),
):
    model = BertSentenceClassification(num_classes)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    (
        model,
        train_losses,
        val_losses,
        train_f1,
        val_f1,
        train_acc,
        val_acc,
    ) = train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        num_classes,
        loss_fn,
        optimizer,
        device,
        num_epochs,
    )

    # logger.debug(f"Train evalution report{evaluate(model, train_dataloader, device)}")
    name = train_dataloader.dataset.__str__()
    plots(num_epochs, train_losses, val_losses, "Loss", name)
    plots(num_epochs, train_f1, val_f1, "F1 score", name)
    plots(num_epochs, train_acc, val_acc, "Accuracy", name)
    model_path = name + "_word_classification.pth"

    # Save the model to a file
    torch.save(model.state_dict(), model_path)
    return model


def FUNSD(train, download):
    pass


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
    train_embedding_subparser(subparsers)
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
        num_classes = 26
    elif args.dataset == "WILDRECEIPT":
        train_set = WILDRECEIPT(train=True, download=True)
        test_set = WILDRECEIPT(train=False, download=True)
        num_classes = 26
    else:
        logger.debug("Dataset not recognized")

    train_dataloader = word_embedding_dataloader(train_set)
    test_dataloader = word_embedding_dataloader(test_set)
    # logger.debug(f"train dataset {train_dataloader.__str__()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = main(
        train_dataloader, test_dataloader, num_epochs=args.epochs, num_classes=num_classes, device=device
    )
    logger.debug(f"Test evalution report{evaluate(model, test_dataloader, device)}")

