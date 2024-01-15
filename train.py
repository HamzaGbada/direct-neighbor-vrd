# TODO: - Create multiple model for GNN
import argparse

import torch
from dgl import add_self_loop
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics.functional.classification import (
    multilabel_accuracy,
)

from args import train_subparser
from src.dataloader.graph_dataset import GraphDataset
from src.graph_pack.graph_model import WGCN
from src.utils.setup_logger import logger
from src.utils.utils import compute_f1_score


def train(
    g,
    model,
    edge_weight,
    train_mask,
    val_mask,
    test_mask,
    num_class,
    lr=0.01,
    epochs=50,
):
    loss_fct = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_val_f1 = 0
    best_test_f1 = 0

    # FIXME: Convert this in graph creattion to float32
    features = g.ndata["features"].to(torch.float64)
    labels = g.ndata["label"]

    train_list, val_list, test_list = [], [], []
    loss_train, loss_val, loss_test = [], [], []
    for e in tqdm(range(epochs)):
        # Forward

        logits = model(g, features, edge_weight)
        f1_score_train = compute_f1_score(
            labels[train_mask].view(-1), logits[train_mask].view(-1)
        )
        accuracy_train = multilabel_accuracy(
            logits[train_mask].squeeze(dim=1),
            labels[train_mask].squeeze(dim=1),
            num_labels=num_class,
            average="macro",
        )

        loss = loss_fct(labels[train_mask], logits[train_mask].squeeze(dim=1))
        loss_v = loss_fct(labels[val_mask], logits[val_mask].squeeze(dim=1))
        loss_t = loss_fct(labels[test_mask], logits[test_mask].squeeze(dim=1))
        loss_train.append(loss)
        loss_val.append(loss_v)
        loss_test.append(loss_t)

        f1_score_val = compute_f1_score(
            labels[val_mask].view(-1), logits[val_mask].view(-1)
        )
        accuracy_val = multilabel_accuracy(
            logits[val_mask].squeeze(dim=1),
            labels[val_mask].squeeze(dim=1),
            num_labels=num_class,
            average="macro",
        )

        f1_score_test = compute_f1_score(
            labels[test_mask].view(-1), logits[test_mask].view(-1)
        )
        accuracy_test = multilabel_accuracy(
            logits[test_mask].squeeze(dim=1),
            labels[test_mask].squeeze(dim=1),
            num_labels=num_class,
            average="macro",
        )
        train_list.append(f1_score_train)
        val_list.append(f1_score_val)
        test_list.append(f1_score_test)
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < accuracy_val:
            best_val_acc = accuracy_val
            # best_test_acc = test_acc
        if best_val_f1 < f1_score_val:
            best_val_f1 = f1_score_val

        if best_test_f1 < f1_score_test:
            best_test_f1 = f1_score_test
        if best_val_acc < accuracy_test:
            best_val_acc = accuracy_test

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            logger.debug(
                f"Epochs: {e}/{epochs}, Train F1-score: {f1_score_train}, Val F1-score: {f1_score_val}, Train Accuracy: "
                f"{accuracy_train}, Val Accuracy: {accuracy_val}, Best Accuracy: {best_val_acc}, Best F1-score: {best_val_f1}, Best Test F1-score: {best_test_f1}"
            )
    return train_list, val_list, test_list, loss_train, loss_val, loss_test


device = "cuda"
if __name__ == "__main__":
    torch.manual_seed(0)
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
    logger.debug(f"dataname {data_name}")
    dataset = GraphDataset(data_name, path=path)

    graph_train = dataset[True].to(device)
    graph_train = add_self_loop(graph_train)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    model = WGCN(
        graph_train.ndata["features"].shape[2],
        hidden_size,
        dataset.num_classes,
        nbr_hidden_layer,
        relu,
    ).to(device)
    # TODO: here sometime float some time double
    model.double()
    edge_weight = graph_train.edata["weight"].double().to(device)

    train_list, val_list, test_list, loss, loss_val, loss_test = train(
        graph_train,
        model,
        edge_weight,
        train_mask,
        val_mask,
        test_mask,
        dataset.num_classes,
        lr,
        epochs,
    )
