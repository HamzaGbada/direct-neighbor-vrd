# TODO: - Create multiple model for GNN
import argparse

from dgl import add_self_loop
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam
from torchmetrics.functional.classification import (
    multiclass_f1_score,
    multilabel_accuracy,
)

from args import train_subparser
from src.dataloader.graph_dataset import GraphDataset
from src.graph_builder.graph_model import WGCN
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
    optimizer = Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_val_f1 = 0
    best_test_f1 = 0

    features = g.ndata["features"]
    labels = g.ndata["label"]
    # label_binarizer = LabelBinarizer()
    # label_binarizer.fit(range(max(labels) + 1))
    # labels = torch.from_numpy(label_binarizer.transform(labels.to('cpu'))).to('cuda')

    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask
    train_list, val_list, test_list = [], [], []
    loss_train, loss_val, loss_test = [], [], []
    for e in range(epochs):
        # Forward

        logits = model(g, features, edge_weight)

        f1_score_train = compute_f1_score(labels.view(-1), logits.view(-1))
        accuracy_train = multilabel_accuracy(
            logits, labels, num_labels=num_class, average="macro"
        )
        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        logger.debug(f"feature shape labels[train_mask]{labels[train_mask].shape}")
        logger.debug(f"feature shape logits[train_mask]{logits[train_mask].shape}")
        # TODO: the error of shape (check the output of the below) is due to the multiclass classification (change
        #  the label)
        loss = cross_entropy(logits[train_mask], labels[train_mask])
        loss_train.append(loss.to("cpu").detach().numpy())
        loss_val.append(
            cross_entropy(logits[val_mask], labels[val_mask]).to("cpu").detach().numpy()
        )
        loss_test.append(
            cross_entropy(logits[test_mask], labels[test_mask])
            .to("cpu")
            .detach()
            .numpy()
        )

        # Compute accuracy on training/validation/test
        train_f1 = multiclass_f1_score(
            pred[train_mask], labels[train_mask], num_classes=num_class, average="micro"
        )
        val_f1 = multiclass_f1_score(
            pred[val_mask], labels[val_mask], num_classes=num_class, average="micro"
        )
        test_f1 = multiclass_f1_score(
            pred[test_mask], labels[test_mask], num_classes=num_class, average="micro"
        )
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        train_list.append(train_f1.to("cpu"))
        val_list.append(val_f1.to("cpu"))
        test_list.append(test_f1.to("cpu"))
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            # best_test_acc = test_acc
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1

        if best_test_f1 < test_f1:
            best_test_f1 = test_f1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"Best Test f1-score {best_test_f1}")
        if e % 10 == 0:
            logger.debug(
                f"Epochs: {e}/{epochs}, Train F1-score: {train_f1}, Val F1-score: {val_f1}, Train Accuracy: "
                f"{train_acc}, Val Accuracy: {val_acc}, Best Accuracy: {best_val_acc}, Best F1-score: {best_val_f1}, Best Test F1-score: {best_test_f1}"
            )
    return train_list, val_list, test_list, loss_train, loss_val, loss_test


if __name__ == "__main__":
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
    epochs = 2

    dataset = GraphDataset(data_name, path=path)

    graph_train = dataset[True].to("cuda")
    graph_train = add_self_loop(graph_train)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # test_index = torch.arange(graph_test.number_of_nodes())
    logger.debug(f"data training indexes {train_mask}")
    logger.debug(f"data validiation indexes {val_mask}")
    # logger.debug(f"data testing indexes {test_index}")

    model = WGCN(
        graph_train.ndata["feat"].shape[2],
        hidden_size,
        dataset.num_classes,
        nbr_hidden_layer,
        relu,
    ).to("cuda")
    # TODO: here sometime float some time double
    model.double()
    edge_weight = graph_train.edata["weight"].double().to("cuda")

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
