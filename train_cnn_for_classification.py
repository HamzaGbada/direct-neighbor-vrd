import os
import warnings

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_accuracy
from torchvision import transforms
from tqdm import tqdm

from src.cnn_embedding.unet_embedding import EfficientNetV2MultiClass
from src.dataloader.cord_dataloader import CORD
from src.dataloader.image_classification_dataloader import ImageDataset
from src.utils.setup_logger import logger
from src.utils.utils import plots, process_labels, compute_f1_score

warnings.filterwarnings("ignore")


def compute_accuracy(label, pred):
    # Convert predicted probabilities to class labels by selecting the class with the highest probability
    predicted_labels = torch.argmax(pred)

    # Compute accuracy by comparing the predicted labels to the true labels
    correct_predictions = (predicted_labels == label).float()

    # Calculate the overall accuracy (we did not divde by len(label) because we will do it later
    return correct_predictions.sum() / len(label)


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

        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.unsqueeze(0)
            optimizer.zero_grad()

            outputs = model(inputs)

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
            for val_inputs, val_labels in tqdm(val_dataloader):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)

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
    total_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # logger.debug(f"shape of output {outputs.shape}")
        # logger.debug(f"shape of labels {labels.shape}")
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predictions = np.argmax(outputs.cpu().numpy(), axis=1)

            all_labels.extend(labels.cpu())
            all_predictions.extend(predictions)

    # report = classification_report(all_labels, all_predictions,
    #                                target_names=["0", "1", "3", "4", "5"])  # Replace with your class names
    return all_predictions


def image_dataloader(dataset, batch_size=1):
    convert_tensor = transforms.ToTensor()
    cropped_images = [
        convert_tensor(
            Image.open(os.path.join(dataset.root, dataset.data[doc_index][0]))
            .convert("L")
            .crop(bbox)
        )
        for doc_index in range(len(dataset))
        for bbox in dataset.data[doc_index][1]["boxes"]
    ]
    labels, name = process_labels(dataset)
    image_dataset = ImageDataset(cropped_images, labels, name)
    dataloader = DataLoader(
        image_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataloader


def main(
    train_dataloader,
    val_dataloader,
    num_classes=5,
    num_epochs=10,
    device=torch.device("cuda"),
):
    model = EfficientNetV2MultiClass(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    # for epoch in range(num_epochs):
    #     train_loss = train(model, dataloader, loss_fn, optimizer, device)
    #     logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

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
    # report = classification_report(all_labels, all_predictions,
    #                                target_names=["0", "1", "3", "4", "5"])  # Replace with your class names
    # logger.debug(f"classification report {report}")
    # logger.debug(f"Train evalution report{evaluate(model, train_dataloader, device)}")
    name = train_dataloader.dataset.__str__()
    plots(num_epochs, train_losses, val_losses, "Loss", name)
    plots(num_epochs, train_f1, val_f1, "F1 score", name)
    plots(num_epochs, train_acc, val_acc, "Accuracy", name)
    model_path = name + "_image_classification.pth"

    # Save the model to a file
    torch.save(model.state_dict(), model_path)

    return model


if __name__ == "__main__":
    dataset_train = CORD(train=True, download=True)
    dataset_test = CORD(train=False, download=True)
    train_dataloader = image_dataloader(dataset_train)
    test_dataloader = image_dataloader(dataset_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = main(
        train_dataloader, test_dataloader, num_epochs=1, num_classes=30, device=device
    )
    logger.debug(f"Test evalution report{evaluate(model, test_dataloader, device)}")
