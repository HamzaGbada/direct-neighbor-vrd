import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from src.cnn_embedding.unet_embedding import UNet
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.image_classification_dataloader import ImageDataset
from src.utils.setup_logger import logger


def train_and_evaluate(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, num_epochs):
    train_losses = []  # To store training loss for each epoch
    val_losses = []  # To store validation loss for each epoch
    all_labels = []
    all_predictions = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loss calculation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_labels)
                predictions = np.argmax(val_outputs.cpu().numpy(), axis=1)

                all_labels.extend(val_labels.cpu())
                all_predictions.extend(predictions.cpu())

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Print and plot the losses
        logger.debug(
            f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}')

    return model, all_labels, all_predictions, train_losses, val_losses


def plot_loss(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(np.arange(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    plt.show()


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        logger.debug(f"shape of output {outputs.shape}")
        logger.debug(f"shape of labels {labels.shape}")
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


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
            all_predictions.extend(predictions.cpu())

    report = classification_report(all_labels, all_predictions,
                                   target_names=["0", "1", "3", "4", "5"])  # Replace with your class names
    return report


def image_dataloader(dataset, batch_size=1):
    convert_tensor = transforms.ToTensor()
    cropped_images = [
        convert_tensor(Image.open(os.path.join(dataset.root, dataset.data[doc_index][0])).convert("L").crop(bbox)) for
        doc_index in range(len(dataset)) for bbox in dataset.data[doc_index][1]['boxes']]
    labels = [x for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['labels']]
    labels = torch.tensor(labels).reshape(-1, 1)
    # TODO: Change THE dependening on Dataset (In this case SROIE)
    X = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1)
    enc = OneHotEncoder(sparse=False)
    enc.fit(X)
    labels = torch.from_numpy(enc.transform(labels))
    image_dataset = ImageDataset(cropped_images, labels)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main(train_dataloader, val_dataloader, num_classes=5, num_epochs=10, device=torch.device('cuda')):
    model = UNet(in_channels=1, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    # for epoch in range(num_epochs):
    #     train_loss = train(model, dataloader, loss_fn, optimizer, device)
    #     logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    model, all_labels, all_predictions, train_losses, val_losses = train_and_evaluate(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, num_epochs)
    report = classification_report(all_labels, all_predictions,
                                   target_names=["0", "1", "3", "4", "5"])  # Replace with your class names
    logger.debug(f"classification report {report}")
    logger.debug(f"Train evalution report{evaluate(model, train_dataloader, device)}")
    plot_loss(num_epochs, train_losses, val_losses)
    model_path = 'Unet_classification.pth'

    # Save the model to a file
    torch.save(model.state_dict(), model_path)

    return model


if __name__ == '__main__':
    dataset_train = SROIE(train=True)
    dataset_test = SROIE(train=False)
    train_dataloader = image_dataloader(dataset_train)
    test_dataloader = image_dataloader(dataset_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = main(train_dataloader, test_dataloader)
    logger.debug(f"Test evalution report{evaluate(model, test_dataloader, device)}")
