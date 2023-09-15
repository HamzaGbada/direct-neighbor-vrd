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

from src.cnn_embedding.unet_embedding import UNet
from src.dataloader.SROIE_dataloader import SROIE
from src.dataloader.cord_dataloader import CORD
from src.dataloader.image_classification_dataloader import ImageDataset
from src.utils.setup_logger import logger


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
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits = model(input_ids, attention_mask)
            predictions = np.argmax(logits.cpu().numpy(), axis=1)

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    report = classification_report(all_labels, all_predictions,
                                   target_names=["0", "1", "3", "4", "5"])  # Replace with your class names
    return report


def image_dataloader(dataset, batch_size=1):
    convert_tensor = transforms.ToTensor()
    cropped_images = [convert_tensor(Image.open(os.path.join(dataset.root, dataset.data[doc_index][0])).convert("L").crop(bbox)) for doc_index in range(len(dataset)) for bbox in dataset.data[doc_index][1]['boxes']]
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


def main(dataloader, num_classes=5, num_epochs = 10, device = torch.device('cuda')):
    model = UNet(in_channels=1, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, loss_fn, optimizer, device)
        logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    logger.debug(f"Train evalution report{evaluate(model, dataloader, device)}")
    return model


if __name__ == '__main__':
    dataset_train = SROIE(train=True)
    dataset_test = SROIE(train=False)
    train_dataloader = image_dataloader(dataset_train)
    test_dataloader = image_dataloader(dataset_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = main(train_dataloader)
    logger.debug(f"Test evalution report{evaluate(model, test_dataloader, device)}")
