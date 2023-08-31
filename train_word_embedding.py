import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from transformers import BertTokenizer, AdamW

from src.dataloader.cord_dataloader import CORD
from src.utils.setup_logger import logger
from src.dataloader.sentence_classification_dataloader import create_dataloader
from src.word_embedding.BERT_embedding import BertForSentenceClassification


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits = model(input_ids, attention_mask)
            predictions = np.argmax(logits.cpu().numpy(), axis=1)

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    report = classification_report(all_labels, all_predictions,
                                   target_names=["CLASS_1", "CLASS_2", ...])  # Replace with your class names
    return report


def word_embedding_dataloader(dataset, max_len=128, batch_size=16):
    sentences = [x for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['text_units']]
    labels = [x for doc_index in range(len(dataset)) for x in dataset.data[doc_index][1]['labels']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataloader = create_dataloader(sentences, labels, tokenizer, max_len, batch_size)
    return dataloader


def main(train_dataloader, num_classes=5, num_epochs = 10, device = torch.device('cpu')):
    model = BertForSentenceClassification(num_classes)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    logger.debug(f"Train evalution report{evaluate(model, train_dataloader, device)}")
    return model


if __name__ == '__main__':
    dataset_train = CORD(train=True)
    dataset_test = CORD(train=False)
    train_dataloader = word_embedding_dataloader(dataset_train)
    test_dataloader = word_embedding_dataloader(dataset_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = main(train_dataloader)
    logger.debug(f"Test evalution report{evaluate(model, test_dataloader, device)}")
