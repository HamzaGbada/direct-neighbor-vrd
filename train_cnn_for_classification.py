import torch
from torch.utils.data import DataLoader


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


def image_dataloader(dataset, batch_size=1):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
