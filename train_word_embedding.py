import torch
from torch import nn

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


# Step 5: Training Loop
def main():
    sentences = [...]  # List of sentences
    labels = [...]  # List of labels (numeric values)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16
    num_classes = 5  # Replace with the actual number of classes

    train_dataloader = create_dataloader(sentences, labels, tokenizer, max_len, batch_size)

    model = BertForSentenceClassification(num_classes)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')


if __name__ == '__main__':
    main()