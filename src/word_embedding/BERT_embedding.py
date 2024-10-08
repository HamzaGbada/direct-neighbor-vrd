from pathlib import Path

import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class BertSentenceClassification(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="bert-base-uncased"):
        super(BertSentenceClassification, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return self.softmax(logits)


class TextEmbeddingModel:
    device: str

    def __init__(self, model_path, num_classes=30, feat_size=500, device="cuda"):
        # Initialize the model and load the state_dict

        self.model = BertSentenceClassification(num_classes)
        self.device = device
        self.load_model(Path(model_path))

        # Define the reshaping layers
        self.reshaping_layers = nn.Sequential(
            nn.Linear(
                num_classes, feat_size
            ),  # Linear layer to reshape from feat_size to 500 features
            nn.Tanh(),  # You can add activation functions as needed
        )

        # Transfer the model and reshaping layers to the GPU
        self.model.to(device=self.device)
        self.reshaping_layers.to(device=self.device)

        # Set the model to evaluation mode
        self.model.eval()

    def load_model(self, model_path):
        # Load the model state_dict
        state_dict = torch.load(model_path, map_location=torch.device(self.device))

        # Load the state_dict to the model
        self.model.load_state_dict(state_dict)

    def to_device(self, device):
        # Transfer the model and reshaping layers to the specified device
        self.model = self.model.to(device=device)
        self.reshaping_layers = self.reshaping_layers.to(device=device)

    def embed_text(self, sentence):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        max_len = 128

        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_len,
            return_tensors="pt",
            pad_to_max_length=True,
            truncation=True,
        )

        batch = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        input_ids = batch["input_ids"].to(device=self.device)
        attention_mask = batch["attention_mask"].to(device=self.device)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        # Forward pass through the model
        outputs = self.model(input_ids, attention_mask)

        # Apply reshaping layers
        reshaped_output = self.reshaping_layers(outputs)

        return reshaped_output
