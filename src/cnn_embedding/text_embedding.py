import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from pathlib import Path


class TextEmbeddingModel:
    def __init__(self, model_path, num_classes=30, feat_size=500, device="cuda"):
        # Initialize the model and load the state_dict
        self.model = BertForSequenceClassification(num_classes)
        self.load_model(Path(model_path))

        # Define the reshaping layers
        self.reshaping_layers = nn.Sequential(
            nn.Linear(
                num_classes, feat_size
            ),  # Linear layer to reshape from feat_size to 500 features
            nn.Tanh(),  # You can add activation functions as needed
        )

        # Transfer the model and reshaping layers to the GPU
        self.to_device(device)

        # Set the model to evaluation mode
        self.model.eval()

    def load_model(self, model_path):
        # Load the model state_dict
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        # Load the state_dict to the model
        self.model.load_state_dict(state_dict)

    def to_device(self, device):
        # Transfer the model and reshaping layers to the specified device
        self.model.to(device=device)
        self.reshaping_layers.to(device=device)

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
        input_ids = batch["input_ids"].to(device="cuda")
        attention_mask = batch["attention_mask"].to(device="cuda")
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        # Forward pass through the model
        outputs = self.model(input_ids, attention_mask)

        # Apply reshaping layers
        reshaped_output = self.reshaping_layers(outputs)

        return reshaped_output
