# TODO: Train BERT for Classification then remove the last layer for embedding
#       (Same For Ngram)
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


class BertForSentenceClassification(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='bert-base-uncased'):
        super(BertForSentenceClassification, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_classes)

        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return self.softmax(logits)
