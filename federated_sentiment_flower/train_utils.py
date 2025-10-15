from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def get_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    return model

def train(model, dataset, epochs=1, batch_size=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=8)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])
    return correct / total