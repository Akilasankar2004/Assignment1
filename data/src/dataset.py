# Python
import torch
from torch.utils.data import Dataset

class BankingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.as_tensor(label, dtype=torch.long)
        }