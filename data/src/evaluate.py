# Python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.dataset import BankingDataset
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "banking77.csv"
MODEL_PATH = PROJECT_ROOT / "model"

data = pd.read_csv(DATA_PATH)
texts = data['text'].tolist()
labels = data['label'].tolist()

_, test_texts, _, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
max_length = 128

test_dataset = BankingDataset(test_texts, test_labels, tokenizer, max_length)


def collate_batch(batch):
    batch_length = max(int(item['attention_mask'].sum()) for item in batch)
    return {
        'input_ids': torch.stack([item['input_ids'][:batch_length] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'][:batch_length] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_batch)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    torch.set_num_threads(min(4, torch.get_num_threads()))
model.to(device)

model.eval()
predictions, true_labels = [], []

with torch.inference_mode():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(batch_labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(true_labels, predictions))