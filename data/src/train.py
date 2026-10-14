# Python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from src.dataset import BankingDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "banking77.csv"
MODEL_PATH = PROJECT_ROOT / "model"

# Load Dataset
data = pd.read_csv(DATA_PATH)
texts = data['text'].tolist()
labels = data['label'].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

# Split Data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = BankingDataset(train_texts, train_labels, tokenizer, max_length)


def collate_batch(batch):
    batch_length = max(int(item['attention_mask'].sum()) for item in batch)
    return {
        'input_ids': torch.stack([item['input_ids'][:batch_length] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'][:batch_length] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch
)

# Load Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    torch.set_num_threads(min(4, torch.get_num_threads()))
model.to(device)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_index, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_index % 50 == 0:
            print(
                f"Epoch {epoch + 1}, Batch {batch_index}/{len(train_loader)}, "
                f"Loss: {total_loss / batch_index:.4f}",
                flush=True
            )
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}", flush=True)
    model.save_pretrained(MODEL_PATH)

# Save Model
tokenizer.save_pretrained(MODEL_PATH)