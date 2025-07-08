import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import load_phrasebank
from train_utils import train_fn, eval_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FinDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def main():
    df = load_phrasebank()
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3).to(device)

    train_dataset = FinDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    test_dataset = FinDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(4):
        loss = train_fn(model, train_loader, optimizer, criterion, device)
        acc, f1 = eval_fn(model, test_loader, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    model.save_pretrained("./finbert_model")

if __name__ == '__main__':
    main()
