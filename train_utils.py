import torch
from sklearn.metrics import accuracy_score, f1_score

def train_fn(model, data_loader, optimizer, criterion, device):
    model.train()
    losses = []
    for batch in data_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def eval_fn(model, data_loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs).logits
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            true.extend(labels.cpu().numpy())
    return accuracy_score(true, preds), f1_score(true, preds, average='macro')
