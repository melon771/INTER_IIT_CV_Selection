import csv
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def save_predictions(filepath, ids, preds):
    """
    Saves predictions in format:
    id,pred
    0,3
    ...
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        for pid, pred in zip(ids, preds):
            writer.writerow([int(pid), int(pred)])
    print(f"Saved predictions to {filepath} ({len(preds)} rows)")

def predict_test(model, test_loader, device):
    """
    Generates test predictions (ids 0..9999).
    """
    model.eval()
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_ids.extend(ids.numpy())
            all_preds.extend(preds)
    # Ensure ordered by id
    paired = sorted(zip(all_ids, all_preds), key=lambda x: x[0])
    ids_sorted = [p[0] for p in paired]
    preds_sorted = [p[1] for p in paired]
    return ids_sorted, preds_sorted

def evaluate_accuracy(model, data_loader, device):
    """
    Computes classification accuracy (percentage 0..100).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0
