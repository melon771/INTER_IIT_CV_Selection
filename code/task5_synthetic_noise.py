import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ResNet18
from dataset import load_cifar_data, get_transforms, CIFAR10IndexedDataset, CIFAR10TestDataset
from utils import set_seed, save_predictions, predict_test

# Identical training setup to Tasks 1-4
BATCH_SIZE = 128
EPOCHS = 60
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same CSR settings as Task 3/4 on real noise
ENSEMBLE_EPOCHS = [25, 30, 35, 38, 41]
SCORE_THRESH = 0.65
CONF_THRESH = 0.70


def predict_labels(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _, _ in loader:
            preds.append(model(images.to(DEVICE, non_blocking=True)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def get_probs(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _, _ in loader:
            probs.append(torch.softmax(model(images.to(DEVICE, non_blocking=True)), dim=1).cpu().numpy())
    return np.vstack(probs)


def train_ce(tag, train_images, labels, train_idx, val_loader, val_labels, train_transform, save_epochs=()):
    set_seed(42)
    dataset = CIFAR10IndexedDataset(train_images, labels, train_idx, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, y, _ in loader:
            images, y = images.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        val_acc = (predict_labels(model, val_loader) == val_labels).mean() * 100.0
        history.append(val_acc)
        if epoch in save_epochs:
            torch.save(model.state_dict(), f'checkpoints/synth_epoch_{epoch}.pth')
        print(f"[{tag}] Epoch {epoch:2d}/{EPOCHS} | Loss {total_loss/total:.4f} | "
              f"Train Acc {correct/total*100:.2f}% | Val(Synth) {val_acc:.2f}%", flush=True)
    return model, history


def run_task5():
    print("=" * 70)
    print("TASK 5: Real vs Synthetic Noise (CE baseline vs CSR, synthetic labels only)")
    print("=" * 70)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    data = load_cifar_data('data')
    train_images = data['train_images']
    labels_df = data['labels_df'].set_index('index')
    train_idx = data['train_indices']
    val_idx = data['val_indices']
    train_transform, eval_transform = get_transforms()

    synth_train = labels_df.loc[train_idx, 'synthetic_label'].values
    synth_val = labels_df.loc[val_idx, 'synthetic_label'].values

    test_loader = DataLoader(CIFAR10TestDataset(data['test_images'], transform=eval_transform),
                             batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(CIFAR10IndexedDataset(train_images, synth_val, val_idx, transform=eval_transform),
                            batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(CIFAR10IndexedDataset(train_images, synth_train, train_idx, transform=eval_transform),
                                   batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # 1) Task 1 CE baseline on synthetic labels (final epoch, as in Task 1)
    base_model, base_hist = train_ce('Baseline-CE', train_images, synth_train, train_idx,
                                     val_loader, synth_val, train_transform, save_epochs=ENSEMBLE_EPOCHS)
    ids, preds = predict_test(base_model, test_loader, DEVICE)
    save_predictions('predictions/t5_baseline_synth.csv', ids, preds)
    base_final = base_hist[-1]
    del base_model
    torch.cuda.empty_cache()

    # 2) Best Task 4 method (CSR): early-learning ensemble scores from the synthetic baseline run
    probs_list = []
    for ep in ENSEMBLE_EPOCHS:
        m = ResNet18(num_classes=10).to(DEVICE)
        m.load_state_dict(torch.load(f'checkpoints/synth_epoch_{ep}.pth', map_location=DEVICE))
        probs_list.append(get_probs(m, train_eval_loader))
        del m
    mean_probs = np.mean(probs_list, axis=0)
    scores = 1.0 - mean_probs[np.arange(len(synth_train)), synth_train]

    corrected = synth_train.copy()
    relabel_mask = (scores > SCORE_THRESH) & (mean_probs.max(axis=1) > CONF_THRESH)
    corrected[relabel_mask] = mean_probs.argmax(axis=1)[relabel_mask]
    n_relabel = int(relabel_mask.sum())
    print(f"CSR relabeled {n_relabel} samples ({n_relabel / len(corrected) * 100:.2f}%) on synthetic labels.")

    csr_model, csr_hist = train_ce('CSR', train_images, corrected, train_idx,
                                   val_loader, synth_val, train_transform)
    ids, preds = predict_test(csr_model, test_loader, DEVICE)
    save_predictions('predictions/t5_best_synth.csv', ids, preds)
    csr_final = csr_hist[-1]

    summary = {
        'val_acc_synthetic_labels__baseline': round(float(base_final), 2),
        'val_acc_synthetic_labels__best_method': round(float(csr_final), 2),
        'baseline_peak_val': round(float(max(base_hist)), 2),
        'baseline_peak_epoch': int(np.argmax(base_hist) + 1),
        'csr_peak_val': round(float(max(csr_hist)), 2),
        'csr_num_relabeled': n_relabel,
        'baseline_val_history': [round(float(v), 2) for v in base_hist],
        'csr_val_history': [round(float(v), 2) for v in csr_hist],
    }
    with open('checkpoints/t5_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Baseline CE (synthetic) final val: {base_final:.2f}% (peak {max(base_hist):.2f}% @ ep {np.argmax(base_hist)+1})")
    print(f"CSR (synthetic) final val:         {csr_final:.2f}% (peak {max(csr_hist):.2f}%)")
    print("=" * 70)


if __name__ == '__main__':
    run_task5()
