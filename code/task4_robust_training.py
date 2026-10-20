import os
import sys
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ResNet18
from dataset import load_cifar_data, get_transforms, CIFAR10IndexedDataset, CIFAR10TestDataset
from utils import set_seed, save_predictions, predict_test

BATCH_SIZE = 128
EPOCHS = 60
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------
class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy (Wang et al., ICCV 2019)
    L_SCE = alpha * L_CE + beta * L_RCE
    """
    def __init__(self, alpha=0.5, beta=1.0, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, targets):
        ce_loss = self.ce(pred, targets)
        
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        
        targets_onehot = F.one_hot(targets, self.num_classes).float()
        targets_onehot = torch.clamp(targets_onehot, min=1e-4, max=1.0)
        
        # RCE: - \sum pred * log(target_onehot)
        rce_loss = (-1.0 * torch.sum(pred_softmax * torch.log(targets_onehot), dim=1)).mean()
        return self.alpha * ce_loss + self.beta * rce_loss

def evaluate_val_acc(model, val_loader, noisy_val_labels, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
    all_preds = np.array(all_preds)
    return (all_preds == noisy_val_labels).mean() * 100.0

# -----------------------------------------------------------------------------
# Method 1: Symmetric Cross Entropy (SCE)
# -----------------------------------------------------------------------------
def train_method_sce(train_images, noisy_train_labels, train_idx, val_loader, noisy_val_labels, test_loader, train_transform):
    print("\n" + "="*70)
    print("TASK 4 - Method 1: Symmetric Cross Entropy (SCE)")
    print("="*70)
    set_seed(42)
    dataset = CIFAR10IndexedDataset(train_images, noisy_train_labels, train_idx, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet18(num_classes=10).to(DEVICE)
    criterion = SCELoss(alpha=0.5, beta=1.0, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc = evaluate_val_acc(model, val_loader, noisy_val_labels, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"SCE Epoch {epoch:2d}/{EPOCHS} | Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | Val(Noisy): {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    # Final epoch predictions as per standard practice, but we also save best checkpoint
    model.eval()
    test_ids, test_preds = predict_test(model, test_loader, DEVICE)
    save_predictions('predictions/t4_sce.csv', test_ids, test_preds)
    val_acc_final = evaluate_val_acc(model, val_loader, noisy_val_labels, DEVICE)
    return {
        'name': 'Symmetric Cross Entropy (SCE)',
        'file': 't4_sce.csv',
        'val_acc_noisy_labels': round(float(val_acc_final), 4),
        'best_val_acc': round(float(best_val_acc), 4)
    }

# -----------------------------------------------------------------------------
# Method 2: Confident Sample Relabeling (CSR) using Task 3 Scores
# -----------------------------------------------------------------------------
def train_method_relabel(train_images, noisy_train_labels, train_idx, val_loader, noisy_val_labels, test_loader, train_transform):
    print("\n" + "="*70)
    print("TASK 4 - Method 2: Confident Sample Relabeling (CSR)")
    print("="*70)
    set_seed(42)

    # Load Task 3 scores and ensemble probabilities
    t3_data = np.load('checkpoints/t3_noise_scores.npy', allow_pickle=True).item()
    scores = t3_data['scores']
    mean_probs = t3_data['mean_probs']
    
    # Construct cleaned/relabeled dataset
    # If high noise score (> 0.65) and confident prediction (max prob > 0.70), relabel with argmax
    corrected_labels = np.copy(noisy_train_labels)
    pred_labels = mean_probs.argmax(axis=1)
    max_conf = mean_probs.max(axis=1)
    
    relabel_mask = (scores > 0.65) & (max_conf > 0.70)
    corrected_labels[relabel_mask] = pred_labels[relabel_mask]
    num_relabeled = relabel_mask.sum()
    print(f"Relabeled {num_relabeled} high-confidence noisy samples ({num_relabeled/len(corrected_labels)*100:.2f}%) using Task 3 ensemble.")

    dataset = CIFAR10IndexedDataset(train_images, corrected_labels, train_idx, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc = evaluate_val_acc(model, val_loader, noisy_val_labels, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"CSR Epoch {epoch:2d}/{EPOCHS} | Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | Val(Noisy): {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    model.eval()
    test_ids, test_preds = predict_test(model, test_loader, DEVICE)
    save_predictions('predictions/t4_relabel.csv', test_ids, test_preds)
    val_acc_final = evaluate_val_acc(model, val_loader, noisy_val_labels, DEVICE)
    torch.save(best_state, 'checkpoints/model_t4_relabel_best.pth')

    return {
        'name': 'Task3 Confident Relabeling (CSR)',
        'file': 't4_relabel.csv',
        'val_acc_noisy_labels': round(float(val_acc_final), 4),
        'best_val_acc': round(float(best_val_acc), 4)
    }

# -----------------------------------------------------------------------------
# Method 3: Co-Teaching
# -----------------------------------------------------------------------------
def train_method_coteaching(train_images, noisy_train_labels, train_idx, val_loader, noisy_val_labels, test_loader, train_transform):
    print("\n" + "="*70)
    print("TASK 4 - Method 3: Co-Teaching (2 Networks)")
    print("="*70)
    set_seed(42)

    dataset = CIFAR10IndexedDataset(train_images, noisy_train_labels, train_idx, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    net1 = ResNet18(num_classes=10).to(DEVICE)
    net2 = ResNet18(num_classes=10).to(DEVICE)

    opt1 = optim.SGD(net1.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    opt2 = optim.SGD(net2.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS, eta_min=1e-4)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS, eta_min=1e-4)

    # Noise rate schedule: ramp up to ~0.40 over 15 epochs
    noise_rate = 0.40
    num_gradual = 15

    best_val_acc = -1.0
    best_state1 = None

    for epoch in range(1, EPOCHS + 1):
        net1.train()
        net2.train()
        # Keep rate: 1 - R(t)
        r_t = min(noise_rate * (epoch / num_gradual), noise_rate)
        keep_rate = 1.0 - r_t

        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            b_size = images.size(0)
            num_remember = int(keep_rate * b_size)

            out1 = net1(images)
            loss1_all = F.cross_entropy(out1, labels, reduction='none')
            ind1_sorted = torch.argsort(loss1_all)
            ind1_update = ind1_sorted[:num_remember]

            out2 = net2(images)
            loss2_all = F.cross_entropy(out2, labels, reduction='none')
            ind2_sorted = torch.argsort(loss2_all)
            ind2_update = ind2_sorted[:num_remember]

            # Cross-update: net1 trains on ind2_update, net2 trains on ind1_update
            opt1.zero_grad(set_to_none=True)
            loss1 = F.cross_entropy(out1[ind2_update], labels[ind2_update])
            loss1.backward()
            opt1.step()

            opt2.zero_grad(set_to_none=True)
            loss2 = F.cross_entropy(out2[ind1_update], labels[ind1_update])
            loss2.backward()
            opt2.step()

        sched1.step()
        sched2.step()

        val_acc1 = evaluate_val_acc(net1, val_loader, noisy_val_labels, DEVICE)
        val_acc2 = evaluate_val_acc(net2, val_loader, noisy_val_labels, DEVICE)
        avg_val_acc = (val_acc1 + val_acc2) / 2.0

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_state1 = copy.deepcopy(net1.state_dict())

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Co-Teaching Epoch {epoch:2d}/{EPOCHS} | Val1: {val_acc1:.2f}% | Val2: {val_acc2:.2f}% | KeepRate: {keep_rate*100:.1f}%")

    # Predict test with ensemble of net1 and net2
    net1.eval()
    net2.eval()
    all_ids, all_preds = [], []
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(DEVICE)
            p1 = F.softmax(net1(images), dim=1)
            p2 = F.softmax(net2(images), dim=1)
            ensemble_p = (p1 + p2) / 2.0
            all_preds.extend(ensemble_p.argmax(dim=1).cpu().numpy())
            all_ids.extend(ids.numpy())
    
    paired = sorted(zip(all_ids, all_preds), key=lambda x: x[0])
    save_predictions('predictions/t4_coteaching.csv', [p[0] for p in paired], [p[1] for p in paired])
    
    val_acc_final = (val_acc1 + val_acc2) / 2.0
    return {
        'name': 'Co-Teaching',
        'file': 't4_coteaching.csv',
        'val_acc_noisy_labels': round(float(val_acc_final), 4),
        'best_val_acc': round(float(best_val_acc), 4)
    }

def run_task4(methods=('sce', 'relabel', 'coteaching')):
    data = load_cifar_data('data')
    train_images = data['train_images']
    test_images = data['test_images']
    labels_df = data['labels_df'].set_index('index')
    train_idx = data['train_indices']
    val_idx = data['val_indices']

    train_transform, eval_transform = get_transforms()

    noisy_train_labels = labels_df.loc[train_idx, 'noisy_label'].values
    noisy_val_labels = labels_df.loc[val_idx, 'noisy_label'].values

    test_dataset = CIFAR10TestDataset(test_images, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    val_dataset = CIFAR10IndexedDataset(train_images, noisy_val_labels, val_idx, transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    trainers = {
        'sce': train_method_sce,
        'relabel': train_method_relabel,
        'coteaching': train_method_coteaching,
    }
    args = (train_images, noisy_train_labels, train_idx, val_loader, noisy_val_labels, test_loader, train_transform)

    # Merge with previously saved results so methods can be (re)run individually
    summary_json = 'checkpoints/t4_summary.json'
    saved = {}
    if os.path.exists(summary_json):
        with open(summary_json) as f:
            saved = {m['file']: m for m in json.load(f)['methods']}
    for key in methods:
        res = trainers[key](*args)
        saved[res['file']] = res
        with open(summary_json, 'w') as f:
            json.dump({'methods': list(saved.values())}, f, indent=2)
    results = list(saved.values())

    # Determine best method based on highest val_acc_noisy_labels
    best_method = max(results, key=lambda x: x['val_acc_noisy_labels'])
    print("\n" + "="*70)
    print("TASK 4 SUMMARY:")
    for r in results:
        print(f"  - {r['name']:35s}: Val Acc (Noisy) = {r['val_acc_noisy_labels']:.2f}% (File: {r['file']})")
    print(f"\nBest Task 4 Method: {best_method['name']} with {best_method['val_acc_noisy_labels']:.2f}% noisy val acc")
    print("="*70)

    np.save('checkpoints/t4_summary.npy', {
        'methods': results,
        'best_method': best_method['name']
    })

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['sce', 'relabel', 'coteaching'],
                        choices=['sce', 'relabel', 'coteaching'])
    run_task4(parser.parse_args().methods)
