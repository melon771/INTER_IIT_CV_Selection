import os
import sys
import copy
import time
import pandas as pd
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

BATCH_SIZE = 128
EPOCHS = 60
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    scheduler.step()
    return running_loss / total, (correct / total) * 100.0

def evaluate_both_val(model, val_loader, noisy_labels, clean_labels, device):
    """
    Evaluates both noisy and clean validation accuracy in a single forward pass.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    all_preds = np.array(all_preds)
    val_acc_noisy = (all_preds == noisy_labels).mean() * 100.0
    val_acc_clean = (all_preds == clean_labels).mean() * 100.0
    return val_acc_noisy, val_acc_clean

def evaluate_train_subsets(model, train_eval_loader, noisy_train_labels, clean_train_labels, device):
    """
    Computes accuracy on correct (noisy==clean) and mislabeled (noisy!=clean) subsets of training set.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _, _ in train_eval_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    all_preds = np.array(all_preds)

    correct_mask = (noisy_train_labels == clean_train_labels)
    mislabeled_mask = (noisy_train_labels != clean_train_labels)

    acc_correct = (all_preds[correct_mask] == noisy_train_labels[correct_mask]).mean() * 100.0
    # Memorization is measured against the given noisy labels (targets)
    acc_mislabeled = (all_preds[mislabeled_mask] == noisy_train_labels[mislabeled_mask]).mean() * 100.0
    return acc_correct, acc_mislabeled

def main():
    set_seed(42)
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    data = load_cifar_data('data')
    train_images = data['train_images']
    test_images = data['test_images']
    labels_df = data['labels_df'].set_index('index')
    train_idx = data['train_indices']
    val_idx = data['val_indices']

    train_transform, eval_transform = get_transforms()

    # Pre-extract label arrays
    noisy_train_labels = labels_df.loc[train_idx, 'noisy_label'].values
    clean_train_labels = labels_df.loc[train_idx, 'clean_label'].values
    noisy_val_labels = labels_df.loc[val_idx, 'noisy_label'].values
    clean_val_labels = labels_df.loc[val_idx, 'clean_label'].values

    # Test loader
    test_dataset = CIFAR10TestDataset(test_images, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Val loader
    val_dataset = CIFAR10IndexedDataset(train_images, noisy_val_labels, val_idx, transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Train eval loader (no augmentations, for logging exact training dynamics)
    train_eval_dataset = CIFAR10IndexedDataset(train_images, noisy_train_labels, train_idx, transform=eval_transform)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # =========================================================================
    # Task 1 (Part A): Clean Reference Model
    # =========================================================================
    print("\n" + "="*70)
    print("TASK 1 (Part A): Training Clean Reference Model (60 epochs)")
    print("="*70)

    clean_train_dataset = CIFAR10IndexedDataset(train_images, clean_train_labels, train_idx, transform=train_transform)
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model_clean = ResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_clean = optim.SGD(model_clean.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_clean = optim.lr_scheduler.CosineAnnealingLR(optimizer_clean, T_max=EPOCHS, eta_min=1e-4)

    start_clean = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(model_clean, clean_train_loader, criterion, optimizer_clean, scheduler_clean, DEVICE)
        if epoch % 5 == 0 or epoch == EPOCHS:
            val_noisy_acc, val_clean_acc = evaluate_both_val(model_clean, val_loader, noisy_val_labels, clean_val_labels, DEVICE)
            print(f">>> [DONE EPOCH {epoch:2d}/{EPOCHS}] | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val(Clean): {val_clean_acc:.2f}% | Val(Noisy): {val_noisy_acc:.2f}%")
        else:
            print(f"  Epoch {epoch:2d}/{EPOCHS} finished (Train Loss: {loss:.4f}, Train Acc: {acc:.2f}%)")

    clean_time = time.time() - start_clean
    val_noisy_clean_ref, val_clean_clean_ref = evaluate_both_val(model_clean, val_loader, noisy_val_labels, clean_val_labels, DEVICE)
    print(f"Clean Ref finished in {clean_time:.1f}s. Val(Noisy): {val_noisy_clean_ref:.2f}%, Val(Clean): {val_clean_clean_ref:.2f}%")

    test_ids, test_preds_clean = predict_test(model_clean, test_loader, DEVICE)
    save_predictions('predictions/t1_clean_ref.csv', test_ids, test_preds_clean)
    torch.save(model_clean.state_dict(), 'checkpoints/model_clean_ref.pth')

    # =========================================================================
    # Task 1 (Part B) & Task 2: Noisy CE Baseline & Training Dynamics
    # =========================================================================
    print("\n" + "="*70)
    print("TASK 1 (Part B) & TASK 2: Training Noisy CE Model & Logging Dynamics (60 epochs)")
    print("="*70)

    noisy_train_dataset = CIFAR10IndexedDataset(train_images, noisy_train_labels, train_idx, transform=train_transform)
    noisy_train_loader = DataLoader(noisy_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model_noisy = ResNet18(num_classes=10).to(DEVICE)
    optimizer_noisy = optim.SGD(model_noisy.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_noisy = optim.lr_scheduler.CosineAnnealingLR(optimizer_noisy, T_max=EPOCHS, eta_min=1e-4)

    dynamics_log = []
    best_noisy_val_acc = -1.0
    best_epoch = -1
    best_model_state = None

    start_noisy = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_epoch(model_noisy, noisy_train_loader, criterion, optimizer_noisy, scheduler_noisy, DEVICE)

        # Training dynamics
        acc_correct, acc_mislabeled = evaluate_train_subsets(model_noisy, train_eval_loader, noisy_train_labels, clean_train_labels, DEVICE)
        val_noisy_acc, val_clean_acc = evaluate_both_val(model_noisy, val_loader, noisy_val_labels, clean_val_labels, DEVICE)

        dynamics_log.append({
            'epoch': epoch,
            'train_acc_correct': round(acc_correct, 4),
            'train_acc_mislabeled': round(acc_mislabeled, 4),
            'val_acc_noisy': round(val_noisy_acc, 4),
            'val_acc_clean': round(val_clean_acc, 4)
        })

        # Save checkpoint if best noisy validation accuracy
        if val_noisy_acc > best_noisy_val_acc:
            best_noisy_val_acc = val_noisy_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model_noisy.state_dict())

        torch.save(model_noisy.state_dict(), f'checkpoints/noisy_epoch_{epoch}.pth')

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f">>> [DONE EPOCH {epoch:2d}/{EPOCHS}] | Train(Corr): {acc_correct:5.2f}% | Train(Mis): {acc_mislabeled:5.2f}% | Val(Noisy): {val_noisy_acc:5.2f}% | Val(Clean): {val_clean_acc:5.2f}%")
        else:
            print(f"  Epoch {epoch:2d}/{EPOCHS} finished (Train Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%)")

    noisy_time = time.time() - start_noisy
    print(f"Noisy CE training finished in {noisy_time:.1f}s.")

    # Save Task 2 dynamics CSV
    dynamics_df = pd.DataFrame(dynamics_log)
    dynamics_df.to_csv('predictions/t2_dynamics.csv', index=False)
    print("Saved predictions/t2_dynamics.csv")

    # Task 1 Noisy CE final epoch predictions
    test_ids, test_preds_noisy_final = predict_test(model_noisy, test_loader, DEVICE)
    save_predictions('predictions/t1_noisy_ce.csv', test_ids, test_preds_noisy_final)

    val_acc_noisy_final = dynamics_log[-1]['val_acc_noisy']
    print(f"Task 1 Noisy CE Final (Epoch 60): Val Acc (Noisy) = {val_acc_noisy_final:.2f}%")

    # Task 2 Early-stopping predictions
    print(f"\nTask 2 Early Stopping: Best epoch = {best_epoch} with Noisy Val Acc = {best_noisy_val_acc:.2f}%")
    model_noisy.load_state_dict(best_model_state)
    test_ids, test_preds_early_stop = predict_test(model_noisy, test_loader, DEVICE)
    save_predictions('predictions/t2_early_stop.csv', test_ids, test_preds_early_stop)

    # Determine memorization start epoch
    # We analyze the mislabeled subset accuracy curve
    mis_accs = dynamics_df['train_acc_mislabeled'].values
    # In initial epochs, mislabeled accuracy is flat/near random guess.
    # Memorization starts when mislabeled accuracy begins sustained increase.
    # Let's find first epoch where mislabeled accuracy increases by > 1.0% in consecutive epochs or exceeds 15%
    diffs = np.diff(mis_accs)
    inflection_candidates = [i + 1 for i in range(1, len(diffs)) if diffs[i] > 1.0 and diffs[i-1] > 0.5]
    memorization_start = inflection_candidates[0] if inflection_candidates else int(np.where(mis_accs > 12.0)[0][0] + 1)

    print(f"\nTask 2 Summary:")
    print(f"  - Memorization Start Epoch: {memorization_start}")
    print(f"  - Early Stop Epoch: {best_epoch}")
    print(f"  - Val Acc (Noisy) at Early Stop: {best_noisy_val_acc:.2f}%")

    # Save summary for results.json
    summary = {
        'val_acc_noisy_clean_ref': val_noisy_clean_ref,
        'val_acc_noisy_noisy_ce': val_acc_noisy_final,
        'memorization_start_epoch': int(memorization_start),
        'early_stop_epoch': int(best_epoch),
        'val_acc_noisy_at_early_stop': best_noisy_val_acc
    }
    np.save('checkpoints/t1_t2_summary.npy', summary)
    print("Saved Task 1 & 2 summary to checkpoints/t1_t2_summary.npy")

if __name__ == '__main__':
    main()
