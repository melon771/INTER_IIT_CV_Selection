import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ResNet18
from dataset import load_cifar_data, get_transforms, CIFAR10IndexedDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_task3_scores():
    print("Computing Task 3 Noise Scores...")
    data = load_cifar_data('data')
    labels_df = data['labels_df'].set_index('index')
    train_idx = data['train_indices']
    noisy_labels = labels_df.loc[train_idx, 'noisy_label'].values
    clean_labels = labels_df.loc[train_idx, 'clean_label'].values
    true_noise = (noisy_labels != clean_labels).astype(int)
    k = true_noise.sum()

    _, eval_transform = get_transforms()
    eval_dataset = CIFAR10IndexedDataset(data['train_images'], noisy_labels, train_idx, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    def get_probs(checkpoint_path):
        model = ResNet18().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        all_probs = []
        with torch.no_grad():
            for imgs, _, _ in eval_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    # Multi-checkpoint early-learning ensemble across pre-memorization epochs
    checkpoint_epochs = [25, 30, 35, 38, 41]
    print(f"Aggregating predictions from epochs: {checkpoint_epochs}...")
    probs_list = [get_probs(f'checkpoints/noisy_epoch_{ep}.pth') for ep in checkpoint_epochs]
    mean_probs = np.mean(probs_list, axis=0)

    # Probability assigned to the given noisy label
    noisy_label_probs = mean_probs[np.arange(len(noisy_labels)), noisy_labels]

    # Error score: 1 - P(noisy_label | x)
    # Higher score = more likely wrong
    scores = 1.0 - noisy_label_probs

    # Internal validation of AUROC and Precision@k (allowed under Task 3 rules)
    auroc = roc_auc_score(true_noise, scores)
    top_k_indices = np.argsort(scores)[::-1][:k]
    prec_at_k = true_noise[top_k_indices].mean() * 100.0

    print(f"\nTask 3 Noise Score Results:")
    print(f"  - AUROC: {auroc:.4f} ({auroc * 100:.2f}%)")
    print(f"  - Precision@{k}: {prec_at_k:.2f}%")

    # Save to predictions/t3_noise_scores.csv with header index,score
    scores_df = pd.DataFrame({
        'index': train_idx,
        'score': scores
    })
    scores_df.to_csv('predictions/t3_noise_scores.csv', index=False)
    print(f"Saved {len(scores_df)} scores to predictions/t3_noise_scores.csv")

    # Save numpy scores for reuse in Task 4 sample selection / relabeling
    np.save('checkpoints/t3_noise_scores.npy', {
        'indices': train_idx,
        'scores': scores,
        'mean_probs': mean_probs,
        'auroc': auroc,
        'prec_at_k': prec_at_k
    })
    print("Saved Task 3 scores and ensemble probabilities to checkpoints/t3_noise_scores.npy")

if __name__ == '__main__':
    compute_task3_scores()
