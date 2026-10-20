import os
import pandas as pd
import numpy as np

def analyze():
    labels_df = pd.read_csv('data/labels.csv')
    with open('data/classes.txt') as f:
        classes = [line.strip() for line in f if line.strip()]

    train_idx = set(np.loadtxt('data/train_idx.txt', dtype=int))
    val_idx = set(np.loadtxt('data/val_idx.txt', dtype=int))

    train_df = labels_df[labels_df['index'].isin(train_idx)].copy()
    val_df = labels_df[labels_df['index'].isin(val_idx)].copy()

    train_noise_rate = (train_df['noisy_label'] != train_df['clean_label']).mean()
    val_noise_rate = (val_df['noisy_label'] != val_df['clean_label']).mean()
    overall_noise_rate = (labels_df['noisy_label'] != labels_df['clean_label']).mean()
    synth_train_noise_rate = (train_df['synthetic_label'] != train_df['clean_label']).mean()

    print(f"Train Noise Rate: {train_noise_rate*100:.2f}% ({train_noise_rate:.6f})")
    print(f"Val Noise Rate: {val_noise_rate*100:.2f}% ({val_noise_rate:.6f})")
    print(f"Overall Noise Rate: {overall_noise_rate*100:.2f}% ({overall_noise_rate:.6f})")
    print(f"Synthetic Train Noise Rate: {synth_train_noise_rate*100:.2f}% ({synth_train_noise_rate:.6f})")

    # 10x10 Confusion Matrix (rows = clean_label, cols = noisy_label)
    cm = pd.crosstab(train_df['clean_label'], train_df['noisy_label'], rownames=['Clean_True'], colnames=['Noisy_Annotated'])
    print("\n10x10 Confusion Matrix (Clean vs Noisy) on Train Set:")
    print(cm)

    # Class names
    cm_named = cm.copy()
    cm_named.index = [classes[i] for i in cm_named.index]
    cm_named.columns = [classes[i] for i in cm_named.columns]
    print("\nNamed Confusion Matrix:")
    print(cm_named)

    # Mismatches
    mismatches = train_df[train_df['clean_label'] != train_df['noisy_label']].copy()
    total_noise_count = len(mismatches)
    print(f"\nTotal mislabeled training samples: {total_noise_count} / {len(train_df)}")

    # Directional errors
    pair_counts = mismatches.groupby(['clean_label', 'noisy_label']).size().reset_index(name='count')
    pair_counts = pair_counts.sort_values(by='count', ascending=False)
    pair_counts['clean_name'] = pair_counts['clean_label'].map(lambda x: classes[x])
    pair_counts['noisy_name'] = pair_counts['noisy_label'].map(lambda x: classes[x])
    pair_counts['pct_of_noise'] = (pair_counts['count'] / total_noise_count) * 100

    print("\nTop 15 Directional Error Pairs (Clean -> Noisy):")
    for _, row in pair_counts.head(15).iterrows():
        print(f"  {row['clean_name']:10s} -> {row['noisy_name']:10s}: {row['count']:4d} samples ({row['pct_of_noise']:.2f}% of noise)")

    # Unordered pairs
    mismatches['pair'] = mismatches.apply(lambda r: tuple(sorted([classes[r['clean_label']], classes[r['noisy_label']]])), axis=1)
    unordered = mismatches['pair'].value_counts()
    print("\nTop 10 Unordered Confused Class Pairs:")
    for pair, cnt in unordered.head(10).items():
        pct = (cnt / total_noise_count) * 100
        print(f"  {pair[0]:10s} <-> {pair[1]:10s}: {cnt:4d} samples ({pct:.2f}% of noise)")

    # Per-class error rates
    print("\nPer-class Noise Rates (Clean class -> % labeled wrong):")
    for c_idx, c_name in enumerate(classes):
        class_samples = train_df[train_df['clean_label'] == c_idx]
        class_noise = (class_samples['noisy_label'] != c_idx).mean()
        print(f"  {c_name:10s}: {class_noise*100:.2f}% noisy ({len(class_samples)} total)")

if __name__ == '__main__':
    analyze()
