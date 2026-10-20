# CIFAR-10N CV Bootcamp: How to reproduce

Run all commands from the submission root, which is the folder containing `code/`, `model.py`, and `data/`.
The data files go in `data/`: `train_images.npy`, `test_images.npy`, `labels.csv`, `train_idx.txt`, `val_idx.txt`, and `classes.txt`.
Scripts write test predictions to `predictions/`. They write checkpoints and summaries to `checkpoints/`.

## Environment

- Python 3.10, PyTorch 2.6 (CUDA 12.4), torchvision 0.21, numpy, pandas, scikit-learn
- GPU: NVIDIA RTX 4060 Laptop (8 GB). Each 60-epoch ResNet-18 run takes about 50 minutes.

## Common training setup (used by every model)

- `model.py` ResNet-18, trained from scratch. Seed 42.
- SGD with momentum 0.9 and weight decay 5e-4. LR 0.1 with cosine annealing to 1e-4 over 60 epochs. Batch size 128.
- Augmentation: random crop 32 with padding 4, then a random horizontal flip.
- Model selection uses noisy validation labels for Tasks 1â€“4 and synthetic validation labels for Task 5.

## Commands

```bash
# Task 1 noise statistics and 10x10 confusion matrix (clean vs noisy)
python code/analyze_noise.py

# Task 1 + Task 2: clean reference model, noisy CE model, per-epoch dynamics, early stop
#   -> predictions/t1_clean_ref.csv, t1_noisy_ce.csv, t2_dynamics.csv, t2_early_stop.csv
#   -> checkpoints/noisy_epoch_{1..60}.pth (reused by Task 3)
python code/train_tasks1_and_2.py

# Task 3: early-learning ensemble noise scores (epochs 25,30,35,38,41 of the noisy CE run)
#   -> predictions/t3_noise_scores.csv, checkpoints/t3_noise_scores.npy
python code/task3_generate_scores.py

# Task 4: robust training (SCE and Task-3-based confident relabeling)
#   -> predictions/t4_sce.csv, predictions/t4_relabel.csv, checkpoints/t4_summary.json
python code/task4_robust_training.py --methods sce relabel

# Task 5: CE baseline and CSR (best Task 4 method) on synthetic_label
#   -> predictions/t5_baseline_synth.csv, predictions/t5_best_synth.csv, checkpoints/t5_summary.json
python code/task5_synthetic_noise.py

# Format check
python check_format.py .
```

## Notes

- Clean labels are used only to train the Task 1 reference model and to measure the following: the confusion matrix, the Task 2 curves, and the Task 3 AUROC/precision@k self-check. They are never used for training, filtering, relabeling, or model selection in Tasks 2â€“5.
- In Task 5, CSR is rebuilt entirely from synthetic labels. Its noise scores come from epochs 25/30/35/38/41 of the synthetic CE baseline run, with the same thresholds as Task 4: relabel if score > 0.65 and ensemble confidence > 0.70. Then a fresh CE model is retrained on the corrected labels.
