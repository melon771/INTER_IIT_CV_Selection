import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, eval_transform

class CIFAR10IndexedDataset(Dataset):
    """
    Dataset wrapper for CIFAR-10 training/validation images.
    Returns (image_tensor, label, original_index).
    """
    def __init__(self, images, labels, indices, transform=None):
        self.images = images  # shape: (N, 32, 32, 3), uint8
        self.labels = np.array(labels, dtype=np.int64)
        self.indices = np.array(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        label = self.labels[idx]
        orig_idx = self.indices[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, orig_idx

class CIFAR10TestDataset(Dataset):
    """
    Dataset wrapper for test images.
    Returns (image_tensor, id).
    """
    def __init__(self, images, transform=None):
        self.images = images  # shape: (10000, 32, 32, 3), uint8
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, idx

def load_cifar_data(data_dir='data'):
    train_images = np.load(os.path.join(data_dir, 'train_images.npy'))
    test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))

    with open(os.path.join(data_dir, 'train_idx.txt')) as f:
        train_indices = [int(line.strip()) for line in f if line.strip()]

    with open(os.path.join(data_dir, 'val_idx.txt')) as f:
        val_indices = [int(line.strip()) for line in f if line.strip()]

    with open(os.path.join(data_dir, 'classes.txt')) as f:
        classes = [line.strip() for line in f if line.strip()]

    return {
        'train_images': train_images,
        'test_images': test_images,
        'labels_df': labels_df,
        'train_indices': np.array(train_indices),
        'val_indices': np.array(val_indices),
        'classes': classes,
    }
