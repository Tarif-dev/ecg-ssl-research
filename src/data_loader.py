"""Data preprocessing, PyTorch Datasets, DataLoaders, and few-shot sampling."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal


# ============================================================================
# Preprocessing Functions
# ============================================================================


def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=5):
    """
    Butterworth bandpass filter to remove noise from ECG.
    - 0.5 Hz: removes baseline wander (breathing, electrode drift)
    - 40 Hz: removes muscle noise & powerline interference
    - filtfilt: zero-phase (no distortion)
    """
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 1.0 - 1e-5)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    filtered = scipy_signal.filtfilt(b, a, sig)
    return filtered.astype(np.float32)


def zscore_normalize(sig, eps=1e-8):
    """Per-signal z-score normalization (zero mean, unit variance)."""
    mean = np.mean(sig)
    std = np.std(sig)
    return ((sig - mean) / (std + eps)).astype(np.float32)


def preprocess_signal(sig, fs, lowcut=0.5, highcut=40.0, order=5):
    """Full preprocessing: bandpass filter → z-score normalization."""
    filtered = bandpass_filter(sig, fs, lowcut, highcut, order)
    normalized = zscore_normalize(filtered)
    return normalized


def patchify(segment, patch_size=50):
    """Divide 1D segment into non-overlapping patches. Returns [N, patch_size]."""
    trim_len = len(segment) - (len(segment) % patch_size)
    return segment[:trim_len].reshape(-1, patch_size)


def patchify_batch(signals, patch_size=50):
    """Patchify a batch: [B, L] → [B, N, patch_size]."""
    B, L = signals.shape
    trim_len = L - (L % patch_size)
    return signals[:, :trim_len].reshape(B, -1, patch_size)


# ============================================================================
# Data Augmentation (for pre-training)
# ============================================================================


class ECGAugmentation:
    """Random augmentations for ECG signals during self-supervised pre-training."""

    def __init__(self, scale_range=(0.9, 1.1), noise_std=0.01, time_shift_ratio=0.1):
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.time_shift_ratio = time_shift_ratio

    def __call__(self, signal):
        # Random amplitude scaling
        scale = np.random.uniform(*self.scale_range)
        signal = signal * scale
        # Random circular time shift
        max_shift = int(len(signal) * self.time_shift_ratio)
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift)
            signal = np.roll(signal, shift)
        # Additive Gaussian noise
        noise = np.random.normal(0, self.noise_std, signal.shape)
        signal = signal + noise
        return signal.astype(np.float32)


# ============================================================================
# PyTorch Datasets
# ============================================================================


class ECGPretrainDataset(Dataset):
    """Dataset for self-supervised pre-training (no labels). Returns (signal,)."""

    def __init__(self, segments, transform=None):
        self.segments = segments.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        if self.transform is not None:
            segment = self.transform(segment)
        return (torch.tensor(segment, dtype=torch.float32),)


class ECGFinetuneDataset(Dataset):
    """Dataset for supervised fine-tuning. Returns (signal, label)."""

    def __init__(self, segments, labels, transform=None):
        assert len(segments) == len(labels)
        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        label = self.labels[idx]
        if self.transform is not None:
            segment = self.transform(segment)
        return (
            torch.tensor(segment, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ============================================================================
# Few-Shot Sampling
# ============================================================================


def create_few_shot_split(labels, n_per_class, seed=42):
    """
    Stratified sampling: select n_per_class examples from each class.

    Returns: np.ndarray of selected indices.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    indices = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_per_class:
            print(
                f"  ⚠ Class {cls}: only {len(cls_idx)} available "
                f"(requested {n_per_class})"
            )
            selected = cls_idx
        else:
            selected = rng.choice(cls_idx, size=n_per_class, replace=False)
        indices.extend(selected)
    return np.array(indices)


def create_data_splits(labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Stratified train/val/test split.

    Returns: dict with 'train', 'val', 'test' index arrays.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val :])

    return {
        "train": np.array(train_idx),
        "val": np.array(val_idx),
        "test": np.array(test_idx),
    }


# ============================================================================
# DataLoader Factory
# ============================================================================


def create_pretrain_dataloader(segments, batch_size=256, num_workers=2, augment=True):
    """Create DataLoader for self-supervised pre-training."""
    transform = ECGAugmentation() if augment else None
    dataset = ECGPretrainDataset(segments, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(
        f"✓ Pre-train DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, {len(loader)} batches"
    )
    return loader


def create_finetune_dataloaders(
    segments, labels, n_shot=None, batch_size=32, num_workers=2, seed=42
):
    """
    Create train/val/test DataLoaders for fine-tuning.

    Args:
        segments: np.ndarray [N, L]
        labels: np.ndarray [N]
        n_shot: int or None — if set, subsample training to n_shot per class
        batch_size: int
        num_workers: int
        seed: int

    Returns: dict with 'train', 'val', 'test' DataLoaders
    """
    splits = create_data_splits(labels, seed=seed)

    if n_shot is not None:
        train_labels = labels[splits["train"]]
        few_shot_idx = create_few_shot_split(train_labels, n_shot, seed)
        train_indices = splits["train"][few_shot_idx]
    else:
        train_indices = splits["train"]

    train_ds = ECGFinetuneDataset(segments[train_indices], labels[train_indices])
    val_ds = ECGFinetuneDataset(segments[splits["val"]], labels[splits["val"]])
    test_ds = ECGFinetuneDataset(segments[splits["test"]], labels[splits["test"]])

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    print(
        f"✓ Fine-tune splits — Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return loaders
