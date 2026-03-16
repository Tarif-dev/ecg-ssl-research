"""Data preprocessing, datasets, and dataloader factories for ECG SSL pipeline."""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=5):
    """Butterworth bandpass filter for ECG denoising."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 1.0 - 1e-5)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    filtered = scipy_signal.filtfilt(b, a, sig)
    return filtered.astype(np.float32)


def zscore_normalize(sig, eps=1e-8):
    """Per-sample z-score normalization."""
    mean = np.mean(sig)
    std = np.std(sig)
    return ((sig - mean) / (std + eps)).astype(np.float32)


def preprocess_signal(sig, fs, lowcut=0.5, highcut=40.0, order=5):
    """Bandpass then z-score normalize."""
    return zscore_normalize(bandpass_filter(sig, fs, lowcut=lowcut, highcut=highcut, order=order))


def patchify(segment, patch_size=50):
    """Patchify a 1D segment into [N, patch_size]."""
    trim_len = len(segment) - (len(segment) % patch_size)
    return segment[:trim_len].reshape(-1, patch_size)


def patchify_batch(signals, patch_size=50):
    """Patchify a batch [B, L] into [B, N, patch_size]."""
    bsz, length = signals.shape
    trim_len = length - (length % patch_size)
    return signals[:, :trim_len].reshape(bsz, -1, patch_size)


class ECGAugmentation:
    """Pretraining augmentations for unlabeled segments."""

    def __init__(self, scale_range=(0.8, 1.2), noise_std=0.01, time_shift_ratio=0.1, sign_flip_prob=0.1):
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.time_shift_ratio = time_shift_ratio
        self.sign_flip_prob = sign_flip_prob

    def __call__(self, signal):
        x = signal.astype(np.float32, copy=True)

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        x = x * scale

        max_shift = int(len(x) * self.time_shift_ratio)
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift + 1)
            x = np.roll(x, shift)

        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=x.shape).astype(np.float32)
        x = x + noise

        if np.random.rand() < self.sign_flip_prob:
            x = -x

        return x.astype(np.float32)


class ECGAmplitudeOnlyAugmentation:
    """Light augmentation for supervised fine-tuning training split."""

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, signal):
        x = signal.astype(np.float32, copy=True)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return (x * scale).astype(np.float32)


class ECGPretrainDataset(Dataset):
    """Dataset for self-supervised pretraining (returns signal only)."""

    def __init__(self, segments, transform=None):
        self.segments = segments.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].astype(np.float32, copy=True)
        if self.transform is not None:
            segment = self.transform(segment)
        return (torch.tensor(segment, dtype=torch.float32),)


class ECGFinetuneDataset(Dataset):
    """Dataset for supervised finetuning (returns signal and label)."""

    def __init__(self, segments, labels, transform=None):
        if len(segments) != len(labels):
            raise ValueError("segments and labels must have the same length")
        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].astype(np.float32, copy=True)
        if self.transform is not None:
            segment = self.transform(segment)
        label = int(self.labels[idx])
        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def create_few_shot_split(labels, n_per_class, seed=42):
    """Return indices with exactly n_per_class examples sampled per class when available."""
    rng = np.random.RandomState(seed)
    selected_indices = []
    classes = np.unique(labels)

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} examples, requested n_per_class={n_per_class}."
            )
        picked = rng.choice(cls_idx, size=n_per_class, replace=False)
        selected_indices.append(picked)

    return np.concatenate(selected_indices)


def create_data_splits(labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Backward-compatible stratified split utility."""
    indices = np.arange(len(labels))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    remaining = 1.0 - train_ratio
    val_size_in_temp = val_ratio / max(remaining, 1e-8)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        train_size=val_size_in_temp,
        stratify=y_temp,
        random_state=seed,
    )

    return {
        "train": np.array(train_idx),
        "val": np.array(val_idx),
        "test": np.array(test_idx),
    }


def _pad_beats_to_length(segments: np.ndarray, target_len: int = 400) -> np.ndarray:
    """Pad/truncate beat windows to a fixed length for consistent patching."""
    if segments.ndim != 2:
        raise ValueError(f"Expected 2D array [N, L], got shape {segments.shape}")

    n, length = segments.shape
    if length == target_len:
        return segments.astype(np.float32)

    if length > target_len:
        return segments[:, :target_len].astype(np.float32)

    padded = np.zeros((n, target_len), dtype=np.float32)
    start = (target_len - length) // 2
    padded[:, start : start + length] = segments.astype(np.float32)
    return padded


def create_pretrain_dataloader(segments, batch_size=256, num_workers=2, augment=True):
    """Create pretraining DataLoader with optional augmentation."""
    transform = ECGAugmentation() if augment else None
    dataset = ECGPretrainDataset(segments=segments, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(augment),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    mode = "train" if augment else "val"
    print(
        f"{mode.capitalize()} pretrain loader | samples={len(dataset):,} | "
        f"batch_size={batch_size} | batches={len(loader):,} | shuffle={bool(augment)}"
    )
    return loader


def create_finetune_dataloaders(
    segments,
    labels,
    n_shot=None,
    batch_size=32,
    num_workers=2,
    seed=42,
):
    """Create stratified few-shot train/val/test dataloaders for fine-tuning.

    Strategy:
      1) Choose exactly n_shot samples per class for train.
      2) On remaining samples, use 10% stratified for validation.
      3) Use the rest for test.
      4) Pad beat inputs to length 400 so patch_size=100 yields 4 patches.
    """
    rng = np.random.RandomState(seed)
    segments = np.asarray(segments)
    labels = np.asarray(labels)

    if segments.ndim != 2:
        raise ValueError(f"Expected segments shape [N, L], got {segments.shape}")

    padded_segments = _pad_beats_to_length(segments, target_len=400)

    unique_classes = np.unique(labels)
    if n_shot is None:
        raise ValueError("n_shot must be provided for create_finetune_dataloaders")

    train_indices_per_class = []
    remaining_indices = []

    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_shot:
            raise ValueError(
                f"Class {cls} has {len(cls_idx)} samples but n_shot={n_shot} was requested"
            )

        picked = rng.choice(cls_idx, size=n_shot, replace=False)
        train_indices_per_class.append(picked)

        cls_remaining = np.setdiff1d(cls_idx, picked, assume_unique=False)
        remaining_indices.append(cls_remaining)

    train_idx = np.concatenate(train_indices_per_class)
    rem_idx = np.concatenate(remaining_indices)
    rem_labels = labels[rem_idx]

    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=0.10,
        stratify=rem_labels,
        random_state=seed,
    )

    train_transform = ECGAmplitudeOnlyAugmentation(scale_range=(0.9, 1.1))
    train_ds = ECGFinetuneDataset(padded_segments[train_idx], labels[train_idx], transform=train_transform)
    val_ds = ECGFinetuneDataset(padded_segments[val_idx], labels[val_idx], transform=None)
    test_ds = ECGFinetuneDataset(padded_segments[test_idx], labels[test_idx], transform=None)

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

    def _counts(idxs):
        vals, cnts = np.unique(labels[idxs], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    print("Fine-tune split summary")
    print(f"  Train: {len(train_ds):,} samples | class counts: {_counts(train_idx)}")
    print(f"  Val:   {len(val_ds):,} samples | class counts: {_counts(val_idx)}")
    print(f"  Test:  {len(test_ds):,} samples | class counts: {_counts(test_idx)}")
    print("  Beat length padded to 400 samples for patch_size=100 compatibility")

    return loaders