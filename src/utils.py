"""General utilities: seed management, config loading, device detection, checkpointing."""

import os
import sys
import random
import yaml
import numpy as np
import torch


def set_seed(seed=42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """Detect and return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✓ Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU available — using CPU (training will be slow)")
    return device


def load_config(config_path):
    """Load a YAML configuration file and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
        },
        path,
    )
    print(f"  → Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load a training checkpoint from disk."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"✓ Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["loss"]


def count_parameters(model):
    """Count and print trainable vs total model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total parameters:     {total:>10,}")
    print(f"  Trainable parameters: {trainable:>10,}")
    print(f"  Frozen parameters:    {frozen:>10,}")
    return total, trainable


def get_env_info():
    """Collect environment info for reproducibility logging."""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "numpy_version": np.__version__,
    }
    return info
