"""Training, evaluation, metrics, and visualization utilities."""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def _build_warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def _masked_patch_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute correlation between reconstruction and target on masked patches."""
    mask_bool = mask > 0.5
    if mask_bool.sum().item() == 0:
        return 0.0

    pred_sel = pred[mask_bool].reshape(-1)
    target_sel = target[mask_bool].reshape(-1)
    if pred_sel.numel() < 2:
        return 0.0

    pred_centered = pred_sel - pred_sel.mean()
    target_centered = target_sel - target_sel.mean()
    denom = torch.sqrt((pred_centered.square().sum() * target_centered.square().sum()).clamp(min=1e-12))
    corr = (pred_centered * target_centered).sum() / denom
    return float(corr.detach().cpu().item())


def pretrain(model, train_loader, val_loader, config, device, save_dir):
    """Stage 1 self-supervised pretraining with masked-only reconstruction loss."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 10))
    weight_decay = 0.05
    grad_clip = float(config["training"].get("gradient_clip", 1.0))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = min(total_steps - 1, warmup_epochs * len(train_loader)) if total_steps > 1 else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_warmup_cosine_lambda(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=1e-6 / max(lr, 1e-12),
        ),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "reconstruction_quality": [],
    }

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        train_losses: List[float] = []
        for batch in train_loader:
            x = batch[0].to(device).float()
            target_patches = model.patchify(x)

            optimizer.zero_grad(set_to_none=True)
            pred, mask = model(x)
            loss = model.compute_loss(pred, target_patches, mask)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses: List[float] = []
        val_corrs: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device).float()
                target_patches = model.patchify(x)
                pred, mask = model(x)
                val_loss = model.compute_loss(pred, target_patches, mask)
                val_losses.append(float(val_loss.detach().cpu().item()))
                val_corrs.append(_masked_patch_correlation(pred, target_patches, mask))

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        recon_quality = float(np.mean(val_corrs)) if val_corrs else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["reconstruction_quality"].append(recon_quality)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )
        print(f"  Reconstruction quality (masked corr): {recon_quality:.4f}")

        if epoch % 10 == 0:
            periodic_ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                    "history": history,
                    "config": config,
                },
                periodic_ckpt_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "reconstruction_quality": recon_quality,
                    "history": history,
                    "config": config,
                },
                best_ckpt_path,
            )

    return model, history


def _compute_class_weights_from_loader(train_loader, device: torch.device) -> Optional[torch.Tensor]:
    label_list: List[np.ndarray] = []
    for _, labels in train_loader:
        label_list.append(labels.numpy())

    if not label_list:
        return None

    y = np.concatenate(label_list, axis=0)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = counts.sum()
    weights = np.zeros(n_classes, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = float(total) / float(n_classes * max(1, cnt))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def finetune(model, train_loader, val_loader, config, device, save_dir):
    """Stage 2 supervised fine-tuning with class weighting and early stopping."""
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = config["training"]
    model_cfg = config.get("model", {})

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("patience", 20))
    use_class_weights = bool(model_cfg.get("use_class_weights", True))

    class_weights = _compute_class_weights_from_loader(train_loader, device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_f1": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_val_f1 = -float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        train_preds: List[np.ndarray] = []
        train_labels: List[np.ndarray] = []

        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            x = x.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            train_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_labels.append(y.detach().cpu().numpy())

        y_true_train = np.concatenate(train_labels, axis=0)
        y_pred_train = np.concatenate(train_preds, axis=0)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_acc = float(accuracy_score(y_true_train, y_pred_train)) if len(y_true_train) > 0 else 0.0

        val_results = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_results, class_names=["Normal", "Arrhythmia"])

        val_f1 = float(val_metrics["macro_f1"])
        val_acc = float(val_metrics["accuracy"])
        val_auc = val_metrics["auc_roc"]

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc if val_auc is not None else 'None'}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": best_val_f1,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "history": history,
                    "config": config,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f})")
                break

    return model, history


def evaluate(model, dataloader, device):
    """Run model inference and return predictions, probabilities, and labels."""
    model.eval()
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    return {
        "preds": np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64),
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32),
        "labels": np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64),
    }


def compute_metrics(results, class_names):
    """Compute standard classification metrics from evaluate() outputs."""
    preds = results.get("preds", results.get("predictions"))
    probs = results.get("probs", results.get("probabilities"))
    labels = results.get("labels")

    metrics = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "auc_roc": None,
        "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
        "classification_report": "",
    }

    if labels is None or len(labels) == 0:
        return metrics

    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(labels, preds)
    metrics["classification_report"] = classification_report(
        labels,
        preds,
        target_names=class_names,
        zero_division=0,
    )

    unique_pred = np.unique(preds)
    if probs is not None and len(unique_pred) > 1:
        try:
            if probs.shape[1] == 2:
                metrics["auc_roc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                )
        except Exception:
            metrics["auc_roc"] = None

    return metrics


def visualize_reconstruction(model, signal, device, patch_size, save_path):
    """Plot original, masked input, and reconstructed signal with masked region shading."""
    import matplotlib.pyplot as plt

    model.eval()
    x = torch.tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        pred, mask = model(x)
        target_patches = model.patchify(x)

    pred_patches = pred.squeeze(0).detach().cpu().numpy()
    target_patches = target_patches.squeeze(0).detach().cpu().numpy()
    mask_np = mask.squeeze(0).detach().cpu().numpy().astype(bool)

    original = target_patches.reshape(-1)
    reconstructed = pred_patches.reshape(-1)

    sample_mask = np.repeat(mask_np, patch_size)
    masked_input = original.copy()
    masked_input[sample_mask] = np.nan

    t = np.arange(len(original))
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(t, original, color="royalblue", linewidth=1.0)
    axes[0].set_title("Original ECG Signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, masked_input, color="gray", linewidth=1.0)
    axes[1].set_title("Masked Input (visible in gray, masked regions are gaps)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t, original, color="royalblue", linewidth=1.0, alpha=0.7, label="Original")
    axes[2].plot(t, reconstructed, color="crimson", linewidth=1.0, alpha=0.9, label="Reconstructed")
    axes[2].fill_between(
        t,
        np.minimum(original, reconstructed),
        np.maximum(original, reconstructed),
        where=sample_mask,
        color="limegreen",
        alpha=0.18,
        label="Masked regions",
    )
    axes[2].set_title("Reconstruction Overlay (green shading shows masked regions)")
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_few_shot_experiment(
    mae_model,
    segments,
    labels,
    n_shots,
    seeds,
    config,
    device,
    save_dir,
):
    """Run repeated few-shot experiments and report mean+-std metrics."""
    from .data_loader import create_finetune_dataloaders
    from .models import ECGClassifier
    from .utils import set_seed

    os.makedirs(save_dir, exist_ok=True)
    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for n_shot in n_shots:
        shot_metrics = {"accuracy": [], "macro_f1": [], "weighted_f1": [], "auc_roc": []}

        for seed in seeds:
            set_seed(seed)
            run_dir = os.path.join(save_dir, f"{n_shot}shot_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            loaders = create_finetune_dataloaders(
                segments,
                labels,
                n_shot=n_shot,
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                seed=seed,
            )

            clf = ECGClassifier.from_pretrained(
                mae_model,
                num_classes=config["model"]["num_classes"],
                freeze_layers=config["model"].get("freeze_layers", 0),
                dropout=config["model"].get("dropout", 0.5),
            ).to(device)

            clf, _ = finetune(
                clf,
                loaders["train"],
                loaders["val"],
                config,
                device,
                run_dir,
            )

            best_ckpt = torch.load(
                os.path.join(run_dir, "best_model.pt"),
                map_location=device,
                weights_only=False,
            )
            clf.load_state_dict(best_ckpt["model_state_dict"])

            eval_results = evaluate(clf, loaders["test"], device)
            m = compute_metrics(eval_results, class_names=["Normal", "Arrhythmia"])
            for key in shot_metrics:
                if m[key] is not None:
                    shot_metrics[key].append(float(m[key]))

        results[n_shot] = {}
        for key, values in shot_metrics.items():
            if len(values) == 0:
                results[n_shot][key] = {"mean": float("nan"), "std": float("nan")}
            else:
                results[n_shot][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

    return results


def plot_training_history(history, title="Training History", save_path=None):
    """Plot training curves from a history dictionary."""
    import matplotlib.pyplot as plt

    keys = [k for k, vals in history.items() if len(vals) > 0 and any(v is not None for v in vals)]
    n = len(keys)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        vals = history[key]
        xs = np.arange(1, len(vals) + 1)
        plot_vals = [np.nan if v is None else v for v in vals]
        ax.plot(xs, plot_vals, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig