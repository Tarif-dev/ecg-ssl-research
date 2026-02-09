"""
Training loops, loss functions, evaluation, and visualization.

Stage 1: Self-supervised pre-training (reconstruction)
Stage 2: Supervised fine-tuning (classification)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm

# Optional: Weights & Biases
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# Loss Functions
# ============================================================================


def reconstruction_loss(pred, target, mask):
    """
    MSE loss on masked regions only.

    pred: [B, N, patch_size], target: [B, N, patch_size], mask: [B, N] (1=masked)
    """
    loss = ((pred - target) ** 2).mean(dim=-1)  # [B, N]
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)  # avg over masked
    return loss


# ============================================================================
# LR Scheduler
# ============================================================================


def get_cosine_schedule_with_warmup(
    optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01
):
    """Cosine decay with linear warmup."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Stage 1: Pre-training
# ============================================================================


def pretrain_one_epoch(
    model, dataloader, optimizer, device, patch_size, gradient_clip=1.0
):
    """One epoch of self-supervised pre-training. Returns avg loss."""
    model.train()
    total_loss, n = 0, 0
    for batch in tqdm(dataloader, desc="  Pre-train", leave=False):
        signals = batch[0].to(device)
        B, L = signals.shape
        trim_len = L - (L % patch_size)
        target = signals[:, :trim_len].reshape(B, -1, patch_size)

        pred, mask = model(signals)
        loss = reconstruction_loss(pred, target, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


def validate_pretrain(model, dataloader, device, patch_size):
    """Validation reconstruction loss."""
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            signals = batch[0].to(device)
            B, L = signals.shape
            trim_len = L - (L % patch_size)
            target = signals[:, :trim_len].reshape(B, -1, patch_size)
            pred, mask = model(signals)
            total_loss += reconstruction_loss(pred, target, mask).item()
            n += 1
    return total_loss / max(1, n)


def pretrain(model, train_loader, val_loader, config, device, save_dir):
    """
    Full pre-training loop with logging, scheduling, and checkpointing.

    Returns: (model, history_dict)
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config["training"]["warmup_epochs"],
        total_epochs=config["training"]["epochs"],
    )

    patch_size = config["model"]["patch_size"]
    grad_clip = config["training"]["gradient_clip"]
    epochs = config["training"]["epochs"]
    save_interval = config["logging"]["save_interval"]

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"  STAGE 1: Self-Supervised Pre-training ({epochs} epochs)")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = pretrain_one_epoch(
            model, train_loader, optimizer, device, patch_size, grad_clip
        )
        val_loss = (
            validate_pretrain(model, val_loader, device, patch_size)
            if val_loader
            else None
        )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        msg = (
            f"Epoch {epoch+1:3d}/{epochs} │ "
            f"Train Loss: {train_loss:.4f} │ LR: {lr:.2e} │ "
            f"Time: {time.time()-t0:.1f}s"
        )
        if val_loss is not None:
            msg += f" │ Val Loss: {val_loss:.4f}"
        print(msg)

        if HAS_WANDB and wandb.run is not None:
            log = {
                "pretrain/train_loss": train_loss,
                "pretrain/lr": lr,
                "pretrain/epoch": epoch + 1,
            }
            if val_loss is not None:
                log["pretrain/val_loss"] = val_loss
            wandb.log(log)

        # Periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(save_dir, f"ckpt_ep{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                path,
            )
            print(f"  → Checkpoint: {path}")

        # Best model
        check = val_loss if val_loss is not None else train_loss
        if check < best_loss:
            best_loss = check
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                os.path.join(save_dir, "best_model.pt"),
            )

    print(f"\n✓ Pre-training done! Best loss: {best_loss:.4f}")
    print(f"  Best model: {os.path.join(save_dir, 'best_model.pt')}")
    return model, history


# ============================================================================
# Stage 2: Fine-tuning
# ============================================================================


def finetune_one_epoch(model, dataloader, optimizer, criterion, device):
    """One epoch of supervised training. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, labels in tqdm(dataloader, desc="  Fine-tune", leave=False):
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (torch.argmax(logits, 1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), correct / max(1, total)


def evaluate(model, dataloader, device):
    """Run inference, return predictions / labels / probabilities."""
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for signals, lbls in dataloader:
            logits = model(signals.to(device))
            probs.extend(torch.softmax(logits, 1).cpu().numpy())
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(lbls.numpy())
    return {
        "predictions": np.array(preds),
        "labels": np.array(labels),
        "probabilities": np.array(probs),
    }


def compute_metrics(results, class_names=None):
    """Compute accuracy, F1, AUC-ROC, confusion matrix from evaluate() output."""
    preds = results["predictions"]
    labels = results["labels"]
    probs = results["probabilities"]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(probs.shape[1])]

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    try:
        auc = (
            roc_auc_score(labels, probs[:, 1])
            if probs.shape[1] == 2
            else roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = None

    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def finetune(model, train_loader, val_loader, config, device, save_dir):
    """
    Full fine-tuning loop with early stopping.

    Returns: (model, history_dict)
    """
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"].get("label_smoothing", 0.1)
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=1e-7
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    history = {"train_loss": [], "train_acc": [], "val_f1": [], "val_acc": []}
    best_f1, wait = 0, 0

    print(f"\n{'='*60}")
    print(f"  STAGE 2: Fine-tuning ({epochs} epochs, patience={patience})")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        train_loss, train_acc = finetune_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_res = evaluate(model, val_loader, device)
        val_m = compute_metrics(val_res)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_f1"].append(val_m["macro_f1"])
        history["val_acc"].append(val_m["accuracy"])

        print(
            f"Epoch {epoch+1:3d}/{epochs} │ "
            f"Loss: {train_loss:.4f} │ Acc: {train_acc:.3f} │ "
            f"Val F1: {val_m['macro_f1']:.3f} │ Val Acc: {val_m['accuracy']:.3f}"
        )

        if HAS_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "ft/train_loss": train_loss,
                    "ft/train_acc": train_acc,
                    "ft/val_f1": val_m["macro_f1"],
                    "ft/val_acc": val_m["accuracy"],
                    "ft/epoch": epoch + 1,
                }
            )

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            wait = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                    "config": config,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  → New best! F1={best_f1:.3f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

    print(f"\n✓ Fine-tuning done! Best Val F1: {best_f1:.3f}")
    return model, history


# ============================================================================
# Few-Shot Evaluation Protocol
# ============================================================================


def run_few_shot_experiment(
    mae_model, segments, labels, n_shots, seeds, config, device, save_dir
):
    """
    Run few-shot evaluation: for each (n_shot, seed), fine-tune & test.

    Returns: dict  n_shot → {metric → {mean, std}}
    """
    from .models import ECGClassifier
    from .data_loader import create_finetune_dataloaders
    from .utils import set_seed

    all_results = {}

    for n_shot in n_shots:
        print(f"\n{'='*60}")
        print(f"  {n_shot}-shot evaluation")
        print(f"{'='*60}")
        shot_metrics = []

        for seed in seeds:
            print(f"\n--- seed={seed} ---")
            set_seed(seed)

            loaders = create_finetune_dataloaders(
                segments,
                labels,
                n_shot=n_shot,
                batch_size=config["training"]["batch_size"],
                seed=seed,
            )
            clf = ECGClassifier.from_pretrained(
                mae_model,
                num_classes=config["model"]["num_classes"],
                freeze_layers=config["model"]["freeze_layers"],
                dropout=config["model"]["dropout"],
            ).to(device)

            run_dir = os.path.join(save_dir, f"{n_shot}shot_s{seed}")
            clf, _ = finetune(
                clf, loaders["train"], loaders["val"], config, device, run_dir
            )

            best = torch.load(
                os.path.join(run_dir, "best_model.pt"),
                map_location=device,
                weights_only=False,
            )
            clf.load_state_dict(best["model_state_dict"])

            res = evaluate(clf, loaders["test"], device)
            m = compute_metrics(res, ["Normal", "Arrhythmia"])
            print(
                f"  Test — Acc: {m['accuracy']:.3f}, "
                f"F1: {m['macro_f1']:.3f}, AUC: {m['auc_roc']}"
            )
            shot_metrics.append(m)

        all_results[n_shot] = {}
        for key in ("accuracy", "macro_f1", "auc_roc"):
            vals = [m[key] for m in shot_metrics if m[key] is not None]
            all_results[n_shot][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        print(f"\n{n_shot}-shot summary:")
        for k, v in all_results[n_shot].items():
            print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    return all_results


# ============================================================================
# Visualization Helpers
# ============================================================================


def visualize_reconstruction(model, signal, device, patch_size, save_path=None):
    """Plot original vs masked vs reconstructed signal."""
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        pred, mask = model(x)
        pred_sig = pred.squeeze(0).cpu().numpy().reshape(-1)
        mask_np = mask.squeeze(0).cpu().numpy()

    trim_len = len(signal) - (len(signal) % patch_size)
    orig = signal[:trim_len]
    mask_samples = np.repeat(mask_np, patch_size)[: len(orig)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(orig, color="steelblue", lw=0.5)
    axes[0].set_title("Original Signal")

    masked_disp = orig.copy()
    masked_disp[mask_samples == 1] = np.nan
    axes[1].plot(masked_disp, color="green", lw=0.5)
    axes[1].set_title(f"Masked Input ({int(mask_np.mean()*100)}% masked)")

    axes[2].plot(orig, color="steelblue", lw=0.5, alpha=0.3, label="Original")
    axes[2].plot(pred_sig[: len(orig)], color="red", lw=0.5, label="Reconstructed")
    axes[2].set_title("Reconstruction vs Original")
    axes[2].legend()
    axes[2].set_xlabel("Sample")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_history(history, title="Training History", save_path=None):
    """Plot training curves from history dict."""
    import matplotlib.pyplot as plt

    keys = [k for k, v in history.items() if any(x is not None for x in v)]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        valid = [(i, v) for i, v in enumerate(history[key]) if v is not None]
        if valid:
            ep, vals = zip(*valid)
            ax.plot(ep, vals, marker=".", markersize=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
