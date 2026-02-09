# ECG Self-Supervised Learning Research

**Self-Supervised Transformer-Based ECG Analysis for Few-Shot Arrhythmia Detection**

## Overview

This project implements a **Masked Autoencoder (MAE)** for ECG signals using a Transformer architecture. The key idea:

1. **Stage 1 — Pre-train** on 100,000+ _unlabeled_ ECG segments (reconstruct masked patches)
2. **Stage 2 — Fine-tune** with only 10–100 _labeled_ examples for arrhythmia detection
3. **Goal:** >90% accuracy with ≤100 labels (vs <70% training from scratch)

## Quick Start (Google Colab)

### Step 1: Open notebooks in Colab

Upload or clone this repo, then open the notebooks in order:

| Notebook                    | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| `01_data_exploration.ipynb` | Download MIT-BIH data, explore & visualize      |
| `02_preprocessing.ipynb`    | Filter, normalize, segment, save processed data |
| `03_pretraining.ipynb`      | Self-supervised pre-training (Stage 1)          |
| `04_finetuning.ipynb`       | Few-shot fine-tuning & evaluation (Stage 2)     |

### Step 2: Run cells top-to-bottom

Each notebook is self-contained — just run every cell in order.

## Project Structure

```
ecg-ssl-research/
├── configs/
│   ├── pretrain_config.yaml    # Stage 1 hyperparameters
│   └── finetune_config.yaml    # Stage 2 hyperparameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pretraining.ipynb
│   └── 04_finetuning.ipynb
├── src/
│   ├── __init__.py
│   ├── download_data.py        # Download MIT-BIH / PTB-XL
│   ├── data_loader.py          # Preprocessing, Datasets, DataLoaders
│   ├── models.py               # PatchEmbed, Transformer, MAE, Classifier
│   ├── training.py             # Training loops, evaluation, visualization
│   └── utils.py                # Seed, device, config, checkpoints
├── requirements.txt
└── .gitignore
```

## Datasets

- **MIT-BIH Arrhythmia Database** — 48 half-hour ECG recordings at 360 Hz (~115 MB)
- **PTB-XL** (optional) — 21,837 12-lead ECGs at 500 Hz (~22 GB)

## Technical Stack

- **PyTorch 2.0+** — model & training
- **WFDB** — ECG data loading from PhysioNet
- **scikit-learn** — evaluation metrics
- **W&B** (optional) — experiment tracking
- **Google Colab** — GPU training (T4/V100)

## Setup (Local)

```bash
pip install -r requirements.txt
```

## License

This project is for academic research purposes.
