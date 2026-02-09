"""
Model architectures for ECG Self-Supervised Learning.

Contains:
  - PatchEmbedding: raw 1D signal → patch embeddings
  - TransformerBlock: single Pre-LN encoder layer
  - ECGTransformerEncoder: stack of N blocks
  - ECGMaskedAutoencoder: Stage 1 (self-supervised)
  - ECGClassifier: Stage 2 (fine-tuning)
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# Patch Embedding
# ============================================================================


class PatchEmbedding(nn.Module):
    """Convert raw 1D ECG signal into a sequence of patch embeddings."""

    def __init__(self, patch_size=50, in_channels=1, embed_dim=128, max_patches=200):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """[B, L] → [B, N, D] where N = L // patch_size."""
        B, L = x.shape
        trim_len = L - (L % self.patch_size)
        x = x[:, :trim_len]
        x = x.reshape(B, -1, self.patch_size)  # [B, N, patch_size]
        N = x.size(1)
        x = self.proj(x)  # [B, N, D]
        x = x + self.pos_embed[:, :N, :]  # add positional encoding
        return x


# ============================================================================
# Transformer Blocks
# ============================================================================


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block (more stable training)."""

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """[B, N, D] → [B, N, D]"""
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ECGTransformerEncoder(nn.Module):
    """Stack of Transformer blocks + final LayerNorm."""

    def __init__(self, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================================
# Masking Strategy
# ============================================================================


def random_masking(x, mask_ratio=0.75):
    """
    Random masking by per-sample shuffling (MAE-style).

    Args:
        x: [B, N, D] embedded patches
        mask_ratio: fraction to mask (0.75 = 75%)

    Returns:
        x_masked: [B, N_keep, D] visible patches
        mask: [B, N] binary (0=keep, 1=masked)
        ids_restore: [B, N] unshuffle indices
    """
    B, N, D = x.shape
    N_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :N_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :N_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


# ============================================================================
# Reconstruction Decoder (lightweight, for pre-training)
# ============================================================================


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder that reconstructs masked patches."""

    def __init__(
        self,
        embed_dim=128,
        patch_size=50,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.reconstruction_head = nn.Linear(embed_dim, patch_size)

    def forward(self, x, ids_restore):
        """
        x: [B, N_keep, D] encoded visible patches
        ids_restore: [B, N] unshuffle indices
        Returns: [B, N, patch_size] reconstructed patches
        """
        B, N_keep, D = x.shape
        N = ids_restore.size(1)

        mask_tokens = self.mask_token.expand(B, N - N_keep, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)
        return self.reconstruction_head(x_full)


# ============================================================================
# Classification Head (for fine-tuning)
# ============================================================================


class ClassificationHead(nn.Module):
    """Global-average-pooling + MLP classifier."""

    def __init__(self, embed_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        """[B, N, D] → [B, num_classes]"""
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.pool(x).squeeze(-1)  # [B, D]
        return self.classifier(x)


# ============================================================================
# Complete Model: Stage 1 — Masked Autoencoder
# ============================================================================


class ECGMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for self-supervised ECG pre-training.

    Flow: Signal → Patches → Mask 75% → Encode visible → Decode all → Reconstruct
    Loss: MSE on masked positions only.
    """

    def __init__(
        self,
        patch_size=50,
        embed_dim=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        decoder_depth=2,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbedding(patch_size, embed_dim=embed_dim)
        self.encoder = ECGTransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio, dropout
        )
        self.decoder = ReconstructionDecoder(
            embed_dim, patch_size, decoder_depth, num_heads, mlp_ratio, dropout
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [B, L] raw ECG signal
        Returns: (pred [B, N, patch_size], mask [B, N])
        """
        x_embed = self.patch_embed(x)
        x_masked, mask, ids_restore = random_masking(x_embed, self.mask_ratio)
        latent = self.encoder(x_masked)
        pred = self.decoder(latent, ids_restore)
        return pred, mask

    def get_encoder_for_finetuning(self):
        """Return (patch_embed, encoder) for downstream use."""
        return self.patch_embed, self.encoder


# ============================================================================
# Complete Model: Stage 2 — Fine-tuning Classifier
# ============================================================================


class ECGClassifier(nn.Module):
    """
    Classifier using pre-trained encoder for few-shot fine-tuning.

    Strategy: Freeze first `freeze_layers` blocks, train the rest + classifier.
    """

    def __init__(
        self, patch_embed, encoder, num_classes=2, freeze_layers=4, dropout=0.5
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        embed_dim = encoder.norm.normalized_shape[0]
        self.classifier = ClassificationHead(embed_dim, num_classes, dropout)

        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze first N encoder blocks
        for i, block in enumerate(self.encoder.blocks):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

        total = len(self.encoder.blocks)
        print(
            f"Encoder: {total} blocks — {freeze_layers} frozen, "
            f"{total - freeze_layers} trainable + classifier head"
        )

    def forward(self, x):
        """[B, L] → [B, num_classes]"""
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.classifier(x)

    @classmethod
    def from_pretrained(cls, mae_model, num_classes=2, freeze_layers=4, dropout=0.5):
        """Create classifier from a trained ECGMaskedAutoencoder."""
        patch_embed, encoder = mae_model.get_encoder_for_finetuning()
        return cls(patch_embed, encoder, num_classes, freeze_layers, dropout)
