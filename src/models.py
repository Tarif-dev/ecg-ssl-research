"""Model architectures for ECG self-supervised learning and fine-tuning."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbedding1D(nn.Module):
    """Split 1D signals into non-overlapping patches and project to embeddings."""

    def __init__(self, patch_size: int = 100, embed_dim: int = 256, max_patches: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_patches = max_patches

        self.proj = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, seq_len] into [B, num_patches, embed_dim]."""
        if x.dim() != 2:
            raise ValueError(f"Expected input [B, L], got shape {tuple(x.shape)}")

        x = x.float()
        bsz, seq_len = x.shape
        trim_len = seq_len - (seq_len % self.patch_size)
        if trim_len <= 0:
            raise ValueError("Input sequence is shorter than patch_size")

        x = x[:, :trim_len]
        patches = x.reshape(bsz, trim_len // self.patch_size, self.patch_size)
        num_patches = patches.shape[1]
        if num_patches > self.max_patches:
            raise ValueError(
                f"num_patches={num_patches} exceeds max_patches={self.max_patches}. "
                "Increase max_patches in PatchEmbedding1D."
            )

        emb = self.proj(patches)
        emb = emb + self.pos_embed[:, :num_patches, :]
        return emb


class PatchEmbedding(PatchEmbedding1D):
    """Backward-compatible alias used by older notebook cells."""


def _random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE-style random masking.

    Returns:
      visible_tokens: [B, N_keep, D]
      mask: [B, N] with 1 for masked, 0 for visible
      ids_restore: [B, N] to unshuffle tokens back to original order
    """

    bsz, num_patches, dim = x.shape
    n_keep = max(1, int(num_patches * (1.0 - mask_ratio)))

    noise = torch.rand(bsz, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]

    visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    mask = torch.ones(bsz, num_patches, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return visible, mask, ids_restore


class ECGTransformerEncoder(nn.Module):
    """Transformer encoder wrapper exposing layer list for optional freezing."""

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.blocks:
            x = layer(x)
        return self.norm(x)


class ECGMaskedAutoencoder(nn.Module):
    """Masked autoencoder for ECG patch reconstruction."""

    def __init__(
        self,
        patch_size: int = 100,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        decoder_depth: int = 4,
        mask_ratio: float = 0.60,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbedding1D(
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_patches=128,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder = ECGTransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        decoder_embed_dim = max(64, embed_dim // 2)
        ff_dim = int(decoder_embed_dim * mlp_ratio)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 128, decoder_embed_dim))
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_embed_dim,
                    nhead=max(1, num_heads // 2),
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify [B, L] into [B, N, patch_size]."""
        x = x.float()
        bsz, seq_len = x.shape
        trim_len = seq_len - (seq_len % self.patch_size)
        x = x[:, :trim_len]
        return x.reshape(bsz, trim_len // self.patch_size, self.patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed patches and binary mask."""
        x = x.float()
        patch_tokens = self.patch_embed(x)  # [B, N, D]
        bsz, n_patches, _ = patch_tokens.shape

        visible_tokens, mask, ids_restore = _random_masking(patch_tokens, self.mask_ratio)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        enc_in = torch.cat([cls_tokens, visible_tokens], dim=1)
        enc_out = self.encoder(enc_in)

        latent_visible = enc_out[:, 1:, :]

        dec_visible = self.decoder_embed(latent_visible)
        n_keep = dec_visible.shape[1]
        n_mask = n_patches - n_keep
        if n_mask > 0:
            mask_tokens = self.mask_token.expand(bsz, n_mask, -1)
            dec_tokens = torch.cat([dec_visible, mask_tokens], dim=1)
        else:
            dec_tokens = dec_visible

        dec_tokens = torch.gather(
            dec_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]),
        )

        dec_tokens = dec_tokens + self.decoder_pos_embed[:, :n_patches, :]
        for layer in self.decoder_layers:
            dec_tokens = layer(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)
        pred = self.decoder_pred(dec_tokens)
        return pred, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full signal without masking and return CLS embeddings [B, D]."""
        x = x.float()
        tokens = self.patch_embed(x)
        bsz = tokens.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        enc_in = torch.cat([cls_tokens, tokens], dim=1)
        enc_out = self.encoder(enc_in)
        return enc_out[:, 0, :]

    @staticmethod
    def compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked-only MSE with per-patch target normalization."""
        target = target.float()
        pred = pred.float()
        mask = mask.float()

        patch_mean = target.mean(dim=-1, keepdim=True)
        patch_std = target.std(dim=-1, keepdim=True).clamp(min=1e-6)
        target_norm = (target - patch_mean) / patch_std

        per_patch = ((pred - target_norm) ** 2).mean(dim=-1)
        denom = mask.sum().clamp(min=1.0)
        return (per_patch * mask).sum() / denom

    def get_encoder_for_finetuning(self) -> Tuple[PatchEmbedding1D, ECGTransformerEncoder]:
        """Return patch embedding and encoder modules for compatibility."""
        return self.patch_embed, self.encoder


class ClassificationHead(nn.Module):
    """Classification head used by both pretrained and baseline classifiers."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGClassifier(nn.Module):
    """Fine-tuning classifier built on top of a pre-trained MAE encoder."""

    def __init__(
        self,
        patch_embed: PatchEmbedding1D,
        encoder: ECGTransformerEncoder,
        num_classes: int = 2,
        freeze_layers: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        embed_dim = encoder.norm.normalized_shape[0]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        freeze_layers = int(max(0, freeze_layers))
        for idx, layer in enumerate(self.encoder.blocks):
            if idx < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        patch_tokens = self.patch_embed(x)
        bsz = patch_tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        enc_in = torch.cat([cls, patch_tokens], dim=1)
        enc_out = self.encoder(enc_in)
        return enc_out[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.classifier(features)

    @classmethod
    def from_pretrained(
        cls,
        mae_model: ECGMaskedAutoencoder,
        num_classes: int = 2,
        freeze_layers: int = 0,
        dropout: float = 0.5,
    ) -> "ECGClassifier":
        patch_embed = copy.deepcopy(mae_model.patch_embed)
        encoder = copy.deepcopy(mae_model.encoder)
        model = cls(
            patch_embed=patch_embed,
            encoder=encoder,
            num_classes=num_classes,
            freeze_layers=freeze_layers,
            dropout=dropout,
        )
        with torch.no_grad():
            model.cls_token.copy_(mae_model.cls_token)
        return model


class BaselineClassifier(nn.Module):
    """Same downstream architecture as ECGClassifier but randomly initialized."""

    def __init__(
        self,
        patch_size: int = 100,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_patches=128,
        )
        self.encoder = ECGTransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.1,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        patch_tokens = self.patch_embed(x)
        bsz = patch_tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        enc_out = self.encoder(torch.cat([cls, patch_tokens], dim=1))
        return enc_out[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))