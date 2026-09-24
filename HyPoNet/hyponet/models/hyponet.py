"""Dual-stream HyPoNet model.

The public forward method returns logits. Training applies sigmoid inside the
Bernoulli KL loss, while evaluation applies it when producing risk scores.
"""

from __future__ import annotations

import math

import numpy as np
import pywt
import torch
from torch import nn
from torch.nn import functional as F

from .xresnet1d import xresnet1d50


def _sinusoidal_encoding(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a positional encoding with shape (1, length, width)."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding.unsqueeze(0).to(dtype=dtype)


class SignalProcessor(nn.Module):
    """Risk-informed three-input network for ABP, ECG, PPG and 81 descriptors.

    The paper fixes the reference attention width at 64 and the DWT level at
    three. Both remain constructor arguments so the same class can run the
    reported wavelet-depth ablation. The paper does not specify a spectral
    patch length or fusion hidden width; the defaults here are 64 and 128.
    """

    def __init__(
        self,
        seq_len: int = 6000,
        patch_len: int = 64,
        stride: int = 32,
        d_model: int = 64,
        n_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 1,
        wp_level: int = 3,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("HyPoNet requires three channels ordered ABP, ECG, PPG.")
        if d_model <= 0 or d_model % 2 or d_model % n_heads:
            raise ValueError("d_model must be positive, even, and divisible by n_heads.")
        if patch_len <= 0 or stride <= 0 or wp_level <= 0:
            raise ValueError("patch_len, stride, and wp_level must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.wp_level = wp_level  # Retained name for the public constructor.
        self.input_norm = nn.BatchNorm1d(input_channels)

        # DWT-Former: each modality has its own wavelet basis. Each DWT
        # subband is partitioned into patches and all subband tokens attend
        # to one another within that modality.
        self.wavelets = ("db4", "db2", "db4")
        self.spectral_patch_embedding = nn.Linear(patch_len, d_model)
        self.subband_embedding = nn.Embedding(wp_level + 1, d_model)
        self.modality_embedding = nn.Embedding(input_channels, d_model)
        self.spectral_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.spectral_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.channel_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Patch-Conv: ABP supplies queries; the 2N ECG and PPG patches
        # supply keys and values. MultiheadAttention shares its Q/K/V
        # projections across all patches.
        self.signal_patch_embedding = nn.Linear(patch_len, d_model)
        self.signal_patch_dropout = nn.Dropout(dropout)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(d_model)
        # Hierarchical residual blocks give progressively larger receptive
        # fields; the final XResNet1d50 feature map has 128 channels.
        self.time_resnet = xresnet1d50(
            input_channels=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)

        self.handcrafted_mlp = nn.Sequential(
            nn.Linear(81, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _spectral_tokens(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        """Apply channel-specific DWT and embed patches from every subband."""
        # PyWavelets operates on NumPy arrays and has no autograd support.
        signal = x[:, channel].detach().to(device="cpu", dtype=torch.float32).numpy()
        coefficients = pywt.wavedec(
            signal,
            wavelet=self.wavelets[channel],
            mode="symmetric",
            level=self.wp_level,
            axis=-1,
        )
        tokens = []
        for subband, values in enumerate(coefficients):
            band = torch.as_tensor(
                np.ascontiguousarray(values), device=x.device, dtype=x.dtype
            )
            pad = (-band.shape[-1]) % self.patch_len
            if pad:
                band = F.pad(band, (0, pad))
            patches = band.unfold(-1, self.patch_len, self.patch_len)
            embedded = self.spectral_patch_embedding(patches)
            embedded = embedded + self.subband_embedding.weight[subband]
            tokens.append(embedded)

        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.modality_embedding.weight[channel]
        sequence = sequence + _sinusoidal_encoding(
            sequence.shape[1], sequence.shape[2], sequence.device, sequence.dtype
        )
        return self.spectral_dropout(sequence)

    def _frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        modality_vectors = []
        for channel in range(3):
            encoded = self.spectral_encoder(self._spectral_tokens(x, channel))
            modality_vectors.append(encoded.mean(dim=1))
        modalities = torch.stack(modality_vectors, dim=1)  # B x 3 x d
        weights = torch.softmax(self.channel_score(modalities), dim=1)
        pooled = (weights * modalities).sum(dim=1)
        return self.frequency_projection(pooled)  # B x 128

    def _time_features(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, self.stride), mode="replicate")
        patches = padded.unfold(-1, self.patch_len, self.stride)
        if patches.shape[2] == 0:
            raise ValueError("The input is shorter than one temporal patch.")
        embedded = self.signal_patch_embedding(patches)  # B x 3 x N x d
        embedded = embedded + _sinusoidal_encoding(
            embedded.shape[2], embedded.shape[3], embedded.device, embedded.dtype
        ).unsqueeze(1)
        embedded = self.signal_patch_dropout(embedded)

        primary = embedded[:, 0]  # ABP: B x N x d
        auxiliary = torch.cat((embedded[:, 1], embedded[:, 2]), dim=1)
        attended, _ = self.cross_attention(
            primary, auxiliary, auxiliary, need_weights=False
        )
        refined = self.attention_norm(primary + attended)
        feature_map = self.time_resnet(refined.transpose(1, 2))  # B x 128 x N'
        return self.time_pool(feature_map).flatten(1)  # B x 128

    def forward(self, x: torch.Tensor, ml_feature: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError("x must have shape (batch, 3, samples): ABP, ECG, PPG.")
        if ml_feature is None or ml_feature.ndim != 2 or ml_feature.shape != (x.shape[0], 81):
            raise ValueError("ml_feature must have shape (batch, 81).")
        x = self.input_norm(x)
        time_features = self._time_features(x)
        frequency_features = self._frequency_features(x)
        handcrafted_features = self.handcrafted_mlp(ml_feature)
        fused = torch.cat(
            (time_features, frequency_features, handcrafted_features), dim=1
        )  # B x 384
        return self.classifier(fused)
