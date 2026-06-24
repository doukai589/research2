from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Compact EEGNet baseline for `[B, C, T]` inputs."""

    def __init__(
        self,
        n_chans: int = 22,
        n_classes: int = 4,
        input_time_len: int = 800,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        drop_prob: float = 0.25,
    ):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(drop_prob),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(drop_prob),
        )
        out_time_len = input_time_len // 32
        self.classifier = nn.Linear(F2 * out_time_len, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

