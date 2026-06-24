from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sas_cert.metrics.classification import classification_metrics


class LinearHead(torch.nn.Module):
    def __init__(self, in_dim: int = 200, n_classes: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_head(
    head: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    sample_weight: Optional[np.ndarray] = None,
) -> torch.nn.Module:
    head.to(device)
    head.train()
    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).long()
    if sample_weight is None:
        tw = torch.ones(len(y), dtype=torch.float32)
    else:
        tw = torch.from_numpy(sample_weight.astype(np.float32))
    loader = DataLoader(TensorDataset(tx, ty, tw), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (loss_fn(head(xb), yb) * wb).mean()
            if torch.isnan(loss):
                raise RuntimeError("Head training loss became NaN.")
            loss.backward()
            opt.step()
    return head


def evaluate_head(head: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> Dict[str, float]:
    head.to(device)
    head.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()), batch_size=batch_size, shuffle=False)
    logits = []
    with torch.no_grad():
        for xb, _ in loader:
            logits.append(head(xb.to(device)).detach().cpu().numpy())
    return classification_metrics(np.concatenate(logits, axis=0), y, n_classes=4)


def clone_head(head: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(head)

