from __future__ import annotations

import copy
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics
from .models import EEGNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(device: str) -> EEGNet:
    return EEGNet(n_chans=22, n_classes=4, input_time_len=800).to(device)


def train_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> torch.nn.Module:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def evaluate_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    logits = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits.append(model(xb.to(device)).detach().cpu().numpy())
    logits_np = np.concatenate(logits, axis=0)
    return classification_metrics(logits_np, y, n_classes=4), logits_np


def clone_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    cloned = copy.deepcopy(model)
    cloned.to(device)
    return cloned

