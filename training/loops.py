"""Core training and validation loops.

Pure functions — no I/O, no side effects beyond model weight updates.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training set.

    Args:
        model: Model to train (set to train mode internally).
        loader: Training DataLoader.
        optimizer: Optimizer with zero_grad called here each step.
        criterion: Loss function.
        device: Device to move batches onto.

    Returns:
        Tuple of (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation or test DataLoader.

    Args:
        model: Model to evaluate (set to eval mode internally).
        loader: Validation/test DataLoader.
        criterion: Loss function.
        device: Device to move batches onto.

    Returns:
        Tuple of (avg_loss, accuracy) over the full loader.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="    Val", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
