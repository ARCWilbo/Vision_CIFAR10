"""Test-set evaluation and prediction collection."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.loops import validate


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all ground-truth labels and model predictions.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to iterate over.
        device: Device the model is on.

    Returns:
        Tuple of (y_true, y_pred) as numpy int arrays.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run full evaluation on the held-out test set.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        criterion: Loss function (same as used during training).
        device: Device the model is on.

    Returns:
        Dict with keys: loss, accuracy, y_true, y_pred.
    """
    loss, accuracy = validate(model, test_loader, criterion, device)
    y_true, y_pred = get_predictions(model, test_loader, device)
    return {"loss": loss, "accuracy": accuracy, "y_true": y_true, "y_pred": y_pred}
