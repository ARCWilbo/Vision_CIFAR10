"""Visualization utilities — all figures saved to disk via matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import MetricsTracker

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_loss_curves(metrics: MetricsTracker, save_path: Path) -> None:
    """Plot training and validation loss over epochs.

    Args:
        metrics: Populated MetricsTracker instance.
        save_path: File path (including filename) to save the figure.
    """
    epochs = range(1, len(metrics) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, metrics.train_losses, label="Train Loss")
    ax.plot(epochs, metrics.val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_accuracy_curves(metrics: MetricsTracker, save_path: Path) -> None:
    """Plot training and validation accuracy over epochs.

    Args:
        metrics: Populated MetricsTracker instance.
        save_path: File path (including filename) to save the figure.
    """
    epochs = range(1, len(metrics) + 1)
    train_pct = [a * 100 for a in metrics.train_accs]
    val_pct = [a * 100 for a in metrics.val_accs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_pct, label="Train Acc")
    ax.plot(epochs, val_pct, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot and save a normalized confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Ordered list of class label strings.
        save_path: File path to save the figure.
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix (normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    save_path: Path,
    num_samples: int = 16,
) -> None:
    """Plot a grid of sample images with predicted and true labels.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to sample images from.
        class_names: Ordered list of class label strings.
        device: Device the model is on.
        save_path: File path to save the figure.
        num_samples: Number of images to display (must be a perfect square friendly number).
    """
    model.eval()
    images_shown = 0
    imgs, preds, labels = [], [], []

    # CIFAR-10 normalization stats for denormalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            batch_preds = outputs.argmax(dim=1).cpu()

            for img, pred, label in zip(batch_imgs.cpu(), batch_preds, batch_labels):
                imgs.append(img)
                preds.append(pred.item())
                labels.append(label.item())
                images_shown += 1
                if images_shown >= num_samples:
                    break
            if images_shown >= num_samples:
                break

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for ax, img, pred, label in zip(axes, imgs, preds, labels):
        img_display = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        color = "green" if pred == label else "red"
        ax.set_title(f"P: {class_names[pred]}\nT: {class_names[label]}", color=color, fontsize=8)
        ax.axis("off")

    for ax in axes[len(imgs):]:
        ax.axis("off")

    fig.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
