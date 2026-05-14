"""Entry point for training and evaluating CIFAR-10 models.

Usage:
    python main.py
    python main.py --config configs/config.yaml
    python main.py --arch resnet18
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from training.evaluate import evaluate_on_test
from training.train import get_data_loaders, train
from utils.plots import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_sample_predictions,
)
from utils.seed import set_seed

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["cnn", "resnet18"],
        default=None,
        help="Override the architecture in the config ('cnn' or 'resnet18')",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training, evaluation, and visualization pipeline."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.arch is not None:
        config["model"]["architecture"] = args.arch

    set_seed(config["training"]["seed"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    arch = config["model"]["architecture"]
    figures_dir = Path(config["paths"]["figures_dir"]) / arch
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model, metrics = train(config, device)

    # Visualize training curves
    plot_loss_curves(metrics, figures_dir / "loss_curves.png")
    plot_accuracy_curves(metrics, figures_dir / "accuracy_curves.png")
    print(f"Training curves saved to {figures_dir}/")

    # Evaluate on held-out test set
    _, _, test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()
    results = evaluate_on_test(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss:     {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")

    # Confusion matrix
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        CIFAR10_CLASSES,
        figures_dir / "confusion_matrix.png",
    )

    # Sample predictions
    plot_sample_predictions(
        model,
        test_loader,
        CIFAR10_CLASSES,
        device,
        figures_dir / "sample_predictions.png",
    )

    print(f"Evaluation figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
