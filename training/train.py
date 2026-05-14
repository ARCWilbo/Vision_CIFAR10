"""Training orchestration: data loading, model building, and the training loop."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.cnn import CustomCNN
from models.resnet import build_resnet18
from training.loops import train_one_epoch, validate
from utils.metrics import MetricsTracker

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_data_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 train, validation, and test DataLoaders.

    Train split applies augmentation when config['data']['augmentation'] is true.
    Val and test splits use only normalization.

    Args:
        config: Full config dict loaded from config.yaml.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config["data"]
    data_dir = Path(config["paths"]["data_dir"])
    batch_size: int = data_cfg["batch_size"]
    val_split: float = data_cfg["val_split"]
    num_workers: int = data_cfg["num_workers"]

    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    if data_cfg["augmentation"]:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # Validation split shares the augmentation transform from full_train.
    # Override it with the eval transform by wrapping the subset.
    val_set.dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate the model specified in config.

    Args:
        config: Full config dict.
        device: Target device for the model.

    Returns:
        Model moved to device.
    """
    arch: str = config["model"]["architecture"]
    dropout: float = config["model"]["dropout"]

    if arch == "cnn":
        model = CustomCNN(num_classes=10, dropout=dropout)
    elif arch == "resnet18":
        model = build_resnet18(num_classes=10)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Choose 'cnn' or 'resnet18'.")

    return model.to(device)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    """Construct the optimizer from config.

    Args:
        model: Model whose parameters will be optimized.
        config: Full config dict.

    Returns:
        Configured optimizer instance.
    """
    train_cfg = config["training"]
    lr: float = train_cfg["learning_rate"]
    wd: float = train_cfg["weight_decay"]
    opt_name: str = train_cfg["optimizer"]

    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r}. Choose 'adam' or 'sgd'.")


def build_scheduler(optimizer: Optimizer, config: dict[str, Any]):
    """Construct the LR scheduler from config.

    Args:
        optimizer: Optimizer to attach the scheduler to.
        config: Full config dict.

    Returns:
        Scheduler instance, or None if scheduler is 'none'.
    """
    train_cfg = config["training"]
    sched_name: str = train_cfg["scheduler"]
    epochs: int = train_cfg["epochs"]

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name!r}. Choose 'cosine', 'step', or 'none'.")


def train(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, MetricsTracker]:
    """Run the full training loop with early stopping and checkpointing.

    Args:
        config: Full config dict loaded from config.yaml.
        device: Device to train on.

    Returns:
        Tuple of (best_model, metrics_tracker).
    """
    train_cfg = config["training"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs: int = train_cfg["epochs"]
    patience: int = train_cfg["early_stopping_patience"]

    train_loader, val_loader, _ = get_data_loaders(config)
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()
    metrics = MetricsTracker()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    arch = config["model"]["architecture"]

    print(f"\nTraining {arch} for up to {epochs} epochs on {device}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        metrics.update(train_loss, val_loss, train_acc, val_acc)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch [{epoch:>3}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "val_acc": val_acc},
                checkpoint_dir / f"best_{arch}.pth",
            )
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        checkpoint_dir / f"final_{arch}.pth",
    )
    print(f"\nTraining complete. Best val acc: {metrics.best_val_acc*100:.2f}%")

    # Reload best weights into model before returning
    best_ckpt = torch.load(checkpoint_dir / f"best_{arch}.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model, metrics
