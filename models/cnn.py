"""Custom CNN for CIFAR-10 classification.

Architecture: three conv blocks with progressively wider channels,
followed by a two-layer classifier head.
~700K parameters — well within the 5M target.
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Small convolutional network designed for 32x32 RGB inputs.

    Block structure:
        Conv2d → BatchNorm → ReLU → MaxPool  (×3, channels 3→32→64→128)
    Classifier:
        Flatten → Dropout → Linear(2048→256) → ReLU → Dropout → Linear(256→num_classes)

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).
        dropout: Dropout probability applied before each linear layer.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)
