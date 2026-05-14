"""ResNet-18 baseline for CIFAR-10.

Uses torchvision's pretrained-compatible ResNet-18 architecture with the
final fully-connected layer replaced to output num_classes logits.
"""

import torch.nn as nn
import torchvision.models as tv_models


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 with a custom classification head.

    Loads the standard torchvision ResNet-18 (random weights) and
    replaces the final linear layer so the model outputs num_classes logits.

    Args:
        num_classes: Number of output classes.

    Returns:
        Modified ResNet-18 model ready for training.
    """
    model = tv_models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
