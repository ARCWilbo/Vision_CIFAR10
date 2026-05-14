"""Deterministic seeding for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds to ensure reproducible results.

    Args:
        seed: Integer seed value to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
