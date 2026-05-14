"""Metrics tracking for training and validation history."""

from dataclasses import dataclass, field


@dataclass
class MetricsTracker:
    """Accumulates per-epoch loss and accuracy for train and validation sets."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)

    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
    ) -> None:
        """Append one epoch of metrics.

        Args:
            train_loss: Average training loss for the epoch.
            val_loss: Average validation loss for the epoch.
            train_acc: Training accuracy (0–1) for the epoch.
            val_acc: Validation accuracy (0–1) for the epoch.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

    @property
    def best_val_acc(self) -> float:
        """Return the highest validation accuracy recorded so far."""
        return max(self.val_accs) if self.val_accs else 0.0

    @property
    def best_val_loss(self) -> float:
        """Return the lowest validation loss recorded so far."""
        return min(self.val_losses) if self.val_losses else float("inf")

    def __len__(self) -> int:
        return len(self.train_losses)
