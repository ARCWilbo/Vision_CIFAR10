# CIFAR-10 CNN Training Pipeline

An educational, production-style deep learning project for image classification using PyTorch. Trains a custom CNN and a ResNet-18 baseline on CIFAR-10 with full logging, checkpointing, and visualization.

---

## Learning Goals

- Understand CNN fundamentals (conv blocks, batch norm, dropout, pooling)
- Build a reproducible ML training pipeline from scratch
- Analyze training dynamics via loss/accuracy curves
- Compare a hand-built CNN against a standard ResNet baseline

---

## Environment Setup

**Option A — conda**
```bash
conda create -n cnn python=3.11 -y
conda activate cnn
pip install -r requirements.txt
```

**Option B — venv**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

Train the custom CNN (default):
```bash
python main.py
```

Train ResNet-18:
```bash
python main.py --arch resnet18
```

Use a custom config:
```bash
python main.py --config configs/config.yaml --arch cnn
```

All hyperparameters (batch size, learning rate, optimizer, scheduler, epochs, dropout, seed) are controlled via `configs/config.yaml` — no hardcoded values in the code.

---

## Project Structure

```
CNN/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── models/
│   ├── cnn.py               # Custom 3-block CNN (~700K params)
│   └── resnet.py            # ResNet-18 with custom head
├── training/
│   ├── loops.py             # train_one_epoch / validate (pure functions)
│   ├── evaluate.py          # Test-set evaluation and prediction collection
│   └── train.py             # Data loading, model/optimizer/scheduler builders, training loop
├── utils/
│   ├── seed.py              # Deterministic seeding
│   ├── metrics.py           # MetricsTracker dataclass
│   └── plots.py             # Loss curves, accuracy curves, confusion matrix, sample predictions
├── outputs/
│   ├── checkpoints/         # best_<arch>.pth, final_<arch>.pth
│   └── figures/             # All saved plots
├── main.py                  # Entry point
└── requirements.txt
```

---

## Configuration

Key options in `configs/config.yaml`:

| Key | Default | Options |
|-----|---------|---------|
| `model.architecture` | `"cnn"` | `"cnn"`, `"resnet18"` |
| `model.dropout` | `0.3` | float |
| `data.batch_size` | `128` | int |
| `training.optimizer` | `"adam"` | `"adam"`, `"sgd"` |
| `training.scheduler` | `"cosine"` | `"cosine"`, `"step"`, `"none"` |
| `training.epochs` | `50` | int |
| `training.early_stopping_patience` | `10` | int |

---

## Results

| Model | Val Accuracy | Train Accuracy | Epochs | Notes |
|-------|-------------|----------------|--------|-------|
| Custom CNN | **~84%** | ~83% | 50 | ~700K params · val loss ≈ 0.44 · no overfitting |
| ResNet-18 | **~86%** | ~94% | 44 (early stop) | ~11M params · val loss ≈ 0.50 · overfitting visible |

**Custom CNN — Accuracy & Loss**

![CNN Accuracy](outputs/figures/cnn/accuracy_curves.png)
![CNN Loss](outputs/figures/cnn/loss_curves.png)

> Val accuracy tracks above train accuracy throughout — expected behavior when dropout is active during training but disabled at eval time.

**ResNet-18 — Accuracy & Loss**

![ResNet18 Accuracy](outputs/figures/resnet18/accuracy_curves.png)
![ResNet18 Loss](outputs/figures/resnet18/loss_curves.png)

> Classic overfitting signature: train loss drops to ~0.16 while val loss plateaus at ~0.50. Early stopping triggered at epoch 44. Data augmentation or stronger regularization would close this gap.

---

## Outputs

After training, the following are saved automatically:

- `outputs/checkpoints/best_<arch>.pth` — best model by validation loss
- `outputs/checkpoints/final_<arch>.pth` — model state at end of training
- `outputs/figures/<arch>/loss_curves.png`
- `outputs/figures/<arch>/accuracy_curves.png`
- `outputs/figures/<arch>/confusion_matrix.png`
- `outputs/figures/<arch>/sample_predictions.png`

---

## Hardware

Designed for Apple Silicon (MPS acceleration). Falls back to CUDA or CPU automatically.
