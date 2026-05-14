# CLAUDE.md

## Project Goal

Build a clean, educational, production-style deep learning project for CIFAR-10 image classification using PyTorch.

The objective is to:

- Learn CNN fundamentals deeply
- Build a reproducible ML training pipeline
- Understand training dynamics and experimentation
- Train locally on Apple Silicon (MPS acceleration)

The codebase should prioritize:

- readability
- modularity
- reproducibility
- experimentation
- strong engineering practices

Do NOT generate overly abstract enterprise architecture.

Keep the project educational and concise.

---

# Environment

Hardware:

- Apple Silicon MacBook
- MPS acceleration available

Python:

- Python 3.11

Frameworks:

- PyTorch
- torchvision
- matplotlib
- numpy
- pandas
- scikit-learn

Use MPS if available:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

# Primary Architecture

Implement:

1. A small custom CNN
2. ResNet-18 baseline using torchvision

The custom CNN should include:

- Conv2d
- ReLU
- MaxPool2d
- BatchNorm
- Dropout
- Linear layers

Do NOT begin with transformers or extremely advanced architectures.

---

# Dataset

Use torchvision CIFAR-10.

Train/test split:

- training set
- validation split
- test set

Include:

- normalization
- random horizontal flips
- random crops

---

# Required Project Structure

The repository should contain:

```text
CNN/
│
├── data/
├── models/
│   ├── cnn.py
│   └── resnet.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── loops.py
│
├── utils/
│   ├── metrics.py
│   ├── seed.py
│   └── plots.py
│
├── configs/
│   └── config.yaml
│
├── outputs/
│   ├── checkpoints/
│   └── figures/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Engineering Requirements

All code should:

- include type hints
- include docstrings
- avoid global variables
- use functions/classes cleanly
- avoid notebook-style scripting
- use pathlib instead of raw strings where possible

Use:

- tqdm progress bars
- deterministic seeds
- model checkpoint saving
- training/validation metrics
- clean logging

---

# Training Requirements

Track:

- train loss
- validation loss
- train accuracy
- validation accuracy

Save:

- best model checkpoint
- final model checkpoint

Implement:

- early stopping
- learning rate scheduler
- optimizer selection

Support:

- Adam
- SGD

---

# Visualization Requirements

Generate:

1. loss curves
2. accuracy curves
3. confusion matrix
4. sample predictions

Save all plots to:

```text
outputs/figures/
```

Use matplotlib only.

---

# CNN Requirements

The custom CNN should:

- accept 32x32 RGB images
- progressively increase channels
- flatten correctly
- use dropout before classifier
- remain small enough to train quickly locally

Target parameter count:

- under 5 million parameters

---

# ResNet Requirements

Implement:

```python
torchvision.models.resnet18
```

Modify final layer for:

```python
num_classes = 10
```

Allow switching between:

- custom CNN
- ResNet18

through config.

---

# Config System

Use a YAML config file.

The config should control:

- batch size
- learning rate
- epochs
- optimizer
- architecture
- scheduler
- augmentation
- dropout
- seed

Avoid hardcoding hyperparameters.

---

# README Requirements

The README should include:

- setup instructions
- environment creation
- training commands
- project structure explanation
- sample results
- learning goals

---

# Training Expectations

The custom CNN should achieve:

- 75–85% validation accuracy

ResNet-18 should achieve:

- 90%+ validation accuracy

---

# Important Constraints

DO NOT:

- generate giant frameworks
- introduce unnecessary abstractions
- use Lightning initially
- use Hydra initially
- create Docker/Kubernetes configs
- use distributed training
- use transformers

DO:

- keep everything educational
- explain architectural choices
- write concise clean code
- prefer clarity over cleverness

---

# Workflow Expectations

When generating code:

1. First explain the purpose of the file
2. Then generate the file
3. Keep files modular
4. Ensure imports are correct
5. Ensure the code runs end-to-end

When debugging:

- reason step-by-step
- explain likely root causes
- propose minimal fixes first

When improving performance:

- explain WHY the change helps

---

# Stretch Goals

After baseline training:

- add mixed precision
- add Grad-CAM visualization
- add simple hyperparameter sweeps
- compare optimizers
- compare augmentations

But only after the baseline works correctly.

---

# Final Objective

The final repository should resemble:

- a strong beginner/intermediate deep learning portfolio project
- something suitable for quant research / ML internship discussions
- a clean educational implementation of CNN training fundamentals
