# MLOps Assignment 2 - CNN Training on CIFAR-10

**Name:** Nisarg Upadhyay  
**Roll Number:** B23CS1075

---

## Overview

Training a SimpleCNN model on CIFAR-10 with:
- Custom DataLoader implementation
- FLOPs counting using ptflops
- Gradient flow visualization
- Weight update tracking
- All visualizations logged to Weights & Biases

---

## Model: SimpleCNN

| Component | Description |
|:----------|:------------|
| Architecture | 3 Conv Blocks + 2 FC Layers |
| Parameters | ~500K |
| Input Size | 3 × 32 × 32 (CIFAR-10) |
| Output | 10 classes |

### Training Configuration

| Parameter | Value |
|:----------|:------|
| Epochs | 30 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| LR Scheduler | Cosine Annealing |

---

## Visualizations (W&B)

1. **Gradient Flow** - Bar charts showing gradient magnitudes per layer
2. **Weight Histograms** - Distribution of weights logged every 5 epochs
3. **Weight Updates** - Tracking weight changes between epochs
4. **Training Curves** - Loss, accuracy, and learning rate curves

---

## Files

- `MLOps_Assignment2_CNN_CIFAR10.ipynb` - Main notebook with all code

---

## Links

- **GitHub Pages:** [Portfolio Site](https://nisargupadhyayiitj.github.io/MLOps-NisargUpadhyay-B23CS1075/)
- **W&B Dashboard:** *(Update after running notebook)*

---

## Requirements

```
torch
torchvision
wandb
ptflops
matplotlib
numpy
```
