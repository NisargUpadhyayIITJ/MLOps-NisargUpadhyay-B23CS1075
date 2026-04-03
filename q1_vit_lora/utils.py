"""
Q1: Utility functions for CIFAR-100 data loading, metrics, and WandB helpers.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_loaders(batch_size: int = 64, num_workers: int = 4, data_dir: str = "./data"):
    """Get CIFAR-100 train, validation, and test data loaders with ViT-appropriate transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    # Split train into train/val (45000/5000)
    train_size = 45000
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val uses test transforms (no augmentation)
    val_dataset_clean = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=test_transform)
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_clean, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_accuracy(outputs, targets):
    """Compute top-1 accuracy."""
    _, preds = outputs.max(1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)


def get_classwise_accuracy(model, test_loader, device, num_classes=100):
    """Compute per-class accuracy on the test set."""
    model.eval()
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            for c in range(num_classes):
                mask = targets == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()

    class_acc = class_correct / (class_total + 1e-8)
    return class_acc.numpy()


def plot_classwise_histogram(class_acc, title, save_path):
    """Plot class-wise accuracy as a histogram."""
    plt.figure(figsize=(20, 6))
    colors = plt.cm.viridis(class_acc)
    plt.bar(range(len(class_acc)), class_acc * 100, color=colors, edgecolor='none')
    plt.xlabel("Class Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_training_curves(history, title, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_gradient_norms(grad_history, title, save_path):
    """Plot gradient norms for LoRA weights during training."""
    plt.figure(figsize=(12, 6))
    for name, norms in grad_history.items():
        short_name = name.split(".")[-2] + "." + name.split(".")[-1]
        plt.plot(norms, label=short_name, alpha=0.7)

    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
