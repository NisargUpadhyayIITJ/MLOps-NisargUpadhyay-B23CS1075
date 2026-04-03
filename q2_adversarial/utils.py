"""
Q2: Utility functions for CIFAR-10 adversarial experiments.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir="./data"):
    """Get CIFAR-10 train and test data loaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def get_cifar10_raw_loaders(batch_size=128, num_workers=4, data_dir="./data"):
    """Get CIFAR-10 loaders with only ToTensor (no normalization) for ART."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def plot_adversarial_comparison(
    clean_imgs, adv_scratch_imgs, adv_art_imgs, clean_labels, pred_scratch, pred_art,
    class_names, save_path, num_samples=5
):
    """Plot comparison: original vs adversarial (from scratch) vs adversarial (ART)."""
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))

    titles = ["Clean", "FGSM (Scratch)", "FGSM (IBM ART)"]
    img_sets = [clean_imgs, adv_scratch_imgs, adv_art_imgs]
    pred_sets = [clean_labels, pred_scratch, pred_art]

    for row, (title, imgs, preds) in enumerate(zip(titles, img_sets, pred_sets)):
        for col in range(num_samples):
            img = imgs[col].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_title(
                f"{class_names[preds[col]]}", fontsize=9
            )
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(title, fontsize=12, rotation=0, labelpad=80)

    plt.suptitle("FGSM Adversarial Attack Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_perturbation_analysis(epsilons, clean_acc, scratch_accs, art_accs, save_path):
    """Plot perturbation strength vs accuracy drop."""
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, [clean_acc] * len(epsilons), "g--", label="Clean Accuracy", linewidth=2)
    plt.plot(epsilons, scratch_accs, "r-o", label="FGSM (Scratch)", linewidth=2, markersize=6)
    plt.plot(epsilons, art_accs, "b-s", label="FGSM (IBM ART)", linewidth=2, markersize=6)
    plt.xlabel("Perturbation Strength (ε)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Perturbation Strength vs Accuracy", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_detector_comparison(pgd_metrics, bim_metrics, save_path):
    """Plot comparison of PGD vs BIM adversarial detection performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    attacks = ["PGD", "BIM"]
    accs = [pgd_metrics["accuracy"], bim_metrics["accuracy"]]
    colors = ["#4facfe", "#f5576c"]
    axes[0].bar(attacks, accs, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Detection Accuracy (%)", fontsize=12)
    axes[0].set_title("Detection Accuracy by Attack Type", fontsize=13)
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    # Precision/Recall comparison
    metrics_names = ["Precision", "Recall", "F1-Score"]
    pgd_vals = [pgd_metrics["precision"], pgd_metrics["recall"], pgd_metrics["f1"]]
    bim_vals = [bim_metrics["precision"], bim_metrics["recall"], bim_metrics["f1"]]

    x = np.arange(len(metrics_names))
    width = 0.35
    axes[1].bar(x - width / 2, pgd_vals, width, label="PGD", color="#4facfe")
    axes[1].bar(x + width / 2, bim_vals, width, label="BIM", color="#f5576c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Detection Metrics Comparison", fontsize=13)
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
