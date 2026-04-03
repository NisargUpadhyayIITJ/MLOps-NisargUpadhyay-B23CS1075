"""
Q2(ii): Adversarial Detection Model - PGD and BIM attacks with ResNet-34 detectors.

Uses NormalizedModel wrapper so attacks operate in [0,1] pixel space
while the target model receives properly normalized input.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_resnet18_cifar10, create_resnet34_detector
from utils import (
    get_cifar10_raw_loaders,
    plot_detector_comparison,
    CIFAR10_MEAN,
    CIFAR10_STD,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class NormalizedModel(nn.Module):
    """Wraps a model with input normalization so attacks work in [0,1] space."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def generate_adversarial_dataset(model, images, targets, attack_type, eps, device, batch_size=256):
    """
    Generate adversarial images using IBM ART attacks.
    model: NormalizedModel that accepts [0,1] inputs.
    """
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device == "cuda" else "cpu",
    )

    if attack_type == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier, eps=eps, eps_step=eps / 4,
            max_iter=10, targeted=False,
        )
    elif attack_type == "bim":
        attack = BasicIterativeMethod(
            estimator=classifier, eps=eps, eps_step=eps / 4,
            max_iter=10,
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    print(f"  Generating {attack_type.upper()} adversarial images...")
    # Process in batches to avoid OOM
    adv_list = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].cpu().numpy()
        adv_batch = attack.generate(x=batch)
        adv_list.append(torch.from_numpy(adv_batch))
        print(f"    Batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}", flush=True)

    return torch.cat(adv_list, dim=0)


def create_detection_dataset(clean_images, adv_images, max_samples=10000):
    """Create binary dataset: clean (0) vs adversarial (1)."""
    n = min(len(clean_images), len(adv_images), max_samples)
    images = torch.cat([clean_images[:n], adv_images[:n]], dim=0)
    labels = torch.cat([torch.zeros(n), torch.ones(n)], dim=0).long()

    perm = torch.randperm(len(images))
    return images[perm], labels[perm]


def train_detector(model, train_loader, val_loader, epochs, lr, device):
    """Train the adversarial detector (ResNet-34, binary classification)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        val_loss, val_acc, _, _ = evaluate_detector(model, val_loader, device)
        scheduler.step()

        print(f"    Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def evaluate_detector(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return total_loss / total, acc, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, \
           (all_preds, all_labels)


def log_samples_wandb(clean_images, adv_images, attack_name, num_samples=10):
    """Log sample clean and adversarial images to WandB."""
    wandb_images = []
    for i in range(min(num_samples, len(clean_images))):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(clean_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0].set_title("Clean", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(adv_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1].set_title(f"{attack_name} Adversarial", fontsize=10)
        axes[1].axis("off")
        plt.tight_layout()
        fig_path = f"/tmp/{attack_name}_sample_{i}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close()
        wandb_images.append(wandb.Image(fig_path, caption=f"{attack_name} Sample {i}"))
    wandb.log({f"{attack_name}_samples": wandb_images})


def main():
    parser = argparse.ArgumentParser(description="Q2: Adversarial Detection Training")
    parser.add_argument("--model-path", type=str, default="weights/q2/resnet18_cifar10_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--detector-epochs", type=int, default=15)
    parser.add_argument("--detector-lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--results-dir", type=str, default="results/q2")
    parser.add_argument("--weights-dir", type=str, default="weights/q2")
    parser.add_argument("--wandb-project", type=str, default="assignment5-adversarial")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    # WandB
    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-',
        project=args.wandb_project,
        name="adversarial_detectors",
        config=vars(args),
        reinit='finish_previous',
    )

    # Load target model with normalization wrapper
    base_model = create_resnet18_cifar10(num_classes=10)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))
    target_model = NormalizedModel(base_model, CIFAR10_MEAN, CIFAR10_STD).to(device)
    target_model.eval()

    # Load raw [0,1] test data
    _, test_loader_raw = get_cifar10_raw_loaders(batch_size=args.batch_size)
    all_images = []
    all_targets = []
    for images, targets in test_loader_raw:
        all_images.append(images)
        all_targets.append(targets)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    results = {}

    for attack_type in ["pgd", "bim"]:
        print(f"\n{'='*60}")
        print(f"Processing {attack_type.upper()} Attack")
        print(f"{'='*60}")

        # Generate adversarial images
        adv_images = generate_adversarial_dataset(
            target_model, all_images, all_targets, attack_type, args.eps, device
        )

        # Log samples to WandB
        log_samples_wandb(all_images, adv_images, attack_type, num_samples=10)

        # Create detection dataset
        det_images, det_labels = create_detection_dataset(all_images, adv_images)

        # Split 80/20
        n_train = int(0.8 * len(det_images))
        train_dataset = TensorDataset(det_images[:n_train], det_labels[:n_train])
        val_dataset = TensorDataset(det_images[n_train:], det_labels[n_train:])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Train detector
        print(f"\nTraining {attack_type.upper()} detector (ResNet-34)...")
        detector = create_resnet34_detector(num_classes=2).to(device)
        detector, best_acc = train_detector(
            detector, train_loader, val_loader,
            epochs=args.detector_epochs, lr=args.detector_lr, device=device,
        )

        # Final evaluation
        _, final_acc, metrics, _ = evaluate_detector(detector, val_loader, device)
        print(f"\n{attack_type.upper()} Detector - Accuracy: {final_acc:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        results[attack_type] = metrics

        wandb.log({
            f"{attack_type}_detection_accuracy": final_acc,
            f"{attack_type}_precision": metrics["precision"],
            f"{attack_type}_recall": metrics["recall"],
            f"{attack_type}_f1": metrics["f1"],
        })

        save_path = os.path.join(args.weights_dir, f"detector_{attack_type}_best.pth")
        torch.save(detector.state_dict(), save_path)

        if final_acc < 70.0:
            print(f"  ⚠️  WARNING: {attack_type.upper()} detection accuracy < 70%!")
        else:
            print(f"  ✓ {attack_type.upper()} detection accuracy ≥ 70%")

    # Comparison plot
    comp_path = plot_detector_comparison(
        results["pgd"], results["bim"],
        os.path.join(args.results_dir, "detector_comparison.png")
    )
    wandb.log({"detector_comparison": wandb.Image(comp_path)})

    with open(os.path.join(args.results_dir, "detector_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    table = wandb.Table(columns=["Attack", "Detection Accuracy", "Precision", "Recall", "F1"])
    for attack, m in results.items():
        table.add_data(attack.upper(), round(m["accuracy"], 2), round(m["precision"], 4),
                       round(m["recall"], 4), round(m["f1"], 4))
    wandb.log({"detector_results_table": table})

    run.finish()
    print("\nAdversarial detection training complete!")


if __name__ == "__main__":
    main()
