"""
Q2(i): Train ResNet-18 from scratch on CIFAR-10 to achieve ≥72% test accuracy.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_resnet18_cifar10
from utils import get_cifar10_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="  Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Q2: Train ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights-dir", type=str, default="weights/q2")
    parser.add_argument("--results-dir", type=str, default="results/q2")
    parser.add_argument("--wandb-project", type=str, default="assignment5-adversarial")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # WandB
    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-', 
        project=args.wandb_project,
        name="resnet18_cifar10_training",
        config=vars(args),
        reinit='finish_previous',
    )

    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Model
    model = create_resnet18_cifar10(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18 parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.weights_dir, "resnet18_cifar10_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (test_acc={best_acc:.2f}%)")

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    wandb.log({"best_test_accuracy": best_acc})

    # Save history
    with open(os.path.join(args.results_dir, "resnet18_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    run.finish()

    if best_acc < 72.0:
        print("⚠️  WARNING: Did not achieve ≥72% test accuracy. Consider more epochs.")
    else:
        print(f"✓ Achieved ≥72% test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
