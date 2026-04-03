"""
Q1: Training script for ViT-S on CIFAR-100 with and without LoRA.
Runs all 10 experiments (1 baseline + 9 LoRA configurations).
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model_for_experiment, count_parameters
from utils import (
    get_cifar100_loaders,
    compute_accuracy,
    get_classwise_accuracy,
    plot_classwise_histogram,
    plot_training_curves,
    plot_gradient_norms,
)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_history=None):
    """Train for one epoch, optionally tracking LoRA gradient norms."""
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

        # Track gradient norms of LoRA weights
        if grad_history is not None:
            for name, param in model.named_parameters():
                if "lora" in name.lower() and param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    grad_history[name].append(norm)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="  Eval", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def run_experiment(
    rank=None,
    alpha=None,
    dropout=0.1,
    epochs=10,
    lr=1e-4,
    batch_size=64,
    device="cuda",
    results_dir="results/q1",
    weights_dir="weights/q1",
    wandb_project="assignment5-vit-lora",
):
    """Run a single training experiment."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Experiment name
    if rank is not None:
        exp_name = f"lora_r{rank}_a{alpha}_d{dropout}"
    else:
        exp_name = "baseline_no_lora"

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Init WandB
    config = {
        "experiment": exp_name,
        "lora": rank is not None,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "model": "vit_small_patch16_224",
        "dataset": "CIFAR-100",
    }

    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-', 
        project=wandb_project,
        name=exp_name,
        config=config,
        reinit='finish_previous',
    )

    # Data
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_dir="./data"
    )

    # Model
    model = get_model_for_experiment(rank=rank, alpha=alpha, dropout=dropout)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Track history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    grad_history = defaultdict(list) if rank is not None else None
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_history
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(weights_dir, f"{exp_name}_best.pth")
            if hasattr(model, "save_pretrained"):
                # PEFT model
                model.save_pretrained(os.path.join(weights_dir, f"{exp_name}_peft"))
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_acc={best_val_acc:.2f}%)")

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

    # Class-wise accuracy
    class_acc = get_classwise_accuracy(model, test_loader, device)
    hist_path = plot_classwise_histogram(
        class_acc,
        f"Class-wise Test Accuracy - {exp_name}",
        os.path.join(results_dir, f"{exp_name}_classwise.png"),
    )
    wandb.log({"classwise_accuracy": wandb.Image(hist_path)})

    # Log class-wise accuracy histogram to WandB
    wandb.log({
        "classwise_accuracy_histogram": wandb.Histogram(class_acc * 100),
    })

    # Training curves
    curve_path = plot_training_curves(
        history, exp_name, os.path.join(results_dir, f"{exp_name}_curves.png")
    )
    wandb.log({"training_curves": wandb.Image(curve_path)})

    # Gradient norms plot (LoRA only)
    if grad_history and len(grad_history) > 0:
        grad_path = plot_gradient_norms(
            dict(grad_history),
            f"LoRA Gradient Norms - {exp_name}",
            os.path.join(results_dir, f"{exp_name}_gradients.png"),
        )
        wandb.log({"gradient_norms": wandb.Image(grad_path)})

    # Save experiment results
    exp_result = {
        "experiment": exp_name,
        "lora": rank is not None,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
    }

    result_path = os.path.join(results_dir, f"{exp_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(exp_result, f, indent=2)

    run.finish()
    return exp_result


def main():
    parser = argparse.ArgumentParser(description="Q1: ViT-S LoRA Training on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--results-dir", type=str, default="results/q1")
    parser.add_argument("--weights-dir", type=str, default="weights/q1")
    parser.add_argument("--wandb-project", type=str, default="assignment5-vit-lora")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific experiment, e.g. 'baseline' or 'r4_a8'")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define all experiment configurations
    experiments = [
        {"rank": None, "alpha": None, "dropout": 0.0, "label": "baseline"},  # Baseline
    ]

    # LoRA combinations: rank in {2, 4, 8}, alpha in {2, 4, 8}, dropout = 0.1
    for r in [2, 4, 8]:
        for a in [2, 4, 8]:
            experiments.append({
                "rank": r, "alpha": a, "dropout": 0.1, "label": f"r{r}_a{a}"
            })

    all_results = []

    for exp in experiments:
        # Filter if --only specified
        if args.only and args.only != exp["label"]:
            continue

        result = run_experiment(
            rank=exp["rank"],
            alpha=exp["alpha"],
            dropout=exp["dropout"],
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            results_dir=args.results_dir,
            weights_dir=args.weights_dir,
            wandb_project=args.wandb_project,
        )
        all_results.append(result)

    # Save summary
    summary_path = os.path.join(args.results_dir, "all_results_summary.json")
    summary = []
    for r in all_results:
        summary.append({
            "experiment": r["experiment"],
            "lora": r["lora"],
            "rank": r["rank"],
            "alpha": r["alpha"],
            "dropout": r["dropout"],
            "trainable_params": r["trainable_params"],
            "test_acc": r["test_acc"],
            "best_val_acc": r["best_val_acc"],
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Experiment':<30} {'Test Acc':>10} {'Trainable':>12}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['experiment']:<30} {r['test_acc']:>9.2f}% {r['trainable_params']:>12,}")


if __name__ == "__main__":
    main()
