"""
Q1: Optuna hyperparameter search for best LoRA configuration.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import optuna
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model_for_experiment, count_parameters
from utils import get_cifar100_loaders, compute_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
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


def objective(trial, args, train_loader, val_loader, device):
    """Optuna objective: train with sampled LoRA hyperparameters, return val accuracy."""
    rank = trial.suggest_int("rank", 2, 16)
    alpha = trial.suggest_int("alpha", 1, 16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    print(f"\n  Trial {trial.number}: rank={rank}, alpha={alpha}, dropout={dropout:.3f}")

    model = get_model_for_experiment(rank=rank, alpha=alpha, dropout=dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.optuna_epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.optuna_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        best_val_acc = max(best_val_acc, val_acc)

        # Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"    Epoch {epoch}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

    del model
    torch.cuda.empty_cache()

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Q1: Optuna LoRA Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--optuna-epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--full-epochs", type=int, default=10, help="Epochs for best config")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-dir", type=str, default="results/q1")
    parser.add_argument("--weights-dir", type=str, default="weights/q1")
    parser.add_argument("--wandb-project", type=str, default="assignment5-vit-lora")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size, data_dir="./data"
    )

    # WandB run for Optuna search
    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-', 
        project=args.wandb_project,
        name="optuna_search",
        config=vars(args),
        reinit='finish_previous',
    )

    # Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, device),
        n_trials=args.n_trials,
    )

    # Report best
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best Trial: {best.number}")
    print(f"  Value (Val Acc): {best.value:.2f}%")
    print(f"  Params: {best.params}")
    print(f"{'='*60}")

    # Log to WandB
    wandb.log({
        "best_trial": best.number,
        "best_val_acc": best.value,
        "best_rank": best.params["rank"],
        "best_alpha": best.params["alpha"],
        "best_dropout": best.params["dropout"],
    })

    # Save best config
    best_config = {
        "rank": best.params["rank"],
        "alpha": best.params["alpha"],
        "dropout": best.params["dropout"],
        "val_accuracy": best.value,
        "trial_number": best.number,
    }
    config_path = os.path.join(args.results_dir, "optuna_best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    run.finish()

    # Retrain best config for full epochs
    print(f"\nRetraining best config for {args.full_epochs} epochs...")
    from train import run_experiment

    run_experiment(
        rank=best.params["rank"],
        alpha=best.params["alpha"],
        dropout=best.params["dropout"],
        epochs=args.full_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        results_dir=args.results_dir,
        weights_dir=args.weights_dir,
        wandb_project=args.wandb_project,
    )

    print("Optuna search and retraining complete!")


if __name__ == "__main__":
    main()
