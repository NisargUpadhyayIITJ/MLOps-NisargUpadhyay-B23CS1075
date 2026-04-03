"""
Q1: Testing script for ViT-S on CIFAR-100.
Loads all trained models and generates summary tables and plots.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model_for_experiment, count_parameters
from utils import (
    get_cifar100_loaders,
    get_classwise_accuracy,
    plot_classwise_histogram,
)


@torch.no_grad()
def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def load_and_test(rank, alpha, dropout, weights_dir, test_loader, device, results_dir):
    """Load a trained model and evaluate it."""
    if rank is not None:
        exp_name = f"lora_r{rank}_a{alpha}_d{dropout}"
    else:
        exp_name = "baseline_no_lora"

    model = get_model_for_experiment(rank=rank, alpha=alpha, dropout=dropout)

    # Load weights
    if rank is not None:
        peft_dir = os.path.join(weights_dir, f"{exp_name}_peft")
        if os.path.exists(peft_dir):
            from peft import PeftModel
            # We need to load the base model and then load PEFT adapter
            import timm
            base_model = timm.create_model("vit_small_patch16_224", pretrained=False)
            base_model.head = nn.Linear(base_model.head.in_features, 100)
            model = PeftModel.from_pretrained(base_model, peft_dir)
        else:
            weight_path = os.path.join(weights_dir, f"{exp_name}_best.pth")
            model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        weight_path = os.path.join(weights_dir, f"{exp_name}_best.pth")
        model.load_state_dict(torch.load(weight_path, map_location=device))

    model = model.to(device)
    test_loss, test_acc = test_model(model, test_loader, device)
    total, trainable = count_parameters(model)

    # Class-wise accuracy
    class_acc = get_classwise_accuracy(model, test_loader, device)
    plot_classwise_histogram(
        class_acc,
        f"Class-wise Test Accuracy - {exp_name}",
        os.path.join(results_dir, f"{exp_name}_test_classwise.png"),
    )

    return {
        "experiment": exp_name,
        "lora": rank is not None,
        "rank": rank if rank else "N/A",
        "alpha": alpha if alpha else "N/A",
        "dropout": dropout if rank else "N/A",
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_params": total,
        "trainable_params": trainable,
    }


def main():
    parser = argparse.ArgumentParser(description="Q1: Test all trained models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights-dir", type=str, default="weights/q1")
    parser.add_argument("--results-dir", type=str, default="results/q1")
    parser.add_argument("--wandb-project", type=str, default="assignment5-vit-lora")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    _, _, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size, data_dir="./data"
    )

    experiments = [
        {"rank": None, "alpha": None, "dropout": 0.0},
    ]
    for r in [2, 4, 8]:
        for a in [2, 4, 8]:
            experiments.append({"rank": r, "alpha": a, "dropout": 0.1})

    results = []
    for exp in experiments:
        try:
            result = load_and_test(
                rank=exp["rank"],
                alpha=exp["alpha"],
                dropout=exp["dropout"],
                weights_dir=args.weights_dir,
                test_loader=test_loader,
                device=device,
                results_dir=args.results_dir,
            )
            results.append(result)
            print(f"{result['experiment']:<30} Test Acc: {result['test_acc']:.2f}%  "
                  f"Trainable: {result['trainable_params']:,}")
        except Exception as e:
            print(f"Failed to test {exp}: {e}")

    # Save test summary
    summary_path = os.path.join(args.results_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print formatted table
    print(f"\n{'='*90}")
    print(f"{'LoRA':<8} {'Rank':<6} {'Alpha':<6} {'Dropout':<8} {'Test Acc':>10} {'Trainable Params':>18}")
    print("-" * 90)
    for r in results:
        lora_str = "Yes" if r["lora"] else "No"
        print(f"{lora_str:<8} {str(r['rank']):<6} {str(r['alpha']):<6} "
              f"{str(r['dropout']):<8} {r['test_acc']:>9.2f}% {r['trainable_params']:>18,}")

    # Log summary table to WandB
    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-', project=args.wandb_project, name="test_summary", reinit='finish_previous')
    table = wandb.Table(
        columns=["LoRA", "Rank", "Alpha", "Dropout", "Test Accuracy", "Trainable Params"]
    )
    for r in results:
        table.add_data(
            "Yes" if r["lora"] else "No",
            str(r["rank"]), str(r["alpha"]),
            str(r["dropout"]),
            round(r["test_acc"], 2),
            r["trainable_params"],
        )
    wandb.log({"test_results_table": table})
    run.finish()

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
