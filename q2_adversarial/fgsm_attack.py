"""
Q2(i): FGSM Attack - From Scratch vs IBM ART comparison.
Uses a trained ResNet-18 on CIFAR-10.

Key design: The model was trained with normalized inputs, but adversarial
attacks must operate in raw [0,1] pixel space. We wrap the model with a
normalization layer so attacks see [0,1] inputs while the model still
receives properly normalized tensors.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np

import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_resnet18_cifar10
from utils import (
    get_cifar10_raw_loaders,
    plot_adversarial_comparison,
    plot_perturbation_analysis,
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


# ========== FGSM FROM SCRATCH ==========
def fgsm_attack_scratch(model, images, targets, epsilon, criterion, device):
    """
    Implement FGSM attack from scratch.
    x_adv = x + epsilon * sign(grad_x(loss))
    Model should accept [0,1] inputs (use NormalizedModel wrapper).
    """
    images = images.clone().detach().to(device).requires_grad_(True)
    targets = targets.to(device)

    outputs = model(images)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.data.sign()
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


# ========== FGSM WITH IBM ART ==========
def fgsm_attack_art(classifier, images, epsilon):
    """
    Implement FGSM attack using IBM ART.
    classifier: pre-built ART PyTorchClassifier wrapping NormalizedModel.
    """
    from art.attacks.evasion import FastGradientMethod

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    images_np = images.cpu().numpy()
    adv_images_np = attack.generate(x=images_np)
    return torch.from_numpy(adv_images_np)


def evaluate_accuracy(model, images, targets, device):
    """Evaluate model accuracy on given images."""
    model.eval()
    correct = 0
    total = targets.size(0)
    bs = 256
    all_preds = []
    with torch.no_grad():
        for i in range(0, total, bs):
            batch = images[i:i+bs].to(device)
            out = model(batch)
            _, preds = out.max(1)
            correct += preds.eq(targets[i:i+bs].to(device)).sum().item()
            all_preds.append(preds.cpu())
    return 100.0 * correct / total, torch.cat(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Q2: FGSM Attack Comparison")
    parser.add_argument("--model-path", type=str, default="weights/q2/resnet18_cifar10_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--results-dir", type=str, default="results/q2")
    parser.add_argument("--wandb-project", type=str, default="assignment5-adversarial")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    # Load base model and wrap with normalization
    base_model = create_resnet18_cifar10(num_classes=10)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = NormalizedModel(base_model, CIFAR10_MEAN, CIFAR10_STD).to(device)
    model.eval()

    # Build ART classifier once (reuse for all epsilons)
    from art.estimators.classification import PyTorchClassifier
    art_classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device == "cuda" else "cpu",
    )

    # WandB
    run = wandb.init(entity='b23cs1075-indian-institute-of-technology-',
        project=args.wandb_project,
        name="fgsm_attack_comparison",
        config=vars(args),
        reinit='finish_previous',
    )

    # Load raw [0,1] test data
    _, test_loader_raw = get_cifar10_raw_loaders(batch_size=args.batch_size)
    all_images = []
    all_targets = []
    for images, targets in test_loader_raw:
        all_images.append(images)
        all_targets.append(targets)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    criterion = nn.CrossEntropyLoss()

    # ===== Clean accuracy =====
    clean_acc, clean_preds = evaluate_accuracy(model, all_images, all_targets, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # ===== Perturbation sweep =====
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.2, 0.3]
    scratch_accs = []
    art_accs = []

    for eps in epsilons:
        print(f"\nEpsilon: {eps}")

        if eps == 0:
            scratch_accs.append(clean_acc)
            art_accs.append(clean_acc)
            continue

        scratch_correct = 0
        art_correct = 0
        total = 0

        for i in range(0, len(all_images), args.batch_size):
            batch_imgs = all_images[i:i + args.batch_size]
            batch_tgts = all_targets[i:i + args.batch_size]

            # FGSM from scratch
            adv_scratch = fgsm_attack_scratch(
                model, batch_imgs, batch_tgts, eps, criterion, device
            )
            with torch.no_grad():
                out_s = model(adv_scratch.to(device))
                _, pred_s = out_s.max(1)
                scratch_correct += pred_s.eq(batch_tgts.to(device)).sum().item()

            # FGSM with ART
            adv_art = fgsm_attack_art(art_classifier, batch_imgs, eps)
            with torch.no_grad():
                out_a = model(adv_art.to(device))
                _, pred_a = out_a.max(1)
                art_correct += pred_a.eq(batch_tgts.to(device)).sum().item()

            total += batch_tgts.size(0)

        s_acc = 100.0 * scratch_correct / total
        a_acc = 100.0 * art_correct / total
        scratch_accs.append(s_acc)
        art_accs.append(a_acc)

        print(f"  Scratch Acc: {s_acc:.2f}%  |  ART Acc: {a_acc:.2f}%")
        wandb.log({"epsilon": eps, "fgsm_scratch_accuracy": s_acc, "fgsm_art_accuracy": a_acc})

    # ===== Perturbation analysis plot =====
    plot_path = plot_perturbation_analysis(
        epsilons, clean_acc, scratch_accs, art_accs,
        os.path.join(args.results_dir, "perturbation_analysis.png")
    )
    wandb.log({"perturbation_analysis": wandb.Image(plot_path)})

    # ===== Visual comparison (eps=0.04) =====
    eps_vis = 0.04
    sample_imgs = all_images[:10]
    sample_tgts = all_targets[:10]

    adv_scratch = fgsm_attack_scratch(model, sample_imgs, sample_tgts, eps_vis, criterion, device)
    adv_art = fgsm_attack_art(art_classifier, sample_imgs, eps_vis)

    _, preds_clean = evaluate_accuracy(model, sample_imgs, sample_tgts, device)
    _, preds_scratch = evaluate_accuracy(model, adv_scratch, sample_tgts, device)
    _, preds_art = evaluate_accuracy(model, adv_art, sample_tgts, device)

    comp_path = plot_adversarial_comparison(
        sample_imgs[:5], adv_scratch.cpu()[:5], adv_art.cpu()[:5],
        sample_tgts[:5].tolist(), preds_scratch[:5].tolist(), preds_art[:5].tolist(),
        CIFAR10_CLASSES,
        os.path.join(args.results_dir, "fgsm_comparison.png"),
    )
    wandb.log({"fgsm_visual_comparison": wandb.Image(comp_path)})

    # ===== Log 10 sample images to WandB =====
    wandb_images = []
    for i in range(10):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [sample_imgs[i], adv_scratch[i].cpu(), adv_art[i].cpu()],
            [f"Clean: {CIFAR10_CLASSES[sample_tgts[i]]}",
             f"Scratch: {CIFAR10_CLASSES[preds_scratch[i]]}",
             f"ART: {CIFAR10_CLASSES[preds_art[i]]}"]
        ):
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        fig_path = os.path.join(args.results_dir, f"fgsm_sample_{i}.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close()
        wandb_images.append(wandb.Image(fig_path, caption=f"Sample {i}"))

    wandb.log({"fgsm_samples": wandb_images})

    # ===== Summary =====
    summary = {
        "clean_accuracy": clean_acc,
        "epsilons": epsilons,
        "scratch_accuracies": scratch_accs,
        "art_accuracies": art_accs,
    }
    with open(os.path.join(args.results_dir, "fgsm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    table = wandb.Table(columns=["Epsilon", "Clean Acc", "FGSM Scratch Acc", "FGSM ART Acc"])
    for i, eps in enumerate(epsilons):
        table.add_data(eps, clean_acc, scratch_accs[i], art_accs[i])
    wandb.log({"fgsm_results_table": table})

    run.finish()
    print("\nFGSM attack comparison complete!")


if __name__ == "__main__":
    main()
