"""
Q1: ViT-S Model with LoRA injection for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
import timm
from peft import get_peft_model, LoraConfig, TaskType


def create_vit_model(num_classes: int = 100, pretrained: bool = True):
    """Create a ViT-Small model pretrained on ImageNet with a new classification head."""
    model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def apply_lora(model, rank: int, alpha: int, dropout: float = 0.1):
    """
    Apply LoRA to attention Q, K, V weights using PEFT.
    Returns the PEFT-wrapped model.
    """
    # Target the attention qkv projection in ViT
    target_modules = ["attn.qkv"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    return peft_model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model):
    """Freeze all parameters except the classification head."""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False


def get_model_for_experiment(rank=None, alpha=None, dropout=0.1, num_classes=100):
    """
    Create model for an experiment.
    If rank is None, returns baseline (no LoRA, only head trainable).
    Otherwise, applies LoRA with given hyperparameters.
    """
    model = create_vit_model(num_classes=num_classes, pretrained=True)

    if rank is not None and alpha is not None:
        # Freeze backbone, then apply LoRA (which makes LoRA params trainable)
        freeze_backbone(model)
        model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
        # PEFT freezes all non-LoRA params; unfreeze the classification head
        # as required: "keeping the classification head trainable for 100 classes"
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True
    else:
        # Baseline: freeze everything except the head
        freeze_backbone(model)

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

    return model
