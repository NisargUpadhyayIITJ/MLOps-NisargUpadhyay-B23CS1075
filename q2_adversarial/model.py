"""
Q2: ResNet model definitions for CIFAR-10 adversarial experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_resnet18_cifar10(num_classes=10, pretrained=False):
    """Create a ResNet-18 for CIFAR-10 (non-pretrained, from scratch)."""
    model = models.resnet18(pretrained=False)
    # Modify for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_resnet34_detector(num_classes=2):
    """Create a ResNet-34 for adversarial detection (binary classification)."""
    model = models.resnet34(pretrained=False)
    # Modify for CIFAR-10 sized inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
