"""
fedpatterns/core/domain/cnn_model.py

Small CNN for MedMNIST 28×28 grayscale images.
Designed for PneumoniaMNIST (binary) in M1, and compatible with multi-label
tasks (e.g. NIH ChestX-ray14) by adjusting num_classes and the output layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Lightweight CNN for 28×28 grayscale medical images.

    Architecture: Conv→BN→Pool → Conv→BN→Pool → FC → FC → output
    Designed to be fast for federated simulation on constrained GPU (RTX 3050 4GB).

    Args:
        num_classes: Number of output classes.
                     Set to 1 for binary (sigmoid) or >1 for multi-class/multi-label.
        in_channels:  Input channels. Default 1 for grayscale MedMNIST.
    """

    def __init__(self, num_classes: int = 1, in_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # 28 → 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # 14 → 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def get_weights(self) -> list[torch.Tensor]:
        """Return model parameters as a list of tensors (for Flower NDArrays)."""
        return [param.data.clone() for param in self.parameters()]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        """Set model parameters from a list of tensors."""
        for param, weight in zip(self.parameters(), weights):
            param.data.copy_(weight)
