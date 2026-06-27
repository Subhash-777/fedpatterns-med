"""
fedpatterns/datasets/medmnist/loader.py

PneumoniaMNIST data loader with Dirichlet non-IID partitioning.
Returns per-client DataLoaders for use inside ClientApp.

Designed to be multi-dataset-compatible:
  - M1: PneumoniaMNIST (binary, 28×28 grayscale)
  - M5: NIH ChestX-ray14 (multi-label, 14 classes) — loader.py to be updated
  - M6: ISIC 2020 (binary, colour) — separate loader
"""

import pathlib
from typing import Optional

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# ── Constants ──────────────────────────────────────────────────────────────────

import os
DATA_CACHE = pathlib.Path.home() / ".medmnist"


def get_transforms(train: bool = True) -> transforms.Compose:
    base = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    if train:
        base = [transforms.RandomHorizontalFlip()] + base
    return transforms.Compose(base)


def get_full_dataset(split: str = "train") -> medmnist.PneumoniaMNIST:
    """Download/cache and return the full PneumoniaMNIST split."""
    dataset_name = os.environ.get("FED_DATASET", "PneumoniaMNIST").lower()
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass(
        split=split,
        transform=get_transforms(train=(split == "train")),
        download=True,
        root=str(DATA_CACHE),
    )


def dirichlet_partition(
    dataset: medmnist.PneumoniaMNIST,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices across clients using Dirichlet distribution.

    Args:
        dataset:     Full MedMNIST dataset.
        num_clients: Number of federated clients.
        alpha:       Dirichlet concentration parameter (lower = more heterogeneous).
        seed:        RNG seed for reproducibility.

    Returns:
        List of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([int(s[0]) for s in dataset.labels])
    num_classes = len(np.unique(labels))

    # Collect indices per class
    class_indices: dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(num_classes)
    }

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c_indices in class_indices.values():
        rng.shuffle(c_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (proportions * len(c_indices)).astype(int)
        # Ensure all samples are assigned
        proportions[-1] = len(c_indices) - proportions[:-1].sum()
        start = 0
        for client_id, n in enumerate(proportions):
            client_indices[client_id].extend(c_indices[start : start + n].tolist())
            start += n

    return client_indices


def get_client_dataloader(
    partition_id: int,
    num_clients: int = 5,
    alpha: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
    split: str = "train",
) -> DataLoader:
    """
    Return a DataLoader for a specific client partition.

    Args:
        partition_id: Client index (0-indexed).
        num_clients:  Total number of clients (default 5).
        alpha:        Dirichlet alpha (default 0.5).
        batch_size:   Mini-batch size.
        seed:         Reproducibility seed.
        split:        'train' or 'val' or 'test'.
    """
    dataset = get_full_dataset(split=split)
    partitions = dirichlet_partition(dataset, num_clients=num_clients, alpha=alpha, seed=seed)
    client_subset = Subset(dataset, partitions[partition_id])
    shuffle = split == "train"
    # pin_memory=False: Ray simulation spawns worker processes without direct CUDA context
    return DataLoader(client_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def get_test_dataloader(batch_size: int = 64) -> DataLoader:
    """Return the global test DataLoader (used by server-side evaluation)."""
    dataset = get_full_dataset(split="test")
    # pin_memory=False: Ray simulation spawns worker processes without direct CUDA context
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
