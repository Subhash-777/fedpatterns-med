"""
baselines/b3_architectural_pattern_baseline/client_app.py

Flower 2.x ClientApp for B3 Architectural Pattern baseline.
Uses Vanilla FedAvg client logic.
"""

from collections import OrderedDict
import random

import numpy as np
import torch
import torch.nn as nn
from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from fedpatterns.core.domain.cnn_model import SmallCNN
from fedpatterns.datasets.medmnist.loader import get_client_dataloader, get_test_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1
NUM_LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

class VanillaClient(NumPyClient):
    def __init__(self, partition_id: int, num_clients: int = 5) -> None:
        self.partition_id = partition_id
        self.num_clients = num_clients
        self.model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
        self.train_loader = get_client_dataloader(
            partition_id=partition_id,
            num_clients=num_clients,
            split="train",
        )
        self.test_loader = get_test_dataloader()
        self.scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        set_seed(42 + self.partition_id + (round_num * 1000))
        
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        total_loss = 0.0
        for _ in range(NUM_LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()

        num_batches = len(self.train_loader) * NUM_LOCAL_EPOCHS
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return self.get_parameters({}), len(self.train_loader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        set_seed(42 + self.partition_id + (round_num * 1000))
        
        self.set_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
                total += len(labels)

        accuracy = correct / total if total > 0 else 0.0
        return float(total_loss / len(self.test_loader)), total, {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    return VanillaClient(partition_id=partition_id, num_clients=num_partitions).to_client()

app = ClientApp(client_fn=client_fn)
