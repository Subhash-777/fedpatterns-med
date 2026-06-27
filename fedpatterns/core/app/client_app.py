from collections import OrderedDict
import random
import logging
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
from fedpatterns.patterns.decorator.differential_decorator import DifferentialDecorator
from fedpatterns.patterns.proxy.secure_aggregation_proxy import SecureAggregationProxy
from fedpatterns.patterns.observer.consent_observer import ConsentObserver

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

class PatternizedClient(NumPyClient):
    def __init__(self, partition_id: int, num_clients: int, dataset_name: str, num_classes: int, alpha: float, enable_straggler: bool) -> None:
        self.partition_id = partition_id
        self.num_clients = num_clients
        self.num_local_epochs = NUM_LOCAL_EPOCHS
        self.num_classes = num_classes
        self.enable_straggler = enable_straggler
        self.model = SmallCNN(num_classes=num_classes).to(DEVICE)
        
        # Override FED_DATASET so loader.py knows what to load
        import os
        os.environ["FED_DATASET"] = dataset_name
        self.train_loader = get_client_dataloader(
            partition_id=partition_id,
            num_clients=num_clients,
            alpha=alpha,
            split="train",
        )
        self.test_loader = get_test_dataloader()

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def train_loop(self, model, train_loader, optimizer, epochs, device) -> float:
        model.train()
        total_loss = 0.0
        use_amp = getattr(optimizer, "dp_enabled", False) is False and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        criterion = nn.BCEWithLogitsLoss()
        mask_data = getattr(self, "mask_data", False)
        max_batches = len(train_loader) // 2 if mask_data else len(train_loader)

        for _ in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                if i >= max_batches: break
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
        return total_loss

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        circuit_state = config.get("circuit_state", "CLOSED")
        if circuit_state == "OPEN":
            return self.get_parameters({}), 0, {"train_loss": 0.0, "is_straggler": False}
            
        set_seed(42 + self.partition_id + (round_num * 1000))
        rng = random.Random(42 + self.partition_id + (round_num * 1000))
        is_straggler = (rng.random() < 0.3) if self.enable_straggler else False
        
        self.set_parameters(parameters)
        optimizer = self.create_optimizer()
        total_loss = self.train_loop(self.model, self.train_loader, optimizer, self.num_local_epochs, DEVICE)

        mask_data = getattr(self, "mask_data", False)
        max_batches = len(self.train_loader) // 2 if mask_data else len(self.train_loader)
        num_batches = max_batches * self.num_local_epochs
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        num_examples = len(self.train_loader.dataset) // 2 if mask_data else len(self.train_loader.dataset)
        return self.get_parameters({}), num_examples, {"train_loss": avg_loss, "is_straggler": is_straggler}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        set_seed(42 + self.partition_id + (round_num * 1000))
        
        self.set_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()
        total_loss, total = 0.0, 0
        all_labels, all_outputs = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
                total += len(labels)

        avg_loss = total_loss / total if total > 0 else 0.0
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)
        
        if self.num_classes == 14:
            from sklearn.metrics import roc_auc_score
            try:
                metric = roc_auc_score(all_labels, all_outputs, average="macro")
            except ValueError:
                metric = 0.0
            return float(avg_loss), total, {"accuracy": metric}
        else:
            preds = (all_outputs > 0.0).astype(float)
            correct = (preds == all_labels).all(axis=1).sum()
            accuracy = correct / total if total > 0 else 0.0
            return float(avg_loss), total, {"accuracy": accuracy}

def client_fn(context: Context) -> Client:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    run_config = context.run_config
    dataset_name = run_config.get("fed_dataset", "PneumoniaMNIST").lower()
    num_classes = 14 if dataset_name == "chestmnist" else 1
    alpha = float(run_config.get("fed_alpha", "0.5"))
    enable_dp = run_config.get("fed_enable_dp", "True").lower() == "true"
    noise_multiplier = float(run_config.get("fed_noise_multiplier", "1.0"))
    enable_straggler = run_config.get("fed_enable_straggler", "True").lower() == "true"
    
    base_client = PatternizedClient(
        partition_id=partition_id, 
        num_clients=num_partitions,
        dataset_name=dataset_name,
        num_classes=num_classes,
        alpha=alpha,
        enable_straggler=enable_straggler
    )
    client = base_client
    if enable_dp:
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(base_client.model):
            base_client.model = ModuleValidator.fix(base_client.model)
        client = DifferentialDecorator(
            client, partition_id=partition_id,
            noise_multiplier=noise_multiplier, max_grad_norm=1.0, device=DEVICE
        )
        client = ConsentObserver(client, partition_id=partition_id, drop_probability=0.2)
        client = SecureAggregationProxy(client)
    return client.to_client()

app = ClientApp(client_fn=client_fn)
