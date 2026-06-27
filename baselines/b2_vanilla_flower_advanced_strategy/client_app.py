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

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

class B2Client(NumPyClient):
    def __init__(self, partition_id: int, num_clients: int, dataset_name: str, num_classes: int, alpha: float) -> None:
        self.partition_id = partition_id
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.model = SmallCNN(num_classes=num_classes).to(DEVICE)
        
        import os
        os.environ["FED_DATASET"] = dataset_name
        self.train_loader = get_client_dataloader(
            partition_id=partition_id,
            num_clients=num_clients,
            alpha=alpha,
            split="train",
        )
        self.test_loader = get_test_dataloader()
        self.global_parameters = None

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        mu = config.get("proximal_mu", 0.0)
        set_seed(42 + self.partition_id + (round_num * 1000))
        
        # We only want trainable parameters, so we clone them from the model AFTER setting parameters
        self.set_parameters(parameters)
        self.global_parameters = [p.clone().detach() for p in self.model.parameters()]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        total_loss = 0.0
        use_amp = DEVICE.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _ in range(NUM_LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.model(images)
                    base_loss = criterion(outputs, labels)
                    proximal_term = 0.0
                    if mu > 0.0 and self.global_parameters is not None:
                        for local_param, global_param in zip(self.model.parameters(), self.global_parameters):
                            proximal_term += torch.square(torch.norm(local_param - global_param))
                    loss = base_loss + (mu / 2) * proximal_term

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
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
    
    return B2Client(
        partition_id=partition_id, 
        num_clients=num_partitions,
        dataset_name=dataset_name,
        num_classes=num_classes,
        alpha=alpha
    ).to_client()

app = ClientApp(client_fn=client_fn)
