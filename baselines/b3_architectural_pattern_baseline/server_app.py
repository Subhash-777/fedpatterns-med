"""
baselines/b3_architectural_pattern_baseline/server_app.py

Flower 2.x ServerApp for B3 Architectural Pattern baseline.
Re-implements Lo et al.'s system-level architectural patterns:
- Client Registry
- Client Selector
"""

import logging
import os
import time
from typing import Optional

import mlflow
import numpy as np
import random

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

from fedpatterns.core.domain.cnn_model import SmallCNN
from fedpatterns.utils.mlflow_utils import log_round_metrics

log = logging.getLogger(__name__)

NUM_CLASSES = 1
NUM_ROUNDS = 3
NUM_CLIENTS = 5
NO_DP_EPSILON = -1.0

_round_start_time: float = 0.0

class ClientRegistry:
    """Architectural Pattern: Client Registry (Lo et al.)"""
    def __init__(self):
        self._registry = {}
        
    def register_client(self, cid: str, capabilities: dict = None):
        self._registry[cid] = capabilities or {}
        
    def get_registered_clients(self) -> list[str]:
        return list(self._registry.keys())

class ClientSelector:
    """Architectural Pattern: Client Selector (Lo et al.)"""
    def __init__(self, registry: ClientRegistry):
        self.registry = registry
        
    def select_clients(self, client_manager, num_clients: int, server_round: int):
        """Selects clients using the ClientManager and updates registry."""
        clients = client_manager.sample(num_clients, min_num_clients=num_clients)
        for client in clients:
            self.registry.register_client(client.cid, {"last_round": server_round})
        return clients


class ArchitecturalFedAvg(FedAvg):
    """FedAvg that utilizes the Client Selector and Client Registry."""

    def __init__(self, run_id: Optional[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        self.model = SmallCNN(num_classes=NUM_CLASSES)
        
        # Architectural Patterns instantiation
        self.client_registry = ClientRegistry()
        self.client_selector = ClientSelector(self.client_registry)
        
        if run_id:
            log.info("[ArchitecturalFedAvg] MLflow run_id: %s", run_id)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Override to use ClientSelector
        config = {"server_round": server_round}
        from flwr.common import FitIns
        fit_ins = FitIns(parameters, config)
        
        clients = self.client_selector.select_clients(client_manager, self.min_fit_clients, server_round)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        config = {"server_round": server_round}
        from flwr.common import EvaluateIns
        evaluate_ins = EvaluateIns(parameters, config)
        
        clients = self.client_selector.select_clients(client_manager, self.min_evaluate_clients, server_round)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        global _round_start_time

        results = sorted(results, key=lambda x: x[0].cid)

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        elapsed = time.perf_counter() - _round_start_time
        num_params = sum(p.numel() for p in self.model.parameters())
        comm_bytes = num_params * 4 * 2 * NUM_CLIENTS

        train_losses = [
            (res.num_examples, res.metrics.get("train_loss", 0.0))
            for _, res in results if res.metrics
        ]
        
        total_examples = sum(num for num, _ in train_losses)
        if total_examples > 0:
            avg_train_loss = float(sum(num * loss for num, loss in train_losses) / total_examples)
        else:
            avg_train_loss = 0.0

        metrics = {
            "train_loss": avg_train_loss,
            "round_time_seconds": elapsed,
            "total_comm_bytes": float(comm_bytes),
            "epsilon": NO_DP_EPSILON,
            "noise_multiplier": 0.0,
            "compliance_coverage": 0.0,
            "straggler_rate": 0.0,
            "rollback_count": 0.0,
        }

        if self.run_id:
            try:
                log_round_metrics(metrics, step=server_round, run_id=self.run_id)
            except Exception as exc:
                log.error("[ArchitecturalFedAvg] Failed to log fit metrics: %s", exc)

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        global _round_start_time
        _round_start_time = time.perf_counter()

        results = sorted(results, key=lambda x: x[0].cid)

        aggregated_loss, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracies = [
            res.metrics.get("accuracy", 0.0)
            for _, res in results
            if res.metrics
        ]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

        eval_metrics = {
            "val_loss": float(aggregated_loss) if aggregated_loss is not None else 0.0,
            "accuracy": avg_accuracy,
        }

        if self.run_id:
            try:
                log_round_metrics(eval_metrics, step=server_round, run_id=self.run_id)
            except Exception as exc:
                log.error("[ArchitecturalFedAvg] Failed to log eval metrics: %s", exc)

        return aggregated_loss, metrics_aggregated


def server_fn(context) -> ServerAppComponents:
    tracking_uri = context.run_config.get("mlflow-tracking-uri", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = context.run_config.get("mlflow-run-id", "")
    run_id = run_id if run_id else None

    model = SmallCNN(num_classes=NUM_CLASSES)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )

    strategy = ArchitecturalFedAvg(
        run_id=run_id,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
