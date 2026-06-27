"""
fedpatterns/patterns/decorator/differential_decorator.py

P3: DifferentialDecorator

Wraps a Flower NumPyClient to dynamically inject Differential Privacy (DP)
noise during the local training phase using Opacus, without modifying the
inner client's core aggregation logic.
"""

import logging
import torch
import numpy as np
from flwr.client import NumPyClient
from opacus import PrivacyEngine

log = logging.getLogger(__name__)


class DifferentialDecorator(NumPyClient):
    """
    Decorator pattern that intercepts the `fit` method to attach an
    Opacus PrivacyEngine to the inner client's PyTorch components.
    """

    def __init__(
        self,
        client: NumPyClient,
        partition_id: int,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._client = client
        self.partition_id = partition_id
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize PrivacyEngine once so the RDP accountant accumulates epsilon across rounds
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.optimizer = self._client.create_optimizer()
        
        # Seed the noise generator once per client to ensure a reproducible continuous noise stream
        seed = 42 + self.partition_id
        self.noise_generator = torch.Generator(device=self.device).manual_seed(seed)
        
        self.dp_model, self.dp_optimizer, self.dp_train_loader = self.privacy_engine.make_private(
            module=self._client.model,
            optimizer=self.optimizer,
            data_loader=self._client.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            noise_generator=self.noise_generator
        )
        
        # In Flower simulation, clients are ephemeral and recreated every round.
        # We load the accountant history from disk so epsilon correctly accumulates.
        import os
        self.history_file = f"/tmp/fedpatterns_dp_history_{self.partition_id}.pt"
        if os.path.exists(self.history_file):
            self.privacy_engine.accountant.history = torch.load(self.history_file)
            
        # Signal inner loop to avoid unsupported AMP operations if DP is enabled
        setattr(self.dp_optimizer, "dp_enabled", True)

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return self._client.get_parameters(config)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self._client.set_parameters(parameters)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        # P7: Check if circuit is OPEN from server
        circuit_state = config.get("circuit_state", "CLOSED")
        if circuit_state == "OPEN":
            # Return empty parameters and 0 weight
            return self.get_parameters({}), 0, {"train_loss": 0.0, "is_straggler": False, "epsilon": -1.0, "noise_multiplier": 0.0}

        # Deterministic seeding for global PyTorch RNG
        round_num = config.get("server_round", 1)
        seed = 42 + self.partition_id + (round_num * 1000)
        from fedpatterns.core.app.client_app import set_seed
        set_seed(seed)
        self.noise_generator.manual_seed(seed)

        # P7: Deterministically simulate straggler flag (30% probability)
        import random
        import os
        enable_straggler = os.environ.get("FED_ENABLE_STRAGGLER", "True").lower() == "true"
        rng = random.Random(seed)
        is_straggler = (rng.random() < 0.3) if enable_straggler else False

        self.set_parameters(parameters)
        
        # Clear optimizer state to simulate a fresh optimizer per round (standard FedAvg behavior)
        self.dp_optimizer.state.clear()

        # Delegate the actual training execution back to the inner client's logic
        epochs = getattr(self._client, "num_local_epochs", 1)
        total_loss = self._client.train_loop(self.dp_model, self.dp_train_loader, self.dp_optimizer, epochs, self.device)
        
        # Extract accumulated epsilon for logging
        try:
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        except Exception:
            epsilon = -1.0
            
        num_batches = len(self.dp_train_loader) * epochs
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Extract updated weights from the base module
        self._client.model.load_state_dict(self.dp_model._module.state_dict())
        
        # Save accountant history so it persists for the next round
        torch.save(self.privacy_engine.accountant.history, self.history_file)
        
        out_params = self.get_parameters({})
        if is_straggler:
            # Simulate a corrupt straggler update to guarantee performance degradation
            out_params = [np.zeros_like(p) for p in out_params]
        
        return out_params, len(self.dp_train_loader.dataset), {
            "train_loss": avg_loss, 
            "epsilon": epsilon,
            "noise_multiplier": self.noise_multiplier,
            "is_straggler": is_straggler
        }

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        return self._client.evaluate(parameters, config)
