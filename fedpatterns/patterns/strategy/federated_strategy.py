"""
fedpatterns/patterns/strategy/federated_strategy.py

P2: FederatedStrategy

Defines the mathematical aggregation behavior of the server.
By decoupling the strategy from the mediator (which handles logging/observability),
we ensure that strategies remain pure, modular, and easy to swap
(e.g., FedAvg -> FedProx -> FedNova) without duplicating side-effect logic.
"""

from typing import Optional

from flwr.common import Parameters
from flwr.server.strategy import FedAvg

class PatternizedFedAvg(FedAvg):
    """
    A pure implementation of Federated Averaging.
    
    Inherits directly from Flower's FedAvg. Unlike the monolithic LoggingFedAvg 
    used in earlier baselines, this class is entirely devoid of side-effects 
    like MLflow logging, making it a strict manifestation of the Strategy pattern.
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
        initial_parameters: Optional[Parameters] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )
