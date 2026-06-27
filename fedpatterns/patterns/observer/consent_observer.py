"""
fedpatterns/patterns/observer/consent_observer.py

P5: ConsentObserver

An Observer pattern implementation that listens to patient consent events.
To satisfy Flower's rigid `min_fit_clients` quorum without crashing the server, 
this observer simulates consent revocation by deterministically masking 
50% of the client's dataset (or reducing total examples to 0 effectively) 
during the `fit` method.
"""

import logging
import numpy as np
from flwr.client import NumPyClient

log = logging.getLogger(__name__)


class ConsentObserver(NumPyClient):
    """
    Observer pattern wrapping a NumPyClient to intercept the `fit` hook
    and deterministically simulate partial/full data consent revocation.
    """

    def __init__(self, client: NumPyClient, partition_id: int, drop_probability: float = 0.2) -> None:
        self._client = client
        self.partition_id = partition_id
        self.drop_probability = drop_probability

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return self._client.get_parameters(config)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self._client.set_parameters(parameters)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        round_num = config.get("server_round", 1)
        
        # Deterministic simulation of a consent event
        seed = 42 + self.partition_id + (round_num * 1000)
        rng = np.random.default_rng(seed)
        
        is_revoked = rng.random() < self.drop_probability
        
        curr = self._client
        while hasattr(curr, '_client'):
            curr = curr._client
            
        if is_revoked:
            log.warning(
                "[ConsentObserver] Client %d consent REVOKED in round %d. Masking 50%% of local dataset.", 
                self.partition_id, round_num
            )
            curr.mask_data = True
        else:
            curr.mask_data = False
            
        return self._client.fit(parameters, config)

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        return self._client.evaluate(parameters, config)
