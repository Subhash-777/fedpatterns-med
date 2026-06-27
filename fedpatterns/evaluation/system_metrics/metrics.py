"""
fedpatterns/evaluation/system_metrics/metrics.py

FL system-level metrics: round timing and communication cost.
"""

import time
from dataclasses import dataclass, field


@dataclass
class RoundTimer:
    """Context manager for timing a single FL round."""
    _start: float = field(default=0.0, init=False, repr=False)
    elapsed_seconds: float = field(default=0.0, init=False)

    def __enter__(self) -> "RoundTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start


def estimate_communication_bytes(model_num_params: int, dtype_bytes: int = 4) -> int:
    """
    Estimate bytes sent/received per client per round (upload + download).

    Args:
        model_num_params: Total number of model parameters.
        dtype_bytes:      Bytes per parameter (4 for float32, 2 for float16).

    Returns:
        Total bytes for one client (upload + download).
    """
    params_bytes = model_num_params * dtype_bytes
    return 2 * params_bytes  # upload (gradients) + download (global model)


def build_system_metrics(
    round_num: int,
    elapsed_seconds: float,
    model_num_params: int,
    num_clients: int = 5,
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """
    Build system metrics dict for one round.

    Returns:
        Dict with keys: round_time_seconds, comm_bytes_per_client, total_comm_bytes.
    """
    comm_per_client = estimate_communication_bytes(model_num_params, dtype_bytes)
    return {
        "round_time_seconds": elapsed_seconds,
        "comm_bytes_per_client": float(comm_per_client),
        "total_comm_bytes": float(comm_per_client * num_clients),
    }
