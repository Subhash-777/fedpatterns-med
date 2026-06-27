"""
fedpatterns/resilience/straggler/straggler_manager.py  [P7 — StragglerCircuitBreaker]

P7: StragglerCircuitBreaker (Resilience pattern — Circuit Breaker)
"""

from enum import Enum, auto
from typing import Dict

class CircuitState(Enum):
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Client excluded (circuit tripped)
    HALF_OPEN = auto()  # Probing for recovery

class StragglerManager:
    """
    Tracks the circuit breaker state of each client (cid).
    """
    def __init__(self):
        self.client_states: Dict[str, CircuitState] = {}
        # Keep track of when a client transitioned to OPEN to enforce cool-off
        self.open_round_start: Dict[str, int] = {}
        
    def get_state(self, cid: str, current_round: int) -> CircuitState:
        """Get the current circuit state, handling transitions based on cool-off period."""
        state = self.client_states.get(cid, CircuitState.CLOSED)
        
        # If OPEN, check if 1-round cool-off is over (transition to HALF_OPEN)
        if state == CircuitState.OPEN:
            start_round = self.open_round_start.get(cid, 0)
            if current_round > start_round:  # 1 round lag
                self.client_states[cid] = CircuitState.HALF_OPEN
                state = CircuitState.HALF_OPEN
                
        return state

    def record_metrics(self, cid: str, is_straggler: bool, current_round: int) -> None:
        """Update circuit state based on reported straggler status from the current round."""
        state = self.get_state(cid, current_round)
        
        if state == CircuitState.CLOSED:
            if is_straggler:
                self.client_states[cid] = CircuitState.OPEN
                self.open_round_start[cid] = current_round
                
        elif state == CircuitState.HALF_OPEN:
            if is_straggler:
                # Failed probe, back to OPEN
                self.client_states[cid] = CircuitState.OPEN
                self.open_round_start[cid] = current_round
            else:
                # Success probe, back to CLOSED
                self.client_states[cid] = CircuitState.CLOSED
                
        # If already OPEN, we ignore metrics (they should be blank anyway)
