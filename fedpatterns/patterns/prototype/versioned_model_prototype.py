"""
fedpatterns/patterns/prototype/versioned_model_prototype.py  [P6 — VersionedModelPrototype]

P6: VersionedModelPrototype (GoF Prototype + Memento)
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

from flwr.common import Parameters

@dataclass
class ModelMemento:
    """
    Memento capturing the internal state of the model parameters at a specific round,
    along with its associated validation loss to support best-so-far tracking.
    """
    round_num: int
    val_loss: float
    parameters: Parameters
    filepath: str

class VersionedModelPrototype:
    """
    Originator and Prototype for the global federated model.
    Creates ModelMementos and can restore its state from them.
    Saves snapshots to disk to survive ephemeral process crashes.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.checkpoint_dir = "/tmp"  # Use /tmp for ephemeral storage, cleaned by run_m4.py

    def save_memento(self, round_num: int, val_loss: float, parameters: Parameters) -> ModelMemento:
        """Create a deep copy of the current parameters as a new memento and persist to disk."""
        filepath = os.path.join(self.checkpoint_dir, f"fedpatterns_memento_{self.run_id}_r{round_num}.pt")
        
        memento = ModelMemento(
            round_num=round_num,
            val_loss=val_loss,
            parameters=parameters,  # Parameters object has tensors explicitly serialized as bytes
            filepath=filepath
        )
        
        with open(filepath, "wb") as f:
            pickle.dump(memento, f)
            
        return memento

    def restore_memento(self, memento: ModelMemento) -> Parameters:
        """Restore model weights from a named snapshot."""
        if not os.path.exists(memento.filepath):
            raise FileNotFoundError(f"Memento file not found: {memento.filepath}")
            
        with open(memento.filepath, "rb") as f:
            loaded_memento: ModelMemento = pickle.load(f)
            
        return loaded_memento.parameters
