import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class RoundCommand:
    """
    Command Pattern (ConcreteCommand): Encapsulates all data and context 
    associated with the completion of a single federated learning round.
    """
    server_round: int
    val_loss: float
    val_accuracy: float
    train_loss: float
    straggler_rate: float
    rollback_triggered: bool
    epsilon: float
    metrics: Dict[str, Any]
    
    def execute(self, audit_log_path: str) -> None:
        """
        Executes the command by persisting it to the audit log.
        This provides a permanent, append-only record for compliance (GDPR Art. 30).
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        
        # Append as JSONL for crash safety
        with open(audit_log_path, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")


class RoundCommandHistory:
    """
    Command Pattern (Invoker): Maintains the history of commands executed
    during the FL session and manages the persistence path.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        # Use output/logs/ for persistent compliance records, not /tmp
        # flwr run isolates apps in ~/.flwr/apps/, so __file__ resolution is fragile.
        # Default to the known absolute path if PROJECT_ROOT env var is missing.
        project_root = os.environ.get("PROJECT_ROOT", "/home/subhash/projects/fedpatterns-med")
        self.audit_log_path = os.path.join(project_root, "output", "logs", f"audit_{run_id}.jsonl")
        self.history: List[RoundCommand] = []

    def add_command(self, command: RoundCommand) -> None:
        """
        Logs a new command to history and immediately executes it for persistence.
        """
        self.history.append(command)
        command.execute(self.audit_log_path)
