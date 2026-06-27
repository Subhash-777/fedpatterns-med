"""
fedpatterns/patterns/mediator/federated_mediator.py

P1: FederatedMediator

The FederatedMediator coordinates the federated learning lifecycle on the server side.
It implements Flower's Strategy interface to plug directly into ServerApp, but delegates
pure aggregation mathematics to an inner FederatedStrategy (e.g., FedAvg).

Responsibilities:
  - Orchestration of the FL round
  - MLflow observability and metric logging
  - Hook points for resilience (rollback, circuit breakers)
  - Hook points for privacy (DP budget tracking, consent observation)
"""

import logging
import time
from typing import Optional

import mlflow
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fedpatterns.core.domain.cnn_model import SmallCNN
from fedpatterns.utils.mlflow_utils import log_round_metrics

# M4 Patterns
from fedpatterns.patterns.prototype.versioned_model_prototype import VersionedModelPrototype, ModelMemento
from fedpatterns.resilience.straggler.straggler_manager import StragglerManager, CircuitState
from fedpatterns.patterns.command.round_command_history import RoundCommandHistory, RoundCommand

log = logging.getLogger(__name__)

# Constants
NUM_CLASSES = 1
NUM_CLIENTS = 5
NO_DP_EPSILON = -1.0


class FederatedMediator(Strategy):
    """
    Mediates interaction between Flower ServerApp, MLflow observability,
    and mathematical aggregation strategies.
    """

    def __init__(self, strategy: Strategy, run_id: Optional[str] = None) -> None:
        """
        Args:
            strategy: The underlying aggregation strategy (e.g., FedAvg).
            run_id: Optional MLflow run ID for logging.
        """
        self.strategy = strategy
        self.run_id = run_id
        self.model = SmallCNN(num_classes=NUM_CLASSES)
        self._round_start_time: float = time.perf_counter()
        
        # P6: VersionedModelPrototype
        self.versioned_prototype = VersionedModelPrototype(run_id=run_id or "default")
        self.best_memento: Optional[ModelMemento] = None
        self.rollback_count = 0
        self._pending_rollback_parameters: Optional[Parameters] = None
        self._latest_aggregated_parameters: Optional[Parameters] = None
        
        # P7: StragglerCircuitBreaker
        self.straggler_manager = StragglerManager()
        
        # P8: RoundCommandHistory
        self.command_history = RoundCommandHistory(run_id=run_id or "default")
        self._last_train_metrics: dict[str, float] = {}

        if run_id:
            log.info("[FederatedMediator] Bound to MLflow run_id: %s", run_id)
        else:
            log.warning("[FederatedMediator] No MLflow run_id provided.")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        client_instructions = self.strategy.configure_fit(server_round, parameters, client_manager)
        
        # Circuit state is set based on previous round's is_straggler signals.
        # Round 1: all clients CLOSED (no prior signal).
        # straggler_rate > 0 will only appear from round 2 onwards.
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            state = self.straggler_manager.get_state(client_proxy.cid, server_round)
            fit_ins.config["circuit_state"] = state.name
            
            # P6: If a rollback was triggered in the previous evaluation phase,
            # override the server-provided parameters with the rolled-back parameters
            if self._pending_rollback_parameters:
                fit_ins.parameters = self._pending_rollback_parameters
                
            updated_instructions.append((client_proxy, fit_ins))
            
        if self._pending_rollback_parameters:
            self._pending_rollback_parameters = None
            
        return updated_instructions

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        # We also need to distribute the rolled-back parameters for evaluation if a rollback just happened
        # However, configure_evaluate runs right after aggregate_fit (which produces the NEW parameters).
        # We only set _pending_rollback_parameters in aggregate_evaluate, so it won't affect configure_evaluate
        # until the next round. This is correct because we want to evaluate the newly aggregated model first.
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        
        # Sort results by CID to guarantee deterministic floating-point accumulation order
        results = sorted(results, key=lambda x: x[0].cid)
        
        active_results = []
        for proxy, res in results:
            is_straggler = res.metrics.get("is_straggler", False) if res.metrics else False
            self.straggler_manager.record_metrics(proxy.cid, is_straggler, server_round)
            
            # P7: Explicitly filter out OPEN-state payloads (num_examples == 0)
            if res.num_examples > 0:
                active_results.append((proxy, res))
                
        # Guard: if all results have num_examples=0, skip aggregation this round
        total_examples = sum(res.num_examples for _, res in active_results)
        if total_examples == 0:
            log.warning("All clients in OPEN state — skipping aggregation round %d", server_round)
            if self.best_memento:
                # Sourced directly from the Memento, strongly coupling P6 and P7
                last_good_parameters = self.versioned_prototype.restore_memento(self.best_memento)
            else:
                last_good_parameters = Parameters(tensors=[], tensor_type="")
            return last_good_parameters, {}

        # 1. Delegate math aggregation to the inner strategy with filtered active_results
        aggregated_parameters, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, active_results, failures
        )
        
        # Save the aggregated parameters so they can be snapshotted in aggregate_evaluate if they improve val_loss
        self._latest_aggregated_parameters = aggregated_parameters

        # 2. Extract and log metrics
        elapsed = time.perf_counter() - self._round_start_time
        num_params = sum(p.numel() for p in self.model.parameters())
        comm_bytes = num_params * 4 * 2 * NUM_CLIENTS

        train_losses = [
            (res.num_examples, res.metrics.get("train_loss", 0.0))
            for _, res in active_results if res.metrics
        ]
        total_examples = sum(num for num, _ in train_losses)
        avg_train_loss = float(sum(num * loss for num, loss in train_losses) / total_examples) if total_examples > 0 else 0.0

        epsilons = [
            res.metrics.get("epsilon", NO_DP_EPSILON)
            for _, res in results if res.metrics and "epsilon" in res.metrics
        ]
        valid_epsilons = [e for e in epsilons if e != NO_DP_EPSILON]
        max_epsilon = max(valid_epsilons) if valid_epsilons else NO_DP_EPSILON

        noise_multipliers = [
            res.metrics.get("noise_multiplier", 0.0)
            for _, res in results if res.metrics and "noise_multiplier" in res.metrics
        ]
        max_noise_multiplier = max(noise_multipliers) if noise_multipliers else 0.0
        
        open_or_half_open = sum(
            1 for cid in self.straggler_manager.client_states.keys()
            if self.straggler_manager.get_state(cid, server_round) in [CircuitState.OPEN, CircuitState.HALF_OPEN]
        )
        straggler_rate = float(open_or_half_open / NUM_CLIENTS)

        metrics = {
            "train_loss": avg_train_loss,
            "round_time_seconds": elapsed,
            "total_comm_bytes": float(comm_bytes),
            "epsilon": max_epsilon,
            "noise_multiplier": max_noise_multiplier,
            "compliance_coverage": 0.0,
            "straggler_rate": straggler_rate,
            "rollback_count": float(self.rollback_count),
        }
        
        self._last_train_metrics = {
            "train_loss": avg_train_loss,
            "straggler_rate": straggler_rate,
            "epsilon": max_epsilon,
        }

        if self.run_id:
            try:
                log_round_metrics(metrics, step=server_round, run_id=self.run_id)
                log.info("[FederatedMediator] Round %d fit metrics logged.", server_round)
            except Exception as exc:
                log.error("[FederatedMediator] Failed to log fit metrics: %s", exc)

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        
        # Reset timer for the next round
        self._round_start_time = time.perf_counter()

        # Sort results by CID to guarantee deterministic floating-point accumulation order
        results = sorted(results, key=lambda x: x[0].cid)

        # 1. Delegate math aggregation
        aggregated_loss, metrics_aggregated = self.strategy.aggregate_evaluate(
            server_round, results, failures
        )
        
        val_loss = float(aggregated_loss) if aggregated_loss is not None else 0.0
        
        # P6: Evaluate Best-So-Far Semantics
        if self.best_memento is None or val_loss < self.best_memento.val_loss:
            if self._latest_aggregated_parameters is not None:
                self.best_memento = self.versioned_prototype.save_memento(
                    server_round, val_loss, self._latest_aggregated_parameters
                )
        elif self.best_memento and val_loss > self.best_memento.val_loss + 0.02:
            log.warning("[FederatedMediator] Rollback triggered! val_loss %.4f > best-so-far %.4f + 0.02", val_loss, self.best_memento.val_loss)
            self.rollback_count += 1
            self._pending_rollback_parameters = self.versioned_prototype.restore_memento(self.best_memento)

        # 2. Extract and log eval metrics
        accuracies = [
            res.metrics.get("accuracy", 0.0)
            for _, res in results
            if res.metrics
        ]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

        eval_metrics = {
            "val_loss": val_loss,
            "accuracy": avg_accuracy,
        }

        if self.run_id:
            try:
                log_round_metrics(eval_metrics, step=server_round, run_id=self.run_id)
                log.info("[FederatedMediator] Round %d eval metrics logged.", server_round)
            except Exception as exc:
                log.error("[FederatedMediator] Failed to log eval metrics: %s", exc)

        # P8: Construct and execute the round command
        rollback_triggered = (val_loss > self.best_memento.val_loss + 0.02) if (self.best_memento and val_loss > self.best_memento.val_loss + 0.02) else False
        command = RoundCommand(
            server_round=server_round,
            val_loss=val_loss,
            val_accuracy=avg_accuracy,
            train_loss=self._last_train_metrics.get("train_loss", 0.0),
            straggler_rate=self._last_train_metrics.get("straggler_rate", 0.0),
            rollback_triggered=rollback_triggered,
            epsilon=self._last_train_metrics.get("epsilon", NO_DP_EPSILON),
            metrics={"total_comm_bytes": float(sum(p.numel() for p in self.model.parameters()) * 4 * 2 * NUM_CLIENTS)}
        )
        self.command_history.add_command(command)

        return aggregated_loss, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters):
        return self.strategy.evaluate(server_round, parameters)
