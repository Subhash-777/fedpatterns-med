# P2 — FederatedStrategy

| Field | Value |
|---|---|
| **1. Name** | FederatedStrategy |
| **2. Intent** | Define a family of federated aggregation algorithms and make them interchangeable at runtime without modifying client code. |
| **3. Context** | Server-side aggregation where different rounds or experiments may require different algorithms (FedAvg, FedProx, FedNova). |
| **4. Problem** | Hardcoding FedAvg makes switching or comparing aggregation algorithms require code changes across the server and experiments. |
| **5. Forces** | Experiment reproducibility; algorithmic diversity; code reuse. |
| **6. Solution** | `FederatedAggregationStrategy` interface with pluggable concrete implementations loaded at runtime via config. |
| **7. Structure/UML** | See `[docs/uml/m2_patternized_architecture.md](file:///home/subhash/projects/fedpatterns-med/docs/uml/m2_patternized_architecture.md)` for Mermaid diagram. |
| **8. Participants** | FederatedStrategy (interface via `flwr.server.strategy.Strategy`), PatternizedFedAvg (concrete M2 impl), ServerApp. |
| **9. Implementation Notes** | `PatternizedFedAvg` implements `aggregate_fit` and `aggregate_evaluate` focusing exclusively on math (e.g., weighted averaging). Side-effects like logging are completely removed, as they are handled by the Mediator. |
| **10. Privacy/Compliance Mapping** | GDPR Art.25: Full. GDPR Art.32: Partial. HIPAA §164.312: Partial. DPDP: Partial. Residual Risk: Low. |
| **11. Benefits** | Algorithm hot-swap without code changes; enables A/B comparison in experiments. |
| **12. Liabilities** | Interface must be stable across all strategies; some algorithms may not fit the interface cleanly. |
| **13. Code Reference** | `fedpatterns/patterns/strategy/federated_strategy.py` |
| **14. Evaluation Metrics Impacted** | Accuracy, F1, AUC (algorithm-dependent); maintainability (CC). |
