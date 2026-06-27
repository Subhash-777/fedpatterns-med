# P1 — FederatedMediator

| Field | Value |
|---|---|
| **1. Name** | FederatedMediator |
| **2. Intent** | Encapsulate how federated nodes interact during training through a central coordinator, reducing coupling and making coordination policy explicit and replaceable. |
| **3. Context** | Multi-institutional federated learning where N hospitals/clients must synchronise model updates without direct peer-to-peer communication. |
| **4. Problem** | Direct client-to-server communication tightly couples each client to the server's coordination protocol, making protocol changes expensive and error-prone. |
| **5. Forces** | Scalability vs. coupling; auditability of coordination decisions; heterogeneous client capabilities. |
| **6. Solution** | Introduce a `FederatedMediator` that all clients communicate through. The mediator orchestrates round lifecycle, client selection, and message routing. |
| **7. Structure/UML** | See `[docs/uml/m2_patternized_architecture.md](file:///home/subhash/projects/fedpatterns-med/docs/uml/m2_patternized_architecture.md)` for Mermaid diagram. |
| **8. Participants** | FederatedMediator (concrete), PatternizedFedAvg (concrete strategy), ClientApp, ServerApp. |
| **9. Implementation Notes** | In M2: Wraps Flower's Strategy interface. Mediator handles MLflow logging hooks (`aggregate_fit`/`aggregate_evaluate`) and time tracking, then dispatches mathematical aggregation to the pure inner `FederatedStrategy`. |
| **10. Privacy/Compliance Mapping** | GDPR Art.25: Partial — coordinator does not hold raw data. GDPR Art.32: Partial. HIPAA §164.312: Partial. DPDP: Partial. Residual Risk: Medium. |
| **11. Benefits** | Decouples clients from coordination logic; enables hot-swap of coordination strategies; single point for logging and auditing. |
| **12. Liabilities** | Single point of failure if mediator crashes; bottleneck if coordination is CPU-heavy; adds one layer of indirection. |
| **13. Code Reference** | `fedpatterns/patterns/mediator/federated_mediator.py` |
| **14. Evaluation Metrics Impacted** | Maintainability (CC reduction), round coordination overhead (round_time_seconds). |
