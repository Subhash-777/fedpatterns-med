# P3 — DifferentialDecorator

| Field | Value |
|---|---|
| **1. Name** | DifferentialDecorator |
| **2. Intent** | Transparently wrap client local training to inject DP noise without modifying the model or training loop. |
| **3. Context** | Cross-institutional federated learning on medical imaging data. |
| **4. Problem** | *[PLACEHOLDER — to be filled in M2]* |
| **5. Forces** | *[PLACEHOLDER]* |
| **6. Solution** | *[PLACEHOLDER]* |
| **7. Structure/UML** | See `docs/uml/` (M2). |
| **8. Participants** | *[PLACEHOLDER]* |
| **9. Implementation Notes** | Opacus PrivacyEngine is re-instantiated per round due to Flower's ephemeral Ray worker model; cumulative RDP accounting is preserved by checkpointing accountant.history to disk between rounds. **Crucially, if a P6 rollback occurs, model weights are restored but epsilon accounting continues monotonically—privacy budget is "spent" on the computation and cannot be refunded.** |
| **10. Privacy/Compliance Mapping** | GDPR Art.25: Full. GDPR Art.32: *[TBD]*. HIPAA §164.312: *[TBD]*. DPDP: *[TBD]*. Residual Risk: Low. |
| **11. Benefits** | *[PLACEHOLDER]* |
| **12. Liabilities** | *[PLACEHOLDER]* |
| **13. Code Reference** | `fedpatterns/patterns/decorator/differential_decorator.py` |
| **14. Evaluation Metrics Impacted** | *[PLACEHOLDER]* |
