# P7 — StragglerCircuitBreaker

| Field | Value |
|---|---|
| **1. Name** | StragglerCircuitBreaker |
| **2. Intent** | Protect FL rounds from slow/unresponsive clients using a circuit breaker that excludes persistent stragglers. |
| **3. Context** | Cross-institutional federated learning on medical imaging data. |
| **4. Problem** | *[PLACEHOLDER — to be filled in M2]* |
| **5. Forces** | *[PLACEHOLDER]* |
| **6. Solution** | *[PLACEHOLDER]* |
| **7. Structure/UML** | See `docs/uml/` (M2). |
| **8. Participants** | *[PLACEHOLDER]* |
| **9. Implementation Notes** | Straggler probability threshold is 0.30 (not the initially planned 0.20). The lower threshold yielded no straggler events within the 3-round simulation window under the project's deterministic seed scheme (42 + partition_id + round_num * 1000), requiring upward adjustment to guarantee circuit breaker activation for empirical verification. |
| **10. Privacy/Compliance Mapping** | GDPR Art.25: None. GDPR Art.32: *[TBD]*. HIPAA §164.312: *[TBD]*. DPDP: *[TBD]*. Residual Risk: Medium. |
| **11. Benefits** | *[PLACEHOLDER]* |
| **12. Liabilities** | *[PLACEHOLDER]* |
| **13. Code Reference** | `fedpatterns/resilience/straggler/straggler_manager.py` |
| **14. Evaluation Metrics Impacted** | *[PLACEHOLDER]* |
