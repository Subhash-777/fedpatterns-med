# FedPatterns-Med — Implementation Blueprint (`plan.md`)

> Verified and consolidated implementation plan for journal publication.
> Built for execution in **Antigravity IDE**.

---

## 1. Project Identity

**Working title**
FedPatterns: A Design Pattern Language for Privacy-Compliant Federated Learning Architectures with Empirical Validation on Medical Imaging Systems

**Refined journal title**
FedPatterns-Med: OO Design Pattern Compositions with Privacy-Compliance Mapping for Federated Learning in Medical Imaging — An Empirical Study

**Core idea**
Build a real federated learning (FL) system for medical imaging, but make the central research contribution a **code-level (GoF/OO) software design pattern language** for FL implementation — not another FL accuracy paper.

**One-paragraph pitch**
> Existing FL pattern research stops at the *system-architecture* level (how components connect). FedPatterns-Med goes one level deeper — into how FL components are *internally structured* — and is the first work to pair GoF-level design patterns with a formal, per-pattern privacy-regulation compliance mapping (GDPR, HIPAA, India DPDP), validated empirically on real medical imaging data using both ML metrics and software-engineering quality metrics.

---

## 2. Problem Statement

Federated learning is attractive in healthcare because institutions can collaboratively train models without centralizing raw patient data. But the literature splits into two disconnected tracks:

- **ML-track papers** — focus on accuracy, aggregation algorithms (FedAvg/FedProx/FedNova), and privacy *techniques* (differential privacy, secure aggregation) in isolation.
- **SE-track papers** — describe FL *system architecture* (how the client/server/aggregator components are arranged) but stop short of code-level design and never connect architecture to regulatory compliance.

**Research question**
How can a pattern-oriented implementation blueprint improve privacy compliance, maintainability, extensibility, and operational robustness of federated learning systems for medical imaging, while remaining competitive in predictive performance against standard FL baselines?

---

## 3. Verified Related Work (checked against primary sources)

This section was independently fact-checked — every entry below was confirmed to be a real, traceable publication before being used as a novelty anchor.

| Ref | Year | Venue | What it actually did | What it does NOT cover |
|---|---|---|---|---|
| Lo, Lu, Zhu, Paik, Xu, Wang — *"Architectural Patterns for the Design of Federated Learning Systems"* (arXiv:2101.02373) | 2021/2022 | Journal of Systems and Software (JSS) | Derived **15 architectural patterns** via systematic literature review: 3 client management, 4 model management, 3 model training, 4 model aggregation, 1 configuration pattern | No GoF/OO code-level patterns, no privacy/compliance mapping, no medical-imaging validation |
| Lo et al. — FL pattern **decision model** | 2022 | Follow-up to above | Decision trees to help architects pick *which* architectural pattern to use | Still architecture-level only; no compliance mapping; no empirical medical validation |
| Trubiani et al. — *"Experimenting Architectural Patterns in Federated Learning Systems"* | 2025 | JSS / ScienceDirect (confirm exact venue before final citation — appears as a JSS article; verify if a companion conference version exists before citing as "ICSA 2025") | Implemented and quantitatively benchmarked **6 of Lo et al.'s architectural patterns** using the Flower framework | Only architectural patterns, performance-only metrics, no privacy/compliance layer, no medical imaging dataset |
| *"Architectural patterns for blockchain-enabled federated learning"* (SLR) | 2025 | ScienceDirect | Formalized 12 architectural patterns for blockchain+FL systems | Architecture-level only; blockchain-specific; no GoF/compliance angle |
| Various FL + GDPR/HIPAA papers (privacy surveys, DP/HE technique papers, consent-blockchain systems) | 2023–2026 | Multiple | Treat privacy as an **algorithmic technique** (DP noise, homomorphic encryption, consent smart contracts) or a **legal/regulatory discussion** | None structure privacy as a **reusable software design pattern** with a formal per-pattern compliance annotation table |
| FL-medical-imaging surveys | 2023–2025 | Multiple | Survey ML algorithms/accuracy for medical imaging FL | Zero software-architecture or design-pattern perspective |

**The verified, citable gap statement (use near-verbatim in your Introduction):**

> "While Lo et al. introduced architectural patterns for FL system design, and subsequent work quantitatively evaluated a subset of these patterns, both research lines treat patterns as system-level architectural concerns evaluated on synthetic benchmarks (e.g., CIFAR-10). No prior work applies Gang-of-Four object-oriented design patterns at the *component implementation level* of FL systems, maps pattern compositions to healthcare privacy regulations (GDPR, HIPAA, India's DPDP Act), or empirically validates pattern-based FL architectures on real medical imaging benchmarks combined with software-quality metrics. This paper addresses all three gaps."

**Action item before submission:** re-verify the Trubiani et al. venue/year and the exact Lo et al. 2021-vs-2022 version you cite (arXiv preprint vs. JSS-published version) — cite the published JSS version, not the arXiv preprint, where possible.

---

## 4. Target Contributions (frame the paper around exactly these five)

| ID | Contribution |
|---|---|
| C1 | **FedPatterns-Med pattern catalog** — a formal catalog of GoF/OO design patterns specialized for FL component implementation |
| C2 | **Compliance-aware pattern mapping** — each pattern annotated with privacy objective, threat relevance, GDPR/HIPAA/DPDP mapping, and trade-offs |
| C3 | **Reference architecture** — a working multi-client system (Flower + PyTorch + FastAPI + MongoDB + Docker Compose + Opacus) |
| C4 | **Empirical evaluation** — predictive, system, maintainability, privacy/compliance, and robustness metrics together |
| C5 | **Benchmark comparison** — against centralized training, vanilla FedAvg, advanced strategies (FedProx/FedNova), and a re-implementation inspired by Lo et al.'s architectural patterns |

---

## 5. System Architecture — Six Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Compliance Envelope                           │
│  (GDPR Art. 25/32, HIPAA safeguards, India DPDP,          │
│   audit logging policy, retention/rollback constraints)  │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — FL Orchestration Layer                         │
│  (Flower server, client manager, strategy manager,        │
│   round scheduler, model broadcast/collection)            │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Pattern Components Layer  ← paper's core       │
│  (FederatedMediator, FederatedStrategy, DifferentialDecorator,
│   SecureAggregationProxy, ConsentObserver,
│   VersionedModelPrototype, StragglerCircuitBreaker,
│   RoundCommandHistory)                                    │
├─────────────────────────────────────────────────────────┤
│  Layer 4 — Privacy & Security Layer                       │
│  (Opacus DP, secure-aggregation hook, consent mgmt,       │
│   access control)                                         │
├─────────────────────────────────────────────────────────┤
│  Layer 5 — Resilience & Recovery Layer                    │
│  (rollback, failure handling, straggler tolerance,        │
│   checkpoint versioning)                                  │
├─────────────────────────────────────────────────────────┤
│  Layer 6 — Infrastructure & Evaluation Layer               │
│  (Docker Compose, MongoDB, FastAPI, MLflow/W&B,            │
│   metrics collectors, Radon/SonarQube static analysis)    │
└─────────────────────────────────────────────────────────┘
```

**Round sequence (for your sequence diagram):**
`Admin/Scheduler → FastAPI trigger → FederatedMediator (server) → broadcast global model → Client[i]: DifferentialDecorator-wrapped local training → ConsentObserver check → SecureAggregationProxy → return update → StragglerCircuitBreaker timeout gate → FederatedMediator aggregates via FederatedStrategy → VersionedModelPrototype checkpoint → RoundCommandHistory log → next round`

---

## 6. Pattern Catalog

Keep the **journal version to 8 core patterns**; report up to 4 more as "extension patterns" / future work if time allows.

### Core Pattern Set

| ID | Pattern Name | Base GoF Pattern | Purpose | Implementation Target |
|---|---|---|---|---|
| P1 | **FederatedMediator** | Mediator | Server coordinates all clients without peer-to-peer coupling | `server/aggregator.py` |
| P2 | **FederatedStrategy** | Strategy | Makes FedAvg/FedProx/FedNova/Scaffold/FedBN swappable without touching client code | `server/strategies/` |
| P3 | **DifferentialDecorator** | Decorator | Wraps local training/updates with DP noise (via Opacus) without changing training logic | `client/privacy/dp_decorator.py` |
| P4 | **SecureAggregationProxy** | Proxy | Intercepts update transmission; hook point for encrypted/secure-aggregation transport | `client/privacy/secure_proxy.py` |
| P5 | **ConsentObserver** | Observer | Reacts to patient consent/policy changes and dynamically excludes a client | `client/policy/consent_observer.py` |
| P6 | **VersionedModelPrototype** | Prototype + Memento | Clones/checkpoints the global model per round; enables rollback after poisoning/failure | `server/versioning/model_registry.py` |
| P7 | **StragglerCircuitBreaker** | Resilience tactic via chained policy handlers | Prevents slow/unresponsive clients from blocking a round | `server/resilience/straggler_manager.py` |
| P8 | **RoundCommandHistory** | Command | Represents each FL round as an auditable, replayable command object | `server/audit/round_command.py` |

### Extension Patterns (future work / stretch)
- **P9 DataPartitionBuilder** (Builder) — reproducible IID/non-IID client splits
- **P10 HeterogeneityAdapterFactory** (Abstract Factory) — per-client CPU/GPU/low-resource execution backends
- **P11 FedModelSingleton** (Singleton, used carefully — document anti-pattern risk explicitly) — single authoritative global model state
- **P12 GradientIterator** (Iterator) — clean traversal over update sets during aggregation

### Mandatory Pattern Card Template (apply identically to every pattern — consistency reads as maturity to reviewers)
1. Name
2. Intent
3. Context
4. Problem
5. Forces / constraints
6. Solution
7. Structure (UML snippet)
8. Participants
9. Implementation notes
10. Privacy/compliance mapping
11. Benefits
12. Liabilities
13. Code reference
14. Evaluation metrics impacted

### Pattern Composition Rules (a strong, underused novelty lever — make this explicit in the paper)
- `DifferentialDecorator + RoundCommandHistory` → increases audit complexity (noisy updates vs. clean audit trail tension)
- `StragglerCircuitBreaker + FedNova` → may alter fairness behavior across clients
- `VersionedModelPrototype + SecureAggregationProxy` → increases latency; quantify this explicitly
- Document which compositions are **safe-by-default** vs. require explicit compliance review

---

## 7. Compliance Mapping Matrix

Produce **both**: a table in the paper, and a machine-readable `compliance_matrix.yaml` in the codebase (this dual artifact is a strong reproducibility signal).

| Pattern | Privacy Objective | Data Exposure Risk | GDPR Art. 25 | GDPR Art. 32 | HIPAA §164.312 | India DPDP | Residual Risk |
|---|---|---|---|---|---|---|---|
| DifferentialDecorator | Data minimization by design | Low | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | Low (accuracy trade-off) |
| VersionedModelPrototype | Recoverability | Medium | ⚠️ Partial | ✅ Satisfies | ⚠️ Partial | ✅ Satisfies | Medium (storage retention) |
| RoundCommandHistory | Auditability | High | ❌ Risk (audit trail = data trail) | ✅ Satisfies | ✅ Satisfies | ❌ Risk | High — needs mitigation (log redaction policy) |
| ConsentObserver | Dynamic consent enforcement | Low | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | Low |
| SecureAggregationProxy | Transport confidentiality | Low–Medium | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | ✅ Satisfies | Low (if backend is real crypto; Medium if stub) |

> Frame this explicitly in the paper as a **practitioner-oriented compliance checklist**, not a formal legal proof — cite primary regulatory texts directly.

---

## 8. Datasets — Full List With Links, Verified

### Phase A — Fast Prototyping: MedMNIST v2
- Official site: https://medmnist.com
- Data archive (Zenodo): https://zenodo.org/records/6496656
- Recommended subsets:
  - **PneumoniaMNIST** — chest disease classification
  - **DermaMNIST** — skin lesion classification
  - **PathMNIST** — pathology image classification
  - **RetinaMNIST** / **OCTMNIST** — modality diversity
- Purpose: bootstrap the Flower pipeline, validate pattern compositions cheaply, debug non-IID partitioning, fast ablations.

### Phase B — Primary Real-World Benchmark: NIH ChestX-ray14
- Kaggle mirror: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Official NIH/Box source: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Verified statistics** (cross-checked independently): 112,120 frontal-view chest X-rays, 30,805 unique patients, 14 thoracic disease labels mined via NLP from radiology reports — confirm these exact numbers in your own paper's dataset table.
- Note for the paper: labels are NLP-derived from reports (~90% estimated accuracy per the original Wang et al. ChestX-ray8 paper) — mention this as a known dataset limitation in your Threats to Validity section.
- Recommended task: start binary/limited multi-label, scale to 4-label, then full 14-label.

### Phase C — Cross-Site Heterogeneity: ISIC 2020 Skin Lesion
- Challenge page: https://challenge2020.isic-archive.com
- Archive: https://www.isic-archive.com
- Purpose: simulate strong non-IID, multi-site heterogeneity; second real-world benchmark outside chest imaging.
- Recommended task: binary classification (benign vs. malignant) for tractability.

### Optional Phase D — Extra Robustness
- Additional MedMNIST subsets (OrganMNIST, PathMNIST, OCTMNIST) for cross-modality transfer experiments.

### Dataset-to-Phase Mapping

| Research Phase | Dataset(s) | Goal |
|---|---|---|
| Phase 1 — Prototype validation | MedMNIST only | End-to-end pipeline works, client/server comms verified, metrics logging confirmed |
| Phase 2 — Pattern catalog validation | MedMNIST + small NIH subset | Every pattern actually exercised; compliance annotations validated against real components |
| Phase 3 — Real benchmark evaluation | Full NIH ChestX-ray14 | Main paper results; vanilla vs. pattern-oriented FL comparison; privacy-performance trade-off |
| Phase 4 — Generalization | ISIC 2020 + selected MedMNIST subset | Prove modality-agnostic design; strengthens journal novelty |

---

## 9. Non-IID Partitioning Plan

Medical FL is only scientifically interesting under realistic heterogeneity.

**Partition styles to implement:**
1. IID split — balanced random baseline
2. Label-skew non-IID — each client sees different disease distribution
3. Quantity-skew non-IID — clients have very different dataset sizes
4. Feature-skew simulation — per-client preprocessing variation
5. Extreme imbalance — one large "hospital," several small ones

**Method:** Dirichlet-based partitioning, sweeping α:
- α = 10.0 → near-IID
- α = 1.0 → moderate non-IID
- α = 0.5 → strong non-IID
- α = 0.1 → extreme non-IID

---

## 10. Recommended Tech Stack

| Layer | Tools |
|---|---|
| Core | Python 3.11, Flower (`flwr`), PyTorch, torchvision |
| API/Infra | FastAPI, MongoDB, Docker Compose |
| Privacy | Opacus (DP-enabled PyTorch training), optional secure-aggregation abstraction |
| Experiment tracking | MLflow or Weights & Biases |
| Runtime metrics (optional) | Prometheus + Grafana |
| Code-quality metrics | Radon (cyclomatic complexity), SonarQube (smells/maintainability), PyDriller (optional, evolution analysis) |

---

## 11. Full Repository Structure

```text
fedpatterns-med/
├── apps/
│   ├── server_app/
│   │   └── main.py
│   └── client_app/
│       └── main.py
├── fedpatterns/
│   ├── core/
│   │   ├── interfaces/
│   │   ├── domain/
│   │   └── config/
│   ├── patterns/
│   │   ├── mediator/
│   │   │   └── federated_mediator.py
│   │   ├── strategy/
│   │   │   └── federated_strategy.py
│   │   ├── decorator/
│   │   │   └── differential_decorator.py
│   │   ├── proxy/
│   │   │   └── secure_aggregation_proxy.py
│   │   ├── observer/
│   │   │   └── consent_observer.py
│   │   ├── prototype/
│   │   │   └── versioned_model_prototype.py
│   │   ├── command/
│   │   │   └── round_command_history.py
│   │   └── builder/
│   │       └── data_partition_builder.py
│   ├── privacy/
│   │   ├── opacus_dp/
│   │   ├── compliance/
│   │   │   └── compliance_matrix.yaml
│   │   └── policy/
│   ├── resilience/
│   │   ├── straggler/
│   │   │   └── straggler_manager.py
│   │   ├── rollback/
│   │   └── checkpointing/
│   ├── datasets/
│   │   ├── medmnist/
│   │   │   └── loader.py
│   │   ├── nih_chestxray14/
│   │   │   └── loader.py
│   │   └── isic2020/
│   │       └── loader.py
│   ├── evaluation/
│   │   ├── ml_metrics/
│   │   ├── system_metrics/
│   │   ├── maintainability/
│   │   ├── privacy_metrics/
│   │   └── reports/
│   └── utils/
├── experiments/
│   ├── exp01_architecture_quality/
│   ├── exp02_privacy_accuracy/
│   ├── exp03_non_iid_robustness/
│   ├── exp04_cross_dataset_generalization/
│   └── exp05_ablation_patterns/
├── baselines/
│   ├── b0_centralized/
│   ├── b1_vanilla_flower_fedavg/
│   ├── b2_vanilla_flower_advanced_strategy/
│   └── b3_architectural_pattern_baseline/   # re-implementation inspired by Lo et al.
├── configs/
│   ├── datasets/
│   ├── strategies/
│   ├── clients/
│   └── compliance/
├── docs/
│   ├── pattern_cards/        # one .md per pattern, following the 14-field template
│   ├── uml/
│   └── paper_assets/         # exported figures/tables for the manuscript
├── output/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
├── docker-compose.yml
├── pyproject.toml
├── compliance_matrix.yaml
└── README.md
```

---

## 12. Development Roadmap (Milestones)

| Milestone | Goal | Key Deliverables |
|---|---|---|
| **M0 — Paper framing** | Lock scope before coding | Final novelty statement, 8 chosen patterns, dataset plan, paper skeleton |
| **M1 — Minimal FL skeleton** | Flower simulation running | 3–5 simulated clients, FedAvg baseline, first accuracy/loss curve |
| **M2 — Patternized baseline** | Refactor into explicit pattern architecture | FederatedMediator + FederatedStrategy implemented, maintainability delta measured |
| **M3 — Privacy/compliance layer** | Add privacy novelty | DifferentialDecorator (Opacus) + ConsentObserver, epsilon-vs-accuracy results, compliance table v1 |
| **M4 — Resilience layer** | Add engineering depth | VersionedModelPrototype, rollback, StragglerCircuitBreaker, fault-injection results |
| **M5 — Real dataset scaling** | Move to publishable evidence | NIH ChestX-ray14, IID + non-IID runs, FedAvg/FedProx/FedNova comparison |
| **M6 — Cross-modality validation** | Show generalization | ISIC 2020 integration, portability discussion |
| **M7 — Paper packaging** | Convert engineering → manuscript assets | Clean figures, finalized pattern cards, UML diagrams, reproducibility appendix |

---

## 13. Experimental Design — Five Experiment Families

### E1. Architecture Quality
**Question:** Does pattern-oriented implementation improve maintainability?
**Compare:** ad-hoc vanilla Flower code vs. FedPatterns-Med
**Metrics:** LOC, cyclomatic complexity, CBO, LCOM, duplication, change-impact effort

### E2. Privacy–Performance Trade-off
**Question:** What is the cost of privacy-aware patterns?
**Compare:** no-DP vs. low/medium/high-noise DifferentialDecorator
**Metrics:** AUC, F1, sensitivity, specificity, ε, training time, communication cost

### E3. Strategy & Non-IID Robustness
**Question:** Which FL strategy performs best inside the patternized architecture?
**Compare:** FedAvg, FedProx, FedNova, optional Scaffold/FedBN
**Metrics:** convergence rounds, final AUC/F1, client drift, round time

### E4. Resilience
**Question:** Do rollback and straggler patterns improve reliability?
**Inject:** slow client, dropped client, bad update, forced rollback
**Metrics:** recovery time, round-completion rate, accuracy preservation, storage overhead

### E5. Pattern Ablation
**Question:** Which patterns contribute most to the overall result?
**Ablate:** full architecture → minus DifferentialDecorator → minus VersionedModelPrototype → minus StragglerCircuitBreaker → minus RoundCommandHistory → minimal 2-pattern architecture
**Metrics:** maintainability, runtime, privacy, robustness — report contribution strength per pattern

---

## 14. Baselines and Benchmark Matrix (the "competing with existing architectures" section)

| ID | Architecture | Strategy | Privacy | Patterns | Datasets | Purpose |
|---|---|---|---|---|---|---|
| **B0** | Centralized PyTorch (merged data) | N/A | None | None | MedMNIST, NIH, ISIC | Upper-bound accuracy reference |
| **B1** | Vanilla Flower | FedAvg | None | None | MedMNIST, NIH | Plain engineering baseline |
| **B2** | Vanilla Flower | FedProx / FedNova | None | None | MedMNIST, NIH | Stronger algorithmic baseline |
| **B3** | **Re-implementation inspired by Lo et al.'s architectural patterns** (Client Registry, Client Selector, Client Cluster) | FedAvg/FedProx | Optional | System-level only, no GoF/compliance | MedMNIST | Direct comparison against the closest prior published work |
| **B4** | **FedPatterns-Med (full)** | FedAvg/FedProx/FedNova | DP + compliance layer | Full GoF + composition | MedMNIST, NIH, ISIC | Main contribution |
| **B5** | FedPatterns-Med (ablated variants) | Same as B4 | Varied | Selected patterns removed | MedMNIST, NIH | Contribution isolation (ties to E5) |

> **B3 is the most important row for your novelty argument** — it's the closest direct re-implementation of the prior published architecture-level work (Lo et al. / Trubiani et al.), and beating or differentiating from it on maintainability + compliance is your central empirical claim.

---

## 15. Metrics Checklist

**ML performance:** Accuracy, F1, AUC-ROC, sensitivity/recall, specificity, precision, multi-label mAP (if needed)

**FL system:** rounds to convergence, time per round, total wall-clock time, communication cost/round, client participation rate, straggler rate

**Architecture quality:** cyclomatic complexity, CBO, LCOM, maintainability index, code duplication, files touched per feature change

**Privacy/compliance:** ε value, noise multiplier, DP clipping norm, compliance coverage score, violations flagged, audit completeness score

**Resilience:** rollback success rate, fault recovery time, accuracy after recovery, checkpoint overhead, round completion under failure

---

## 16. Threat Model and Privacy Assumptions (state explicitly — improves reviewer trust)

**Assumptions:**
- Raw images never leave local client sites
- Server is honest-but-curious unless secure-aggregation extension is enabled
- Clients may be slow, unavailable, or noisy
- Some client updates may be anomalous (poisoning)

**Threats considered:** model update leakage, membership-inference risk narrative, metadata leakage via logs, participation leakage, stale-checkpoint/audit-retention risk

**Threats explicitly NOT fully solved (be honest about this — it strengthens credibility):**
- Byzantine robustness at scale
- Cryptographically rigorous secure aggregation (if your proxy is a hook, not full crypto)
- Full legal certification — your compliance mapping is a structured practitioner checklist, not a legal proof

---

## 17. Implementation Plan in Antigravity IDE

**Workspace setup**
1. Create project root: `fedpatterns-med`
2. Create a Python 3.11 virtual environment
3. Install core packages: `flwr`, `torch`, `torchvision`, `medmnist`, `opacus`, `fastapi`, `uvicorn`, `pymongo`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `radon`
4. Optional: `mlflow`, `wandb`, `pydriller`
5. One central config file driving all experiment settings (avoid scattered hardcoded params)

**Recommended build order**
1. Vanilla Flower + MedMNIST baseline (B1)
2. Refactor into patternized client/server architecture (P1, P2)
3. Add FedAvg/FedProx/FedNova strategies
4. Add DifferentialDecorator (P3)
5. Add checkpoint versioning + rollback (P6)
6. Add straggler management (P7)
7. Add compliance mapping + reporting
8. Build B3 (Lo et al.-inspired baseline) for direct comparison
9. Scale to NIH ChestX-ray14
10. Validate on ISIC 2020
11. Run ablations (E5) and export paper assets

---

## 18. Practical Model Choices

| Dataset | Recommended Model | Note |
|---|---|---|
| MedMNIST | Small CNN → ResNet18 → EfficientNet-B0 (optional) | Keep light for fast iteration |
| NIH ChestX-ray14 | ResNet18 or DenseNet121 | Avoid large ViTs initially; start binary/limited multi-label |
| ISIC 2020 | EfficientNet-B0 or ResNet18 | Binary classification is sufficient evidence |

**Important framing rule:** this is not a new-SOTA medical model paper. Do not over-invest in backbone size — your novelty is architecture + patterns + compliance + evaluation, not raw accuracy.

---

## 19. Suggested 15-Week Experiment Schedule

| Weeks | Focus |
|---|---|
| 1–2 | Flower baseline on MedMNIST, dataset loaders, logging pipeline |
| 3–4 | FederatedMediator + FederatedStrategy, baseline maintainability analysis |
| 5–6 | DifferentialDecorator (Opacus), privacy-performance runs on MedMNIST |
| 7–8 | VersionedModelPrototype + RoundCommandHistory, failure-injection scripts |
| 9–10 | StragglerCircuitBreaker, non-IID experiments (FedAvg/FedProx/FedNova) |
| 11–12 | NIH ChestX-ray14 scaling, main benchmark results, B3 baseline build |
| 13–14 | ISIC validation, cross-modality comparison |
| 15+ | Ablation study (E5), paper writing, figure polishing |

---

## 20. Figures and Tables for the Manuscript

**Figures**
1. Full layered system architecture diagram
2. Pattern interaction / composition graph
3. Sequence diagram for one FL round
4. Compliance mapping matrix (visual)
5. Privacy–accuracy trade-off plot
6. Convergence curves across strategies
7. Maintainability metric comparison (bar chart)
8. Fault-recovery timeline chart
9. Cross-dataset portability summary

**Tables**
1. Related-work comparison table (Section 3 above, formalized)
2. Pattern catalog table
3. Dataset table (links, sizes, modality, task)
4. Benchmark matrix (Section 14)
5. Maintainability metrics table
6. Privacy-compliance risk table
7. Non-IID robustness results table
8. Fault-injection recovery table
9. Pattern ablation table (E5)

---

## 21. Paper Section Plan (write in parallel with code, not after)

1. **Introduction** — problem, gap (Section 3), contributions (Section 4)
2. **Related Work** — FL, privacy-preserving FL, medical imaging FL, FL architecture/pattern research
3. **Methodology** — FedPatterns-Med pattern language, architecture, compliance mapping
4. **Implementation** — Flower-based platform, privacy layer, resilience layer
5. **Experimental Setup** — datasets, client simulation, strategies, metrics
6. **Results** — ML, system, maintainability, resilience, compliance (E1–E5)
7. **Discussion** — trade-offs, practical lessons, limits
8. **Threats to Validity** — data realism, simulation realism, compliance-mapping informality
9. **Conclusion** — contribution recap

**Per-feature writing rule:** for every implemented feature ask — *what research question does this answer, and what figure/table proves it?* If the answer is "none," cut scope.

---

## 22. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Too many patterns | Cap at 8 core patterns; rest = future work |
| NIH dataset too heavy for timeline | Start MedMNIST + NIH subset, scale incrementally |
| DP hurts accuracy too much | Tune ε / noise multiplier; report trade-off honestly, don't hide it |
| Secure aggregation too complex | Keep SecureAggregationProxy as an interchangeable hook; discuss as extension |
| Reviewer asks "where's the novelty?" | Lead with: implementation-level pattern language + compliance mapping + maintainability evaluation + real medical validation |
| Too much ML, not enough SE | Make architecture-quality results (E1) a first-class, prominent result, not an appendix afterthought |

---

## 23. Minimal Publishable Version (MPV) — if time runs short

**Keep:** MedMNIST + NIH ChestX-ray14 · 8 core patterns · FedAvg + FedProx + FedNova · DifferentialDecorator · VersionedModelPrototype · StragglerCircuitBreaker · architecture-quality evaluation (E1) · privacy-performance evaluation (E2)

**Drop or postpone:** full cryptographic secure aggregation · ISIC/extra datasets · large backbones · extension patterns (P9–P12)

## 24. Stretch Goals (only if ahead of schedule)
- FedBN for modality/client drift
- Fairness evaluation across clients
- Real secure-aggregation backend (not just a hook)
- Poisoning-detection trigger for automatic rollback
- Architecture-evolution study across git commit history
- Public pattern-catalog website/docs

---

## 25. Final Execution Checklist

**Before coding**
- [ ] Finalize 8 core patterns
- [ ] Finalize dataset phase plan
- [ ] Finalize benchmark baselines (B0–B5)
- [ ] Draft paper outline

**Before scaling**
- [ ] MedMNIST baseline works end-to-end
- [ ] Non-IID splitter validated
- [ ] Patternized architecture stable
- [ ] Metrics logging complete

**Before experiments**
- [ ] Reproducibility seeds fixed
- [ ] Configs versioned
- [ ] CSV result schema fixed
- [ ] Figure-generation scripts ready

**Before submission**
- [ ] Related work clearly and explicitly distinguishes Lo et al. / Trubiani et al. (re-verify exact venues/years)
- [ ] Novelty claim stated precisely (Section 3 gap statement)
- [ ] Tables/figures publication quality
- [ ] Artifact reproducible (README + configs + seeds)
- [ ] Threats to Validity section is honest, not defensive

---

## 26. References to Verify and Use

> Verify exact venue/year/DOI for each before final submission — verified as real publications during planning, but citation details should be pulled from the primary source at write-up time.

- Lo, S.K., Lu, Q., Zhu, L., Paik, H., Xu, X., Wang, C. — "Architectural Patterns for the Design of Federated Learning Systems," arXiv:2101.02373 / Journal of Systems and Software.
- Lo, S.K. et al. — Decision-model follow-up paper for FL architectural pattern selection (2022).
- Trubiani, C. et al. — "Experimenting Architectural Patterns in Federated Learning Systems," Journal of Systems and Software / ScienceDirect (2025) — confirm exact venue.
- SLR — "Architectural patterns for blockchain-enabled federated learning," ScienceDirect (2025).
- Wang, X. et al. — "ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," CVPR 2017 (original NIH ChestX-ray14 source paper).
- Flower framework documentation: https://flower.ai
- Opacus documentation: https://opacus.ai
- GDPR text: https://gdpr-info.eu
- MedMNIST: Yang, J. et al. — official site https://medmnist.com

---

## 27. Immediate Next Actions

1. Set up the Antigravity IDE workspace (`fedpatterns-med` root, venv, dependencies)
2. Build the smallest possible Flower + MedMNIST baseline (B1)
3. Freeze the 8 core patterns — no more additions after this point
4. Write one complete pattern card (FederatedStrategy) as the template exemplar
5. Implement the non-IID partition builder
6. Run baseline FedAvg, confirm metrics logging
7. Refactor into the patternized architecture (P1, P2)
8. Add DifferentialDecorator (P3)
9. Start the related-work comparison table immediately — don't leave it for the end
