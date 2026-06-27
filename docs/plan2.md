# FedPatterns-Med — Implementation Blueprint (`plan.md`)

## 1. Project Identity

**Working title**
FedPatterns: A Design Pattern Language for Privacy-Compliant Federated Learning Architectures with Empirical Validation on Medical Imaging Systems

**Refined journal title**
FedPatterns-Med: OO Design Pattern Compositions with Privacy-Compliance Mapping for Federated Learning in Medical Imaging

**Core idea**
Build a real federated learning system for medical imaging, but make the main contribution a **software design pattern language** at the implementation level, not just another FL model paper.

**Why this can publish**
Existing research already covers:
- architectural patterns for federated learning systems,
- decision support for selecting FL architectural patterns,
- performance analysis of a subset of FL architectural patterns.

Your novelty must therefore be:
1. GoF/OO-level implementation patterns inside FL components,
2. privacy-compliance mapping per pattern and per pattern composition,
3. empirical validation on real medical imaging datasets,
4. maintainability and architecture-quality evaluation in addition to ML metrics,
5. composition rules showing which pattern combinations help or hurt.

---

## 2. Problem Statement

Federated learning is attractive in healthcare because institutions can collaboratively train models without sharing raw patient data. But most FL papers focus on **model accuracy, aggregation algorithms, or privacy methods**, while the software architecture and code-level design of FL systems is underexplored.

Current gaps:
- Researchers describe FL architectures, but not a reusable **implementation-level pattern catalog**.
- Privacy is often discussed as a technique, not as a **first-class software design concern**.
- Medical imaging FL papers benchmark algorithms, but rarely evaluate **maintainability, modularity, extensibility, auditability, and compliance readiness**.
- Existing FL pattern work mostly stays at the system architecture level; your work should go one level deeper into **component structure and pattern composition**.

**Research question**
How can a pattern-oriented implementation blueprint improve privacy compliance, maintainability, extensibility, and robustness of federated learning systems for medical imaging, while remaining competitive in predictive performance?

---

## 3. Target Contributions

You should design the paper so that it claims **five contributions**:

### C1. FedPatterns-Med pattern catalog
A formal catalog of OO design patterns specialized for FL implementation.

### C2. Compliance-aware pattern mapping
Each pattern should include:
- privacy objective,
- threat model relevance,
- GDPR mapping,
- HIPAA mapping,
- India DPDP mapping,
- implementation trade-offs.

### C3. Reference architecture
A working multi-client system built with Flower, PyTorch, FastAPI, MongoDB, Docker Compose, and optional Opacus.

### C4. Empirical evaluation
Evaluate using:
- predictive metrics,
- system metrics,
- maintainability metrics,
- privacy/compliance metrics,
- robustness/failure metrics.

### C5. Benchmark comparison
Benchmark against:
- centralized training,
- vanilla Flower FedAvg implementation,
- architectural-pattern-style FL setup inspired by prior FL pattern papers,
- advanced FL strategies such as FedProx and FedNova,
- optionally FedBN/Scaffold for non-IID experiments.

---

## 4. Existing Research You Must Position Against

### 4.1 FL architectural patterns
Prior work already proposed architectural patterns for FL systems and categorized them across client management, model management, training, and aggregation lifecycle stages. This means your paper cannot claim to be the first pattern paper for FL.

### 4.2 FL architectural pattern selection
There is already a decision model for selecting FL architectural patterns. So your novelty should not be “pattern selection” alone.

### 4.3 FL architectural pattern performance analysis
A later study quantitatively analyzed the performance impact of selected FL architectural patterns using Flower-based experimentation. This means you should go beyond raw performance and include implementation-level pattern design, compliance mapping, and medical imaging validation.

### 4.4 Medical imaging FL studies
Medical imaging FL literature already compares algorithms and shows that no single FL method consistently dominates across datasets. Non-IID settings remain difficult, especially for healthcare tasks.

### 4.5 Privacy techniques
Privacy-preserving FL surveys already discuss differential privacy, secure aggregation, and fairness/privacy trade-offs. So your novelty is not “using DP,” but **structuring DP as a reusable design pattern component inside a federated software system**.

---

## 5. Final Novel Positioning

Your final claim should be:

> Existing work studies FL architectural patterns at the system level. FedPatterns-Med introduces a **code-oriented design pattern language** for FL implementations, enriched with privacy-compliance annotations and validated on real medical imaging tasks using both ML and software-quality metrics.

This is the framing you should keep consistent in the codebase, experiments, and paper.

---

## 6. Project Objectives

### Primary objective
Build a federated medical imaging platform whose implementation is explicitly organized using a reusable pattern language.

### Secondary objectives
- Demonstrate privacy-aware design using DP and optional secure aggregation hooks.
- Show that pattern-based design improves maintainability without unacceptable ML degradation.
- Evaluate how pattern compositions behave under non-IID, straggler, and rollback scenarios.
- Produce figures, tables, and metrics suitable for a journal submission.

---

## 7. Recommended Datasets and Exact Usage Plan

Use datasets in **three phases** so the project scales safely.

### Phase A — Fast prototyping dataset
#### MedMNIST v2
- Official: https://medmnist.com
- Zenodo: https://zenodo.org/records/6496656
- Purpose:
  - bootstrap the architecture quickly,
  - validate Flower pipeline,
  - test pattern compositions cheaply,
  - run repeated experiments fast,
  - debug non-IID partitioning.
- Recommended subsets:
  - **PneumoniaMNIST** for chest disease classification,
  - **DermaMNIST** for skin lesion classification,
  - **PathMNIST** for pathology image classification,
  - **RetinaMNIST** or **OCTMNIST** for modality diversity.
- Why use it:
  - standardized, lightweight, easy to run locally,
  - excellent for ablation experiments,
  - quick round-trip for early paper figures.

### Phase B — Primary real-world benchmark
#### NIH ChestX-ray14
- Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Official NIH/Box reference: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Purpose:
  - main flagship evaluation dataset,
  - demonstrate clinical relevance,
  - validate non-IID hospital partitions,
  - run privacy-performance trade-off study.
- Dataset characteristics to note in paper:
  - 112,120 frontal chest X-rays,
  - 14 thoracic disease labels,
  - 30,000+ patients.
- Recommended task:
  - start with binary or limited multi-label task for feasibility,
  - then extend to 4-label or 14-label setup for full paper version.

### Phase C — Cross-site heterogeneity dataset
#### ISIC 2020 Skin Lesion
- Challenge page: https://challenge2020.isic-archive.com
- Archive: https://www.isic-archive.com
- Purpose:
  - simulate strong non-IID, multi-site heterogeneity,
  - test whether pattern-based implementation remains modular across modalities,
  - provide a second real-world benchmark outside chest imaging.
- Recommended task:
  - binary classification (benign vs malignant) for tractable training.

### Optional Phase D — Extra robustness benchmark
#### Additional MedMNIST subsets
Use OrganMNIST, PathMNIST, or OCTMNIST for cross-modality transfer experiments.

---

## 8. Dataset Strategy by Research Phase

### Phase 1 — Prototype validation
Dataset: MedMNIST only.

Goal:
- ensure architecture works end-to-end,
- test client/server communication,
- verify metrics logging,
- benchmark strategies quickly.

### Phase 2 — Pattern catalog validation
Dataset: MedMNIST + small NIH subset.

Goal:
- verify every pattern is actually used,
- validate compliance annotations against real components,
- write first draft of pattern cards.

### Phase 3 — Real benchmark evaluation
Dataset: full NIH ChestX-ray14.

Goal:
- produce main results for paper,
- compare vanilla FL and pattern-oriented FL,
- quantify privacy-performance trade-off.

### Phase 4 — Generalization and heterogeneity
Dataset: ISIC 2020 + selected MedMNIST subset.

Goal:
- prove architecture is modality-agnostic,
- show design patterns scale across domains,
- strengthen journal novelty.

---

## 9. Non-IID Partitioning Plan

This is crucial because medical FL becomes interesting only under realistic heterogeneity.

### Partition styles to implement
1. **IID split** — simple balanced random split across clients.
2. **Label-skew non-IID** — each client sees different disease distribution.
3. **Quantity-skew non-IID** — clients have very different dataset sizes.
4. **Feature-skew simulation** — per-client image preprocessing differs slightly.
5. **Extreme client imbalance** — one large hospital, several small hospitals.

### Recommended method
Use **Dirichlet-based partitioning** with different alpha values.
- alpha = 10.0 → near-IID,
- alpha = 1.0 → moderate non-IID,
- alpha = 0.5 → strong non-IID,
- alpha = 0.1 → extreme non-IID.

### Why it matters
This directly supports experiments on:
- maintainability under heterogeneous settings,
- strategy robustness,
- privacy-performance trade-offs,
- pattern composition usefulness.

---

## 10. Architecture Overview

The full system should have **six layers**.

### Layer 1 — Compliance Envelope
Cross-cutting policy layer that tracks:
- GDPR Art. 25,
- GDPR Art. 32,
- HIPAA safeguards,
- India DPDP compliance notes,
- audit logging policies,
- data retention and rollback constraints.

### Layer 2 — FL Orchestration Layer
- Flower server,
- client manager,
- strategy manager,
- round scheduler,
- model broadcast and collection.

### Layer 3 — Pattern Components Layer
This is the paper's heart.
It contains your implementation-level patterns.

### Layer 4 — Privacy and Security Layer
- differential privacy,
- secure aggregation hooks,
- consent management,
- logging and access control.

### Layer 5 — Resilience and Recovery Layer
- rollback,
- failure handling,
- straggler tolerance,
- checkpoint versioning.

### Layer 6 — Infrastructure and Evaluation Layer
- Docker Compose,
- MongoDB,
- FastAPI,
- Prometheus/W&B/MLflow if needed,
- metrics collectors,
- static analysis tools.

---

## 11. Pattern Catalog You Should Implement

Keep the final paper to **8 core patterns** and optionally report 4 more as extensions.

### Core pattern set for journal paper

#### P1. FederatedMediator
**Base pattern:** Mediator

**Purpose:**
The central server coordinates all clients without peer-to-peer dependency.

**Implementation target:**
`server/aggregator.py`

**Why useful:**
Reduces coupling and simplifies client logic.

**Measured effect:**
- lower coupling,
- easier strategy replacement,
- cleaner orchestration.

---

#### P2. FederatedStrategy
**Base pattern:** Strategy

**Purpose:**
Make FedAvg, FedProx, FedNova, Scaffold, or FedBN swappable without changing client code.

**Implementation target:**
`server/strategies/`

**Why useful:**
Supports reproducible benchmarking and algorithm ablation.

**Measured effect:**
- extensibility,
- lower modification cost,
- strong benchmark flexibility.

---

#### P3. DifferentialDecorator
**Base pattern:** Decorator

**Purpose:**
Wrap local training or uploaded updates with DP logic.

**Implementation target:**
`client/privacy/dp_decorator.py`

**Variants:**
- sample-level DP using Opacus,
- user-level/server-side update noising prototype.

**Measured effect:**
- privacy gain,
- accuracy drop,
- training slowdown.

---

#### P4. SecureAggregationProxy
**Base pattern:** Proxy

**Purpose:**
Intercept model update transmission and apply secure aggregation or encrypted transport hooks.

**Implementation target:**
`client/privacy/secure_proxy.py`

**Note:**
If full secure aggregation is too heavy, implement it as an interface/hook and state in the paper that the proxy is ready for interchangeable secure backends.

---

#### P5. ConsentObserver
**Base pattern:** Observer

**Purpose:**
React to patient consent or policy changes and exclude a client dynamically.

**Implementation target:**
`client/policy/consent_observer.py`

**Novelty value:**
Strong compliance angle for the paper.

---

#### P6. VersionedModelPrototype
**Base pattern:** Prototype + Memento

**Purpose:**
Clone and checkpoint the global model at each round, enabling rollback after poisoning or failure.

**Implementation target:**
`server/versioning/model_registry.py`

**Measured effect:**
- recovery speed,
- storage overhead,
- rollback reliability.

---

#### P7. StragglerCircuitBreaker
**Base pattern:** Circuit-breaker-like resilience tactic, implemented through chainable policy handlers

**Purpose:**
Prevent slow or unresponsive clients from blocking the FL round.

**Implementation target:**
`server/resilience/straggler_manager.py`

**Measured effect:**
- lower wall-clock round time,
- potential fairness/accuracy trade-off.

---

#### P8. RoundCommandHistory
**Base pattern:** Command

**Purpose:**
Represent each federated round as an auditable command object.

**Implementation target:**
`server/audit/round_command.py`

**Novelty value:**
Supports auditability, rollback, and compliance narratives.

---

### Extension patterns
These can strengthen the pattern language, but only if time permits.

#### P9. DataPartitionBuilder
Build reproducible IID/non-IID client splits.

#### P10. HeterogeneityAdapterFactory
Instantiate client execution backends for CPU/GPU/low-resource nodes.

#### P11. FedModelSingleton
Ensure exactly one authoritative global model state.

#### P12. GradientIterator
Iterate over update sets cleanly during aggregation/comparison.

---

## 12. Pattern Card Template

Every pattern in the paper should follow the same card structure:

1. Name
2. Intent
3. Context
4. Problem
5. Forces / constraints
6. Solution
7. Structure / UML snippet
8. Participants
9. Implementation notes
10. Privacy/compliance mapping
11. Benefits
12. Liabilities
13. Example code reference
14. Evaluation metrics impacted

This consistency makes the journal paper look mature.

---

## 13. Compliance Mapping Matrix

Create one table per paper and one machine-readable YAML/JSON file in the codebase.

### Suggested columns
- Pattern name
- Privacy objective
- Data exposure risk
- GDPR Art. 25 relevance
- GDPR Art. 32 relevance
- HIPAA relevance
- DPDP relevance
- Required mitigation
- Residual risk

### Example risks
- DifferentialDecorator → low residual risk, medium performance cost.
- RoundCommandHistory → higher audit value but potential metadata retention risk.
- VersionedModelPrototype → rollback benefit but storage/compliance implications.

---

## 14. Recommended Tech Stack

### Core stack
- **Python 3.11**
- **Flower** for FL orchestration
- **PyTorch** for modeling
- **FastAPI** for API and management endpoints
- **MongoDB** for pattern registry, experiment metadata, audit logs
- **Docker Compose** for multi-client simulation

### Privacy stack
- **Opacus** for DP-enabled PyTorch training
- Optional secure aggregation abstraction layer

### Experiment/monitoring stack
- **MLflow** or **Weights & Biases** for experiment tracking
- **Prometheus + Grafana** optional for runtime metrics
- **Radon** for cyclomatic complexity
- **SonarQube** for code smells and maintainability
- **PyDriller** optional for evolution/change analysis

---

## 15. Recommended Repository Structure

```text
fedpatterns-med/
├── apps/
│   ├── server_app/
│   └── client_app/
├── fedpatterns/
│   ├── core/
│   │   ├── interfaces/
│   │   ├── domain/
│   │   └── config/
│   ├── patterns/
│   │   ├── mediator/
│   │   ├── strategy/
│   │   ├── decorator/
│   │   ├── proxy/
│   │   ├── observer/
│   │   ├── prototype/
│   │   ├── command/
│   │   └── builder/
│   ├── privacy/
│   │   ├── opacus_dp/
│   │   ├── compliance/
│   │   └── policy/
│   ├── resilience/
│   │   ├── straggler/
│   │   ├── rollback/
│   │   └── checkpointing/
│   ├── datasets/
│   │   ├── medmnist/
│   │   ├── nih_chestxray14/
│   │   └── isic2020/
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
├── configs/
│   ├── datasets/
│   ├── strategies/
│   ├── clients/
│   └── compliance/
├── docs/
│   ├── pattern_cards/
│   ├── uml/
│   └── paper_assets/
├── output/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 16. Development Roadmap

## Milestone 0 — Paper framing before coding
**Deliverables**
- final novelty statement,
- 8 chosen core patterns,
- experimental questions,
- baseline list,
- dataset plan,
- paper skeleton.

---

## Milestone 1 — Minimal FL skeleton
**Goal:** get Flower simulation running.

### Tasks
- initialize Flower project,
- create PyTorch classifier for MedMNIST,
- simulate 3–5 clients,
- implement FedAvg baseline,
- produce first accuracy/loss curve.

### Deliverables
- reproducible local FL run,
- CSV logs per round,
- first diagram of raw architecture.

---

## Milestone 2 — Patternized baseline architecture
**Goal:** refactor minimal system into explicit pattern-based structure.

### Tasks
- implement FederatedMediator,
- implement FederatedStrategy,
- abstract client/server interfaces,
- create pattern registry docs,
- measure maintainability before/after.

### Deliverables
- patternized baseline,
- maintainability delta table,
- draft pattern cards for P1 and P2.

---

## Milestone 3 — Privacy/compliance layer
**Goal:** add privacy-aware novelty.

### Tasks
- implement DifferentialDecorator using Opacus,
- implement ConsentObserver event mechanism,
- implement compliance annotation schema,
- document privacy assumptions,
- test epsilon tracking.

### Deliverables
- privacy-enabled FL pipeline,
- epsilon-vs-accuracy results,
- compliance mapping table v1.

---

## Milestone 4 — Resilience layer
**Goal:** add engineering depth.

### Tasks
- implement VersionedModelPrototype,
- add round checkpointing,
- implement rollback after anomaly threshold breach,
- implement StragglerCircuitBreaker,
- simulate dropped/slow clients.

### Deliverables
- fault-injection results,
- recovery/rollback metrics,
- resilience pattern cards.

---

## Milestone 5 — Real dataset scaling
**Goal:** move from toy-scale to publishable evidence.

### Tasks
- move to NIH ChestX-ray14,
- implement multi-label or simplified binary pipeline,
- run IID and non-IID experiments,
- test FedAvg, FedProx, FedNova.

### Deliverables
- main benchmark tables,
- communication-time and convergence charts,
- central paper results draft.

---

## Milestone 6 — Cross-modality validation
**Goal:** show architecture generalizes.

### Tasks
- integrate ISIC 2020,
- optionally add one more MedMNIST modality,
- rerun selected best experiments,
- compare portability effort.

### Deliverables
- cross-modality generalization evidence,
- “architecture portability” discussion for paper.

---

## Milestone 7 — Paper packaging
**Goal:** convert engineering work into publishable paper assets.

### Tasks
- export clean figures,
- finalize pattern cards,
- create UML/component diagrams,
- produce reproducibility appendix,
- prepare artifact README.

### Deliverables
- final journal manuscript package,
- GitHub-ready artifact,
- submission checklist.

---

## 17. Experimental Design

You should run **five experiment families**.

### E1. Architecture quality experiment
**Question:** Does pattern-oriented implementation improve maintainability?

**Compare:**
- ad-hoc vanilla Flower code,
- FedPatterns-Med architecture.

**Metrics:**
- LOC,
- cyclomatic complexity,
- CBO,
- LCOM,
- duplication,
- change impact effort.

**Expected result:**
Pattern-based system is more modular, extensible, and easier to modify.

---

### E2. Privacy-performance experiment
**Question:** What is the cost of privacy-aware patterns?

**Compare:**
- no DP,
- DifferentialDecorator with low noise,
- DifferentialDecorator with medium noise,
- DifferentialDecorator with high noise.

**Metrics:**
- AUC,
- F1,
- sensitivity,
- specificity,
- epsilon,
- training time,
- communication cost.

**Expected result:**
Privacy increases protection but introduces predictable performance trade-offs.

---

### E3. Strategy and non-IID robustness experiment
**Question:** Which FL strategy behaves best when wrapped in the patternized architecture?

**Compare:**
- FedAvg,
- FedProx,
- FedNova,
- optional Scaffold,
- optional FedBN.

**Metrics:**
- convergence rounds,
- final AUC/F1,
- client drift,
- round completion time.

**Expected result:**
FedProx/FedNova or related robust strategies should outperform vanilla FedAvg under strong non-IID settings.

---

### E4. Resilience experiment
**Question:** Do rollback and straggler patterns improve reliability?

**Inject failures:**
- slow client,
- dropped client,
- bad update,
- checkpoint rollback event.

**Metrics:**
- recovery time,
- successful round completion rate,
- accuracy preservation after rollback,
- storage overhead.

---

### E5. Pattern ablation experiment
**Question:** Which patterns contribute the most?

**Ablation plan:**
- full architecture,
- minus DifferentialDecorator,
- minus VersionedModelPrototype,
- minus StragglerCircuitBreaker,
- minus RoundCommandHistory,
- minimal 2-pattern architecture.

**Metrics:**
- maintainability,
- runtime,
- privacy,
- robustness,
- paper contribution strength.

---

## 18. Baselines and Benchmark Targets

You asked for benchmarking against “existing high-end architectures.” Use the following structured baseline ladder.

### B0. Centralized training baseline
Train model on merged public data.
Purpose: upper bound for accuracy.

### B1. Vanilla Flower FL baseline
Simple Flower project using FedAvg with no pattern layer and no DP.
Purpose: engineering baseline.

### B2. Vanilla Flower + advanced strategy baseline
FedProx / FedNova / Scaffold without FedPatterns.
Purpose: algorithmic baseline.

### B3. Prior architectural-pattern-inspired baseline
Implement a lighter architecture inspired by previously published FL architectural patterns.
Purpose: compare against system-level architecture thinking.

### B4. FedPatterns-Med full system
Your full implementation with implementation-level patterns, compliance layer, resilience, and privacy components.
Purpose: main contribution.

### B5. FedPatterns-Med ablated variants
Purpose: isolate contribution of each pattern or layer.

---

## 19. Exact Benchmark Matrix

| Benchmark ID | Architecture | Strategy | Privacy | Patterns | Datasets | Main Purpose |
|---|---|---|---|---|---|---|
| B0 | Centralized PyTorch | N/A | None | None | MedMNIST, NIH, ISIC | Upper-bound reference |
| B1 | Vanilla Flower | FedAvg | None | None | MedMNIST, NIH | Simple FL baseline |
| B2 | Vanilla Flower | FedProx/FedNova | None | None | MedMNIST, NIH | Stronger FL algorithm baseline |
| B3 | Arch-pattern-inspired FL | FedAvg/FedProx | Optional | System-level only | MedMNIST | Compare against prior pattern work |
| B4 | FedPatterns-Med Full | FedAvg/FedProx/FedNova | DP + compliance | OO + composition | MedMNIST, NIH, ISIC | Main contribution |
| B5 | FedPatterns-Med Ablation | Same as B4 | Varied | Selected removed | MedMNIST, NIH | Contribution isolation |

---

## 20. Metrics You Must Collect

## 20.1 ML performance metrics
- Accuracy
- F1-score
- AUC-ROC
- Sensitivity / recall
- Specificity
- Precision
- Multi-label mAP if needed

## 20.2 FL system metrics
- Number of rounds to convergence
- Time per round
- Total wall-clock training time
- Communication cost per round
- Client participation rate
- Straggler rate

## 20.3 Architecture quality metrics
- Cyclomatic complexity
- CBO
- LCOM
- Maintainability index
- Code duplication
- Number of files touched per feature change

## 20.4 Privacy/compliance metrics
- Epsilon value
- Noise multiplier
- DP clipping norm
- Compliance coverage score
- Number of compliance violations flagged
- Audit completeness score

## 20.5 Resilience metrics
- Rollback success rate
- Fault recovery time
- Accuracy after recovery
- Checkpoint overhead
- Round completion under failures

---

## 21. Novel Features That Strengthen Journal Value

These are the features that make the work look more like a journal paper and less like a course project.

### F1. Pattern composition rules
Example:
- DifferentialDecorator + RoundCommandHistory may increase audit complexity.
- StragglerCircuitBreaker + FedNova may change fairness behavior.
- VersionedModelPrototype + SecureAggregationProxy may increase latency.

Write these as explicit composition rules.

### F2. Compliance-aware pattern cards
Each pattern should include a mini legal/compliance note.
This is unusual and publishable.

### F3. Architecture-quality evaluation
Most FL papers ignore software maintainability. This is your major SE contribution.

### F4. Multi-phase dataset strategy
Using lightweight → clinical-scale → heterogeneous multi-modality progression makes the work rigorous.

### F5. Reproducibility package
Provide configs, seeds, dataset scripts, and experiment manifests.
This helps with journal quality.

### F6. Failure injection testing
Most papers stop at accuracy. Fault injection makes the work more real.

### F7. Modeled hospital simulation
Name clients as hospitals with different specialties and distributions. This improves clarity and realism.

---

## 22. Threat Model and Privacy Assumptions

You should explicitly state the threat model in both implementation and paper.

### Assumptions
- Raw images never leave local client sites.
- Server is honest-but-curious unless secure aggregation extension is enabled.
- Clients may be slow, unavailable, or noisy.
- Some client updates may be anomalous.

### Threats considered
- model update leakage,
- membership inference risk narrative,
- metadata leakage through logs,
- participation leakage,
- stale checkpoints and audit retention risks.

### Threats not fully solved
- Byzantine robustness at scale,
- cryptographically rigorous secure aggregation if only proxied/hooked,
- full legal certification.

Being honest here improves paper credibility.

---

## 23. Implementation Plan in Antigravity IDE

Since you are using Antigravity IDE, keep the workflow simple and reproducible.

### Workspace setup
1. Create project root: `fedpatterns-med`
2. Create Python virtual environment
3. Install core packages:
   - `flwr`
   - `torch`
   - `torchvision`
   - `medmnist`
   - `opacus`
   - `fastapi`
   - `uvicorn`
   - `pymongo`
   - `pandas`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`
   - `radon`
4. Add optional tools:
   - `mlflow`
   - `wandb`
   - `pydriller`
5. Use one config file for all experiment settings.

### Recommended coding order
1. Vanilla Flower MedMNIST baseline
2. Refactor to patternized client/server architecture
3. Add FedAvg/FedProx/FedNova strategies
4. Add DifferentialDecorator
5. Add checkpoint versioning and rollback
6. Add straggler management
7. Add compliance mapping and reporting
8. Scale to NIH
9. Validate on ISIC
10. Run ablations and prepare paper assets

---

## 24. Practical Model Choices

Keep the models reasonable for your compute.

### For MedMNIST
- small CNN,
- ResNet18,
- EfficientNet-B0 optional.

### For NIH ChestX-ray14
- start with ResNet18 or DenseNet121,
- avoid giant ViTs at first,
- begin with binary/multi-label subset for feasibility.

### For ISIC 2020
- EfficientNet-B0 or ResNet18,
- binary classification is enough for strong paper evidence.

### Important rule
The paper is not a new SOTA medical model paper.
Do **not** overinvest in huge backbones.
Your novelty is architecture + patterns + compliance + evaluation.

---

## 25. Suggested Experiment Schedule

### Week 1–2
- Flower baseline on MedMNIST
- dataset loaders
- logging pipeline

### Week 3–4
- FederatedMediator and FederatedStrategy
- baseline maintainability analysis

### Week 5–6
- DifferentialDecorator with Opacus
- privacy-performance runs on MedMNIST

### Week 7–8
- VersionedModelPrototype and RoundCommandHistory
- failure injection scripts

### Week 9–10
- StragglerCircuitBreaker
- non-IID experiments with FedAvg/FedProx/FedNova

### Week 11–12
- NIH ChestX-ray14 scaling
- main benchmark results

### Week 13–14
- ISIC validation
- cross-modality comparison

### Week 15+
- ablation study
- paper writing and figure polishing

---

## 26. Figures You Should Produce for the Paper

1. Full layered system architecture diagram.
2. Pattern interaction / composition graph.
3. Sequence diagram for one FL round.
4. Compliance mapping matrix figure.
5. Privacy-accuracy trade-off plot.
6. Convergence curves across strategies.
7. Maintainability metric comparison bar chart.
8. Fault recovery timeline chart.
9. Cross-dataset portability summary table.

---

## 27. Tables You Should Produce for the Paper

1. Related-work comparison table.
2. Pattern catalog table.
3. Dataset table with links, sizes, modality, task.
4. Benchmark matrix table.
5. Maintainability metrics table.
6. Privacy-compliance risk table.
7. Non-IID robustness results table.
8. Fault-injection recovery table.
9. Pattern ablation table.

---

## 28. Writing the Journal Paper Alongside the Code

Write the paper in parallel instead of waiting until the end.

### Section plan
- **Introduction**: problem, gap, contributions.
- **Related Work**: FL, privacy-preserving FL, medical imaging FL, FL architecture/pattern research.
- **Methodology**: FedPatterns-Med pattern language, architecture, compliance mapping.
- **Implementation**: Flower-based platform, privacy layer, resilience layer.
- **Experimental Setup**: datasets, client simulation, strategies, metrics.
- **Results**: ML, system, maintainability, resilience, compliance.
- **Discussion**: trade-offs, practical lessons, limits.
- **Threats to Validity**: data, simulation realism, compliance interpretation.
- **Conclusion**: contribution recap.

### Writing rule
For every implemented feature, ask:
- what research question does this answer?
- what figure/table will prove it?

If the answer is “none,” reduce scope.

---

## 29. Risks and Mitigations

### Risk 1 — Too many patterns
**Mitigation:** keep 8 core patterns.

### Risk 2 — NIH is too heavy
**Mitigation:** start with MedMNIST and NIH subset, then scale.

### Risk 3 — DP hurts accuracy too much
**Mitigation:** tune epsilon/noise multiplier and report trade-off honestly.

### Risk 4 — Secure aggregation too complex
**Mitigation:** keep secure proxy abstraction and discuss as extension if needed.

### Risk 5 — Journal reviewers ask “where is the novelty?”
**Mitigation:** emphasize implementation-level pattern language + compliance mapping + maintainability evaluation + real medical validation.

### Risk 6 — Too much ML, not enough software engineering
**Mitigation:** make maintainability and architecture evaluation first-class results.

---

## 30. Minimal Publishable Version (MPV)

If time becomes tight, this is the smallest version that can still become a paper.

### Keep
- MedMNIST + NIH ChestX-ray14
- 8 core patterns
- FedAvg + FedProx + FedNova
- DifferentialDecorator
- VersionedModelPrototype
- StragglerCircuitBreaker
- architecture-quality evaluation
- privacy-performance evaluation

### Drop or postpone
- full cryptographic secure aggregation
- too many extra datasets
- too many advanced models
- too many optional patterns

---

## 31. Stretch Goals

If implementation goes smoothly, these can lift the paper further.

- add FedBN for modality/client drift,
- add fairness evaluation across clients,
- add secure aggregation backend,
- add poisoning detection trigger for rollback,
- add architecture evolution study across git commits,
- release pattern catalog as website/docs.

---

## 32. Final Execution Checklist

### Before coding
- [ ] Finalize 8 core patterns
- [ ] Finalize dataset phase plan
- [ ] Finalize benchmark baselines
- [ ] Create paper outline

### Before scaling
- [ ] MedMNIST baseline works
- [ ] Non-IID splitter works
- [ ] Patternized Flower structure stable
- [ ] Metrics logging complete

### Before experiments
- [ ] Reproducibility seeds fixed
- [ ] Configs versioned
- [ ] CSV result schema fixed
- [ ] Figures script-ready

### Before submission
- [ ] Related work clearly distinguishes prior FL architectural pattern papers
- [ ] Novelty claim is precise
- [ ] Tables and figures are publication quality
- [ ] Artifact is reproducible
- [ ] Threats to validity are honest

---

## 33. What Success Looks Like

A strong version of this project will show that:
1. pattern-oriented FL implementations are easier to maintain and extend,
2. privacy-aware components can be modularized as reusable patterns,
3. resilience mechanisms such as rollback and straggler handling improve operational robustness,
4. the architecture remains competitive on real medical imaging datasets,
5. software design concerns deserve equal attention alongside ML performance in federated healthcare systems.

That is the core message of the paper.

---

## 34. Immediate Next Actions

1. Set up the Antigravity IDE workspace.
2. Build the smallest Flower + MedMNIST baseline.
3. Freeze the 8 core patterns.
4. Create one pattern card example for FederatedStrategy.
5. Implement non-IID partition builder.
6. Run baseline FedAvg.
7. Refactor into patternized architecture.
8. Add DifferentialDecorator.
9. Start writing the related work table immediately.

---

## 35. Citation Targets to Mention in the Paper Draft

Use these categories in related work:
- FL architectural patterns,
- FL architectural pattern selection,
- FL pattern performance analysis,
- medical imaging FL surveys,
- privacy-preserving FL surveys,
- differential privacy tooling,
- Flower framework documentation.

This makes your paper well-positioned academically and technically.
