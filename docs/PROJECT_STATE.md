# FedPatterns-Med — Project State & Handoff Document

> Generated as a context handoff for a new chat session. This document is the
> single source of truth for everything decided and discovered so far.
> Read this BEFORE looking at plan.md or implementation_plan.md, since this
> file explains what's actually been verified/built vs. what's still just planned.

---

## 1. Project Identity

**Working title:** FedPatterns: A Design Pattern Language for Privacy-Compliant
Federated Learning Architectures with Empirical Validation on Medical Imaging Systems

**Refined journal title:** FedPatterns-Med: OO Design Pattern Compositions with
Privacy-Compliance Mapping for Federated Learning in Medical Imaging — An
Empirical Study

**Core pitch:** A formal catalog of GoF/OO-level design patterns for federated
learning system implementation (not just architecture-level, which is what
prior work covers), paired with a per-pattern privacy-regulation compliance
mapping (GDPR/HIPAA/India DPDP), empirically validated on real medical imaging
data (not synthetic CIFAR-10-style benchmarks like prior work).

**Course context:** Built for a Software Design Patterns course (22AIE460),
targeted for eventual conference → journal publication.

**IDE:** Antigravity IDE. **Path:** `/home/subhash/projects/fedpatterns-med`

---

## 2. Verified Novelty Claim — Do Not Re-Litigate This, It's Already Checked

This was independently fact-checked against real, traceable sources (not assumed):

| Ref | What it actually did | What it does NOT cover |
|---|---|---|
| Lo et al. (arXiv:2101.02373, JSS) | 15 architectural patterns for FL (client mgmt, model mgmt, training, aggregation) | No GoF code-level patterns, no compliance mapping, no medical validation |
| Lo et al. follow-up (2022) | Decision model for picking which architectural pattern to use | Same gaps as above |
| Trubiani et al. (2025, JSS/ScienceDirect — confirm exact venue before citing) | Quantitatively benchmarked 6 architectural patterns via Flower, on CIFAR-10-style data | Performance-only, no privacy/compliance layer, no real medical imaging |
| Various FL+GDPR/HIPAA papers | Treat privacy as algorithm (DP, homomorphic encryption) or legal commentary | None structure privacy as a reusable software design pattern with compliance annotations |

**Citable gap statement (verified, use near-verbatim in the paper's Introduction):**
> "While Lo et al. introduced architectural patterns for FL system design, and
> subsequent work quantitatively evaluated a subset of these patterns, both
> research lines treat patterns as system-level architectural concerns
> evaluated on synthetic benchmarks. No prior work applies Gang-of-Four
> object-oriented design patterns at the component implementation level of FL
> systems, maps pattern compositions to healthcare privacy regulations, or
> empirically validates pattern-based FL architectures on real medical imaging
> benchmarks combined with software-quality metrics."

**Action before final submission:** re-verify Trubiani et al.'s exact venue/year and cite the published version, not an arXiv preprint, wherever possible.

---

## 3. Current Project Status (as of this handoff)

### M0 — Paper Framing & Setup: ✅ COMPLETE
- Conda env `fedpatterns-med` (Python 3.11), `flwr[simulation]==1.31.0` (modern `ClientApp`/`ServerApp` API, not deprecated `start_simulation()`)
- Full repo scaffold per plan.md Section 11, package discovery configured (`apps`, `baselines`, `fedpatterns` all properly installable via `pip install -e .`)
- 8-pattern catalog frozen (table in Section 8 below) — no changes without explicit approval
- `compliance_matrix.yaml` canonical source at `fedpatterns/privacy/compliance/`, root copy auto-generated
- CUDA verified (RTX 3050 Mobile, 4GB VRAM), `num_gpus=0.2` per client, mixed precision via `torch.cuda.amp`

### M1 — Vanilla FedAvg Baseline (B1): ✅ COMPLETE AND FROZEN
- **Canonical run_id:** `165d7ae7eda14086a70414d61d305209` (run name "big-dolphin-82"), tagged `canonical_baseline=true` in MLflow experiment `M1-B1-VanillaFedAvg`
- **Canonical metrics:** accuracy ≈ `0.8189`, val_loss ≈ `0.523–0.566` (range reflects documented ~0.042 residual GPU non-determinism — see Section 6), cyclomatic_complexity = `3.38888888`
- `baselines/b1_vanilla_flower_fedavg/` is **FROZEN** — do not modify further. Any future infra improvements must not retroactively change this reference point.
- MLflow experiment cleaned — contains exactly 2 runs (the final determinism-check pair), 10 stale pre-fix runs were purged

### M2 — Patternized Architecture (B4, P1+P2): 🔶 IN PROGRESS — 2 OPEN ITEMS
- `PatternizedFedAvg` (P2/FederatedStrategy) and `FederatedMediator` (P1) implemented in `fedpatterns/patterns/strategy/` and `fedpatterns/patterns/mediator/`
- Entry points live in `fedpatterns/core/app/` (server_app.py, client_app.py, run_patternized.py) — **NOT** `baselines/b4_.../`. B4 labeling is done via MLflow tag `baseline_id="B4"`, not folder placement. (This was a deliberately resolved decision — see Section 4.)
- `apps/server_app/main.py` re-exports from the currently-active implementation (Option 1 convention)
- Cyclomatic complexity corrected to `2.71` after fixing a radon path bug (was scanning B1's directory instead of M2's)

**OPEN ITEM 1 — stale run cleanup needed:** MLflow experiment `M2-B4-PatternizedFedAvg` currently contains TWO runs: one correct (`spiffy-be...`, CC=2.71) and one stale pre-fix run (`indecisiv...`, CC=3.39, leftover from before the radon path bug was fixed). This needs to be deleted and the correct run tagged `canonical_baseline=true`, same as B1.

**OPEN ITEM 2 — unconfirmed: was the determinism fix ported to M2?** B1's `server_app.py` was modified to sort incoming client results by client ID before aggregation, to fix floating-point accumulation-order non-determinism (this was found AFTER the original M2 plan was written). **It is not yet confirmed whether `FederatedMediator`/`PatternizedFedAvg`'s `aggregate_fit` does the same sorting.** If it doesn't, any variance observed between M1 and M2 results is confounded by this difference, not purely reflecting the architectural refactor — which would undermine the Experiment E1 metric-parity claim. This must be confirmed/fixed and a 2-run determinism check run on M2 itself before M2 is considered done.

**Minor open item:** walkthrough.md states M1 accuracy as `0.821`; canonical run actually logged `0.8189...` — fix for citation precision.

### M3+ — Not yet started
Per plan.md: M3 = Privacy/compliance layer (DifferentialDecorator P3 via Opacus, ConsentObserver P5). Do not start until M2's two open items above are resolved.

---

## 4. Key Technical Decisions Log (all confirmed, do not re-litigate)

| Decision | Resolution |
|---|---|
| Flower version | `flwr[simulation]==1.31.0`, modern `ClientApp`/`ServerApp` simulation API (`flwr run`), not deprecated `start_simulation()`. (Earlier docs said ">=2.0" — that was a documentation error, corrected; 1.31.0 with the modern API is what's actually pinned and working.) |
| Client count | 5 simulated clients, project-wide default, all milestones |
| GPU config | CUDA default (RTX 3050, 4GB VRAM); `num_gpus=0.2` per client (5×0.2 = full allocation); mixed precision via `torch.cuda.amp` |
| MLflow cross-process tracking | `run_id` passed via environment variable (`MLFLOW_RUN_ID`) from the launcher script to the `flwr run` subprocess; metrics logged via `mlflow.log_metrics(..., run_id=run_id)` directly (not `with mlflow.start_run()`) to avoid RUNNING→FINISHED status flapping across rounds |
| MLflow backend | SQLite (`sqlite:///mlflow.db`), migrated from deprecated filesystem backend |
| Flower config injection | `--run-config` argument used to pass `mlflow-run-id`/`mlflow-tracking-uri` into the isolated Flower simulation process (this was a fix needed specifically for 1.31.x's process model) |
| Compliance YAML | Single source of truth at `fedpatterns/privacy/compliance/compliance_matrix.yaml`; root-level copy auto-generated via `scripts/generate_root_compliance.py`, never hand-edited |
| `apps/` role | Stable re-exports (Option 1): root `pyproject.toml` always points to `apps.server_app.main:app`; switching active baseline = editing the one import line inside `apps/server_app/main.py`, not touching `pyproject.toml` |
| Package discovery | Explicit `__init__.py` throughout `apps/`, `baselines/`, `fedpatterns/`; `[tool.setuptools.packages.find]` configured in `pyproject.toml`; **`pip install -e .` must be re-run after scaffold changes** — this was a real bug source (forgetting this step causes `ModuleNotFoundError` even with correct config) |
| `fedavg.yaml` defaults | `num_rounds: 3, fraction_fit: 1.0, min_fit_clients: 5, min_evaluate_clients: 5, min_available_clients: 5, seed: 42` |
| Architecture quality metric | `radon cc -j` (JSON output, not text-scraping — avoids fragile parsing), averaged per-block, logged to MLflow as `cyclomatic_complexity` |
| `run_b1.py`/`run_patternized.py` invocation safety | `PROJECT_ROOT` resolved via `pathlib.Path(__file__).resolve().parents[N]`, passed as `cwd` to all `subprocess.run` calls — makes scripts safe to invoke from any directory |
| Determinism — Ray worker seeding | Per-client-per-round deterministic seed: `set_seed(42 + partition_id + round_num*1000)`, called at the start of `fit`/`evaluate`, with `server_round` injected via `on_fit_config_fn`/`on_evaluate_config_fn`. Fixes a race condition where Ray's worker-process pooling caused seed reuse across mismatched client/round assignments. |
| Determinism — cuDNN | `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` — set in `client_app.py` |
| Determinism — floating-point accumulation order | B1's `server_app.py` sorts incoming client results by client ID before aggregation, since Ray's async client returns vary in order run-to-run and float addition isn't associative. **This eliminated most but not all variance.** |
| Residual variance — accepted and documented | ~0.042 spread in val_loss across "fully" deterministic runs, attributed to PyTorch CUDA mixed-precision (`autocast`/`GradScaler`) atomic operations (e.g. `scatter_add`) that have no deterministic implementation without `torch.use_deterministic_algorithms(True)` (which risks crashes/severe slowdown — deliberately NOT enabled). This is the empirical, verified basis for the "expected variance" disclaimer in M2's verification plan. |
| Epsilon sentinel | `epsilon = -1.0` is the documented sentinel for "no DP applied" in baseline runs (real ∞ isn't cleanly storable in MLflow/CSV numeric fields) — documented in `fedpatterns/evaluation/privacy_metrics/__init__.py` |
| B4 directory placement | Lives in `fedpatterns/core/app/`, NOT `baselines/b4_.../` — because per plan.md Section 14, B4 is the paper's main contribution, not an external comparison baseline (those are B0–B3 in `baselines/`). Tagged `baseline_id="B4"` in MLflow for benchmark-matrix queryability despite the different folder. |
| Straggler threshold | 0.30 (code) — planned 0.20 proved insufficient to trigger circuit breaker in 3 rounds under the deterministic seed scheme; 0.30 locks in M4 and forward. |

---

## 5. Immediate Next Actions for the New Chat (in order)

1. **Resolve M2 Open Item 1:** Delete the stale `indecisiv...` run (CC=3.39) from `M2-B4-PatternizedFedAvg`. Confirm/tag the correct run (`spiffy-be...`, CC=2.71) as `canonical_baseline=true`.
2. **Resolve M2 Open Item 2:** Confirm whether `FederatedMediator`/`PatternizedFedAvg`'s `aggregate_fit` sorts client results by client ID (matching B1's fix). If not present, add it. Then run a 2-run determinism check on M2 itself and report actual numbers (not just "finished") — confirm variance is in a similar ~0.04 range to B1's.
3. **Minor:** Correct walkthrough.md's M1 accuracy figure from `0.821` to the precise canonical value `0.8189...`.
4. Once 1–3 are clean and verified (not just claimed), proceed to drafting the M3 implementation plan: DifferentialDecorator (P3, via Opacus) + ConsentObserver (P5).

---

## 6. Working Methodology — Carry This Forward, It's Been Catching Real Bugs

This is important: **across the M0–M2 review rounds, directly cross-checking MLflow UI screenshots/CSV data against the agent's narrative summary caught a real, distinct issue in the majority of rounds** — including a radon path bug that silently produced identical complexity scores across genuinely different codebases, twice; a metrics aggregation bug inflating train_loss 7x; a non-determinism source the agent initially believed was fixed but wasn't; and stale/orphaned MLflow runs left uncleaned on more than one occasion.

**Rules for the new chat to follow:**
- Never accept "complete"/"verified"/"finished" from the agent's narrative alone — ask for the actual screenshot, CSV, or MLflow query output, and check the specific numbers against what's claimed.
- Watch specifically for: identical metric values across code that should differ (signals a path/scope bug, not a real result), reported deltas that don't arithmetically match the actual data shown, and extra/stale runs sitting in an experiment that weren't mentioned in the summary.
- Require explicit before/after numbers for any claimed fix (e.g., determinism checks) — not just "I verified it works."
- `baselines/b1_vanilla_flower_fedavg/` is frozen. No further edits, regardless of what future milestones seem to require — if something there genuinely needs to change, that's a decision to surface explicitly, not do silently.

---

## 7. Files That Should Also Be Available in the New Chat

Upload these alongside this document:
- `plan.md` — the master 27-section blueprint (architecture, datasets, experiments, benchmarks, full repo tree)
- `implementation_plan.md` — the current M0+M1+M2 task tracker (detailed, file-level)
- `compliance_matrix.yaml` — current state of the 8-pattern × 5-dimension compliance mapping
- `walkthrough.md` — the agent's running narrative log (treat with the skepticism described in Section 6 — verify against MLflow, don't take at face value)

---

## 8. Frozen Pattern Catalog (P1–P8) — Reference

| ID | Pattern Name | Base GoF | Implementation File | Status |
|---|---|---|---|---|
| P1 | FederatedMediator | Mediator | `fedpatterns/patterns/mediator/federated_mediator.py` | ✅ Implemented (M2) |
| P2 | FederatedStrategy | Strategy | `fedpatterns/patterns/strategy/federated_strategy.py` | ✅ Implemented (M2) |
| P3 | DifferentialDecorator | Decorator | `fedpatterns/patterns/decorator/differential_decorator.py` | ⬜ M3 |
| P4 | SecureAggregationProxy | Proxy | `fedpatterns/patterns/proxy/secure_aggregation_proxy.py` | ⬜ Later |
| P5 | ConsentObserver | Observer | `fedpatterns/patterns/observer/consent_observer.py` | ⬜ M3 |
| P6 | VersionedModelPrototype | Prototype + Memento | `fedpatterns/patterns/prototype/versioned_model_prototype.py` | ⬜ M4 |
| P7 | StragglerCircuitBreaker | Resilience chain | `fedpatterns/resilience/straggler/straggler_manager.py` | ⬜ M4 |
| P8 | RoundCommandHistory | Command | `fedpatterns/patterns/command/round_command_history.py` | ⬜ Later |

P1–P8 are frozen — no additions, drops, merges, or substitutions without explicit user approval.

---

## 9. Known Risks to Watch in Upcoming Milestones

- **M3 (DifferentialDecorator/Opacus):** real epsilon values will start being logged (replacing the `-1` sentinel) — confirm the metrics module correctly distinguishes "no DP" (`-1`) from "DP with very large epsilon" (a real but large number) so these aren't conflated in later analysis/plots.
- **Determinism fixes must be ported to every new milestone's aggregation code**, not just M1/M2 — this has already required fixing twice (Ray worker seeding, then float accumulation order); a new pattern composition in M3+ could reintroduce similar issues if the same care isn't applied.
- **Re-verify radon's scan path every time a new entry point is added** — this exact bug occurred twice already (M2 first implementation, caught; recurrence in the cleanup itself was caught too). Make this an explicit, named verification step for every future milestone, not an assumption.
- **MLflow experiment hygiene** — stale/duplicate runs have appeared in both M1 and M2 at least once each. Consider establishing a simple convention now (e.g., a script that lists all runs in an experiment and flags if more than one is untagged/uncleaned) rather than catching this manually every milestone.
