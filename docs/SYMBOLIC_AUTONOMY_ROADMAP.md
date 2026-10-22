# EvolveAI Symbolic Autonomy Roadmap

## Non-negotiable constraint

EvolveAI must not call hosted or local large language models. The core uses explicit symbolic representations, classical/statistical methods, grammar-based language processing, and registered tools.

This does not imply consciousness or unrestricted general intelligence. Capability is measured by reproducible task outcomes.

## Architecture boundaries

- `evolveai/autonomy/models.py`: serializable goals, action specifications, observations, and run results.
- `evolveai/autonomy/planner.py`: deterministic STRIPS-style planning over declared preconditions and effects.
- `evolveai/autonomy/controller.py`: bounded execution, action authorization, observation recording, replanning, and termination.
- `evolveai/autonomy/registry.py`: explicit action registry; no arbitrary tool discovery or execution.
- Existing `EvolveAgent`: remains the conversational entry point while the autonomy engine is developed and tested independently.

## Milestone 1: a tested vertical slice

1. Represent goals as desired predicates and an explicit initial state.
2. Represent actions with preconditions, add effects, delete effects, cost, and risk.
3. Find a plan using bounded breadth-first search.
4. Execute only registered actions whose preconditions hold.
5. Require explicit policy approval for high-risk actions.
6. Verify the goal against the resulting state; do not mark a task complete merely because actions returned successfully.
7. Record each decision and observation so a run can be audited and resumed later.

## Planning semantics

Predicates are canonical strings, for example `file_loaded:ledger.csv` or `report_verified`. An action is applicable when all preconditions are present. Applying it removes delete effects and adds add effects. A plan is valid only if its simulated final state contains every goal predicate.

The initial planner uses breadth-first search, suitable for small action spaces. Before expanding to large search spaces, add state hashing, a node limit, action-cost-aware search, and domain-specific heuristics.

## Safety and reliability

- Set maximum action count, wall-clock deadline, and per-action timeout at the execution adapter.
- Treat action descriptions and data from tools as untrusted input, never as instructions to change policy.
- Keep file, network, shell, and account access deny-by-default.
- Require approval for destructive, financial, externally visible, or privilege-changing actions.
- Make actions idempotent where possible and store action IDs/results to avoid duplicate side effects after retries.
- Separate evidence confidence from task success. Verify postconditions independently.
- Never automatically promote an inferred rule to trusted knowledge without validation and versioning.

## Later milestones

### Milestone 2 — durable experience and belief revision
Move operational state to a shared persistent database. Store claims, source spans, timestamps, confidence model/version, and explicit support/dependency edges. Retract or recompute dependent conclusions when evidence is invalidated.

### Milestone 3 — research and experiments
Add allowlisted search and document adapters, source deduplication, claim extraction, contradiction handling, and sandboxed code execution. Experiments must have a hypothesis, expected observation, resource limit, and recorded result.

### Milestone 4 — learning
Start with calibrated statistical classifiers, rule mining, and outcome-based action statistics. Evaluate candidate rules against held-out examples and counterexamples before deployment. Preserve a rollback path.

### Milestone 5 — non-LLM language interface
Build a parser pipeline with tokenization, grammar/intent classification, entity and slot extraction, ambiguity detection, and grammar-based response realization. Ask clarification questions when multiple interpretations remain plausible.

### Milestone 6 — background operation
Use a durable job queue, leases, checkpoints, cancellation, retry policies, and a detailed append-only activity log. Background tasks must have a declared objective, permissions, budget, and stop conditions.

## Acceptance criteria

- Every plan is explainable in terms of action preconditions and effects.
- The planner never returns a plan that fails simulated goal verification.
- Execution stops on exhausted budgets, denied actions, missing preconditions, or failed postconditions.
- High-risk actions cannot bypass the authorization policy.
- Tests cover unsatisfied goals, impossible plans, cycles, action failure, and budget exhaustion.
- No LLM SDK, model endpoint, or model credential is required by this subsystem.
