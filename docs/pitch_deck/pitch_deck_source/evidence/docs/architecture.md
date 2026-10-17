# MedRelay architecture (as implemented)

```
                      clinical text                         structured + streaming events (replay stream: late, duplicate, out-of-order)
                           |                                                  |
        +------------------v------------------+                               |
        | clinical_extractor.py               |                               |
        |  A rules | B GenAI | C/C2 validated | <- llm_backend.py (llama.cpp, local GGUF, big-endian converted)
        |  D +action classifier | E rules+GenAI|                               |
        +------------------+------------------+                               |
          validated ExtractedObligation records (Pydantic)                    |
                           |   obligation_created events                      |
                           +---------------------------+----------------------+
                                                       v
                                     event_processor.py (idempotent, SQLite WAL, timers, micro-batching)
                                       validate -> graph update -> state recompute -> features -> risk -> alerts -> audit
                                          |            |                      |          |
                                          v            v                      v          v
                           evidence_graph.py   risk_predictor.py      alert_engine.py   SQLite tables
                          (NetworkX graph;     (calibrated ML;        (rules R1-R7 ->   events / obligations /
                           deterministic        features computed      alerts, priority  transitions / alerts /
                           state rules)         BY the graph)          score, grounded   risk_log / rejected
                                                                       explanations)
```

## Separation of responsibilities
| component | decides | never decides |
|---|---|---|
| GenAI extractor | *proposes* obligation records (type, span, deadline, role, dependencies) | workflow state; whether anything was completed |
| validators | accept/reject/repair proposals using the source text and deterministic parsers | clinical meaning |
| evidence graph | workflow **state** (pending / completed with evidence / overdue / unverified / contradictory / not yet assessable) from documented evidence, with the rule id and evidence ids | forecasts |
| ML risk model | calibrated **forecast** that an open obligation is unresolved at its deadline | already-overdue status (deadline logic) |
| alert engine | fixed rules combining state, forecast and dependency facts; priority score; explanation | treatment, orders, diagnoses |

## Workflow state rules (`evidence_graph.derive_state`)
| rule | condition | state |
|---|---|---|
| R2 | a conflicting secondary record exists | contradictory evidence |
| R3 | completion-type event exists but is not verified | unverified / insufficient evidence |
| R1 | verified completion event of **this** obligation's own step type (late flag if after deadline) | completed with evidence |
| R4 | no completion, deadline not reached | pending |
| R5 | deadline passed, no completion, **positive** evidence of non-completion (tracker snapshot "open", documented no-show/cancellation) | overdue without verified completion |
| R6 | deadline passed, no completion, still inside the grace window for delayed evidence | censored / not yet assessable |
| R7 | deadline passed, grace over, no evidence either way | unverified / insufficient evidence |

* A completed upstream step never completes a downstream one. A completion whose parent has no documented completion is flagged `dependency_evidence_gap`.
* States are a pure function of the **set** of ingested evidence ordered by *event time*; duplicates are no-ops; late evidence reconciles the state (and the audit log records both event time and ingestion time).
* `live=True` evaluation counts only evidence already **ingested** by the evaluation time (what the running service could know).

## Graph vocabulary
Node types: Patient, Encounter, ClinicalNote, Obligation, Task, Result, Review, Acknowledgment, Notification, EvidenceEvent, ResponsibleRole (event nodes take the artefact type their event implies).
Edge types: EXTRACTED_FROM, DEPENDS_ON (kept acyclic; cycles are rejected), ASSIGNED_TO, EVIDENCED_BY, COMPLETED_BY, REQUIRES_REVIEW, CONTRADICTS, BELONGS_TO.

## Risk features (strict event-time cutoff)
Groups: `task` (category, step, role, window, time to deadline ...), `own_events` (independent-task view), `workload_history` (prior completions, concurrent obligations, department transitions),
`graph_dependency` (chain position, unresolved dependencies/ancestors, parent state, role hand-offs), `graph_temporal` (chain-propagated event counts/recency, scheduling lead), `evidence` (acknowledgment state, verified/unverified counts, conflicts).
Systems: **B** = task + own_events + workload_history (independent task monitoring); **C** = all groups (dependency-aware graph). Forbidden as features: outcomes, completion timestamps after the cutoff, generator profile/flags, identifiers, split names.

## Persistence and recovery
SQLite (WAL) tables: `events` (unique `event_id`, delivery count), `obligations`, `transitions` (audit log), `alerts`, `risk_log`, `rejected`. The in-memory graph is a cache: `EventProcessor.recover` rebuilds it from the event table.
Batched commits are single atomic transactions.

## Data provenance rules enforced in code
Real-source / synthetic / weak labels are kept apart: action labels have a `label_tier` (gold = 0, silver, synthetic template examples train-only); follow-up outcomes are simulated and described as such;
the no-show label is real attendance and never mixed with obligations; hidden generator fields live only in ground-truth files and are never present in live payloads.
