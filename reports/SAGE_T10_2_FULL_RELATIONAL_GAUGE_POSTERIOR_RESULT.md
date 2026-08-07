# SAGE.T10.2 - Full Relational Gauge Posterior: final result

## Outcome

The exclusive T10.2 verdict is:

`DATA_OR_PROVENANCE_INVALID`

T10.2 is not supported, no integration pilot is authorized, and source
validation, `ar25`, the holdout, and production authority remain closed.

The amended source collection exceeded its registered 5,400-second wall
limit and was externally terminated before the runner returned from the
current lane.  No source ledger, cross-fit audit, or collection report was
persisted.  Consequently, the exact number of executed actions, completed
lanes, outcomes, and coverage cannot be attested.  The resource overrun is a
secondary finding, but the first applicable verdict in the registered ladder
is the higher-priority provenance failure.

## Frozen lineage and publication

| Item | Value |
|---|---|
| T10.1 immutable baseline | `05c1c91b82054af55a03ef962745f4f101cd3c0e` |
| T10.1 manifest | `4d1c4dc8b62973187ea5e1c52e698652fdaeb424ae481b56baded0c0b2b9c1a3` |
| T10.1 fail-closed report | `167649e5a0e27d63668ca20ae98c57dfd50dd469204ce53a7a5af488fae6348a` |
| Initial T10.2 preregistration commit | `c12f7ae` |
| Initial T10.2 manifest | `018bff2edd117d9440c3f86b6a179174464327baac904f52e6576176b89cdd0b` |
| A1 amendment commit | `c4402cc094ef643c2bb3a0c38f44a626f0c27110` |
| Amended T10.2 manifest | `f3f7a433140bb0e89ac641efc32900fc9dbbd6f701bb4b3b0cfdb193f869a8ef` |
| Draft pull request | `https://github.com/tishka04/adaptive_reasoning/pull/7` |

The amended manifest preserved the frozen T7-T10.1 hashes for contracts,
posterior, executor, decision engine, and controller.  It bound 37 code files,
the three authorized source shards, the four structural frames, and the
compact event/quotient schemas.

## Implemented system

T10.2 implements the complete joint particle
`H = (dynamics, goal, frame, transports, option)` without changing the frozen
`JointProgramHypothesis`, `ProgramPosterior`, or `SageTController` contracts.
The implementation includes:

- one physical `PhysicalEventBundle` with a single common outcome and four
  deterministic structural projections;
- the frozen `root_only`, `allocentric_object_relative`,
  `action_aligned_relational`, and `action_rooted_topological` frame bank;
- exact and partial `TransportMap` certificates, inverse and commutativity
  checks, closed structural schemas, and endpoint-free compact quotients;
- mixed `OptionAutomaton` macros with at most four states, two action schemas,
  horizon 16, one real action followed by observation and replanning;
- a `JointGaugeHypothesis`, gauge-equivalence quotient, bounded 256-class
  posterior, 64-class decision evaluation, residual mass preservation, and
  delayed MAP collapse;
- one-outcome likelihood accounting, projection-score averaging,
  commutativity penalties, existing SAGE.T likelihood weights and MDL priors;
- counterfactual decisions, danger veto, incomplete-projection fallback,
  cross-fit isolation, source/validation firewalls, resource gates, and the
  seven registered CLI phases.

Raw grids, raw frames, complete graphs, colors, absolute coordinates,
persistent identifiers, and game identity are not admitted to transferable
programs or persisted model views.

## Pre-collection verification

| Check | Result |
|---|---|
| T10.2 dedicated tests | 287 passed: 286 combined plus the 321.97-second source-trainer test separately |
| Final protocol/runtime subset | 81 passed with the long test isolated above |
| Historical SAGE.T suite | 564 passed, one already-proven long test deselected |
| Reused V4.9/V4.16/V4.19 encoder tests | 18 passed |
| Full repository suite | 2,383 passed, one long T10.2 test deselected |
| Full-suite failures | 2 baseline-reproduced missing optional assets; no T10.2 regression |
| Formatting, lint, bytecode compilation | passed |

The two known full-suite failures are the absent pretrained checkpoints used
by `test_agent_runtime_config.py` and the absent local Qwen safetensors asset
used by `test_sage12_integration_pilot_v4_7.py`.  They match the T10.1 baseline
environment and are not caused by T10.2.

## A1 technical amendment

The first `collect` invocation under manifest `018bff2e...` executed one legal
`bp35` action and then failed before bundle sealing because `_make_bundle`
passed custom-builder-only keywords to the closed default bundle signature.
No event or artifact was retained and no fit ran.

A1 fixed only that dispatch, added a regression test, and introduced a hard
remaining budget of 4,607 new actions so the one earlier action still counted
toward the registered 4,608-action ceiling.  Seeds, splits, frames, gates,
priors, controls, and validation conditions were unchanged.  A1 was committed,
pushed, and recorded in the draft PR before the amended collection began.

## Amended collection attempt

| Evidence | Observation |
|---|---|
| Command | `.sage12_cache/v4_18/runtime/Scripts/python.exe -m theory.sage_t.t10_2_protocol collect` |
| Local process start | 2026-08-07 16:39:20 Europe/Paris, externally observed |
| Registered wall limit | 5,400 s |
| External termination | 5,600.241 s (`command timed out`) |
| Overrun | 200.241 s, about 3.71% |
| Source monotonic timing proof | not available |
| Peak observed working set | about 355 MiB |
| Output directory immediately after termination | absent |
| Residual Python/ARC process | none |
| Collection report | not created |
| Source ledger | not created |
| Cross-fit audit | not created |
| Exact actions executed | unknown and not attested |
| Upper bound from amended runtime | at most 4,607 new actions; at most 4,608 including A1 |

The runner checks elapsed time before and after each lane, not inside the
environment call.  The external timeout therefore terminated the worker while
one lane was still executing, before the runner could issue its own
`ResourceGateError`.  No partial source data is recoverable or admissible.

## Phase disposition

| Phase | Status | Reason |
|---|---|---|
| `freeze` | complete | amended manifest published before the scientific attempt |
| `collect` | `ABORTED_EXTERNAL_TIMEOUT` | wall limit exceeded; no durable provenance |
| `compile` | `NOT_RUN_PREREQUISITE_MISSING` | no collection report or source ledger |
| `replay` | `NOT_RUN_PREREQUISITE_MISSING` | fresh integrity gate unavailable |
| `source-train` | `NOT_RUN_DATA_INVALID` | fitting forbidden before valid combined QA |
| `validate` | `BLOCKED_NOT_OPENED` | no signed `PASS_T10_2_SOURCE_GATE` |
| `report` | complete | terminal fail-closed reports reconstructed and signed |

The compact `compile_report.json` published with this result is a signed
record of the compilation refusal, with `compile_invoked=false`; it is not a
claim that compilation ran.  Likewise, `source_report.json` is a terminal
provenance report built with `trainer_invoked=false`, not a source-training
result.

## Registered controls and oracles

All 23 registered source controls have status
`NOT_RUN_PREREQUISITE_EVIDENCE_ABSENT`.  The requirement to execute every
control after a scientific gate failure does not permit controls to be run on
missing or unauditable physical bundles.

| Control | Status |
|---|---|
| `t10_1_behavior_frozen_baseline` | not run |
| `capacity_matched_independent_posterior` | not run |
| `single_frame_root_only` | not run |
| `single_frame_allocentric_object_relative` | not run |
| `single_frame_action_aligned_relational` | not run |
| `single_frame_action_rooted_topological` | not run |
| `identity_only_transport` | not run |
| `no_transport` | not run |
| `deterministically_permuted_transport` | not run |
| `frame_swap` | not run |
| `binding_swap` | not run |
| `dynamics_swap` | not run |
| `goal_swap` | not run |
| `option_swap` | not run |
| `early_map_collapse` | not run |
| `immediate_noop_deduplication` | not run |
| `best_executed_sequence_oracle` | not run |
| `grammar_oracle` | not run |
| `transport_oracle` | not run |
| `dynamics_oracle` | not run |
| `goal_oracle` | not run |
| `option_oracle` | not run |
| `complete_program_oracle` | not run |

No F1, likelihood, rank, causal degradation, oracle recovery, level rate, or
latency result is reported.  There is no evidence from this run about whether
the relational gauge posterior resolves the three T10.1 `SEQUENCE_MISS`
failures.

## Validation and authority firewall

Paired validation on `re86`, `ls20`, and `sc25` was never opened.  No seed in
2101-2105 was executed, no T10.1/T10.2 pair was created, and no validation
latency or utility statistic exists.  `ar25`, the holdout, and production
authority remained closed throughout.

## Compact artifacts

The post-failure report graph is:

`compile refusal -> terminal source report -> final report -> inventory binding`

| Artifact | Internal checksum |
|---|---|
| `training/sage_t/t10_2_gauge_posterior/compile_report.json` | `fff2d2125075da8708c1adae040b1e8c6e0c1b20825f1e71dea2a7c179f2eb30` |
| `training/sage_t/t10_2_gauge_posterior/source_report.json` | `cbccc5738a7005669e3a3b8fd837591bac574642370ec4eeb6720b0f776e5624` |
| `training/sage_t/t10_2_gauge_posterior/report.json` | `76f9ad0ca976b32c3f36ac132a22a9c9d4984a90e9d4848c3466e1f9997410e0` |
| `training/sage_t/t10_2_omitted_artifacts_inventory.json` | `9c007cb71866ae72a46c158977d8363005834bb7eb6c4f85a1f2a75b967acf95` |
| `training/sage_t/t10_2_report_inventory_binding.json` | `b8eca4f5a38b2b632ec507cdc1bc603160fb3bedb6d4acdc34618806e697f674` |

The inventory contains exactly the three compact lifecycle reports above.
It records zero omitted T10.2 files because the failed collection never
persisted a ledger, cache, checkpoint, projection, or raw graph.  No file was
deleted.  The pre-existing T10.1 raw/regenerable artifacts remain local and
excluded under their separate baseline inventory.

Re-running `report` produced byte-identical compile, source, final, inventory,
and binding files.  All five pass the registered compact-artifact schemas.

## Final interpretation

The implementation and its synthetic/non-regression evidence are complete,
but the registered active experiment is not scientifically interpretable.
The provenance failure is terminal for T10.2.  A new attempt would require a
new version and a new manifest, with incremental lane-level durable accounting
and an in-lane wall-clock stop specified before any new action.  It would not
authorize the T10.3 integration pilot, `ar25`, holdout access, or production
authority.
