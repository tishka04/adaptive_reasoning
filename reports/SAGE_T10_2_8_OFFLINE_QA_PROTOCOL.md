# SAGE.T10.2.8 — Lineage-aware offline QA protocol

## Scope

T10.2.8 is a strictly offline, pre-fit gate over the accepted T10.2.7 source
collection. It makes no environment call, issues no physical action, replays no
action, fits no model, and opens neither source validation nor AR25. Every
T10.2.2–T10.2.7 journal and accepted artifact remains read-only.

The purpose of this phase is to answer one bounded question: does the accepted
mixed-lineage ledger satisfy the original frozen T10.2 provenance and derived-
label QA contract strongly enough to authorize a separate future source-training
protocol?

## Frozen handoff

The handoff receipt binds the exact successful T10.2.7 terminal evidence:

- protocol manifest and migration receipt;
- collection report `T10_2_7_SOURCE_COLLECTION_COMPLETE`;
- recovery report `PASS_T10_2_7_RECOVERY`;
- 18 accepted lanes, 72 complete resets, and 1,370 accepted events;
- accepted ledger byte size and SHA-256;
- passing nine-unit cross-fit audit;
- recovery checkpoint and zero-replay receipts;
- the original frozen T10.2.2 scientific kernel and environment checksum.

It also freezes an 18-entry lineage registry. Seventeen physical lanes retain
the T10.2.2 kernel manifest as their event provenance. The one replacement lane
retains the T10.2.7 manifest and physical recovery seed while remaining linked
to logical confirmation seed 111. No event is rewritten or resealed for QA.

## Lineage validation

Before computing metrics, T10.2.8:

1. authenticates the ledger descriptor and every event checksum;
2. maps every physical `(split, game, seed)` tuple to exactly one registered
   lineage;
3. checks per-lane event counts against the accepted lane reports;
4. validates the 17 parent lanes under the T10.2.2 manifest;
5. validates the recovery lane under the T10.2.7 hybrid execution manifest;
6. requires one environment checksum, globally unique event IDs, closed
   firewalls, and zero physical replay.

Any lineage failure produces `DATA_OR_PROVENANCE_INVALID`, skips scientific QA,
and stops before fit.

## Scientific QA gate

After lineage validation, the runtime computes the original frozen T10.2 QA
metrics without changing thresholds:

- confident correspondence at least 0.90;
- fully ambiguous correspondence below 0.10;
- exact round-trip, permutation-invariant, commutative transport evidence;
- consistent learned-predicate declarations and complete labels;
- each learned predicate prevalence in `[0.005, 0.95]`, at least 32 positives,
  and support in at least two games;
- evaluable nonterminal-prefix fraction at least 0.80;
- multiframe-coherent nonterminal-prefix fraction at least 0.50.

Behavioral progression, goal, terminal, controller, decision-engine, and option-
conditioning counts are reported diagnostically but do not alter the frozen QA
gate.

## Terminal decisions

- `PASS_T10_2_8_QA_READY_FOR_SEPARATE_SOURCE_TRAIN_PROTOCOL`: lineage and QA
  both pass. T10.2.8 still performs no fitting; a new frozen protocol is needed.
- `FAIL_T10_2_8_QA_STOP_BEFORE_FIT`: lineage passes but QA fails. Fitting,
  source validation, and AR25 remain forbidden.
- `DATA_OR_PROVENANCE_INVALID`: lineage or artifact authentication fails.

The lineage audit, QA report, and terminal report are signed and write-once. A
failed gate is a final negative result for this protocol, not permission to edit
labels, thresholds, or accepted events in place.
