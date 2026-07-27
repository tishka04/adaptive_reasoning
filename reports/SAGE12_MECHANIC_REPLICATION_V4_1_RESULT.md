# SAGE12 temporal mechanic replication V4.1 — source-preflight result

Date: 2026-07-27

Status: `FAIL_SOURCE_TRAIN_PREFLIGHT`

Preflight checksum:
`cffa41e2ae980f64dfc76cbe40076809b301da4e8f98dffbc02122eb2bfa147c`

V4.1 stopped at its frozen source-only boundary. No prospective validation
transition was collected, no Qwen hypothesis was generated, and no semantic
world model, EBM, trajectory ranker, or controller was fit or evaluated.

## Executive result

The three intended V4 repairs worked operationally:

- causal actor-role resolution reached 0.9984 globally and at least 0.9870 in
  every source game;
- leave-one-source-game-out calibration reduced structured macro Brier from
  0.0483 to 0.0430 and macro ECE from 0.0683 to 0.0360;
- all 1,911 compact Qwen prompts fit the frozen preflight budget, at 322–345
  tokens against a 384-token limit.

The conjunctive preflight still failed two frozen gates. `actor_displaced`
had only 35 positive source examples, below the required 75, and adding
static family/anchor identity raised source-game classification accuracy by
0.1293 over action alone, above the maximum allowed 0.10. The first failure
means one effect lacks enough source support for a trustworthy replication.
The second shows that the model view still carries too much game-specific
identity before any outcome prediction is attempted.

## Frozen artifacts

| Artifact | Checksum |
|---|---|
| Frozen manifest | `86b3d3b38ba41d0f860169928f6cc5afd6765ccdbf83078e3a09d60da0e07abc` |
| Source priors | `3fb135fcfeff4312955a79fcb074d2e5bb66ff9c72430ce27dce2d44e790794a` |
| Calibration | `473fa223cf46730b2240d03c9d46d3a724c935b43ea659095fd5fca31795c953` |
| Source preflight | `cffa41e2ae980f64dfc76cbe40076809b301da4e8f98dffbc02122eb2bfa147c` |

The immutable V3 source-training corpus supplied 3,040 traces and 1,911
unique eight-transition windows. `source_validation_opened` is false and
`world_model_fit_authorized` is false in the machine-readable result.

## Role and label capacity

The causal tracker produced 930 `translational`, 978 `non_translational`, and
3 `ambiguous` windows. Global resolved coverage was 0.9984. The lowest
per-game result was 0.9870 on `lf52`; every other source game scored 1.00.

| Effect | Positive | Negative | Frozen minimum | Result |
|---|---:|---:|---:|---|
| `actor_displaced` | 35 | 746 | 75 / 75 | fail |
| `target_created` | 87 | 1,490 | 75 / 75 | pass |
| `target_moved` | 90 | 1,190 | 75 / 75 | pass |
| `target_removed` | 226 | 1,054 | 75 / 75 | pass |

## Source-only predictive diagnostics

These are leave-one-source-game-out diagnostics, not prospective evidence.

| Structured model | Macro Brier | Macro ECE | Macro F1 |
|---|---:|---:|---:|
| Raw | 0.0483 | 0.0683 | 0.5557 |
| Calibrated | 0.0430 | 0.0360 | 0.7155 |

Calibration passed both the ECE maximum of 0.10 and the frozen Brier
non-degradation allowance of 0.005. The static identity probe did not pass:
action-only accuracy was 0.1879, while action plus family/anchor features
reached 0.3171, a gain of 0.1293.

## Gate ledger

| Source-preflight gate | Result |
|---|---|
| At least 1,500 source windows | pass |
| Global role resolution at least 0.95 | pass |
| Per-game role resolution at least 0.90 | pass |
| Model-view firewall | pass |
| Qwen prompt budget | pass |
| Calibrated Brier non-degradation | pass |
| Source OOF macro ECE at most 0.10 | pass |
| At least 75 positives and negatives per effect | **fail** |
| Static identity gain at most 0.10 | **fail** |

Because the protocol is conjunctive, seven passing gates cannot compensate
for the two failures.

## Interpretation and next admissible iteration

V4.1 validates the engineering repairs but does not validate the proposed
semantic-planning architecture. It failed before the experiment that could
test prospective transfer. The result narrows the next source-only work:

1. replace or broaden `actor_displaced` with a preregistered,
   semantically defensible actor-state effect, or collect more genuine
   translational source evidence without duplicating rows;
2. coarsen or remove family/anchor conditions until the frozen identity probe
   passes while retaining causal context;
3. retain the causal role contract, source-only calibration, compact Qwen
   compiler, and separate authority ledgers unchanged unless a new versioned
   protocol explicitly amends them.

Any successor must freeze and publish a new source preflight before collecting
prospective outcomes. The stopped V4.1 manifest must not be reused to bypass
this failure.
