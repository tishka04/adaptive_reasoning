# SAGE12 V4.3 causal-binding pilot result

Status: **FAIL_CLOSED at source preflight**

- source preflight checksum:
  `2acbe04b34fb0d22456154e39c326c2e5da2cc13f271cfcc7818ff65601a63a2`
- binding closure checksum:
  `be2d4769af5ef5b21c672e30f29731a05bc8e34e52cf28cc51070318077469f9`
- world-model closure checksum:
  `c0452c1233460d128ca1e8239da6a4910701760965a808e3ea20f056d2df2319`

No source-validation game was opened. No binding model was fit on validation,
and no semantic world model, Qwen model, GNN, EBM, or controller was trained
or executed.

## Outcome

The executed counterfactual collection itself worked: all 352 source roots
were acquired, 2,396 pairs and 4,792 arms were retained, and all replay hashes
matched. The pilot then failed for two independent reasons.

First, the frozen effect-capacity gate failed. `target_moved` occurred only
eight times among 2,638 applicable arms, below the required 75 positives.
Creation and removal had sufficient capacity.

Second, every binding projection failed source-only leave-one-game-out
utility and identity controls:

| Projection | Brier skill vs stronger baseline | Macro-F1 gain | Identity gain over action | Macro ECE | Result |
|---|---:|---:|---:|---:|---|
| minimal | -0.1620 | -0.0400 | +0.2089 | 0.0314 | fail |
| relational | -0.1548 | -0.0222 | +0.2550 | 0.0398 | fail |
| typed | -0.0712 | -0.0083 | +0.5624 | 0.0312 | fail |

The frozen requirements were at least +0.10 Brier skill, +0.05 macro-F1
gain, at most +0.05 identity gain, and at most 0.10 macro ECE. Calibration
passed for all projections, but predictive utility was negative and
game-signature leakage was far above the limit.

The strongest baseline was `binding_only` for minimal and relational, and
action plus history without binding for typed. This means the structured
combination of action, binding, and local temporal evidence did not add
transferable source-game information over simpler marginals.

## Gate ledger

Passed:

- 2,396 source pairs versus the 2,000 minimum;
- strict JSON validity 1.00;
- compiler grounding 1.00;
- replay integrity 1.00;
- support/evidence separation by construction;
- macro ECE for all projections.

Failed:

- target-movement source capacity: 8 versus 75 positives;
- source macro-Brier skill for all projections;
- source macro-F1 gain for all projections;
- game-identity gain for all projections.

Because no projection passed, `projection_freeze.json` contains no selected
projection. The validation collector remains mechanically blocked. The
binding result is `SKIPPED_SOURCE_PREFLIGHT`, and the world-model result is
`SKIPPED_FAIL_CLOSED`.

## Interpretation

V4.3 successfully repaired the experimental weakness of V4.2.1: it created
real executed alternatives from verified identical states rather than a
near-vacuous permutation. The negative result therefore says something
stronger about the current representation and learner.

It rejects this specific combination:

- the current coarse `BindingSignature` vocabulary;
- an eight-event local Beta rule;
- the frozen outcome-independent legal-action coverage policy;
- the three current target-effect labels;
- cross-game source transfer under the SAGE11 split.

It does **not** evaluate or refute a semantic world model, energy model, or
hierarchical controller, because the protocol correctly stopped before those
models were fit. It also does not rule out persistent object-relative binding,
mechanic-specific event discovery, or a representation learned from
within-state intervention equivalence classes. The most immediate evidence is
that adding binding fields primarily increased game identification, while the
collector almost never elicited target movement.

## Consequence

The appropriate next iteration must be designed from source data only and
must not simply increase rows under the same policy. It should first redefine
the intervention unit around persistent object-relative slots and
mechanic-specific effect discovery, then preregister a policy that can produce
adequate positive/negative capacity without adapting to observed effects.
Only after source LOGO utility and identity controls pass should a new
validation corpus or semantic world model be authorized.

The immutable source collection is documented in
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_COLLECTION.md`. The frozen protocol
is `reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_PROTOCOL.md`.

Final software and artifact validation passed 39 focused
V4.3/V4.2/V4.2.1 tests in 31.35 seconds and focused Ruff checks. It reloaded
the preflight and both closure records and confirmed that no validation shard
directory exists.
