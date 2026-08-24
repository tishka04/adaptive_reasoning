# SAGE.T12.6 — Target-local future-viability grounding

## Scientific status

T12.5c established a clean local causal fact at the level-0 stage-3 cursor:
with the same exact anchor and two action slots, the correctly bound
`ACTION3>ACTION3` program progressed in all four trials while the
`ACTION4>ACTION3` binding swap progressed in none. Neither arm terminated.

The natural next target, the exact common level-1 entry, is not unexplored.
T12.4a.4d and T12.4a.4d.1 already spent 24,171 and 36,481 SDK calls there.
Both were integrity-clean search misses with zero progress; their archive
policies were dominated by immediate novelty, contract mismatch, action-family
diversity and terminal-risk control. Repeating another broad search would not
isolate the mechanism supported by T12.5c.

T12.6 therefore performs a zero-SDK temporal-transfer audit before any new
physical experiment. It asks whether a frozen target-local descriptor can
bind an action to **future productive reach**, and whether that binding beats
an immediate-effect model and an equal-capacity score permutation on later
sealed archives.

## Question

> Can pre-action, identity-free local structure predict which grounded action
> leads to the deepest observed continuation of changed, non-terminal states,
> across unseen search seeds and later search policies, with a specific
> advantage that disappears when the same scores are bound to the wrong
> actions?

This is not a level-progress claim. The old archives contain no level-1 to
level-2 witness. Future productive reach is a bounded precursor signal whose
usefulness must be confirmed prospectively later.

## Frozen evidence split

The split follows collection chronology:

- **training/cross-fit corpus:** the 12 signed T12.4a.4d archives from search
  seeds 9101–9103, lineages 8701/8705 and arms `local_archive_control` /
  `contract_regrounded`;
- **sealed evaluation corpus:** the 18 later T12.4a.4d.1 archives from search
  seeds 9201–9203, both lineages and arms `local_archive_control`,
  `diversity_control`, `abstract_hazard_diversity`.

The failed parent verdicts are preserved. Historical code is not reexecuted;
the signed manifests, receipts and every archive hash are verified. Current
T12.6 extraction and scoring code is independently frozen.

The evaluation corpus is never used to fit, update, select a feature or tune a
threshold. Only a passed signed compile receipt exposes the already frozen
model to it.

## Label and descriptor

For every archived state with at least two observed action branches, the label
of a branch is the maximum number of subsequent changed, non-terminal edges
reachable from its target state, capped at horizon four. The immediate edge is
excluded from this future label. Cycles cannot add reach because a cell may
appear only once in a traversed path.

Only decision groups with different observed future-reach labels contribute
to ranking metrics. Missing descendants remain censored archive evidence; they
are not relabelled as level progress.

The pre-action descriptor reuses the existing translation-invariant local
hazard contract:

- action schema and whether it is coordinate-grounded;
- typed attributes and informative roles within radius seven;
- relative row/column offsets only.

Absolute coordinates, entity ids, exact hashes, archive cell ids, game ids and
outcomes are excluded. A signature mean requires at least two training
observations. Missing signature support backs off to action schema plus
coordinate-grounding, then to the frozen global mean.

## Controls

Two equal-capacity controls use the same pre-action descriptors and table
capacity:

1. **immediate model:** predicts `4 × nonterminal + 2 × changed + novel` from
   the training corpus, then ranks later branches by that frozen prediction;
2. **binding swap:** circularly shifts the future-model score vector by one
   position after deterministic action-key ordering. It preserves the exact
   score multiset and candidate catalogue while breaking action binding.

Labels do not depend on any of the three scores.

## Pre-freeze training-only diagnostic

A read-only diagnostic restricted to seeds 9101–9103 found 270 eligible
decision groups. Leave-one-search-seed-out future-binding accuracy was 0.8148,
versus 0.5889 for the immediate model and 0.2074 for the binding swap; selected
target-local signature coverage was 0.5815. These values motivated fixed gates
but are not a signed compile result. No derived future-viability metric was
read from the 9201–9203 evaluation corpus before freeze.

## Compile gate

Integrity requires all 12 registered archive conditions, both lineages, all
three compile seeds, zero conflicting duplicate actions, zero SDK calls and
the wall/storage bounds.

Scientific compile gates are:

- at least 240 eligible decision groups;
- micro future-binding top-1 accuracy at least 0.75;
- gain over the immediate model at least 0.10;
- gain over the binding swap at least 0.30;
- selected target-local signature coverage at least 0.45;
- future binding strictly beats both controls in every held-out-seed fold;
- future-binding accuracy at least 0.70 for each lineage inside every fold.

A compile miss stops evaluation.

## Sealed evaluation gate

The compiled tables are loaded without update. Integrity requires all 18
registered conditions, both lineages, seeds 9201–9203, zero duplicate-action
conflict, zero SDK calls and all bounds.

Scientific evaluation gates are:

- at least 250 eligible decision groups;
- future-binding top-1 accuracy at least 0.70;
- gain over the immediate model at least 0.08;
- gain over the binding swap at least 0.25;
- selected target-local signature coverage at least 0.40;
- future binding strictly beats both controls on every evaluation search seed;
- future-binding accuracy at least 0.65 on each lineage.

## Exclusive outcomes

Compile:

- malformed corpus, conflict, missing condition or bound failure →
  `FAIL_T12_6_COMPILE_INTEGRITY_GATE`;
- inadequate group count or signature coverage →
  `FAIL_T12_6_INSUFFICIENT_FUTURE_VIABILITY_SUPPORT`;
- adequate support but failed discrimination →
  `FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE`;
- all gates pass → `PASS_T12_6_COMPILE_GATE`.

Evaluation:

- malformed or incomplete sealed corpus →
  `FAIL_T12_6_EVALUATION_INTEGRITY_GATE`;
- support does not transfer →
  `FAIL_T12_6_FUTURE_VIABILITY_SUPPORT_TRANSFER_GATE`;
- correct binding does not transfer above controls →
  `FAIL_T12_6_FUTURE_VIABILITY_TRANSFER_GATE`;
- all gates pass → `PASS_T12_6_FUTURE_VIABILITY_GATE`.

Every miss is a signed negative result. No corpus deletion, threshold change,
feature addition, fold substitution or same-version rerun is authorized.

## Bounds and authority

- exactly zero SDK calls;
- maximum one hour per offline phase;
- maximum 3 GiB of new artifacts per phase;
- no archive copies or raw frames;
- source-train `bp35` only;
- no physical `run` command in the CLI.

A final pass supports only temporal transfer of a bounded future-productive-
reach ranking inside previously collected level-1 archives. It does not prove
level progress or controller utility. It may authorize only preparation of a
separately frozen T12.6b prospective physical control. Collection, validation,
holdout, neural training and production authority remain closed.
