# SAGE.11 source-train anti-shortcut audit pre-registration

Status: frozen before empirical execution

Protocol date: 2026-07-26

Format: `sage11-anti-shortcut-logo-v1`

## Purpose

Factorized pilot v2 passed its frozen comparator gate but showed only 0.0078
current-action shuffle degradation. The player-moved gain could be explained
by near-constant action-availability and object-role atoms acting as implicit
game identifiers.

This audit asks whether the shared 77-feature interface learns
action-conditioned changed-cell dynamics across games rather than recognizing
game-specific effect marginals.

## Firewall

- Read only the 76,908 rows from the 11 registered `source_train` shards.
- Verify the manifest checksum and each selected source-train shard checksum
  and row count.
- Do not open, encode, score, or tune on `re86`, `ls20`, `sc25`, historical,
  holdout, or regression-only shards.
- Use leave-one-game-out evaluation across all 11 source-training games.
- Fit the atom vocabulary independently inside the shared source-train loader;
  no validation-derived vocabulary or feature selection is allowed.

## Frozen representation

The audit consumes `sage11-streaming-features-v2`, schema checksum
`39bb692848fba64ef994e0c0a304785128e1a69adaf6308f1d22623a8f0876bd`.
The shared loader and live tracker must remain exactly feature-identical to
the frozen pilot-v2 matrix.

Four views are evaluated:

1. **Action-only:** the six one-hot current actions plus `has_xy`, boundary,
   corner, and diagonal predicates.
2. **State-only:** all features that do not depend on the current candidate
   action. This excludes direct current-action predicates,
   `same_as_current`, and all current-versus-previous target relations.
3. **Full:** all 77 features.
4. **Full without fixed game-signature atoms:** all features except current
   `action:available(...)` and `object:role_present(...)` indicators.

State digest bytes, raw coordinates, game identity, policy arm, and current
outcomes remain excluded.

## Frozen targets and estimators

Two independently scored heads:

- changed-cells bucket: zero, one, few, some, many;
- player moved: false or true.

Every view/head/fold uses one balanced
`HistGradientBoostingClassifier`: learning rate 0.08, maximum depth 4, 100
iterations, early stopping disabled, random state 11. There is no
hyperparameter, seed, threshold, or feature search.

The factorized composite is the unweighted mean of the two head macro-F1
scores. The primary changed-cell comparator is the stronger of action-only
and state-only.

## Conditional action shuffle

For each held-out game:

1. group test rows by their complete current availability/object-role atom
   signature;
2. within each group, permute all current-action-dependent columns as one
   block;
3. preserve state-only columns and do not retrain;
4. use random state `11 + fold_index`.

This holds the coarse current-state signature fixed while breaking the
candidate action/context relation. The reported degradation is full
macro-F1 minus conditionally shuffled macro-F1 on concatenated out-of-fold
predictions.

## Explicit game-identifier test

The audit computes deterministic row-weighted purity of fixed signatures:

1. map each availability/object-role bit signature to its game counts;
2. assign each signature to its majority game;
3. report majority-game accuracy, signature count, shared-signature count,
   and the fraction of rows in signatures exclusive to one game.

It also compares full out-of-fold performance with the same models trained
after removing every fixed signature atom.

A shortcut-reliance failure occurs when signature purity exceeds 0.80 **and**
removing those atoms decreases either changed-cells or factorized composite
macro-F1 by more than 0.02.

## Frozen go/no-go gate

The audit passes only if every condition holds:

1. out-of-fold changed-cells full macro-F1 exceeds the stronger of
   action-only and state-only by at least 0.10;
2. out-of-fold factorized composite degrades by at least 0.10 under
   conditional action shuffling;
3. changed-cells full-minus-best-baseline is non-negative on at least 9 of 11
   held-out games and no game is below -0.05;
4. the explicit game-signature shortcut-reliance test does not fail.

Player movement cannot compensate for a failed changed-cells requirement.
Strong absolute F1, a favorable subset of games, or the earlier v2 result
cannot waive a condition.

## Consequences

- **Pass:** the shared interface is eligible for GPU training of the compact
  factorized PyTorch world model. Training still must pass every source-only
  world-model gate before shadow evaluation.
- **Fail:** do not train the world model. Extend the collector with contact,
  alignment, proximity, and object-relative action relations, then build a
  smaller source-only replacement pilot corpus rather than recollecting
  100,000 rows.

## Required artifact

Publish
`diagnostics/sage/sage11_source_train_anti_shortcut_logo.json` with every fold,
aggregate view/head metrics, conditional-shuffle results, signature purity and
ablation, hardware/timings, exact checksums, and the final decision.
