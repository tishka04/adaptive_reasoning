# SAGE12 constrained effect pilot V2 protocol

Status: frozen before computing any V2 metric on the source-validation games.

Frozen manifest:
`training/sage12/constrained_pilot_v2/frozen_manifest.json`.

Manifest checksum:
`033274922c2f25d3cb7918bc2f308fffdc03d4811f034e9db171147d4aec25aa`.

## Question

Can a frozen Qwen2.5 0.5B encoder plus a native constrained multi-label head
use one deliberately game-light relational motif to predict observed
one-step effects better than action identity and a direct motif classifier?

This is a new Stage A version. It repairs two failures of the first proposal
pilot:

1. code renders the classifier decisions into the typed hypothesis schema, so
   Markdown fences, malformed JSON, unsupported fields, and non-zero support
   are impossible;
2. the model receives no scene inventory. Its state view is one binary
   `actor_interaction` motif.

Passing this pilot would authorize only a separate semantic world-model
protocol. It would not promote controller authority.

## Prior-result disclosure

The V1 validation result was already known when V2 was designed: free
generation failed typed parsing and the full compact scene graph leaked game
identity. No V2 classifier, embedding, threshold, baseline, or validation
metric was fit or computed before this document and its manifest were frozen.

Representation design used only the 1,624 source-training rows. The first
source-training-only preflight rejected a six-bit actor-local motif:

- motif-only game identity: 46.12%;
- selected-action-only identity: 22.29%;
- action plus motif identity: 54.37%;
- motif gain beyond action: +32.08 points.

The retained single bit records only whether a player-role entity has a
`near`, `contact`, or `adjacent` relation with any non-player entity. Its
source-training-only audit is:

- motif-only identity: 19.64% versus 9.85% majority, gain +9.79 points;
- action plus motif identity: 31.28%;
- motif gain beyond selected action: +8.99 points.

These two gains are below the frozen +10-point leakage limits. The initial and
final preflights are preserved separately. Source validation, holdout,
historical environments, and `ar25` were not opened by either preflight.

## Data and firewall

V2 reuses the exact checksummed 2,104-row V1 source corpus rather than
recollecting equivalent transitions:

- 1,624 rows from the 11 SAGE11 source-training games;
- 480 rows from `re86`, `ls20`, and `sc25`;
- combined shard checksum
  `ce5cfe1217f9add9ab250f60315ed66d154ae8ed903e51bb572b69a4b3`;
- no holdout, historical, or `ar25` access;
- no row-wise split and no validation tuning.

The selected legal action name is supplied because the scientific comparison
asks whether state adds information beyond action identity. Action arguments
remain compiler provenance and are not part of the Qwen prompt.

## Representation

The complete model prompt contains:

- the selected action name;
- `actor_interaction=yes|no`;
- the fixed four-effect vocabulary.

It excludes available-action sets, entity identifiers or counts, object roles
other than `player`, shapes, aspects, relation counts or direction, scene
signatures, raw grids, colors, coordinates, game ID, policy information, and
all outcome fields.

The existing deterministic entity-binding permutation produces the relation
control. The one-bit motif is recomputed after permutation; action and entity
inventory remain unchanged.

## Model and constrained output

The encoder is the existing local Qwen2.5 0.5B Instruct weights, checksum
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
It is frozen and runs on `cuda:0`, already selected by the V1 identical-input
CPU/GPU benchmark.

For every row:

1. the last non-padding hidden state is extracted and L2-normalized;
2. one independent class-balanced liblinear logistic head is fit per effect
   on source training only;
3. regularization is `C=1`, maximum iterations 1,000, seed 12, and the fixed
   decision threshold is 0.5;
4. code renders positive decisions as strict JSON hypotheses with
   `support=0`;
5. the unchanged compiler checks the exact selected legal action and grounds
   the `player` role for `player_moved`.

No free-form decoder output is accepted or repaired.

## Targets

Four independently observed transition components are scored:

- `changed`;
- `player_moved`;
- `level_complete`;
- `game_over`.

The primary macro-F1 averages `changed` and `player_moved`. Source training has
only one `level_complete` and 27 `game_over` rows, so both terminal labels are
reported but cannot dominate the primary promotion statistic.

## Baselines

- selected-action-only class-balanced logistic heads;
- selected action plus the same one-bit motif through direct logistic heads,
  without Qwen;
- the existing deterministic SAGE12 template.

The stronger baseline is selected once by aggregate source-validation primary
macro-F1. Per-game comparisons use the stronger baseline on that game and are
reported for all three games.

## Frozen gates

All gates must pass:

- strict JSON validity exactly 1.00;
- `support=0` rate exactly 1.00;
- grounded emitted hypotheses at least 0.99;
- Qwen-head primary macro-F1 gain at least +0.05 over the stronger baseline;
- relation-shuffle primary macro-F1 degradation at least 0.05;
- non-negative Qwen gain on each source-validation game;
- motif-only identity no more than majority +0.10 on source training;
- action-plus-motif identity no more than action-only identity +0.10.

Any failure produces `FAIL_CLOSED`, authorizes no semantic world-model fit,
and must be documented without changing a threshold.

## Reproduction

```powershell
python -m theory.sage12.constrained_pilot preflight
python -m theory.sage12.constrained_pilot evaluate
```

The first command is source-training-only design evidence. The second command
must not run until this protocol, code, tests, preflights, and frozen manifest
are committed.
