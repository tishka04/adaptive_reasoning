# SAGE12 clean temporal-mechanic replication V4.1 protocol

Date frozen: 2026-07-27.

Status: frozen before source-only preflight.

Frozen manifest checksum:
`86b3d3b38ba41d0f860169928f6cc5afd6765ccdbf83078e3a09d60da0e07abc`.

## Question and authority boundary

V4.1 asks whether the positive V4 temporal result replicates after repairing
its two failed gates:

> Can eight observed semantic transitions predict the next effect beyond
> action identity, with causally resolved roles and source-only calibrated
> probabilities?

The structured Beta mechanic inducer remains primary. V4.1 does not change its
rule vocabulary, local-evidence minimum, prior strength, context length,
baselines, or primary causal controls. It adds a causal role-state contract
and a source-only calibration layer.

Qwen2.5 0.5B receives a compact version of the same history. Qwen has separate
authority: a structured pass may authorize a deterministic V5 world-model
protocol, while Qwen must independently pass before LLM proposals can enter
that V5. Neither result authorizes an EBM, controller, holdout, historical
game, or `ar25` execution.

## Role repair

The V4 tracker assumed a one-cell displacement and treated every game as
having the same translational avatar semantics. That made `cd82` and `sp80`
fail coverage even when an action controlled rotation, selection, scrolling,
or a larger object.

V4.1 tracks components causally inside a reset and emits one audit-only role
state:

- `translational`: a unique persistent component has sufficient observed
  displacement or selection evidence;
- `non_translational`: after eight observations, at least three move-family
  actions produced state changes without a uniquely tracked translation;
- `ambiguous`: neither claim is supported.

Components larger than 25% of the grid cannot become actor candidates.
Displacements may have any non-zero magnitude. A candidate needs at least two
evidence points and a two-point lead over the runner-up. The tracker can use
only the current and previous frames; future transitions are forbidden.

`actor_displaced` is applicable only when the same translational component is
matched before and after. Non-translational controls are masked rather than
stored as false displacement labels. The detailed role state is audit
provenance and is excluded from the model view.

## Source-only calibration

For every source game, priors are fitted on the other ten games and used to
predict the held-out game's windows. The pooled out-of-game predictions fit
one Platt mapping per effect and per method:

`sigmoid(slope * logit(clipped_probability) + intercept)`.

The clip is `[1e-6, 1-1e-6]`. Logistic regression uses L2, `C=1`, `lbfgs`,
1,000 iterations, no class weighting, and seed 307. Per-label decision
thresholds maximize F1 on these out-of-game predictions only. Ties choose the
threshold closest to 0.5 and then the higher threshold.

Calibration coefficients, thresholds, metrics, and checksums are published
before prospective collection. Raw V4-style probabilities remain separately
scored, so calibration cannot conceal loss of the original temporal signal.

## Compact Qwen contract

The prompt uses one line per event:

`action/family/anchor/effects/applicable`.

The effect codebook is `A=actor_displaced`, `C=target_created`,
`R=target_removed`, and `M=target_moved`. Qwen returns at most eight compact
records with exact/family scope, value, anchor, effect, and `z=0`. The compiler
derives rule IDs and rejects unknown fields, ungrounded scopes, invalid enums,
or non-zero support without repair.

Every source window is tokenized with the local model before prospective
collection. The maximum must be at most 384 tokens, leaving headroom under
the frozen 512-token runtime cap. Decoding remains temperature zero on
`cuda:0`, with at most 256 output tokens.

## Data and temporal firewall

Source development uses only the immutable 3,040 V3 source-training traces.
The already inspected V3 validation outcomes are non-gating. Windows remain
length eight, query horizon one, contiguous by frame checksum, reset-local,
game-local, and deduplicated only by complete audit digest.

After a passing source preflight, V4.1 collects exactly 768 new transitions:
256 each from `re86`, `ls20`, and `sc25`, using unused policy seeds 307, 347,
389, and 433. Each reset has at most 32 actions and each game at most 24
resets. Legal actions are balanced, collection is never outcome-adaptive, and
chronological repeats remain in the raw audit stream.

Game ID, seeds, reset/step indices, coordinates, values, colours, object IDs,
raw grids, hashes, query outcomes, and role-state details remain outside the
model view. Static action/family/anchor identity leakage is measured before
prospective collection.

## Frozen source preflight

Prospective collection is forbidden unless all source checks pass:

- at least 1,500 windows;
- at least 75 positives and 75 negatives for every effect;
- role state resolved on at least 0.95 of windows globally and 0.90 in every
  source game;
- static identity accuracy no more than action-only +0.10;
- calibrated source out-of-game macro ECE at most 0.10;
- calibration macro-Brier degradation at most 0.005;
- maximum complete Qwen prompt at most 384 tokens;
- model-view firewall passes.

A failed preflight is published as `FAIL_SOURCE_TRAIN_PREFLIGHT` and stops the
iteration before any V4.1 prospective outcome is opened.

## Frozen structured gates

The prospective structured result passes only if every check succeeds:

- at least 500 scored windows;
- at least 30 positives and 30 negatives per effect;
- role state resolved globally at least 0.95 and per game at least 0.90;
- JSON, grounding, and `support=0` exactly 1.00;
- raw and calibrated macro Brier skill both at least +0.10 over the stronger
  calibrated local-action, global-action, or template baseline;
- run-cluster bootstrap 95% lower bound above zero;
- calibrated macro-F1 gain at least +0.05;
- context-outcome shuffle reduces skill by at least 0.05;
- eight-transition context gains at least 0.05 skill over no context;
- non-negative skill in every validation game;
- calibrated macro ECE at most 0.10;
- source preflight passed unchanged.

For future scope, each effect receives its own authority ledger. An effect is
eligible for V5 only when its best preregistered method has Brier at most
0.10, F1 at least 0.20, and at least 30 prospective positives and negatives.

## Separate Qwen gates

The 128 contexts are selected without outcomes by game, action, anchor, and
window digest. Qwen passes separately only with:

- strict JSON validity at least 0.95;
- grounding at least 0.95;
- `support=0` exactly 1.00;
- productive-effect recall@8 at least 0.70;
- outcome-shuffle skill loss at least 0.05;
- non-negative skill in every validation game.

CUDA unavailability or model/runtime failure closes only Qwen authority; it
does not alter the structured verdict.

## Execution and publication order

```powershell
python -m theory.sage12.mechanic_replication preflight
python -m theory.sage12.mechanic_replication_collection
python -m theory.sage12.mechanic_replication evaluate
```

Publication is three immutable checkpoints on `main`:

1. implementation, tests, protocol, and frozen manifest;
2. source windows, priors, calibration, thresholds, token budget, and
   preflight;
3. prospective shards, predictions, Qwen outputs, controls, full result, and
   documentation.

No threshold, seed, rule, calibrator, prompt, schema, gate, or code path may
change after its corresponding outcome boundary is opened. Every pass or
failure is published.

## Source-preflight compatibility amendment

The first source-only preflight attempt reached Qwen tokenization but stopped
before producing a preflight result because `transformers` 5 returns a
mapping containing `input_ids` where the existing adapter expected the tensor
directly. The adapter now unwraps `input_ids` for both token counting and
generation. This changes no prompt, schema, weights, decoding, data, metric,
threshold, or gate. No V4.1 prospective trace or outcome had been opened.
