# SAGE12 V4.4 paired causal-contrast protocol

Status: **frozen before source preflight**

Manifest checksum:
`598cdbca8ef50b05d3c9743cbbf4245c0e4c0495b81fd5fc3fd06e67bc623f5d`

## Question

V4.3 acquired valid executed counterfactual pairs but predicted each arm as an
independent absolute outcome. Its binding fields then acted primarily as game
signatures. V4.4 asks a narrower causal question:

> From two interventions executed at an identical verified pre-state, which
> arm produces the target effect?

The shared state and game-specific background should cancel in a direct
left-minus-right comparison. This pilot tests that claim before any semantic
world model is considered.

## Source corpus and effect authority

V4.4 reuses only the immutable 2,396-pair V4.3 source-training corpus with
collection checksum
`a842c0bdd99a1e10ad48c03ded447e231a6767e6af7410192b2f21c4b2948722`.
No V4.3 validation data exists or may be created by the source preflight.

A source-only design audit found:

- 172 creation-discordant pairs;
- 189 removal-discordant pairs;
- zero movement-discordant pairs.

Creation and removal are authoritative. Movement remains diagnostic-only; it
cannot be promoted under V4.4. This change does not rewrite the V4.3 result.

## Pair view

For each authoritative effect, only pairs with applicable outcomes `(1, 0)`
or `(0, 1)` are scoreable. The label is whether the left arm is positive.
Ties remain audit data and are excluded from directional fitting.

Each arm is encoded independently, then the right feature vector is
subtracted from the left. Model inputs may contain:

- action name and family;
- one frozen binding projection;
- arm-conditioned rates from the shared eight-event context;
- action-by-binding interactions.

Game ID, pair ID, frames, hashes, coordinates, raw arguments, object IDs,
seeds, resets, tree paths, labels, and future outcomes are forbidden.

The projection ladder remains `minimal`, `relational`, and `typed`.
Game-identity accuracy is measured from pair-difference features relative to
action-difference features.

## Antisymmetric model

The primary model is regularized logistic regression with no intercept.
Every training row `(x, y)` is augmented with `(-x, 1-y)`. Therefore swapping
complete arms must satisfy `p(-x) = 1-p(x)` to numerical error at most
`1e-12`.

Temperature calibration is slope-only, preserving antisymmetry, and is fit
from source leave-one-game-out predictions. The frozen grid is 197 values
from 0.10 through 5.00.

Baselines:

1. action plus shared history, without binding;
2. action difference only;
3. binding difference only;
4. deterministic occupied/free template.

## Source gates

All gates are conjunctive:

- at least 75 discordant source pairs per authoritative effect;
- at least two source games with ten discordant pairs per effect;
- macro-Brier skill at least +0.10 versus the stronger baseline;
- macro directional-accuracy gain at least +0.10;
- swapping only the two bindings reduces accuracy by at least 0.10;
- game-identity gain over action difference at most +0.05;
- macro ECE at most 0.10;
- exact arm-swap inversion error at most `1e-12`;
- every scoreable source game is non-negative versus the stronger baseline;
- paired bootstrap accuracy-gain lower 95% bound is positive;
- strict JSON validity, support-zero semantics, and the V4.3 source checksum
  all pass.

Among passing projections, highest source LOGO macro-Brier skill wins; within
0.005, the simpler projection wins. The projection, temperatures, vectorizer
vocabularies, coefficients, baseline, and thresholds are frozen before
validation collection.

Failure writes `FAIL_CLOSED`, selects no projection, and mechanically blocks
validation.

## Conditional validation

Only a source pass may collect fresh counterfactual trees on `re86`, `ls20`,
and `sc25`:

- 64 roots per game;
- depth three, beam-independent binary execution;
- 32-action reset budget;
- seeds 1451, 1499, 1553, and 1601;
- no outcome-adaptive selection or deletion.

Validation is evaluated once with the frozen source model. It requires at
least 30 discordant pairs per effect, +0.10 macro-Brier skill, +0.10
directional-accuracy gain, +0.10 binding-swap accuracy loss, macro ECE at most
0.10, and non-negative transfer in every scoreable validation game.

A validation pass authorizes only preparation of a separately frozen
absolute semantic-world-model protocol limited to creation and removal. It
does not authorize world-model fitting directly, Qwen, a GNN, an EBM,
controller execution, holdout opening, historical evaluation, or `ar25`.

Commands:

```powershell
python -m theory.sage12.pairwise_causal_pilot preflight
python -m theory.sage12.pairwise_causal_pilot collect-validation
python -m theory.sage12.pairwise_causal_pilot evaluate
```
