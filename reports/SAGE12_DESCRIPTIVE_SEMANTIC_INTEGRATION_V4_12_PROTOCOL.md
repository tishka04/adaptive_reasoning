# SAGE12 V4.12 — descriptive semantics and conditional integration protocol

Status at freeze: **protocol frozen; no V4.12 model result observed**.

## Question

V4.11 failed before training because its scalar horizon-progress teacher did
not reach the frozen capacity requirement in enough games. That result did not
test whether object-relative relations can predict the immediate semantic
effects that were abundant in the same panels.

V4.12 asks two questions in order:

1. Can an identity-free, object-relative comparator improve prediction of
   directly observed effects beyond both a root-only anchor and an action-only
   prior on held-out games?
2. If yes, do those semantics help the complete source-only chain
   `predictor → world model → depth-3 trajectories → EBM → controller`?

The second question is evaluated only if the first gate passes.

## Frozen source

V4.12 collects no new transitions. It reuses:

- the 1,056 replay-verified V4.11 panels;
- their 3,914 unique immediate action arms and 5,529 within-state action
  comparisons;
- the unchanged eleven-game SAGE11 source-train split;
- the complete V4.3 depth-three trees used by V4.7;
- the V4.7 world-model, trajectory and EBM hyperparameters.

The manifest fingerprints the V4.11 panel/QA/result artifacts, V4.3
collection artifacts and V4.7 protocol/result artifacts. Source validation,
historical games, holdout games and live play remain closed.

V4.11's collection, minimum-panel and action-aligned firewall checks must
pass. Its failed *progress-capacity* check is intentionally not inherited,
because V4.12 does not train or evaluate a scalar progress target.

## Descriptive teacher target

The target is not a single action-goodness score. Each fresh same-state pair
contributes independent supervision for every applicable effect on which its
two actions differ:

1. `changed`
2. `moved`
3. `target_removed`
4. `target_moved`
5. `local_change`
6. `contact_lost`
7. `productive`
8. `risk`

These are exactly the eight effects that met V4.11's pre-existing eligibility
rule. Scoring an effect separately preserves partial credit and avoids asking
one scalar to combine game-specific notions of progress.

## Student and distillation

The student retains the V4.11 object-relative DeepSets encoder:

- identity-free action/root descriptors;
- unordered action-aligned neighbor relations;
- no game ID, raw coordinates, object IDs, colour/value signatures, future
  frame or outcome fields.

Two models are trained per outer fold: a root-only anchor and the full
object-relative model. Both receive absolute imitation loss over the complete
17-effect teacher vocabulary; their pairwise loss is restricted to the eight
eligible effects. Progress-pair and tie-consistency weights are exactly zero.

The full model may alter the root anchor only through a panel-centred relation
residual. Its per-effect weight is selected inside the training games from
`[0, 0.25, 0.5, 0.75, 1]`; inactive effects receive zero residual. Calibration
uses training-game-balanced logit shifts. Every exported probability is outer
leave-one-game-out.

Frozen training parameters:

| Parameter | Value |
|---|---:|
| seed | 5,120 |
| hash buckets | 2,048 |
| embedding width | 32 |
| hidden width | 96 |
| epochs | 30 |
| samples/game/epoch | 256 |
| maximum pairs/epoch | 4,096 |
| learning rate | 0.0015 |
| weight decay | 0.0001 |
| effect-pair weight | 0.50 |
| progress/tie weights | 0 |

GPU execution is requested when CUDA is available; decoding and data are
unchanged by device choice.

## Semantic evaluation and gate

The primary test rows are only the fresh V4.11 panel arms. Resampling uses
10,000 equal-game bootstrap draws. V4.12 passes the semantic gate only if all
of the following hold:

- effect-pair log-loss gain over root-only has a strictly positive 95% lower
  confidence bound;
- effect-pair log-loss gain over action-only has a strictly positive lower
  bound;
- shuffling action-relative relations worsens effect-pair log loss with a
  strictly positive lower bound;
- active-effect macro-Brier gain over root-only has a strictly positive lower
  bound;
- relation shuffling worsens macro-Brier with a strictly positive lower bound;
- descriptive ECE is no worse than root-only ECE;
- descriptive pair loss is non-worse than root-only in at least 6/11 games;
- descriptive active-effect Brier is non-worse in at least 6/11 games;
- the 95% upper confidence bound of incremental game-identity accuracy over
  root-only is at most 0.02;
- neighbour permutation changes no probability by more than `1e-6`;
- reversing a pair complements its preference probability within `1e-6`.

The zero thresholds are deliberate existence tests: this iteration asks
whether the relations add any reproducible held-game information before
demanding a larger practical margin. The confidence intervals prevent a small
positive point estimate from opening the downstream stack by itself.

## Conditional global architecture test

If and only if the semantic gate passes, V4.12 immediately runs strict LOGO
world-model and EBM fitting on the frozen V4.3 trees. The eight probabilities
are added as numeric semantic features to the existing slot signature; the
seven compatible probabilities also use the unchanged `SlotAnnotation`
contract.

The depth-three controller ladder is:

- deterministic-left;
- action-only;
- action-sequence-only;
- fold-selected primary baseline;
- structured/root-feature world model + EBM;
- root-semantic world model + EBM;
- descriptive semantic world model + fixed heuristic controller;
- descriptive semantic world model + EBM;
- relation-shuffled descriptive semantics + the same fitted EBM;
- oracle semantics + EBM.

The fixed heuristic scores predicted return, completion, failure,
productivity, entropy, uncertainty and contradiction using weights
`[1, +3, -4, +0.05, -0.10, -0.10, -0.50]`.

Exploratory global support requires all of:

- descriptive EBM utility over the fold-selected primary baseline has a
  strictly positive paired-bootstrap lower bound;
- mean utility gains over the structured and root-semantic chains are
  strictly positive;
- relation shuffling causes a strictly positive mean loss;
- descriptive utility is non-worse than the primary baseline in at least
  6/11 games;
- at least one completion trajectory is selected if the frozen trees contain
  any completion opportunities.

Completion and immediate utility are both reported. Oracle performance is a
diagnostic upper bound, not a deployable method.

## Interpretation boundary

A semantic failure rejects this eight-effect object-relative distillation on
the available source panels and skips downstream fitting. It does not refute
all possible semantic representations.

An integration failure after a semantic pass is stronger evidence against the
current global composition, because the full planned architecture is then
executed with informative semantics. A pass is exploratory source-only
support, not authorization for holdout access or live control.

