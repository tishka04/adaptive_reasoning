# SAGE12 action-target effect pilot V3 protocol

Status: frozen before collecting a V3 transition.

Frozen manifest:
`training/sage12/action_target_pilot_v3/frozen_manifest.json`.

Manifest checksum:
`8aff373b2896b13dfafe88a8a8d37d9399088f881386a19c81cb63acb3f487bf`.

## Question and authority boundary

V1 showed that free-form Qwen output was not a usable typed interface. V2
repaired that interface but showed that a single actor-interaction bit did not
predict effects beyond action identity. V3 asks the narrower causal question:
does a representation anchored to the exact destination or click target carry
a transferable effect signal?

The primary model is deliberately cheap and structured. Qwen2.5 0.5B is a
frozen secondary ablation. A Qwen failure cannot veto a structured-model pass.
A V3 pass authorizes only a separately frozen small semantic-world-model
iteration. V3 does not fit a world model or an EBM and grants no controller
authority.

## Data and firewall

V3 executes 4,000 fresh source-only transitions. It does not relabel or append
the immutable V1/V2 corpus:

- 3,040 source-training rows from the 11 SAGE11 source-training games;
- 960 source-validation rows, exactly 320 each from `re86`, `ls20`, and
  `sc25`;
- no historical, holdout, or `ar25` access;
- split by game, never by row;
- source validation stays closed until the projection and structured-model
  family are frozen from source training.

The source-training base is 256 transitions for each non-`lp85` game and all
27 verified unique `lp85` transitions, for 2,587 rows. A deterministic
training-only scheduler allocates the remaining 453 rows, with at most 64
extra rows per game. Twenty percent of adaptive choices preserve balanced
exploration. The other 80% prioritizes action-anchor strata using Beta(1,1)
smoothed observed yield for the still-deficient event labels.

The adaptive targets are 200 actor displacements and 100 each target
creations, removals, and movements. They guide sampling but do not weaken the
frozen promotion capacities. Validation collection is balanced and never
outcome-adaptive. An exact pre-frame/action/argument combination may appear
only once.

## Action-target record

The raw `sage12-action-target-trace-v3` record retains the before/after frames,
exact action arguments, state, reset, seed, and hashes for audit only.

The model-facing projection contains:

- selected action identity and action family;
- requested movement direction where applicable;
- anchor kind: movement destination, clicked object, clicked empty cell, or
  targetless;
- occupied/open status;
- actor-anchor distance relation and relative direction;
- coarse target affordance, size, and aspect in the first projection.

It excludes coordinates, colours, cell values, entity IDs, global counts,
game ID, policy metadata, grid hashes, raw frames, and all outcomes.

The source-training-only leakage ladder is fixed:

1. `full`;
2. `no_shape`, removing size, aspect, and affordance;
3. `coarse`, additionally collapsing direction and distance detail;
4. `LEAKAGE_FAIL`.

The first projection whose game-identity accuracy gains no more than 0.10
beyond action-only is frozen. No projection may be designed after validation
is opened.

## Observed labels and conservative matching

V3 scores independent observed components:

- `actor_displaced`;
- `target_created`;
- `target_removed`;
- `target_moved`.

Terminal success, terminal failure, and no-op remain separately reported.

Objects are matched one-to-one using value, area compatibility, overlap, and
bounded displacement. A near tie within 0.08 is ambiguous and is not used as
supervision. Labels also carry explicit applicability masks: for example,
removal and movement are not applicable when the pre-action anchor contains
no object. Raw ambiguous rows remain published.

## Models and controls

Source-training leave-one-game-out macro-F1 selects once between:

- independent class-balanced liblinear logistic heads;
- independent depth-3 histogram gradient-boosting heads.

Both use the action-target projection. The fixed threshold is 0.5.

Baselines are selected-action-only logistic heads and a deterministic template
using movement openness and click occupancy. Controls are target-feature
permutation within game/action/anchor kind, action permutation within game,
source-training label permutation, game-identity probes, and per-game
reporting.

The frozen Qwen2.5 0.5B encoder receives exactly the selected structured
projection. Unique prompts are embedded on `cuda:0`; independent logistic
heads are fitted on source training. Its performance, target-shuffle
sensitivity, timing, prompt checksum, and embedding checksum are diagnostics,
not gates.

## Frozen gates

All gates must pass:

- 100 positive and 100 negative source-training examples per primary label;
- 20 positive and 20 negative validation examples per label, with positives
  in at least two validation games;
- zero exact duplicate rows;
- non-ambiguous rate at least 0.95 globally and 0.90 in every game;
- strict JSON validity and `support=0` exactly 1.00;
- grounded emitted hypotheses at least 0.99;
- structured macro-F1 gain at least +0.10 over the stronger baseline;
- the 95% stratified-bootstrap lower bound of that gain strictly above zero;
- target-shuffle macro-F1 degradation at least 0.05;
- non-negative gain in each validation game;
- macro ECE at most 0.10;
- selected projection game-identity gain beyond action-only at most +0.10.

Any failure produces `FAIL_CLOSED`. Missing label capacity is reported as a
data-sufficiency failure rather than silently dropping a label.

## Reproduction and publication order

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.action_target_collection source_train
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.action_target_pilot preflight
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.action_target_collection source_validation
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.action_target_pilot evaluate
```

The protocol, code, tests, and collection manifest are committed before the
first command. The training preflight and projection freeze are committed
before the third command. The final collection, predictions, result, and
positive or negative report are then published without changing a gate.

## Source-training collection execution amendment

The first source-training launch completed all 11 base shards, then stopped
before writing the first adaptive top-up because Windows/OneDrive temporarily
locked the existing `bp35.jsonl` destination during `os.replace`. No
source-validation game was opened and no preflight, model fit, metric, or gate
was computed.

Atomic shard and manifest replacement now retries the identical byte-for-byte
operation after transient `PermissionError` failures. Quotas, seeds, action
selection, adaptive allocation, records, labels, projections, models, and
gates are unchanged. The source-training collection resumes from the
checksummed base shards after this amendment is committed.

The resumed run then established that `su15` could supply 290 unique rows but
saturated before its provisional adaptive allocation of 301. The frozen
policy already defines the 453 rows as a global event-deficit top-up with only
a maximum, not a required per-game top-up. The collector now preserves every
unique `su15` row and deterministically reallocates its shortfall to the
highest-yield non-saturated source-training game below the same +64 cap.
The global 3,040/960 budgets, exact-duplicate rule, event targets, firewall,
models, and gates remain unchanged.

The first source-training preflight invocation then stopped before producing
an accuracy or model-selection result because scikit-learn 1.9 rejects direct
multiclass use of the frozen `liblinear` solver. The identity probe now wraps
the same class-balanced binary estimator in an explicit one-vs-rest
classifier. Features, folds, solver, regularization, labels, selection
threshold, and leakage limit are unchanged. Source validation was still
unopened when this compatibility amendment was made.

## Source-training freeze result

The completed 3,040-row source-training collection has manifest checksum
`1ba0b41b2595a1c9f18f613696e97b4397066194fe700117104d1eaa930d3331`.
It contains no exact duplicate and produced 721 actor-displacement, 160
target-creation, 369 target-removal, and 147 target-movement positives.

The source-training-only preflight checksum is
`1ea27b59159bb138cfa7321fbf40d2a5abf6d20e3302c02a05b1ba4c14fccc5a`.
The full projection leaked +0.3793 game-identity accuracy beyond action and
`no_shape` leaked +0.1289. The frozen `coarse` projection passes at +0.0987.
Leave-one-game-out macro-F1 selected shallow gradient boosting at 0.2012 over
logistic at 0.1903. Projection-freeze checksum:
`7e1a93970b5502873bce6c3659ba46f671752adce81a8b2da829a6485b36ce9c`.

The preflight also reports a known quality risk without changing a gate:
actor identification is unavailable on 540 rows, reducing the global
non-ambiguous rate to 0.8224, mostly in `cd82` and `sp80`. This is already
below the frozen 0.95 promotion gate. The protocol nevertheless proceeds to
the once-only source-validation evaluation so transfer, controls, and the
complete negative or positive result can be published.
