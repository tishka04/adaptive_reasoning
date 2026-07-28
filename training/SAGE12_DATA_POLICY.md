# SAGE12 semantic-trajectory data policy

Status: Stage A V1-V4 are complete. V4.1 failed its frozen source-only
preflight and stopped before prospective collection. World-model and EBM
fitting remain unauthorized. V4.2 opened its frozen prospective corpus and
failed closed before producing a metric verdict.

V4.1 derives new versioned windows from the immutable V3 source-training
traces. It may write source priors, leave-one-game-out calibration,
source-only thresholds, and prompt-token audits before prospective
collection. Those artifacts must be committed before any new validation
transition is opened.

Only after a fully passing source preflight may V4.1 collect 768 new
transitions: 256 per `re86`, `ls20`, and `sc25`, using seeds 307, 347, 389,
and 433. The collection remains balanced, non-adaptive to outcomes, reset
bounded, and retains chronological repeats. Its frozen manifest checksum is
`86b3d3b38ba41d0f860169928f6cc5afd6765ccdbf83078e3a09d60da0e07abc`.

That collection was not opened. V4.1 froze 1,911 derived source windows,
source priors, calibration parameters, and its failed machine-readable
preflight under checksum
`cffa41e2ae980f64dfc76cbe40076809b301da4e8f98dffbc02122eb2bfa147c`.
The source-validation shard does not exist by design. Any successor requires
a new versioned manifest and source preflight; V4.1's unused prospective
manifest cannot be treated as approval to collect.

V4.2 re-derives source windows from the same immutable V3 source-training
traces under a new format and checksum. Its authoritative dataset contains
only target creation, removal, and movement; actor displacement is retained
only as an audit count. Coarse anchors are `occupied`, `free`, and `none`.
Prospective collection is forbidden until the V4.2 source preflight passes.
If authorized, it is exactly 768 new transitions on `re86`, `ls20`, and
`sc25`, using seeds 479, 523, 569, and 617. Manifest checksum:
`fba242f31cbc492f44333bcae8c5f9228baee0b79c15f0c68009dce7a76a6210`.

The V4.2 source preflight passed all gates under checksum
`68747717f45289775cd543aaa027eb24164200b255b42b57368e4c6fba0816ff`.
Its source windows, priors, calibration, and token audit are frozen.
Collection of the exact preregistered 768 transitions is therefore
authorized; no other game, seed, row count, or adaptive policy is permitted.

That collection completed at exactly 768 rows under report checksum
`6bdec774c744061e3e5014ced8d3d0191d1cdc13243130817ea9ec84fd50dce7`.
Its three raw shards and combined checksum are frozen before prospective
evaluation. Chronological repeats remain part of the audit corpus.

The frozen evaluator opened those outcomes and stopped at
`FAIL_RUNTIME_CLOSED` before producing predictions or a pilot result. The 576
validation windows and two 128-row Qwen streams remain non-authoritative
audit data. The V4.2 shards are now opened and must not serve as the clean
gating set of a successor.

The next authorized experiment is the offline SAGE12 V4 temporal
mechanic-induction pilot. It derives eight-transition source-training windows
from the immutable V3 training traces, excludes the already inspected V3
validation outcomes from gating, and collects 768 fresh prospective
transitions under a separately checksummed manifest. V4 retains chronological
repeats because they are evidence about a mechanic; scored window digests,
frame continuity, reset boundaries, and all raw/model-view separation remain
audited.

V4 completed with 1,911 source-training windows, 768 fresh prospective
transitions, and 576 prospective windows. Its result is `FAIL_CLOSED`,
checksum
`5987eb9531f568dc814dad46eb9e78d13a3813a9c30db3d6cb1fa8a319e16927`.
The corpus remains audited research data. Its strong temporal prediction
signal does not override the failed source actor-quality and calibration
gates, and it authorizes neither world-model nor EBM fitting.

Format: `sage12-semantic-trajectory-v1`.

The first concrete collection is preregistered as
`sage12-proposal-pilot-v1` in
`training/sage12/proposal_pilot_v1/frozen_manifest.json`, checksum
`0dfdff9a61e45e02b16601a47d987454c991a2d5f99c8964a5486c17ed17aceb`.
It is a 2,104-row source-only proposal pilot and cannot authorize world-model
fitting unless every frozen Stage A gate passes.

The collection completed with exactly 1,624 source-training and 480
source-validation rows. Its compact-shard checksum is
`ce5cfe1217f9de1ef9add9ab250f60315ed66d154ae8ed903e51bb572b69a4b3`.
Stage A subsequently failed all seven gates, so this data remains an audited
negative-result corpus and cannot be used to fit the semantic world model
under this protocol. See `reports/SAGE12_PROPOSAL_PILOT_RESULT.md`.

## Constrained Stage A V2

The same immutable 2,104 rows may be reused by the separately versioned
constrained pilot in
`training/sage12/constrained_pilot_v2/frozen_manifest.json`, checksum
`033274922c2f25d3cb7918bc2f308fffdc03d4811f034e9db171147d4aec25aa`.
This is evaluation reuse, not a new collection or silent relabelling.

V2 fits a linear multi-label head on source training only. Its Qwen prompt
contains the selected action and one binary actor-interaction motif. It may
read source validation exactly once after its implementation, tests, gates,
preflights, protocol, and manifest are committed. The first six-bit motif was
rejected using source-training leakage only; both the rejected and retained
preflights must remain published. V2 still cannot authorize world-model
fitting unless every new frozen gate passes.

V2 completed `FAIL_CLOSED`, checksum
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.
The immutable corpus may remain published as negative-result evidence, but
neither its V1 full-graph view nor its V2 one-bit projection authorizes
semantic world-model labels or fitting. A future action-target-grounded format
must be separately versioned and collected under a new frozen manifest.

## Action-target Stage A V3

The required new format is frozen as `sage12-action-target-trace-v3` in
`training/sage12/action_target_pilot_v3/frozen_manifest.json`, checksum
`8aff373b2896b13dfafe88a8a8d37d9399088f881386a19c81cb63acb3f487bf`.
It collects 4,000 fresh executed transitions: 3,040 source-training and 960
source-validation. V1/V2 rows are not relabelled or mixed into this corpus.

V3 retains raw frames and exact coordinates only as checksummed audit
provenance. Model projections may contain the action, requested direction,
anchor kind, coarse actor-target relation, occupancy, path status, and the
first source-training-approved coarse target descriptors. They may never
contain absolute coordinates, colour/value, IDs, game signatures, raw grids,
policy metadata, or outcomes.

Training collection has a 2,587-row balanced base and a preregistered 453-row
event-deficit top-up. Validation is fixed at 320 rows per source-validation
game and is never outcome-adaptive. Exact pre-frame/action/argument duplicates
are forbidden.

V3 makes the structured action-target classifier primary and Qwen a secondary
ablation. A structured pass can authorize a separately frozen small semantic
world-model protocol even when Qwen fails. Until every V3 gate passes, no
world model or EBM may be fit.

The source-training collection completed at 3,040 unique rows, manifest
checksum
`1ba0b41b2595a1c9f18f613696e97b4397066194fe700117104d1eaa930d3331`.
The source-only preflight selected the `coarse` projection and shallow
gradient boosting; projection-freeze checksum
`7e1a93970b5502873bce6c3659ba46f671752adce81a8b2da829a6485b36ce9c`.
No source-validation metric was seen before this freeze.

V3 subsequently completed `FAIL_CLOSED`, result checksum
`10b1d84b6ff675c3fd05f73ad853d0618658b79045824ad4c2f9e79e6466fdb4`.
The primary structured model reached 0.232 macro-F1 versus 0.237 for
action-only and 0.371 for the deterministic template. Its gain against the
stronger baseline was -0.140, target-shuffle degradation was 0.0005, and
macro ECE was 0.397. Validation creation/removal positives also occurred in
only one game, while actor ambiguity reduced source-training non-ambiguity to
0.822.

The complete result is in
`reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`. These 4,000 rows remain an
audited negative-result corpus. They may support explicitly labelled
post-hoc diagnostics, but they do not authorize fitting, tuning, or promoting
a semantic world model, EBM, or live controller. A sequence-conditioned
mechanic-induction repair must use a separately frozen protocol and must not
silently redefine V3 labels or gates.

## Purpose

SAGE12 data teaches and evaluates three separable questions:

1. Does a grounded hypothesis generator cover productive mechanisms?
2. Does the semantic world model predict observed abstract effects and useful
   multi-step consequences?
3. Does a trajectory energy model rank productive, safe trajectories above
   matched alternatives?

The record keeps proposal text/IDs, compiler rejections, trajectories,
energies, the actually selected action, and observed outcomes separate.
Generated or rolled-out fields are never labels by themselves.

## Collection unit

One record is created for one real decision cycle:

- source game, branch, and step are provenance metadata only;
- the model-facing scene uses structural roles and relations, not game ID;
- hypotheses have `support=0`;
- every compiled option maps to one exact legal action and argument set;
- every rollout is counterfactual and remains candidate-only;
- the outcome fields are populated only after the selected real action is
  observed;
- unsafe means observed `game_over` without terminal success;
- productive means an observed non-noop or level completion. Later protocols
  may add stronger semantic labels, but must version the format.

JSONL writes are append-only. The final collector must publish row counts,
per-game counts, policy mixture, duplicate policy, and SHA-256 checksums before
training.

## Firewall

Until a separately frozen SAGE12 collection manifest exists:

- use only the existing SAGE11 source-training games for fitting;
- use only the existing SAGE11 source-validation games for pilot selection;
- do not open `NEURO_HOLDOUT_V1`;
- do not train on historical report-only games;
- keep `ar25` regression-only;
- do not put source game ID, raw grid hash, absolute layout signature, policy
  arm, outcome, or future state into model inputs;
- split by game, never by row;
- fit vocabularies, normalizers, templates, prompts, and calibration only on
  source training.

If a new game registry or changed split is needed, it must be committed before
collection. SAGE11 rows may be referenced for provenance but are not silently
relabelled as semantic trajectories.

## Required negative controls

Every pilot must include:

- action-only and deterministic-template baselines;
- within-game action shuffling conditioned on the same semantic state;
- relation/entity binding shuffling with action identity preserved;
- fixed-signature game-identity probe;
- matched unsafe and nonproductive decoys;
- leave-one-game-out source-training reporting;
- per-game results, not only an aggregate.

## GPU policy

The 0.5B local proposal model may use the laptop GPU for inference when
`device="auto"` selects CUDA and measured wall time improves without changing
the frozen model, prompt, decoding, or outputs. The pairwise EBM may use CUDA
for a preregistered training run. CPU remains the reference path. Device,
library versions, seeds, wall time, peak memory, and output checksum must be
recorded. Hardware choice may not be changed after outcomes to rescue a gate.

## Promotion boundary

Data collection alone grants no controller authority. Proposal, world-model,
energy, shadow, and final holdout gates in
`reports/SAGE12_VALIDATION_PROTOCOL.md` must pass in order. Any failed stage
stops later evaluation and leaves SAGE12 off or shadow-only.

## V4.2.1 clean recovery corpus

SAGE12 V4.2.1 uses the immutable V3 source-training traces only for its
source rehearsal, priors, and calibration. Its source rehearsal prediction
artifact is derived data and must cover all 1,911 source windows before any
prospective row is collected.

The V4.2 prospective shards are opened and must not be reused, copied, or
relabelled. After both source gates pass, V4.2.1 may collect exactly 768 fresh
transitions, 256 per source-validation game, under unused seeds 661, 709,
757, and 809. Collection remains chronological and outcome-independent;
repeated transitions are retained as mechanic evidence and reported.

Raw prospective shards must be published before evaluation. Predictions and
the structured intermediate verdict must then be written before Qwen
generation. All artifacts remain audit data unless every frozen gate passes.
They cannot directly authorize world-model, EBM, or controller fitting.

The V4.2.1 collection completed with 768 fresh rows and combined shard
checksum
`9cbc1dcb450a71f1a670e515b5adcd7d72af7d9c9fd21549a9b1514917d65a4c`.
It contains 256 rows per validation game and retains 79 chronological exact
repeats. These shards are now opened, immutable prospective audit data and
may be evaluated only by the already frozen V4.2.1 code.

That frozen evaluation is now complete and failed closed. The 768 traces,
576 windows, predictions, structured intermediate, and Qwen streams are
immutable negative-promotion audit data. They may be used only by a separately
labelled post-hoc diagnostic or a newly frozen successor protocol; they do
not authorize fitting or tuning a semantic world model, EBM, or controller.

## V4.3 replayed counterfactual-tree corpus

V4.3 is an independent prospective dataset with format
`sage12-bound-trajectory-v4.3`. It must not reuse, copy, or relabel V4.2 or
V4.2.1 prospective shards.

One row is a pair of executed interventions from an identical pre-state. The
collector restores the state by replaying the reset-local action prefix,
verifies the grid/state/level hash, clones it for the left and right arms, and
keeps both outcomes. Pairs are organized into binary trees of maximum depth
three. Root selection, action selection, and branch selection may use only
pre-action legal actions and semantic binding strata; effect-adaptive sampling
or post-outcome deletion is forbidden.

The authorized source-training cap is 32 roots per game across the 11 frozen
source games: at most 2,464 pairs or 4,928 executed branch transitions. The
authorized source-validation cap is 64 roots per game across `re86`, `ls20`,
and `sc25`: at most 1,344 pairs or 2,688 branch transitions. Seeds are
857/907/953/1009 for source and 1061/1103/1151/1201 for validation.

Raw frames, state hashes, absolute coordinates, action arguments, object IDs,
game IDs, seeds, reset indices, root IDs, paths, and observed outcomes are
audit fields. Model inputs contain only action name/family, the frozen
identity-free binding projection, and the preceding eight semantic events.
Validation collection is forbidden until the source-only projection,
calibration, thresholds, baseline choice, and identity diagnostic are frozen
and published.

Every raw shard and manifest is published before its corresponding evaluation.
A failed binding result remains immutable negative-result audit data and
forbids world-model fitting. A passing binding result authorizes only the
frozen structured world-model evaluation; even a world-model pass does not
authorize an EBM, controller, holdout, historical set, or `ar25`.

The V4.3 source collection is now immutable: 352 roots, 2,396 pairs, 4,792
arms, and report checksum
`a842c0bdd99a1e10ad48c03ded447e231a6767e6af7410192b2f21c4b2948722`.
It contains every observed arm, including terminally truncated trees, and had
zero replay failures. These source shards may be read only by the already
frozen V4.3 source preflight. They do not authorize validation collection or
any downstream model by themselves.

The frozen source preflight failed, so the source shards are now immutable
negative-result audit data. `projection_freeze.json` selects no projection.
The V4.3 validation collector remains blocked and no validation shard may be
created under this version. The binding and world-model closure records grant
no downstream authority.

## V4.4 derived paired-causal view

V4.4 may read the immutable V4.3 source-training shards without copying or
rewriting them. Its derived unit is a directional pair for creation or
removal: both outcomes must be applicable and exactly one arm must be
positive. Same-outcome pairs remain audit data but are not directional labels.
Movement remains diagnostic because the source corpus contains no discordant
movement pair.

Model inputs are left-minus-right action, binding, and arm-conditioned
eight-event history features. Game and pair IDs, frames, hashes, coordinates,
arguments, object IDs, seeds, resets, paths, and outcomes are forbidden.
Source LOGO predictions alone select the projection, temperatures, model,
baseline, and thresholds.

No validation data may be created unless every frozen V4.4 source gate passes.
An authorized validation collection must use fresh seeds 1451, 1499, 1553,
and 1601 and publish raw shards before evaluation. Even a validation pass
authorizes only a separately frozen absolute world-model protocol; it does not
authorize fitting that model, an EBM, or a controller.

The V4.4 source preflight failed every predictive projection. Its
`projection_freeze.json` selects no representation. The validation collection
manifest records `SKIPPED_SOURCE_PREFLIGHT`, and no validation shard directory
exists. V4.4 derived artifacts are immutable negative-result audit data and
grant no downstream authority.

## V4.6 exploratory global-integration view

V4.6 may read, but never rewrite or relabel, the 340 complete binary trees in
the immutable V4.3 source-training shards. This is a separately frozen
architecture experiment, not a continuation of V4.3 authority. It may fit an
experimental leave-one-game-out semantic world model and pairwise EBM solely
to measure the complete offline chain. That permission does not override the
V4.3 fail-closed promotion verdict and cannot authorize source validation,
holdout access, live execution, or controller authority.

No new transition is collected. The only new model input is the existing
pre-action scene graph at a frozen V4.3 root. Future frames and effects are
used only to train other-game folds, create pairwise preferences, and score
the already executed branches. Each held-out source game is excluded from its
world-model, action-only, and EBM training.

Qwen generation is restricted to the 44 outcome-blind root keys frozen in
`integration_pilot_v4_6/frozen_manifest.json`, once with original relations
and once with the frozen relation shuffle. Raw responses, deterministic
normalization, folds, per-root decisions, and the final result are immutable
audit artifacts. The completed negative result grants no downstream
authority and may be superseded only by a new versioned protocol.
