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
