# SAGE12 semantic-trajectory data policy

Status: Stage A collection frozen before outcomes; collection not started.

Format: `sage12-semantic-trajectory-v1`.

The first concrete collection is preregistered as
`sage12-proposal-pilot-v1` in
`training/sage12/proposal_pilot_v1/frozen_manifest.json`, checksum
`0260eb15fd9a0cecb21644160888bde9b6e5be03b4428f1afd989401686c148b`.
It is a 2,104-row source-only proposal pilot and cannot authorize world-model
fitting unless every frozen Stage A gate passes.

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
