# SAGE12 V4.14 — human-trajectory temporal semantics

Status at freeze: **protocol specified; no V4.14 result observed**.

## Question

V4.13 established that the complete world-model → depth-three trajectory →
EBM → controller chain works with correct semantic effects, while the V4.12
snapshot predictor misses the positive, action-dependent effects that the
controller needs. V4.14 tests whether complete human trajectories can supply
the missing supervision when terminal credit is treated as a temporal signal
instead of a rare one-step class.

Every registered condition will run. Component metrics diagnose the first
failing boundary but cannot stop the global evaluation.

## Frozen split

The SAGE11 registry remains unchanged so old results stay reproducible.
V4.14 defines a separate, content-addressed protocol split:

- human training: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`;
- transfer evaluation: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87`, `tu93`;
- bounded active validation: `re86`, `ls20`, `sc25`;
- final confirmation only: the five `NEURO_HOLDOUT_V1` games.

All six games with recorded human play are legitimate V4.14 training games.
Consequently, `ar25`, `cn04`, and `ft09` cease to be independent evaluation
games for this protocol only. The holdout remains closed.

The manifest fingerprints every human JSONL file, the V4.11 teacher panels,
the V4.13 result, the split, seeds, hyperparameters, and forbidden fields
before training.

## Temporal teacher

The teacher converts each non-reset human transition into three disjoint
views:

1. **Audit view:** game, episode, step, before/after digests, source file and
   human annotations.
2. **Student view:** the current object-relative graph, the legal executed
   action and a causal history ending strictly before that action.
3. **Teacher targets:** the seventeen existing semantic effects, persistent
   role evidence, next-belief deltas, and hindsight trajectory targets.

Hindsight targets are:

- a level or WIN within 4, 16, and 64 actions;
- steps to the next level, right-censored at 128;
- discounted progress `0.97 ** distance`;
- GAME_OVER within eight actions;
- preference between observed suffixes from the same pre-state.

Immediate `productive` remains an immediate physical/functional label. Earlier
actions in a successful sequence are not relabelled as immediate wins; their
credit enters only the temporal heads.

The tracked role vocabulary is `controllable`, `movable`, `blocker`,
`consumable`, `hazard`, and `goal_relevant`. Free-form hypotheses, objectives,
post-action grids, raw colours, absolute coordinates, game identity, state
hashes, and persistent object identifiers never enter the student view.

## Student and deployable rollout

The V4.12 object-relative DeepSets encoder is extended with a 128-unit GRU.
Training uses 32-transition truncated histories, 32-wide hashed embeddings,
a 96-wide graph trunk, 30 epochs, AdamW at `0.0015`, and weight decay
`0.0001`.

The frozen loss is:

- immediate masked semantic BCE: `1.0`;
- next-belief transition loss: `1.0`;
- multi-horizon progress BCE: `0.5`;
- observed-suffix ranking: `0.5`;
- censored distance loss: `0.25`;
- game-identity confusion: `0.1`.

Positive weights are computed inside each training fold and capped at `20`.
Every diagnostic prediction on a human game is outer leave-one-game-out.
After that audit, one final checkpoint is fitted on all six human games.

The rollout applies a candidate action to `TemporalBeliefState` and predicts
the next semantic belief without reading any true future descriptor. The
existing `SlotAnnotation` boundary remains available through an adapter.
Qwen2.5 0.5B keeps the frozen V4.7 decoding contract. CUDA is used only when
a same-workload benchmark is at least 20% faster than CPU.

## Unconditional evaluation

The component ladder contains:

- action-only and action-sequence-only;
- deterministic templates;
- V4.12 snapshot semantics;
- V4.14 temporal semantics;
- relation-shuffled and history-shuffled controls;
- oracle semantics;
- true executed world outcomes with the learned EBM;
- the complete Qwen → compiler → temporal rollout → EBM → controller chain.

Primary transfer measurements are balanced positive-effect recovery, Brier,
ECE, observed-suffix ranking, one- and three-step rollout error, game-signature
leakage, per-game utility, completion capture, and paired bootstrap gain over
the action-sequence baseline.

`GLOBAL_CHAIN_SUPPORTED` requires all of:

- paired 95% bootstrap lower bound above zero;
- non-negative transfer on at least five of eight games;
- at least one completion opportunity captured;
- at least 50% of the opportunities captured by the oracle.

If this classification is not reached, the completed ladder assigns the first
failing component. No local metric prevents later conditions from running.

After the checkpoint and analysis are frozen, the active and baseline
controllers are paired on `re86`, `ls20`, and `sc25`, seeds 0–2, budget 1,000
actions and at most 14 resets. Levels, WINs, GAME_OVERs, illegal proposals,
latency, and paired level/WIN progress are descriptive in V4.14 and cannot
open the holdout or promote authority.

## Reproducibility and publication

The implementation must expose `freeze`, `compile`, `train`, `evaluate`, and
`run-all` commands. It writes checksummed JSON/JSONL artifacts under
`training/sage12/human_temporal_semantics_v4_14`, a result report whether the
verdict is positive or negative, focused tests, and an updated SAGE12 README.
Only scoped V4.14 files may be committed; unrelated local edits are preserved.
