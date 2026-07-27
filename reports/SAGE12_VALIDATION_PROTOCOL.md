# SAGE12 preregistered validation protocol

Status: protocol implemented with the software stack; no empirical SAGE12
pilot has been run.

## Research question

Can grounded high-level hypotheses and semantic trajectory scoring add useful
cross-game control information beyond action identity, deterministic
templates, and the existing symbolic controller without importing game
identity or reducing safety?

This protocol replaces the failed SAGE11 relational effect track. Passing a
stage authorizes only the next stage, never retroactively validates SAGE11.

## Frozen system

- Observation: grounded scene entities plus contact, alignment, proximity, and
  directional relations.
- Proposal model: one preregistered local open-weight instruction model,
  deterministic decoding by default, strict typed JSON, maximum eight
  hypotheses.
- Compiler: exact legal action/arguments, bindable roles, true current
  preconditions, bounded bindings, allowlisted effects.
- World model: semantic predicate transitions with maximum depth three and
  beam width eight.
- Baseline energy: explicit goal-distance, observed-danger, uncertainty,
  likelihood, cost, and contradiction terms.
- Learned EBM: optional six-input pairwise MLP trained on observed preferences.
- Controller: receding horizon; at most the first action executes.

Any model, prompt, schema, split, feature, metric, threshold, or seed change
after viewing validation outcomes creates a new version and a new protocol.

## Stage A — grounded proposal pilot

Fit or prompt-tune on source training only, then evaluate once on the frozen
source-validation games.

All gates must pass:

1. strict JSON validity at least 99%;
2. at least 95% of parsed proposals obey `support=0`;
3. at least 90% of parsed proposals either compile or receive the expected
   explicit compiler rejection;
4. productive-mechanism recall@8 exceeds both the deterministic template and
   action-only baselines by at least 0.10;
5. entity/relation binding shuffle reduces recall@8 by at least 0.10;
6. no validation game has negative recall gain versus the stronger baseline;
7. game identity from fixed scene signatures is no better than the
   preregistered majority plus 0.10 absolute accuracy.

Failure stops world-model fitting. The LLM may still be used as a logged
candidate generator in offline diagnostics, never as control authority.

## Stage B — semantic world-model pilot

Train only on executed source-training transitions. Score individual semantic
effects with macro-F1 and Brier/ECE, and score complete trajectories by whether
their predicted subgoal effects occur within the registered horizon.

All gates must pass:

1. effect macro-F1 is at least +0.10 above the stronger action-only or
   state-only baseline;
2. trajectory top-k success is at least +0.10 above one-step/template rollout;
3. within-state action shuffling degrades effect macro-F1 by at least 0.10;
4. binding shuffling degrades trajectory top-k success by at least 0.10;
5. ECE is at most 0.10;
6. at least 9 of 11 source-training LOGO folds and every source-validation game
   are non-negative versus the stronger baseline;
7. unsafe-effect recall is at least 0.90.

Failure leaves SAGE12 off and stops EBM training.

## Stage C — energy pilot

Build preference pairs only from observed outcomes. A preferred trajectory
must be safer first, then show stronger subgoal progress, then lower cost; ties
are excluded. Train the optional EBM on source training and compare it with the
fixed heuristic energy on source validation.

All gates must pass:

1. pairwise accuracy exceeds 0.70;
2. accuracy improves by at least +0.05 over the heuristic energy;
3. unsafe-decoy rejection is at least 0.99;
4. relation/binding shuffling reduces accuracy by at least 0.10;
5. every source-validation game is non-negative versus the heuristic;
6. all inputs are the six documented energy components—no game ID, raw
   coordinates, state hash, outcome leakage, or free-form LLM text.

Failure preserves the heuristic ranker for diagnostics but grants no live
authority.

## Stage D — shadow controller

Run paired source-validation episodes with identical seeds/resets. Shadow must
return the byte-identical symbolic action while logging its counterfactual
choice.

Required gates:

- action identity 100%;
- zero danger-veto violations;
- zero protected-competence preemptions;
- zero illegal action/argument proposals reaching ranking;
- proposal plus rollout p95 latency at most the preregistered live budget;
- productive top-1 and top-3 precision reported with paired bootstrap 95%
  intervals;
- no data or memory survives a game/seed boundary except explicitly frozen
  source knowledge.

Passing authorizes one bounded probe per branch/context. It does not authorize
active mode.

## Stage E — bounded and final holdout

Bounded source-validation must show a positive paired bootstrap lower bound on
level/WIN progress, zero additional unsafe outcomes, zero lost WINs, and no
protected-route preemption. Only then may the existing five-game × five-seed
`NEURO_HOLDOUT_V1` matrix be opened exactly once.

Active promotion requires:

- paired bootstrap 95% lower bound above zero for the primary level/WIN metric;
- at least one new level or WIN;
- zero unsafe regressions;
- zero lost WINs;
- no game with a material preregistered regression;
- complete logs, device metadata, prompt/model checksum, data manifest,
  checkpoint checksum, and software commit.

Any failure is published as a negative result and leaves the effective mode at
the last passed stage.

## Current result

Only software tests have run. No dataset, LLM proposal benchmark, semantic
world-model fit, learned EBM fit, shadow episode, bounded probe, or holdout
evaluation has occurred. All empirical gates are therefore unpassed and the
live default is `off`.
