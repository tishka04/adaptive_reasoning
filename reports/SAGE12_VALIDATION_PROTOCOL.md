# SAGE12 preregistered validation protocol

## Stage A V3 action-target amendment

The V1 and constrained V2 proposal gates are closed negative results. The next
authorized Stage A experiment is the separately frozen action-target V3
protocol in `reports/SAGE12_ACTION_TARGET_PILOT_V3_PROTOCOL.md`.

V3 replaces neither result retroactively. It requires fresh source-only
execution, exact action/destination or click-target anchoring, conservative
component labels, a source-training leakage ladder, action-only and
deterministic baselines, binding/action/label controls, per-game transfer, and
calibration. Qwen2.5 0.5B is diagnostic rather than promotion-critical.

Only a full V3 structured-model pass may authorize a new world-model protocol.
No V3 outcome directly authorizes an EBM, shadow mode, bounded probes, or
controller authority.

Status: Stage A V1-V4 executed and failed closed. V4 found strong temporal
signal but did not pass every gate. V4.1 also failed closed at its source-only
preflight: the role, calibration, prompt-budget, and firewall gates passed,
but effect capacity and game-signature leakage did not. Stages B-E remain
unauthorized. V4.2 passed its target-only source preflight and collection but
failed closed at runtime before a prospective verdict.

V4.1 must stop before prospective collection if its source preflight fails.
If structured V4.1 passes, it authorizes only a separately frozen deterministic
V5 world-model protocol and only for effects admitted by the per-effect
ledger. Qwen must independently pass JSON, grounding, recall, shuffle, and
per-game gates before LLM-generated rules can enter V5. Neither branch
authorizes an EBM or controller.

V4.1 produced 1,911 source windows but no prospective data. Its global role
resolution was 0.9984, calibrated source macro ECE was 0.0360, and Qwen
prompts fit at 322–345 tokens. `actor_displaced` had only 35 positives versus
the required 75, while static identity gained 0.1293 accuracy over action
alone versus the allowed 0.10. Complete result:
`reports/SAGE12_MECHANIC_REPLICATION_V4_1_RESULT.md`.

V4.2 excludes the under-capacity actor effect from authority rather than
weakening its threshold. It tests only `target_created`, `target_removed`,
and `target_moved`, maps anchors to `occupied` / `free` / `none`, tightens
the identity limit to +0.05, and adds source utility plus prospective
anchor-shuffle gates. Its protocol is
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_PROTOCOL.md`.

V4.2 passed source preflight and completed collection, then failed closed
before producing prospective metrics because the frozen serializer omitted
the generic `any` rule anchor. Its opened shards cannot be reused for a clean
replication. Stages B-E remain unauthorized. See
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_RESULT.md`.

SAGE12 V4 is a new Stage A temporal amendment. It tests whether eight observed
transitions can induce a game-local typed rule that predicts the next effect.
Its structured rule inducer is primary and Qwen is secondary. Only a full V4
pass can authorize a separately frozen deterministic-hypothesis world-model
pilot; it grants no EBM or controller authority.

V4 completed `FAIL_CLOSED`. The temporal inducer achieved +0.4676 macro Brier
skill and +0.1526 macro-F1 over the stronger local action-only baseline, with
a positive bootstrap lower bound, strong outcome-shuffle degradation, and
positive transfer in every validation game. Promotion still failed because
source actor-role coverage was only 0.831 and macro ECE was 0.1056 versus the
0.10 maximum. No Stage B model was fit. Full result:
`reports/SAGE12_MECHANIC_INDUCTION_V4_RESULT.md`.

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

Stage A collected 2,104 source-only executed traces and evaluated 224 frozen
Qwen2.5 0.5B generations. All seven gates failed: strict typed parsing and
LLM recall@8 were zero, relation-shuffle degradation was zero, the action-only
baseline reached 0.895 recall, all three validation games transferred
negatively, and game-identity accuracy was 0.999 versus a 0.099 majority.

Result checksum:
`fbb86c17fee57ff46199dd94594936694bf2b0e63b05ece2c9e323813422d35a`.
The complete audit is in `reports/SAGE12_PROPOSAL_PILOT_RESULT.md`.

Failure stopped Stage B before semantic world-model fitting. No learned EBM,
shadow episode, bounded probe, holdout, historical, or `ar25` evaluation
occurred. The live default remains `off`.

A separately versioned Stage A repair is now frozen in
`reports/SAGE12_CONSTRAINED_PILOT_V2_PROTOCOL.md`. It uses constrained typed
outputs and a one-bit actor-interaction representation.

V2 also failed closed. Output validity, grounding, and reduced-leakage gates
passed, but Qwen primary macro-F1 was 0.484 versus 0.549 for action-only,
relation shuffling improved the score, and `re86` transferred negatively.
Result checksum:
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.
Stage B therefore remains unauthorized.

Action-target V3 then collected 4,000 fresh source-only transitions and
selected its coarse projection using source training before opening
validation. Typed output and grounding reached 1.00, and the projection passed
the frozen identity-leakage gate. Predictive transfer failed: the structured
model reached 0.232 macro-F1 versus 0.371 for the stronger deterministic
template, a -0.140 gain with bootstrap 95% interval [-0.155, -0.125].
Target-shuffle degradation was 0.0005, macro ECE was 0.397, and `re86` plus
`sc25` transferred negatively. Validation label capacity and source-training
ambiguity gates also failed.

V3 result checksum:
`10b1d84b6ff675c3fd05f73ad853d0618658b79045824ad4c2f9e79e6466fdb4`.
See `reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`. Stage B remains
unauthorized; no world model or EBM was fit.

## V4.2.1 runtime-safe replication amendment

V4.2.1 is the only authorized successor to the V4.2 runtime failure. It is a
new protocol and dataset version; V4.2 code, artifacts, and opened shards
remain frozen. The scientific model and every V4.2 gate are unchanged.

The added precondition is a source-only full-pipeline rehearsal over all 1,911
source windows. Every generated public rule must round-trip, both exact and
family generic `any` rules must occur, and the full source prediction writer
must successfully serialize selected `any` evidence. Only then may the
ordinary source preflight run.

A passing preflight may collect 768 fresh transitions under seeds 661, 709,
757, and 809. Prospective evaluation must persist predictions and a
structured verdict before Qwen. Qwen retains its V4.2 concrete-anchor
contract and separate authority. An automatic runtime failure artifact
revokes all downstream authority. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_PROTOCOL.md`.

V4.2.1 has now completed `FAIL_CLOSED`. The structured branch passed 18/19
gates but achieved only +0.017061 binding-shuffle loss against the required
+0.020000. Qwen failed all six separate gates. Therefore Stage B and every
later stage remain unauthorized. Full result:
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_RESULT.md`.

## V4.3 causal-binding amendment

V4.2.1's single failed binding-shuffle gate does not authorize a world model.
The separately frozen V4.3 protocol replaces that weak observational control
with replayed executed pairs. It is governed by
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_PROTOCOL.md` and the checksummed
manifest in `training/sage12/bound_mechanic_pilot_v4_3`.

V4.3 must publish its source corpus before source preflight, freeze its
identity-safe projection and all calibration before validation collection,
publish validation shards before scoring, and pass every binding gate before
fitting its structured semantic world model. A binding failure requires an
explicit skipped fail-closed world-model artifact.

Neither a binding pass nor a world-model pass directly authorizes Stage C.
A world-model pass permits only preparation of a new frozen energy/safety
protocol. Qwen, GNN, EBM, controller, holdout, historical, and `ar25` access
remain closed during V4.3.

V4.3 has now stopped at `FAIL_CLOSED` source preflight. Movement capacity,
source LOGO utility, and identity leakage all failed. No projection was
selected, the validation collector stayed blocked, and the binding/world-model
commands wrote skipped closure artifacts. Stages B-E remain unauthorized.
Complete result:
`reports/SAGE12_BOUND_MECHANIC_PILOT_V4_3_RESULT.md`.

## V4.4 paired-causal amendment

V4.4 is a source-only derived repair over the immutable V4.3 pairs. It tests
creation and removal as antisymmetric arm comparisons and keeps movement
diagnostic. The frozen source gates require discordant capacity across at
least two games, positive LOGO utility, binding-swap sensitivity, low identity
gain, calibration, exact arm-swap inversion, non-negative scoreable games,
and a positive bootstrap lower bound.

Only a full source pass may open a new validation collection. A validation
pass may authorize preparation of a separate absolute world-model protocol,
but it does not authorize fitting or control. Protocol:
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_PROTOCOL.md`.

V4.4 has now stopped at source `FAIL_CLOSED`. Capacity and exact
antisymmetry passed, but utility, binding sensitivity, identity, calibration,
per-game transfer, and bootstrap gates failed for every projection. The
validation collection and final result are explicit skipped artifacts; no
validation game was opened. Stages B-E remain unauthorized. Full result:
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_RESULT.md`.

## V4.5 rooted-event feasibility amendment

V4.5 is a design-only successor over the immutable V4.3 source pairs. It
replaces `BindingSignature` supervision with tri-view object correspondence,
common-dynamics cancellation, source-discovered intervention events, and a
target-rooted local graph.

The existing source corpus may be used only for feasibility. Fresh source
collection is blocked unless correspondence confidence, ambiguity, grounding,
event localization and event capacity all pass together with source LOGO
utility, root/relation sensitivity, identity, calibration, per-game,
bootstrap, and exact-antisymmetry gates.

A feasibility pass would trigger an independently seeded source replication
that retains eight full context traces for persistent tracking. Only its
complete pass may open the unchanged SAGE11 source-validation games. A final
pass authorizes preparation of a separate world-model protocol, never model
fitting or control. Qwen, GNN, EBM, controller, holdout, historical, and
`ar25` authority remain closed. Protocol:
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_PROTOCOL.md`.

V4.5 has now stopped at source-only `FAIL_CLOSED`. Root grounding and
exclusive-event localization failed; no direct target event survived event
discovery. Predictive Brier skill was negative, accuracy gain negligible,
identity leakage high, calibration over threshold, per-game transfer
negative in two games, and both root/relation controls moved in the wrong
direction. The fresh-source and validation commands wrote closure artifacts
and created no shard. Stages B-E remain unauthorized. Full result:
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_RESULT.md`.

## V4.18 goal-conditioned trajectory-value amendment

V4.18 is a separately frozen diagnostic over the complete V4.15 human
sequences and existing V4.11 transfer panels. It assigns retrospective credit
at horizons 8, 16, 32 and 64, fits a compact object-relative critic and
compares V4.15, V4.17, action-only, learned, relation-removed, trajectory
oracle and exact-oracle lanes without a blocking intermediate gate.

The storage contract is part of validation: every command records pre/post
inventories, limits scratch and local cache to 5 GiB each, rejects derived
files above 512 MiB, caps the repository at 12 GiB, requires 100 GiB free and
removes command scratch before success.

The completed result is `REPRESENTATION_OR_DATA_BOTTLENECK`. The trajectory
oracle validates the objective, while the learned critic loses to action-only
and relation removal. Nine active runs make no progress and propose no illegal
action. Holdout and authority remain closed. Protocol and result:
`reports/SAGE12_GOAL_CONDITIONED_TRAJECTORY_VALUE_V4_18_PROTOCOL.md` and
`reports/SAGE12_GOAL_CONDITIONED_TRAJECTORY_VALUE_V4_18_RESULT.md`.
