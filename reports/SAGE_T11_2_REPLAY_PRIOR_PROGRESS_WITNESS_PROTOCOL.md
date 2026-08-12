# SAGE.T11.2 — Replay-prior progress-witness protocol

Status: `PREREGISTERED_BEFORE_IMPLEMENTATION_AND_SOURCE_TRAIN`

Date: 2026-08-12

## Scientific question

Can a common posterior over complete causal programs improve `bp35` control after:

1. scoring each program against its own declared local progress predicates;
2. persisting exact-prefix branch evidence and reloading it before paired control;
3. restricting bounded authority to an exactly supported, non-terminal intervention?

This iteration tests a mechanism correction, not a posterior-temperature or likelihood-weight retuning.

## Frozen scope

- Game: `bp35` only.
- Split/stage: `source_train` only.
- Seeds: `5101`, `5102`, `5103`.
- Resets per seed and arm: `2`.
- Action budget per reset: `48`.
- Authority: paired `shadow` baseline and `bounded` causal arms.
- Maximum causal overrides per reset: `1`.
- Maximum predicted terminal risk: `0.05`.
- Minimum exact intervention support: `1` observed branch.
- Minimum top-particle posterior probability: `0.80`.
- Any supported terminal failure vetoes the intervention.
- Maximum generated artifacts per replay run and paired run: `3 * 1024^3` bytes (`3 GiB`), enforced before writes.
- Neural holdout and source-validation games remain closed.

## Frozen inputs

T11.2 creates new immutable inputs under `training/sage_t/causal_inputs_v3/`:

- `programs.raw.json` and `programs.sealed.json`;
- `bundles.raw.json` and `bundles.sealed.json`.

The rival programs retain the T11.1 structural families so the iteration isolates the semantic and replay-to-control changes. Exact-prefix bundles retain the verified T11.1 prefixes and branches. The new sealed files must bind their content to the T11.2 protocol checksum.

## Local progress witness

For each particle `P_k`, observed progress is evaluated by executing every predicate in `P_k.goal.progress_predicates` on the observed `state_after`. The predicted progress is evaluated with the same predicates on the predicted `state_after`.

The progress channel must not use `observed_delta.progress` or `level_change`. Level progression and success remain separate goal/level likelihood channels.

## Replay-to-control closure

The exact-prefix replay phase must:

1. collect every branch as `TransitionEvidence`;
2. update one replay posterior over the sealed rival programs, with repair disabled;
3. append every update to an A40 causal-memory file;
4. include the memory path, SHA-256, record count and evidence count in the signed replay report and receipt.

Every causal paired arm except `no_replay_prior` must verify and reload this replay memory before its first decision. `no_mdl_prior` must replay the persisted evidence with `mdl_beta=0` rather than reuse weights produced under the MDL prior.

## Bounded intervention rule

When the causal choice differs from the historical action, bounded authority applies it only if all conditions hold:

- top particle probability is at least `0.80`;
- the exact grounded action has at least one prior evidence record from the same program-declared causal state projection;
- no matching evidence record is a terminal failure;
- predicted terminal risk is at most `0.05`;
- the causal-state context has not already been overridden in the reset;
- fewer than one overrides have occurred in the reset.

Otherwise the controller executes the historical action and records the veto reason.

## Paired arms

The same seed/reset order and action budget are used for:

1. `baseline`;
2. `posterior_full`;
3. `no_replay_prior`;
4. `no_posterior_update`;
5. `no_information_gain`;
6. `no_a40_memory`;
7. `no_mdl_prior`.

All mutable arm memory is isolated by seed and arm. It is reset between games/replicates. The baseline never receives causal authority.

## Primary gate

T11.2 passes the active source-train gate only if all integrity checks pass and:

- `games_with_progress >= 1` for the full posterior;
- `safety_regressions == 0` relative to the paired baseline;
- `posterior_ablation_advantage == true` against `no_posterior_update`;
- the replay gate passes with at least two exact branches per bundle and positive total entropy reduction;
- replay memory is checksum-valid and actually loaded before bounded decisions;
- both replay and paired artifact trees remain below `3 GiB`.

Any miss is reported as a negative result. It does not authorize source validation, holdout access, retuning, or publication claims.

## Required tests before physical execution

- a matched observed local witness is not penalized when `level_change == 0`;
- different programs can evaluate different observed progress on the same transition;
- replay emits checksum-bound A40 memory and paired control verifies it;
- `no_replay_prior` begins without replay evidence;
- unsupported interventions are vetoed;
- matching supported interventions are allowed;
- any matching terminal failure vetoes the intervention;
- the one-intervention/reset and 3 GiB limits are signed and enforced;
- legacy exact-route protection and existing causal tests remain green.

## Promotion decision

Only a passing signed gate permits a separately preregistered source-validation iteration. A failing gate closes T11.2 after mechanism-level autopsy on already collected artifacts.
