# SAGE.T11/T12 causal-program posterior protocol

## Objective

The target vertical replaces component-local beliefs with one posterior over
complete programs containing bindings, two-slice dynamics, action
interventions, goal/progress/failure predicates, and an observation model. One
runtime owns the only causal executor, posterior, mechanism registry, and
decision engine.

This protocol freezes the implementation before any causal active validation.
It does not open bounded or active authority and does not authorize the neural
holdout.

## Immutable boundaries

- Development: `bp35` and the remaining SAGE.11 source-train games.
- Frozen source validation: `re86`, `ls20`, and `sc25`.
- Historical reporting only: `ft09`, `sb26`, `wa30`, `tn36`, and `cn04`.
- Regression only: `ar25`.
- Single neural confirmation only after a passing source-validation receipt:
  `s5i5`, `vc33`, `m0r0`, `sk48`, and `r11l`.
- Evidence collected in one source-validation game may update that game's
  online posterior, but must not initialize another validation game.

The split checksum and implementation hashes are frozen in
`theory/sage_t/causal/sage_t11_causal_protocol_manifest.json`.

## Gates

1. **Contract and execution**: canonical JSON round-trip; compiler rejects
   cycles, missing parents, type errors, unavailable actions, incomplete
   mechanisms, and unresolved neural modules; `do` cuts the replaced
   mechanism and leaves non-descendants invariant.
2. **Posterior**: noisy evidence concentrates mass without binary collapse;
   programs are merged only by canonical structural identity, never merely
   because their observed-history predictions match.
3. **A38T-A40T closure**: comparison and responsibility diagnostics update the
   posterior; the checksummed append-only memory reloads the same mass and can
   alter a later decision. The no-A40T ablation must remove that cross-episode
   effect.
4. **Replay intervention**: all branch predictions exist before execution;
   every branch replays to the same exact state/action-schema hash; divergence
   cancels the branch before its action; at least two legal actions are
   compared.
5. **Control safety**: exact and protected routes are lexicographically above
   information gain. Causal probes have terminal risk at most `0.05`, one use
   per abstract context, and at most five interventions per reset.
6. **Historical ft09 non-regression**: retain max level 6, 43 cumulative
   levels, 3 wins, and zero protected-route preemptions. Efficiency improvement
   requires a paired reduction in actions per level, frontier probes, or
   multiform selections and must weaken under the no-posterior ablation.
7. **Promotion**: active authority remains closed until real progress is
   observed on source train and then on at least two of the three frozen
   source-validation games, without a safety regression. Only a receipt bound
   to this protocol checksum can open the one-shot neural holdout.

## Neural phase

SAGE.T12 shares graph-masked mechanism heads across particles. A head receives
only declared parents, the grounded action, and an explicit local context. The
preferred form is a symbolic operator with neural parameters and an explicit
symbolic fallback. ARC-LeWM supplies an observation encoder, proposal signal,
or calibrated observation likelihood; it is not the executor and its output is
not evidence.

Training combines structured transition loss, exact-prefix branch loss,
cross-context invariance, parent sparsity/MDL, and calibration. Known
mechanisms remain frozen while bindings and local parameters adapt. A module is
promotable as an inter-game prior only after independent support in at least two
source games and two contexts per game, with no unresolved terminal
contradiction.

## Required ablations

- no common posterior;
- no information gain;
- no A40T memory;
- no MDL prior;
- dynamics and goal in separate particles;
- no inter-game mechanisms;
- parameter-matched monolithic world model;
- symbolic-only versus hybrid/neural mechanisms.

Architectural correctness, prediction accuracy, calibration, active progress,
and cross-game transfer must be reported separately. No static or synthetic
result is sufficient for authority promotion.

## Experimental CLI closure

`python -m theory.sage_t.causal.experiment_cli` implements immutable phases for
sealing complete-program registries, sealing outcome-free exact-prefix plans,
freezing paired matrices, executing replay bundles, running baseline/full/
ablation arms, and verifying signed receipts. Bounded authority requires a
passing replay receipt bound to the same experiment manifest. Source validation
requires a passing source-train receipt; each validation game receives a fresh
game-scoped memory path. The CLI deliberately exposes no holdout phase.
