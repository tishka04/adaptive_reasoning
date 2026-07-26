# SAGE.11 validation and publication protocol

Status: roadmap steps 1-4 complete. Step 4 first produced a documented v1
no-go, then a separately pre-registered factorized v2 formal go. The exact
77-feature/factorized interface is now implemented across archived-row
loading, the model/trainer, and live inference. Step 5 has not started because
the stricter source-train-only anti-shortcut audit is pre-registered but not
yet executed. The first source-capacity gate failed
closed at an optimistic maximum of 98,708/100,000 under the 8,000/game base
cap. On 2026-07-26 the user approved the minimum 1,292-row aggregate overflow
on five source-training games with demonstrated remaining unique capacity.
The amended collection now verifies at exactly 100,000 rows (76,908
source-train / 23,092 source-validation), manifest
`d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`.
The frozen curriculum checksum is
`d11948c5cfcb70ce888b435d63d217b95ce2a0006e4423ae7ac70374d81c630c`.
The cheap effect classifier improved macro-F1 by only 0.0288 over the
train-only per-action majority baseline, below the required 0.10. Authority
remains `off`. Factorized pilot v2 later improved 0.2075 over its learned
action-only comparator and passed its frozen gate, but showed only 0.0078
action-shuffle degradation and does not authorize the unmodified model. No
graph-model training, historical evaluation, or holdout evaluation was
started. See `reports/SAGE11_SOURCE_CAPACITY_RESULT.md`,
`reports/SAGE11_EFFECT_PILOT_RESULT.md`,
`reports/SAGE11_EFFECT_PILOT_V2_RESULT.md`, and
`reports/SAGE11_ANTI_SHORTCUT_AUDIT_PROTOCOL.md`.

Final repository validation after pilot v2 on 2026-07-26: Ruff passed on the
full SAGE.11 package and updated pilot tests, `git diff --check` passed, the
focused SAGE.10g/SAGE.11 suite passed 32 tests, and the complete suite passed
1,652 tests. The sole warning was joblib falling back from physical-core
discovery to the available logical-core count on Windows; it does not affect
results.

Pre-execution validation of the shared-interface/anti-shortcut
pre-registration passed Ruff, `git diff --check`, and 34 focused
SAGE.10g/SAGE.11 tests. The same benign Windows joblib physical-core warning
was the only warning.

## Software evidence

The focused implementation suite covers:

- split disjointness and holdout leakage rejection;
- fixed-mixture determinism, per-game caps, deduplication, ACTION6 coverage,
  strong/weak labels, manifests, and live controller archiving;
- multi-source frozen curricula and holdout provenance rejection;
- persistent transfer-step evidence, effect-to-subgoal-graph bridge, and no
  fabricated edge/terminal credit;
- 1.552M-parameter factorized model shape, terminal-head threshold, composite
  loss, shared streaming-schema checkpoint binding, effect-pilot go/no-go,
  anti-shortcut audit gates, and bounded adaptation;
- typed hypotheses with support zero;
- exact off behavior, byte-identical shadow behavior, bounded-gate downgrade,
  danger veto, protected competence, one-probe budget, demotion, and re-arming.

## Empirical sequence

1. Run SAGE.10g source curriculum over source-train games and freeze each
   library before merging.
2. Run SAGE.10h/10i active/ablation pilots and archive every transition with
   the live collector.
3. Build and verify at least 100,000 transitions.
4. Run the cheap effect pilots, implement the passing versioned interface,
   then run the frozen source-train-only anti-shortcut audit. On failure,
   collect only a smaller object-relational pilot corpus.
5. Only after that audit passes, train five-head world models with per-run
   checkpoints and pass every source-only world-model gate.
6. Run shadow on source validation. Entry to bounded requires action identity,
   zero would-be successful-route preemptions, top-k productivity advantage,
   calibration, and inference budget.
7. Run bounded with one neural probe per branch/context, symbolic danger veto,
   protected competence, two-failure demotion, and explicit re-arming.
8. Run active only after bounded passes.

The data-policy amendment required by step 3 is approved and independently
verified. The first parts of step 4 are complete: v1 failed, v2 formally
passed, and its exact interface is implemented. The anti-shortcut audit is
frozen before execution. Steps 5-8 remain blocked until it passes.

## Step 4 v1 result

The single fixed `HistGradientBoostingClassifier` fit 76,908 source-training
rows using 19 train-fitted binary pre-action atoms and six action features.
It evaluated on all 23,092 frozen source-validation rows:

- per-action majority macro-F1: 0.0490;
- classifier macro-F1: 0.0779;
- absolute improvement: +0.0288, gate requires at least +0.10;
- within-game action-shuffle degradation: +0.0059;
- per-game improvement: `re86` -0.0070, `ls20` -0.0543, `sc25` +0.0234.

All validation effect classes and typed atoms were present in training.
Therefore the no-go is not an unseen-vocabulary artifact. The full result is
checksummed as
`c724aeb6d2ab71154a7c72fa381f3f5f4347a5135644ba64ac82a5542e528136`.
The laptop RTX 4050 was detected but not used because the fixed
scikit-learn estimator is CPU-only and the 100,000-by-25 matrix completed in
9.764 seconds on CPU; changing estimators solely to use CUDA would have
changed the pre-registered pilot.

## Step 4 factorized v2 result

The v2 protocol was committed and pushed as `2660f4b` before fitting. It used
separate changed-cells and player-moved heads, a learned 10-feature
action-only comparator, and a 77-feature full representation with
leakage-free trajectory context. The single source-only execution passed all
frozen conditions:

- full composite macro-F1: 0.5506;
- learned action-only composite macro-F1: 0.3431;
- absolute improvement: +0.2075, gate required +0.10;
- both core heads non-negative against action-only;
- per-game composite improvement: `re86` +0.1084, `ls20` +0.0802,
  `sc25` +0.0329.

The formal pass is not a promotion claim. Changed-cells F1 was only 0.1562 and
remained below its per-action majority. Player-moved supplied nearly all the
gain, and current-action shuffling degraded the composite by only 0.0078,
below the later world-model requirement of 0.10. The current atoms can act as
implicit game-regime signatures. The graph model now consumes the exact v2
interface, but must not train until the frozen source-train-only LOGO audit
passes its changed-cells, action-conditioning, fold-robustness, and
signature-ablation gates.

V2 result checksum:
`45f58d1537a1b1a6800636b77df401ab3bf1f94f4ed6dc3bcf2d107864f0328f`.

## Step 4 source-train anti-shortcut audit

The pre-registered `sage11-anti-shortcut-logo-v1` audit reads only the 76,908
rows in the 11 source-training shards and holds out one source-training game
per fold. It compares action-only, state-only, full 77-feature, and
signature-ablated views for separate changed-cells and player-moved heads.
Current-action-dependent columns are shuffled only within exact
availability/object signatures, so the control preserves the coarse state
regime. Fixed signatures are also scored as game classifiers and ablated.

The audit passes only with at least +0.10 changed-cells macro-F1 over the
stronger action/state baseline, at least 0.10 composite degradation under the
conditional shuffle, non-negative changed gain on at least 9/11 games with no
fold below -0.05, and no measured fixed-signature shortcut reliance.
Player-moved cannot compensate for changed-cells. The protocol and code must
be committed and pushed before the one empirical execution. A fail blocks GPU
training and requires a smaller contact/alignment/proximity/object-relative
pilot corpus.

## Non-regression and report-only matrix

- Off must be identical to the current controller.
- Shadow must execute byte-identical actions.
- `ft09`, seed 0, budget 160, 14 resets must retain at least 43 levels,
  maximum level 6, three WINs, and zero protected-route preemptions.
- Historical report only: five games, seeds 0/1, budgets 500/1500/4000,
  14 resets. No hyperparameter or gate may change from these outcomes.

## Final confirmation

Run `NEURO_HOLDOUT_V1` on seeds 0–4, budget 4,000, 14 resets, active versus
off, with paired reset/action digests. Promotion requires all of:

- paired-bootstrap 95% lower bound of score gain above zero;
- no WIN lost;
- at least one new level or WIN where off fails;
- zero unsafe outcome, controller error, and protected-route preemption;
- all 25 game/seed pairs and both digests present.

Any failure leaves the system in shadow. The negative result, partial run log,
dataset/model checksums, exact commit, and reproduction command must be
published.
