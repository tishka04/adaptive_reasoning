# SAGE.11 validation and publication protocol

Status: implementation tests pass; long-running empirical gates have not been
executed in this change. Authority therefore defaults to `off`, and no trained
model is promoted.

Final repository validation on 2026-07-26: Ruff passed on all changed Python
files, `git diff --check` passed, and the complete suite passed with 1,642
tests. The sole warning was joblib falling back from physical-core discovery
to the available logical-core count on Windows; it does not affect results.

## Software evidence

The focused implementation suite covers:

- split disjointness and holdout leakage rejection;
- fixed-mixture determinism, per-game caps, deduplication, ACTION6 coverage,
  strong/weak labels, manifests, and live controller archiving;
- multi-source frozen curricula and holdout provenance rejection;
- persistent transfer-step evidence, effect-to-subgoal-graph bridge, and no
  fabricated edge/terminal credit;
- 1.54M-parameter model shape, terminal-head threshold, composite loss,
  effect-pilot go/no-go, amended gates, and bounded adaptation;
- typed hypotheses with support zero;
- exact off behavior, byte-identical shadow behavior, bounded-gate downgrade,
  danger veto, protected competence, one-probe budget, demotion, and re-arming.

## Empirical sequence

1. Run SAGE.10g source curriculum over source-train games and freeze each
   library before merging.
2. Run SAGE.10h/10i active/ablation pilots and archive every transition with
   the live collector.
3. Build and verify at least 100,000 transitions.
4. Run the cheap effect pilot. Stop and publish the negative result if it is a
   no-go.
5. Train five-head world models with per-run checkpoints and pass every
   source-only world-model gate.
6. Run shadow on source validation. Entry to bounded requires action identity,
   zero would-be successful-route preemptions, top-k productivity advantage,
   calibration, and inference budget.
7. Run bounded with one neural probe per branch/context, symbolic danger veto,
   protected competence, two-failure demotion, and explicit re-arming.
8. Run active only after bounded passes.

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
