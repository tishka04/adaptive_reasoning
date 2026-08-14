# SAGE.T12.5b — Observed-effect shadow ranking

## Question

T12.5 showed offline that an ordered sequence of typed effects is a better
progress explanation than terminal-only, change-count and unordered models.
Its remaining limitation is explicit: the 64 deletion branches identify
action order, but their artifacts do not contain an observed object-centric
delta for every action.

T12.5b asks a narrower question:

> From the same exactly replayed progress stage, can a small empirical
> action-to-effect model rank the known next causal milestone above the other
> legal actions on an independent replay lineage?

The experiment is shadow-only. Rankings are recorded but never choose an action
sent to the environment.

## Fixed paired design

The parent is the passed `PASS_T12_5_CAUSAL_PROGRESS_GATE` receipt. The experiment
uses only source-train game `bp35` and its two already confirmed lineages:

- 8701: effect-model induction;
- 8705: confirmation, never used to fit the model.

There are five causal stages. At each stage, each of `ACTION3`, `ACTION4`,
`ACTION6` and `ACTION7` is executed twice from the same expected exact hash.
This yields 80 branches:

`2 lineages × 5 stages × 4 actions × 2 repetitions`.

The schedule is fixed before collection. It does not depend on a model score.
The SDK budget is 5,000 calls; the exact planned replay load is approximately
4,840 calls. Raw frames are not persisted, and all artifacts together are
limited to 3 GiB.

## State and effect representation

Each branch stores:

- the expected and observed exact prefix hash for replay integrity;
- the typed effects of the already completed causal prefix;
- the candidate action's typed one-step delta;
- availability, level change and terminal status.

The model may use only six aggregate effect channels:

- `predicate_counts.adjacent`;
- `predicate_counts.aligned`;
- `predicate_counts.contact`;
- `predicate_counts.near`;
- `role_counts.clickable`;
- `role_counts.movable`.

Hashes, pixels, entity identifiers, coordinates, game identity and level
identity are never model inputs. Exact hashes exist only as replay receipts.

The empirical predictor is a bounded `(progress stage, action) → typed delta`
table fitted on lineage 8701. No neural training is added. The frozen T12.5
posterior consumes its predicted deltas to compute expected progress potential.

## Transport rule

Raw aggregate deltas may contain context-sensitive nuisance changes. The sealed
T12.4a.4b traces already show, before T12.5b collection, that `near` differs
between lineages at stages 0 and 4 while the causal milestone is stable.

Therefore:

- exact full-vector transport is reported diagnostically;
- the preregistered transport gate compares the milestone-match signature of
  every predicted and observed effect across all five causal milestones.

This prevents an irrelevant counter difference from rejecting a correctly
transported causal effect, without weakening the goal-relative semantic test.

## Baselines

The same fitted effect table produces five shadow rankings:

- causal posterior progress;
- changed/not-changed only;
- absolute effect magnitude only;
- fixed lexicographic action order;
- the known action sequence as an action-only reconstruction reference.

The action-only reference is necessarily strong because T12.5b evaluates an
already discovered option. It is reported but is not the scientific comparator.
The causal ranking must beat the best of change-only, magnitude-only and
lexicographic ranking by at least 0.05 mean reciprocal rank.

## Gate

T12.5b passes only if:

- all 80 stage prefixes replay exactly;
- all 80 branch actions are available;
- typed effects are deterministic across both repetitions within each lineage;
- predicted and observed milestone signatures agree on every confirmation
  branch;
- causal top-1 accuracy and mean reciprocal rank are both 1.0;
- every expected action has a strictly positive causal margin;
- causal MRR exceeds every non-goal baseline by at least 0.05;
- ranking the confirmation lineage from its observed deltas is also perfect;
- every known next action increases observed posterior progress potential;
- the ranking plan checksum is fixed before confirmation collection;
- there are no terminal failures, schedule deviations, SDK overruns or storage
  violations.

A failure is reported without retuning. A pass removes the typed-delta evidence
gap and may authorize only a separately frozen T12.5c paired control experiment.
It does not show policy improvement, target transfer or holdout performance.
Environment control, source validation, holdout, neural training and production
authority remain closed.

