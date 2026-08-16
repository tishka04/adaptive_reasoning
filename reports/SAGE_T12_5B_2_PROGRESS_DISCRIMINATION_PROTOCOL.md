# SAGE.T12.5b.2 — Affordance-grounded progress discrimination

## Scientific status

T12.5b-r1 is a sealed negative result. Its exact prefixes, deterministic
effects, milestone transport and causal top-1 ranking passed, but two gates
failed: one action did not transport as an executable affordance and effect
magnitude ranked the known option equally well.

T12.5b.2 does not amend or rerun that result. It is a post-hoc, offline
diagnostic over the already observed 60 branches. Because those observations
were inspected before this protocol was written, the audit cannot establish a
new confirmatory performance claim. Its purpose is to decide whether the
existing corpus can discriminate causal progress from raw change magnitude.

## Question

> After grounding the executable action set independently in each exact
> context, do the sealed observations contain at least one action whose change
> magnitude is larger than the next causal milestone but whose causal progress
> is lower?

Such a pair is a hard contrast. Without it, perfect causal ranking and perfect
magnitude ranking are observationally confounded.

## Bound parent

Freeze is allowed only for the signed parent status
`FAIL_T12_5B_PROGRESS_SHADOW_GATE`, with exactly these failed checks:

- `all_candidate_actions_available`;
- `causal_ranking_beats_non_goal_baselines`.

Every other T12.5b-r1 check must be true. A different failure class fails
closed. The parent receipt, manifest, trials, effect model, rankings, report,
progress posterior and program registry are all checksum-bound.

## Local affordance contract

For each lineage, stage and action name, the audit derives one local affordance
from the two exact repetitions:

- executable only if both repetitions are available;
- availability must be deterministic;
- an unavailable action has no effect vector, magnitude or progress score;
- executable effects must agree across repetitions;
- every exact context must expose at least two executable candidates;
- the known progress action must remain executable.

Cross-lineage bindings use only `(stage, milestone_signature)`. Action names,
hashes, coordinates and entity identifiers are excluded from the binding key.
Names are persisted only as provenance for the locally grounded action.

## Hard contrast

For a locally executable action `p` matching the preregistered next milestone
and distractor `d` at the same exact prefix, a registered hard contrast
requires:

1. `p` matches the next milestone and `d` does not;
2. `magnitude(d) >= magnitude(p) + 1`;
3. both effects are observed and deterministic.

The milestone label is therefore independent of the causal potential being
evaluated. Only after the contrast is registered do we test whether
`progress_gain(p) > progress_gain(d)`.

The unit of comparison is the exact context, never a synthetic composition of
effects collected at different states. No new pixels, simulated effects or
cross-stage permutations count as evidence.

## Gate and negative route

The discrimination gate requires:

- all parent and replay integrity checks above;
- full semantic binding coverage across the five milestones;
- at least one hard contrast in each of lineages 8701 and 8705;
- causal accuracy 1.0 on hard contrasts;
- causal accuracy at least 0.5 above magnitude accuracy;
- zero environment calls and artifacts below 3 GiB.

If integrity passes but no hard contrast exists, the signed classification is
`INSUFFICIENT_DISCRIMINATIVE_CONTRASTS`. This is a negative diagnostic, not a
model failure. It may authorize only a separately frozen T12.5b.3 collection
design targeting observed same-prefix distractors. It does not authorize that
collection directly.

Environment collection, T12.5c control, source validation, holdout, neural
training and production authority remain closed in every T12.5b.2 outcome.
