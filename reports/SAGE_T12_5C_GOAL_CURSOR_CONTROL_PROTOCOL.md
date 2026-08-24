# SAGE.T12.5c — Paired goal-cursor binding control

## Scientific status

T12.5b.5 passed its signed evaluation gate on source-train `bp35`. From the
exact stage-3 state, `ACTION3>ACTION3` produced safe level progress in both
route lineages 8701 and 8705, whereas the cursor-mismatch branch
`ACTION4>ACTION3>ACTION3` produced no progress in either lineage. Terminal
risk was not invariant: the mismatch was terminal on 8701 but non-terminal on
8705.

That result establishes transfer of a goal-viability contrast. It does not yet
establish that the **binding of the next action to the goal cursor**, rather
than a larger action budget or an arbitrary continuation, causes the
advantage. T12.5c is the separately frozen paired control authorized by the
signed T12.5b.5 receipt. It does not amend or rerun its parent.

## Question

> At the same exact stage-3 anchor and with the same two action slots, does the
> confirmed goal-cursor binding preserve safe level progress while a single
> preregistered binding swap removes it on both source-game route lineages?

This is a prospective physical control. The outcome of the two-slot
binding-swap arm has not been used to choose the arm, thresholds or schedule.

## Existing SAGE.T grounding

The experiment reuses the confirmed five-step contextual option:

`ACTION4 → ACTION4 → ACTION4 → ACTION3 → ACTION3`.

It replays the sealed witness route, executes the first three option steps,
checks the exact stage-3 state hash, then reacquires every declared action from
the current SDK legal inventory. No new causal hierarchy, posterior, learned
policy or action-effect model is introduced.

The two arms are:

1. **goal cursor** — `ACTION3 → ACTION3`;
2. **binding swap** — `ACTION4 → ACTION3`.

The binding swap changes only slot 0. It substitutes the known
cursor-mismatch action `ACTION4` for the bound next action `ACTION3`, then
forcibly consumes that cursor slot and leaves slot 1 unchanged. Both arms
therefore have the same maximum horizon of two actions. Both start from the
same anchor, receive the same legal inventory and use the same number of
resets and repetitions.

This control differs deliberately from the T12.5b.5 mismatch branch. The
parent retained the complete continuation after a mismatch and therefore used
three actions. T12.5c instead holds opportunity and horizon constant; it asks
whether the correct cursor binding itself matters.

## Frozen matrix

Each arm is repeated twice on lineages 8701 and 8705:

`2 arms × 2 lineages × 2 repetitions = 8 trials`.

The physical order is fixed and counterbalanced:

| Order | Lineage | Arm | Repetition |
|---:|---:|---|---:|
| 0 | 8701 | goal cursor | 0 |
| 1 | 8701 | binding swap | 0 |
| 2 | 8701 | binding swap | 1 |
| 3 | 8701 | goal cursor | 1 |
| 4 | 8705 | binding swap | 0 |
| 5 | 8705 | goal cursor | 0 |
| 6 | 8705 | goal cursor | 1 |
| 7 | 8705 | binding swap | 1 |

No seed replacement, arm addition, extra repetition, adaptive stopping,
threshold change or same-version rerun is authorized.

## Integrity gate

All of the following must pass:

- the eight scheduled trials occur in the frozen order;
- both repetitions exist for every lineage/arm cell;
- the exact stage-3 anchor is reproduced within each lineage;
- availability, projected effects and outcomes are deterministic within every
  cell;
- every declared action is acquired live, unless physical termination or
  observed progress stops the program first;
- both arms retain the same two-slot maximum horizon;
- the SDK-call and wall-time bounds are respected.

An unavailable action is missing evidence and fails integrity; it is never
converted to a zero-effect vector. A deterministic terminal outcome after an
available action is retained as scientific risk evidence and is not itself an
integrity failure.

## Scientific gate

For each lineage separately:

- both goal-cursor repetitions must produce `level_delta > 0` without a
  terminal failure;
- both binding-swap repetitions must produce `level_delta <= 0`;
- the paired level-delta gain must be at least 1.

All three conditions must hold on both lineages. Terminal behavior of the
control may differ across lineages and is reported explicitly; the causal
claim concerns loss of progress, not transfer of terminal hazard.

Labels use only observed level delta, terminal state and completion. No causal
score, immediate effect magnitude or action name ranking selects a label.

## Exclusive outcomes

- malformed, non-exact, nondeterministic, incomplete schedule, missing live
  action or budget violation → `FAIL_T12_5C_COLLECTION_INTEGRITY_GATE`;
- goal-cursor arm fails safe progress →
  `FAIL_T12_5C_GOAL_CURSOR_PROGRESS_GATE`;
- binding-swap arm progresses →
  `FAIL_T12_5C_BINDING_SWAP_CONTROL_GATE`;
- no positive paired advantage on both lineages →
  `FAIL_T12_5C_PAIRED_ADVANTAGE_GATE`;
- all integrity and scientific checks pass →
  `PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE`.

Every miss is a signed negative result. It closes collection and does not
authorize retuning or rerunning this protocol.

## Bounds and claim boundary

- maximum 1,000 SDK calls total;
- maximum 7,200 wall-clock seconds;
- maximum 3 GiB of artifacts;
- no raw-frame persistence;
- source-train `bp35` only;
- sealed lineages 8701 and 8705 only.

A pass supports only the local claim that correct goal-cursor binding is
causally necessary for the observed two-step progress advantage at this
confirmed source-game context. It is not evidence of generic ARC-AGI
improvement, target-game transfer or autonomous controller performance.

A pass may authorize only preparation of a separately frozen T12.6 protocol.
It does not authorize that experiment, source validation, holdout access,
neural training, environment control or production authority.
