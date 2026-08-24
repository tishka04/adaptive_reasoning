# SAGE.T12.5b.5 — Goal-continuation viability

## Scientific status

T12.5b.4 is preserved as the signed negative result
`FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM`. Its 72-trial calibration was
integrity-clean, exact, deterministic and within budget, but every
transportable length-2/3 program executed **after** the extra `ACTION4` detour
ended in `GAME_OVER` and no safe progress program existed.

This result changes the diagnosis. The detour was neutral only under the
instantaneous milestone signature: it changed no sealed progress milestone,
but it destroyed the viability of the confirmed goal continuation. T12.5b.5
is therefore a separate iteration, explicitly scoped after the parent miss. It
does not amend, rerun or reclassify T12.5b.4.

## Question

> From the exact stage-3 state before the fatal detour, does a first action
> that advances the confirmed option cursor preserve safe level progress when
> the remaining goal continuation is re-grounded from the live legal-action
> inventory, while a milestone-neutral cursor-mismatch action does not?

The primary object is now **future goal viability**, not immediate effect
magnitude and not the frozen causal score.

## Existing SAGE.T grounding

The experiment reuses the confirmed contextual minimal option and the existing
exact-prefix collector. The unique five-step option is:

`ACTION4 → ACTION4 → ACTION4 → ACTION3 → ACTION3`.

It was reproduced six times and is context-bound to lineages 8701 and 8705.
After the first three steps, the frozen goal continuation is therefore:

`ACTION3 → ACTION3`.

No new posterior, policy network or parallel causal hierarchy is introduced.
Every declared continuation step is reacquired from the current SDK legal
inventory immediately before execution. Missing actions remain missing; no
unavailable action receives a zero-effect vector.

The branch labels use only observed `level_delta`, terminal state and
completion. Neither immediate magnitude nor a causal score selects or labels a
branch.

## Frozen calibration

Calibration uses source-train lineage 8701 and the exact stage-3 state. Three
first-action interventions are fixed before collection:

1. `ACTION3 → ACTION3`: `ACTION3` matches the next option step, advances the
   goal cursor, and leaves one `ACTION3` to re-ground;
2. `ACTION4 → ACTION3 → ACTION3`: `ACTION4` does not advance the goal cursor,
   so the full two-step continuation remains;
3. `ACTION6 → ACTION3 → ACTION3`: the same cursor-mismatch rule, retained as a
   calibration-only local intervention.

Each branch is repeated twice from an independent reset:

`3 branches × 2 repetitions = 6 calibration trials`.

Only the `ACTION3` and `ACTION4` first-action branches are transport-eligible.
`ACTION6` is not silently deleted if unavailable or unstable.

Candidate terminal outcomes are deterministic scientific risk evidence, not
an integrity failure. A missing or non-exact transport first action is an
integrity failure.

## Calibration labels and selection

A branch is **safe progress** when both repetitions:

- replay the exact stage-3 context;
- acquire the declared first action from the live legal inventory;
- produce `level_delta > 0`;
- have no terminal failure;
- agree on availability, effects and outcome.

A branch is **rejected for viability** when both repetitions acquire the first
action deterministically and produce `level_delta <= 0`. A terminal failure is
allowed and retained as risk evidence.

Calibration selects:

- the shortest transportable cursor-advance safe-progress branch;
- the shortest transportable cursor-mismatch rejected branch.

The tie-break is lexicographic action order. The rule is fixed independently
of every learned score.

Calibration passes only if collection integrity passes, a cursor-advance safe
progress branch exists, and a paired transportable viability contrast exists.

## Two-phase firewall

Calibration is the only phase authorized by the frozen manifest. A passed
signed calibration receipt seals the exact branch pair in
`evaluation_registry.json`; only that receipt authorizes evaluation.

Evaluation uses lineage 8705 and executes exactly the registered pair twice:

`2 branches × 2 repetitions = 4 evaluation trials`.

The gate requires exact deterministic replay, transfer of safe progress for
the cursor-advance branch, deterministic rejection of the cursor-mismatch
control, and transfer of the paired viability contrast.

No calibration pass means no evaluation. No evaluation pass means no T12.5c
freeze preparation.

## Negative-result routes

Calibration outcomes are classified without retuning:

- malformed, non-exact, nondeterministic, incomplete schedule, transport
  first-action miss or budget violation →
  `FAIL_T12_5B_5_CALIBRATION_INTEGRITY_GATE`;
- no safe progress from the confirmed continuation →
  `FAIL_T12_5B_5_GOAL_CONTINUATION_GATE`;
- no transportable cursor-advance/cursor-mismatch viability contrast →
  `FAIL_T12_5B_5_NO_VIABILITY_CONTRAST`;
- full calibration pass → `PASS_T12_5B_5_CALIBRATION_GATE`.

Evaluation outcomes are classified separately:

- replay, registry, determinism or budget failure →
  `FAIL_T12_5B_5_EVALUATION_INTEGRITY_GATE`;
- progress continuation does not transfer →
  `FAIL_T12_5B_5_GOAL_CONTINUATION_TRANSFER_GATE`;
- detour control is no longer rejected →
  `FAIL_T12_5B_5_DETOUR_CONTROL_GATE`;
- the paired contrast does not transfer →
  `FAIL_T12_5B_5_VIABILITY_TRANSFER_GATE`;
- full pass → `PASS_T12_5B_5_GOAL_VIABILITY_GATE`.

No same-version rerun, seed substitution, threshold change, action removal,
continuation edit or post-hoc branch addition is authorized after a miss.

## Bounds and claim boundary

- maximum 1,000 SDK calls for calibration;
- maximum 750 SDK calls for evaluation;
- maximum 1,750 total SDK calls;
- maximum 7,200 wall-clock seconds per physical phase;
- maximum 3 GiB of artifacts per phase;
- no raw frame persistence;
- source-train game `bp35` only;
- calibration lineage 8701 and evaluation lineage 8705 only.

A final pass supports only the claim that the contextual goal continuation
distinguishes a viable first action from the preregistered fatal detour across
the two source-game route lineages. It is not generic ARC-AGI improvement,
target-game generalization or controller authority.

A final pass authorizes only preparation of a separately frozen T12.5c paired
control protocol. It does not authorize that run. Environment control, source
validation, holdout access, neural training, target-game transfer and
production authority remain closed in every outcome.
