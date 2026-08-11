# SAGE T10.3.12d — Executor correspondence diagnostic

Status before `freeze`: implemented, not frozen, no T10.3.12d physical action.

## Purpose and evidence boundary

T10.3.12c was a clean `CROSS_GAME_TRANSFER_MISS`: the frozen source factor was
applicable to five games but won none. Its three path branches ended with a
collapsed eight-action suffix containing at most two distinct grounded
arguments. On `lf52`, the role and transition ablations escaped that collapse
and won.

T10.3.12d tests the post-hoc diagnosis that the path miss came from recomputing
a path on every action and taking its first successor, rather than advancing a
stable option-local plan. This is a diagnostic remediation on already observed
games. Even a PASS is not independent evidence of cross-game or factor
generalization.

## Panel and firewalls

All nine T10.3.12c target games remain in the matrix to avoid post-hoc game
selection. No new game is opened. Every non-path context receives the same
zero-action abstention in every arm. Only path execution is scored.

Sequence games, source validation, `ar25`, holdouts, production authority, and
legacy fallback remain closed. Parent outcomes are used only to motivate and
audit this diagnostic. Parent actions, supports, and grounded paths are not
used to fit or compile an executor.

## Arms

1. `stable_source_cursor`: build the source-oriented path once from the fresh
   reset, retain it only in reset-local memory, advance an option cursor, and
   reacquire each planned action exactly from the current legal set.
2. `stateless_source_replan`: the T10.3.12c control that recomputes the current
   path and takes its first successor on every decision.
3. `stable_reverse_orientation`: the same stable cursor with only the initial
   path orientation reversed.
4. `stable_cursor_hold`: the source-oriented initial plan, but the cursor is
   ablated by repeatedly selecting its first waypoint.

The transferable registry contains only initiation, orientation, continuation,
reacquisition, termination, and horizon fields. No coordinate, action argument,
waypoint identity, parent path, or target support is serialized.

## Matrix and durability

The matrix contains nine games × four arms = 36 fresh resets. Each reset has at
most 16 actions and 180 seconds. The global maximum is 576 physical actions and
7,200 seconds. Work labels do not seed the environment; initial frame hashes
are published.

Intent is written before every physical action and each event is sealed
immediately. No physical action is replayed. Interruption after an intent makes
the run incomplete and integrity-invalid. Each executed action must come from
`sage_t_executor_correspondence`.

## Gates and verdicts

Collection requires 36 complete receipts, clean accounting, no unresolved or
inflight intent, no illegal action, no legacy fallback, and no replay.

The stable source executor must:

- recognize at least three path-applicable games;
- build exactly one plan and perform zero replans per applicable reset;
- reacquire exactly one legal planned waypoint per executed action;
- persist no path or grounded argument;
- win at least one diagnostic game;
- exceed the stateless arm by at least one success;
- exceed cursor-hold by at least one success;
- incur no error or game over.

The reverse-orientation outcome is reported separately. Principal negative
verdicts are `PLAN_REACQUISITION_MISS`, `REVERSE_ORIENTATION_ONLY`,
`STABLE_EXECUTOR_NO_PROGRESS`, `STATELESS_REPLAN_NOT_DISCRIMINATED`,
`CURSOR_NOT_CAUSAL`, and `EXECUTOR_SAFETY_OR_ACCOUNTING_MISS`.

The positive diagnostic verdict is
`PASS_T10_3_12D_EXECUTOR_CORRESPONDENCE_RECOVERED`. It authorizes only a new,
independently frozen validation protocol; it does not promote a program or open
any validation split automatically.

Code 0 means the phase completed. Code 2 means integrity/provenance invalidity.
Code 3 means a complete negative scientific gate. No post-freeze repair or
retuning is allowed.

