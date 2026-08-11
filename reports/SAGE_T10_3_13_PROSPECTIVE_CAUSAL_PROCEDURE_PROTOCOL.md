# SAGE T10.3.13 - prospective causal-procedure confirmation

Status: implemented but dormant.  The protocol is not frozen, no authorization
receipt exists, and no protected game may be instantiated.

T10.3.13 is conditional on a positive signed T10.3.12f terminal report.  It
binds the exact causal engine, prior, thresholds, candidate-selection rule,
budgets, and T10.3.12f report by checksum.  Any code or prior change after the
T10.3.12f result requires a new development iteration and keeps this protocol
closed.

The protected panel is `s5i5`, `vc33`, `m0r0`, `sk48`, and `r11l`.  Merely
listing these predeclared identifiers is not an opening: the runtime must not
read a frame, construct an environment, or authorize an action without a
separate explicit authorization receipt supplied after T10.3.12f passes.
`ar25`, source validation, sequence authority, production, promotion, and
automatic retuning remain closed.

Candidate selection is deterministic:

- a source-informed T10.3.12f PASS selects `source_closed_loop` against
  `uniform_closed_loop`;
- a generic T10.3.12f PASS selects `uniform_closed_loop` against
  `source_open_loop`;
- every other verdict prevents freezing and execution.

The final matrix has one independent reset for each game and arm: 10 resets,
48 actions per reset, and 480 actions maximum.  Initial state hashes must match
inside each candidate/control pair.  Arm order alternates by a manifest-bound
hash.  No scientific retry or replacement reset is allowed.  Interrupted
execution may resume only from sealed receipts and never replays a physical
action.

The candidate must complete a level on at least three games, obtain at least
two net game successes over the control, have greater bounded utility on at
least four games and no lower utility on the fifth, and improve prequential
causal log loss on at least four games.  All accounting, safety, provenance,
no-fallback, and no-replay gates are mandatory.

A PASS is bounded prospective evidence on this panel.  It is not universal
ARC generalization and grants no further authority.
