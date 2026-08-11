# SAGE T10.3.12c — Operator runbook

Run from the repository root in PowerShell with the repository virtual
environment. Do not run `freeze` until all static and focused tests are green.

The scientific order is:

1. static checks and focused regressions;
2. `freeze` exactly once;
3. `audit-parent`;
4. `preflight`;
5. `audit-targets`;
6. `compile-transfer` before any target action;
7. inspect `status` and confirm zero authorized actions;
8. `active-transfer`;
9. `adjudicate`;
10. `report`.

Before `active-transfer`, status must show the first four phase artifacts,
`authorized_actions = 0`, no live lock, no inflight intent, and no unresolved
intent. The compiled registry must say `compiled_before_target_outcomes = true`,
`historical_support_imported = 0`, and `grounded_arguments_imported = false`.

If any phase returns code 2, stop. Do not run another writing phase. If
`adjudicate` returns code 3, run only `report` to seal the registered negative;
do not rerun a target, change a gate, or retune T10.3.12c. If execution is
interrupted after an intent was written, the action is never replayed and the
run is an integrity-invalid incomplete run.

The final terminal report must retain closed firewalls for sequence games,
source validation, holdouts, `ar25`, production authority, program promotion,
and sequence composition.

