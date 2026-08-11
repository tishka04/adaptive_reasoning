# SAGE T10.3.12e runbook

Run from the repository root with the ARC-AGI virtual environment.

## Frozen phase order

1. targeted lint, imports, and tests;
2. `freeze` exactly once;
3. `audit-parent`;
4. `audit-trajectories`;
5. `preflight`;
6. `compile-programs` before any T10.3.12e action;
7. `status` and verify virgin accounting;
8. `active-diagnostic`;
9. `adjudicate`;
10. `report`.

Before `active-diagnostic`, the signed registry must state that it was compiled
before diagnostic actions, used no parent outcomes for fitting, imported no
grounded paths or action checksums, and has zero historical support.

`active-diagnostic` returns 0 only after the bounded matrix is completely and
durably collected.  `adjudicate` and `report` return either 0 for the positive
diagnostic verdict or 3 for a complete negative result.  Code 2 requires an
immediate stop.  After code 3, run only `report`; do not retry, retune, or
promote.

The runtime emits one-line JSON.  In PowerShell, always capture it with an
array wrapper (`@(...)`) before selecting the last JSON line; indexing a scalar
string with `[-1]` returns its final character rather than its final line.
