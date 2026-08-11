# SAGE.T10.3.12 operator runbook

Run from `C:\Users\coudr\projects\adaptive_reasoning` in PowerShell. The
runtime is deliberately phase-separated; do not skip a phase or edit a frozen
file between phases.

## Exit-code policy

- `0`: the requested phase completed.
- `2`: provenance, checksum, accounting or environment integrity failure. Stop
  immediately; do not write further experimental artifacts.
- `3`: complete scientific gate miss. Run only `report` to seal the terminal
  result, then stop. Do not retry, retune or proceed to active collection.

`active-core` returning 0 means collection completed, not that the hypothesis
passed. The scientific verdict is produced by `adjudicate`.

## Phase order

1. static checks and focused tests;
2. `freeze` exactly once;
3. `status` and `audit-parent`;
4. `preflight`, `materialize-offline`, `compile-candidates`,
   `evaluate-offline`;
5. verify the signed offline gate;
6. `active-core`;
7. verify accounting, then `adjudicate`;
8. `report`.

If the process is interrupted during `active-core`, run `status`. An intent
whose physical outcome is unknown is sealed as an interrupted negative receipt
and is never replayed. Do not delete journals or restart the same frozen matrix.

## Scientific interpretation

Do not call a PASS unless `report` exits 0 with the exact PASS verdict. A code 3
is a valid negative result and must remain frozen. Latency is telemetry only.
Initial labels are not evidence of environment diversity; use the published
count of distinct initial-frame hashes.

