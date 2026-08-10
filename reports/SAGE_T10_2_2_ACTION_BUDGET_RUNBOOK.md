# SAGE.T10.2.2 action-budget runbook

The commands below use the repository virtual environment from the repository
root. Do not point them at the T10.2.1 artifact directory.

## Freeze

Run all tests first, then freeze the T10.2.2 manifest:

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage_t.t10_2_2_protocol freeze-t10-2-2 --repo-root .
```

Freezing writes the signed T10.2.2 manifest and its full/smoke compatibility
manifests under `theory/sage_t/`.

## Real smoke

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage_t.t10_2_2_runtime smoke --repo-root .
```

Inspect `training/sage_t/t10_2_2_action_budget_smoke/` and require the smoke
gate in `t10_2_2_collection_report.json` to pass before the full run.  The gate
requires both confirmation controllers (`capacity_matched_independent` and
`learned`) to complete with sealed evidence.

## Full collection (manual user action)

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage_t.t10_2_2_runtime collect --repo-root .
```

The command is resumable. Re-run the same command after an interruption; never
copy or rename a partially written journal. Do not edit a frozen manifest or an
artifact in place.

## Stop conditions

Stop before the full collection if the smoke reports checksum drift, replay,
unknown topology, accounting failure, missing discovery donors, a hot-loop
history scan, or a cold/warm incremental-bookkeeping ratio above 1.10.
