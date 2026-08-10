# SAGE.T10.2.7 — Event-seal recovery runbook

Run all commands from the repository root in PowerShell. Do not rerun T10.2.6
and do not delete or edit its partial journal.

## 1. Verify the frozen migration and event-seal preflight

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_7_runtime status --repo-root .
$LASTEXITCODE
```

Expected status before collection:

- `READY_T10_2_7_EVENT_SEAL_RECOVERY`;
- `migration_verified: true`;
- `t10_2_6_quarantined_intents: 1`;
- `t10_2_6_sealed_events: 0`;
- `first_event_seal_preflight.passed: true`;
- exit code `0`.

Stop if any value differs. In particular, never launch collection if the
first-event seal preflight is absent or false.

## 2. Launch the bounded replacement collection

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_7_runtime collect --repo-root .
$LASTEXITCODE
```

The runtime tries at most three fresh replacement lanes. Each lane contains four
bounded resets. Donor-cache readiness messages are expected before reset work.
The first progress line must use phase
`t10_2_7_event_seal_recovery_reset` and report both
`spawn_child_registry_installed: true` and
`execution_manifest_hybrid: true`.

Exit code `0` is authorized only for
`T10_2_7_SOURCE_COLLECTION_COMPLETE`. Exit code `3` is a scientific or recovery
gate failure with a durable report. Exit code `2` is a protocol, provenance, or
I/O failure and must be investigated before any new run.

## 3. Inspect status after interruption or completion

The same status command is read-only with respect to all scientific journals:

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_7_runtime status --repo-root .
$LASTEXITCODE
```

Safe resumption occurs only at a reset boundary with no prior intent. If a reset
contains any durable intent but lacks a terminal report, the runtime marks its
unsealed intents unresolved and excludes that entire attempt; it never replays
the environment action.

## 4. Handoff after success

After `T10_2_7_SOURCE_COLLECTION_COMPLETE`, preserve these files together:

- `training/sage_t/t10_2_7_event_seal_recovery/recovery_report.json`;
- `training/sage_t/t10_2_7_event_seal_recovery/accepted_source_events.jsonl`;
- `training/sage_t/t10_2_7_event_seal_recovery/accepted_cross_fit_audit.json`;
- `training/sage_t/t10_2_7_event_seal_recovery/t10_2_7_collection_report.json`;
- the complete T10.2.7 source-collection journal.

Do not open source validation or AR25 automatically. The accepted collection
report is only the prerequisite for the next separately frozen compile/train
and evaluation protocol.
