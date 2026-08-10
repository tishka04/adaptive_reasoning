# SAGE.T10.2.8 — Offline QA runbook

Run from the repository root in PowerShell. T10.2.8 is offline and must not
create scorecards or load a game environment.

## 1. Check the frozen handoff

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_8_runtime status --repo-root .
$LASTEXITCODE
```

Before compilation, expect `READY_T10_2_8_OFFLINE_QA`, 1,370 accepted events,
18 lanes, 72 resets, zero authorized physical actions, and exit code `0`.

## 2. Run lineage validation and QA

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_8_runtime compile --repo-root .
$LASTEXITCODE
```

Exit code `0` is reserved for a passing lineage and QA gate. Exit code `3` is an
expected, durably reported scientific gate failure. Exit code `2` indicates a
protocol, provenance, or I/O error rather than a scientific negative result.

## 3. Read the terminal status

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_8_runtime status --repo-root .
$LASTEXITCODE
```

After compilation, status should be `COMPLETE_T10_2_8_OFFLINE_QA` and report
the terminal gate decision. The status command exits `0` even when the frozen
scientific decision is negative, because it is reporting a valid completed
protocol.

## 4. Decision rule

If the terminal decision is `FAIL_T10_2_8_QA_STOP_BEFORE_FIT`, do not run an
older T10.2 compile/train command and do not start source validation. Preserve:

- `training/sage_t/t10_2_8_offline_qa/lineage_audit.json`;
- `training/sage_t/t10_2_8_offline_qa/qa_report.json`;
- `training/sage_t/t10_2_8_offline_qa/t10_2_8_report.json`.

The next scientific iteration must preregister a repair to the representation,
predicate construction, or multiframe projection logic before collecting or
fitting again.
