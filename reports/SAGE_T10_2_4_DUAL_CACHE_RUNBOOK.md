# SAGE.T10.2.4 dual-cache runbook

Run from the repository root.

## 1. Verify migration

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage_t.t10_2_4_runtime status --repo-root .
$LASTEXITCODE
```

Expected: exit code `0`, thirteen frozen lanes, two adopted predecessor caches,
lane 14 as the next lane, and `replay_authorized: false`.

## 2. Build the missing factorized cache

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_4_runtime prepare --repo-root .
$LASTEXITCODE
```

This performs no physical action.  The 546-event gauge cache is adopted from
T10.2.3; the missing factorized state prints progress per event and checkpoints
every eight events.  It may run longer than ten minutes.  Interruption is safe:
rerunning resumes from the last authenticated checkpoint.

## 3. Continue collection

After `prepare` returns `0`:

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_4_runtime collect --repo-root .
$LASTEXITCODE
```

Do not use the T10.2.2 or T10.2.3 collection commands.  Do not edit the parent
journal or either cache root.  A checksum error, unknown record, unresolved
intent or changed frozen report is a fail-closed stop signal.

