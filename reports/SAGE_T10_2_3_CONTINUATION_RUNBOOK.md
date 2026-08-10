# SAGE.T10.2.3 continuation runbook

Run from the repository root with the project virtual environment.

## 1. Verify the frozen migration

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage_t.t10_2_3_runtime status --repo-root .
$LASTEXITCODE
```

Expected exit code: `0`, with `READY_T10_2_3_CONTINUATION`, nine frozen lanes,
lane 10 as the next lane, and `replay_authorized: false`.

## 2. Optional explicit precomputation

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_3_runtime prepare --repo-root .
$LASTEXITCODE
```

This performs no physical action.  It prints one progress record per donor
event and durably checkpoints every eight events.  It is safe to interrupt and
rerun: the authenticated partial cache resumes from its last checkpoint.

## 3. Continue the physical collection

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_3_runtime collect --repo-root .
$LASTEXITCODE
```

The command verifies migration first.  If the next exact cache is absent, it
builds or resumes it with visible progress before starting that reset.  The
existing nine lanes are skipped by their immutable reports; no completed
physical action is replayed.

Do not use the old `t10_2_2_runtime collect` command after migration.  Do not
delete or edit the parent journal, checkpoint, cursor, cache metadata or cache
state.  A non-zero exit, checksum error, unresolved intent, unknown record, or
changed lane report is a fail-closed stop signal.

The complete matrix is not a passing scientific result by itself.  Inspect the
terminal T10.2.2 collection report, checkpoint binding, evidence funnel,
cross-fit audit and registered gates before any claim is promoted.

