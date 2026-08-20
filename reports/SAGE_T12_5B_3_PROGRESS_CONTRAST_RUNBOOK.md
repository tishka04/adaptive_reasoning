# SAGE.T12.5b.3 runbook

T12.5b.3 contains a physical ARC collection. Codex prepares and validates the
implementation but does not run `collect`. The user launches that phase from a
terminal after reviewing the frozen manifest.

Run from the repository root in PowerShell:

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\progress_discrimination_t12_5b_2_bp35"
$Root = ".\training\sage_t\progress_contrast_t12_5b_3_bp35"

& $Py -m pytest -q `
  tests\test_sage_t_progress_discrimination_t12_5b_2.py `
  tests\test_sage_t_progress_contrast_t12_5b_3.py

& $Py -m ruff check `
  theory\sage_t\causal\progress_contrast.py `
  theory\sage_t\causal\progress_contrast_protocol.py `
  theory\sage_t\causal\progress_contrast_experiment.py `
  theory\sage_t\causal\progress_contrast_cli.py `
  tests\test_sage_t_progress_contrast_t12_5b_3.py
```

Commit the implementation before scientific freeze. The worktree must be
clean; do not use `--allow-dirty` for scientific evidence.

```powershell
& $Py -m theory.sage_t.causal.progress_contrast_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\audit\discrimination_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.progress_contrast_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\collection\contrast_receipt.json"
```

Before collection, inspect that status reports:

- `firewall.prospective_contrast_collection_authorized == true`;
- `firewall.environment_collection_authorized == true`;
- `firewall.causal_progress_control_authorized == false`;
- `firewall.t12_5c_control_freeze_authorized == false`;
- `firewall.holdout_opened == false`.

Then launch the complete physical collection manually:

```powershell
& $Py -u -m theory.sage_t.causal.progress_contrast_cli collect `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\collection"
$CollectCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.progress_contrast_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\collection\contrast_receipt.json"

exit $CollectCode
```

Exit code 3 is an expected scientific miss, not necessarily an integrity
error. Inspect `receipt.classification`, every `metrics.checks` entry, SDK and
wall-time accounting, the hard-contrast registry, and the final firewall.

Only `PASS_T12_5B_3_PROSPECTIVE_CONTRAST_GATE` may set
`t12_5c_control_freeze_authorized` to true. Even then, environment control,
source validation, holdout, neural training, and production authority remain
closed.
