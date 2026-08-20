# SAGE.T12.5b.4 runbook

T12.5b.4 has two distinct physical phases. Codex prepares, tests, commits, and
freezes the implementation but does not launch either phase. The user runs
calibration first and inspects its signed gate before deciding whether the
separate evaluation command is authorized.

Run from the repository root in PowerShell:

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\progress_contrast_t12_5b_3_bp35"
$Root = ".\training\sage_t\local_program_utility_t12_5b_4_bp35"

& $Py -m pytest -q `
  tests\test_sage_t_local_program_utility_t12_5b_4.py `
  tests\test_sage_t_progress_contrast_t12_5b_3.py `
  tests\test_sage_t_progress_discrimination_t12_5b_2.py `
  tests\test_sage_t_progress_shadow_t12_5b.py `
  tests\test_sage_t_causal_progress_t12_5.py

& $Py -m ruff check `
  theory\sage_t\causal\local_program_utility.py `
  theory\sage_t\causal\local_program_utility_protocol.py `
  theory\sage_t\causal\local_program_utility_experiment.py `
  theory\sage_t\causal\local_program_utility_cli.py `
  tests\test_sage_t_local_program_utility_t12_5b_4.py
```

Commit the implementation before scientific freeze. The freeze must use a
clean worktree; `--allow-dirty` is only for non-scientific smoke tests.

```powershell
& $Py -m theory.sage_t.causal.local_program_utility_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\collection\contrast_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.local_program_utility_cli status `
  --manifest "$Root\manifest.json"
```

Before calibration, status must report:

- `firewall.calibration_collection_authorized == true`;
- `firewall.evaluation_collection_authorized == false`;
- `firewall.causal_progress_control_authorized == false`;
- `firewall.t12_5c_control_freeze_authorized == false`;
- `firewall.holdout_opened == false`.

## Manual calibration

```powershell
& $Py -u -m theory.sage_t.causal.local_program_utility_cli calibrate `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\calibration"
$CalibrationCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.local_program_utility_cli status `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json"

exit $CalibrationCode
```

Exit code 3 preserves a negative scientific receipt. Do not rerun, substitute
programs, alter thresholds, or launch evaluation. Inspect the classification,
all checks, terminal/missing program counts, SDK accounting, selected pair,
and final firewall.

Only `PASS_T12_5B_4_CALIBRATION_GATE` may set
`firewall.evaluation_collection_authorized` to true.

## Separately authorized manual evaluation

Run this only after reviewing a passed calibration receipt:

```powershell
& $Py -u -m theory.sage_t.causal.local_program_utility_cli evaluate `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json" `
  --output-dir "$Root\evaluation"
$EvaluationCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.local_program_utility_cli status `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json" `
  --evaluation-receipt "$Root\evaluation\evaluation_receipt.json"

exit $EvaluationCode
```

Only `PASS_T12_5B_4_LOCAL_PROGRAM_UTILITY_GATE` may authorize preparation of a
separate T12.5c freeze. It never authorizes environment control, validation,
holdout access, neural training, or production authority.
