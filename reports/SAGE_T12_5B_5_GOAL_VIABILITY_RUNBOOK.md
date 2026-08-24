# SAGE.T12.5b.5 — Goal-continuation viability runbook

Run every command from the repository root. Physical collection remains a
manual user action.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Root = ".\training\sage_t\goal_viability_t12_5b_5_bp35"
$Parent = ".\training\sage_t\local_program_utility_t12_5b_4_bp35"
```

## 1. Focused validation

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_local_program_utility_t12_5b_4.py `
  tests\test_sage_t_goal_viability_t12_5b_5.py

& $Py -m ruff check `
  theory\sage_t\causal\goal_viability.py `
  theory\sage_t\causal\goal_viability_protocol.py `
  theory\sage_t\causal\goal_viability_experiment.py `
  theory\sage_t\causal\goal_viability_cli.py `
  tests\test_sage_t_goal_viability_t12_5b_5.py
```

## 2. Freeze

The worktree must be clean and the code commit must be the intended frozen
commit.

```powershell
& $Py -m theory.sage_t.causal.goal_viability_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\calibration\calibration_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.goal_viability_cli status `
  --manifest "$Root\manifest.json"
```

Expected pre-run status:

- `calibration_collection_authorized=true`;
- `evaluation_collection_authorized=false`;
- `t12_5c_control_freeze_authorized=false`;
- every validation, holdout, neural and production field is false.

## 3. Calibration collection — lineage 8701

This is the first physical phase and the only phase initially authorized.

```powershell
& $Py -u -m theory.sage_t.causal.goal_viability_cli calibrate `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\calibration" `
  --environments-dir ".\environment_files"
```

Exit code `0` means `PASS_T12_5B_5_CALIBRATION_GATE`. Exit code `3` is a
signed negative scientific result. On exit code `3`, stop: do not rerun, edit
the branch set, remove `ACTION6`, change a threshold or launch evaluation.

Inspect:

```powershell
& $Py -m theory.sage_t.causal.goal_viability_cli status `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json"
```

Required calibration artifacts:

- `calibration/calibration_receipt.json`;
- `calibration/calibration_report.json`;
- `calibration/calibration_trials.json`;
- `calibration/viability_branch_registry.json`;
- `calibration/evaluation_registry.json` only after a pass.

## 4. Evaluation — lineage 8705

Run this phase only when signed status explicitly reports
`evaluation_collection_authorized=true`.

```powershell
& $Py -u -m theory.sage_t.causal.goal_viability_cli evaluate `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json" `
  --output-dir "$Root\evaluation" `
  --environments-dir ".\environment_files"

& $Py -m theory.sage_t.causal.goal_viability_cli status `
  --manifest "$Root\manifest.json" `
  --calibration-receipt "$Root\calibration\calibration_receipt.json" `
  --evaluation-receipt "$Root\evaluation\evaluation_receipt.json"
```

Only `PASS_T12_5B_5_GOAL_VIABILITY_GATE` may set
`t12_5c_control_freeze_authorized=true`. That field authorizes preparation of
a new protocol only, not environment control or another physical run.
