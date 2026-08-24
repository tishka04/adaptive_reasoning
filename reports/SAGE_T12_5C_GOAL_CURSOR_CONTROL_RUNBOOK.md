# SAGE.T12.5c — Paired goal-cursor control runbook

Codex prepares, validates, commits and freezes this iteration. Physical
collection remains a manual user action. Run commands from the repository root
in PowerShell.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Root = ".\training\sage_t\goal_cursor_control_t12_5c_bp35"
$Parent = ".\training\sage_t\goal_viability_t12_5b_5_bp35"
```

## 1. Focused validation

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_goal_viability_t12_5b_5.py `
  tests\test_sage_t_goal_cursor_control_t12_5c.py

& $Py -m ruff check `
  theory\sage_t\causal\goal_cursor_control.py `
  theory\sage_t\causal\goal_cursor_control_protocol.py `
  theory\sage_t\causal\goal_cursor_control_experiment.py `
  theory\sage_t\causal\goal_cursor_control_cli.py `
  tests\test_sage_t_goal_cursor_control_t12_5c.py
```

## 2. Freeze

The worktree must be clean and the code commit must be the intended scientific
commit.

```powershell
& $Py -m theory.sage_t.causal.goal_cursor_control_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\evaluation\evaluation_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.goal_cursor_control_cli status `
  --manifest "$Root\manifest.json"
```

Expected pre-run status:

- `paired_control_collection_authorized=true`;
- `environment_collection_authorized=true` only for this fixed run;
- `t12_6_freeze_authorized=false`;
- source validation, holdout, controller, neural and production fields are
  false.

## 3. Manual paired collection

This single command performs exactly eight resets in the frozen
counterbalanced order.

```powershell
& $Py -u -m theory.sage_t.causal.goal_cursor_control_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\control" `
  --environments-dir ".\environment_files"
$ControlCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.goal_cursor_control_cli status `
  --manifest "$Root\manifest.json" `
  --control-receipt "$Root\control\control_receipt.json"

exit $ControlCode
```

Exit code `0` means `PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE`. Exit code `3`
means a signed negative scientific result. In either case, do not rerun,
replace a seed, alter either arm, add repetitions or change a threshold.

Required artifacts:

- `control/control_receipt.json`;
- `control/control_report.json`;
- `control/control_trials.json`;
- `control/control_arm_registry.json`.

Only the exact PASS may set `t12_6_freeze_authorized=true`. That field permits
preparation of a new frozen protocol only. It never grants source validation,
holdout, neural, autonomous controller or production authority.
