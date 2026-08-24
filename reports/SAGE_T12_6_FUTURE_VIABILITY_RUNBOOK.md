# SAGE.T12.6 — Future-viability audit runbook

T12.6 is entirely offline. It has no physical run command and makes zero SDK
calls. Run from the repository root in PowerShell.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Root = ".\training\sage_t\future_viability_t12_6_bp35"
$Authority = ".\training\sage_t\goal_cursor_control_t12_5c_bp35"
$Training = ".\training\sage_t\target_regrounding_t12_4a_4d_bp35"
$Evaluation = ".\training\sage_t\hazard_diversity_t12_4a_4d_1_bp35"
```

## Validation

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_goal_cursor_control_t12_5c.py `
  tests\test_sage_t_future_viability_t12_6.py

& $Py -m ruff check `
  theory\sage_t\causal\future_viability.py `
  theory\sage_t\causal\future_viability_protocol.py `
  theory\sage_t\causal\future_viability_experiment.py `
  theory\sage_t\causal\future_viability_cli.py `
  tests\test_sage_t_future_viability_t12_6.py
```

## Freeze

The worktree must be clean and committed.

```powershell
& $Py -m theory.sage_t.causal.future_viability_cli freeze `
  --authority-manifest "$Authority\manifest.json" `
  --authority-receipt "$Authority\control\control_receipt.json" `
  --training-manifest "$Training\manifest.json" `
  --training-receipt "$Training\paired\target_regrounding_receipt.json" `
  --evaluation-manifest "$Evaluation\manifest.json" `
  --evaluation-receipt "$Evaluation\paired\hazard_diversity_receipt.json" `
  --manifest "$Root\manifest.json"
```

Pre-compile status must authorize only `compile` and keep environment
collection, evaluation, T12.6b, validation, holdout and authority false.

## Offline compile

```powershell
& $Py -m theory.sage_t.causal.future_viability_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\compile"
$CompileCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json"
```

Exit code `3` is a signed negative result and forbids evaluation. Only
`PASS_T12_6_COMPILE_GATE` authorizes the next offline command.

## Sealed offline evaluation

```powershell
& $Py -m theory.sage_t.causal.future_viability_cli evaluate `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --output-dir "$Root\evaluation"
$EvaluationCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --evaluation-receipt "$Root\evaluation\evaluation_receipt.json"
```

Only `PASS_T12_6_FUTURE_VIABILITY_GATE` may set
`t12_6b_physical_freeze_authorized=true`. That field authorizes preparation of
a new protocol only; it never authorizes physical collection by itself.
