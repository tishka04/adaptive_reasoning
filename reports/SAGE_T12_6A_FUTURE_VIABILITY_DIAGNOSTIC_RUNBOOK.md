# SAGE.T12.6a — Runbook du diagnostic post-hoc

T12.6a est entièrement hors ligne. Sa CLI ne possède ni commande physique ni
commande d’évaluation.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Parent = ".\training\sage_t\future_viability_t12_6_bp35"
$Root = ".\training\sage_t\future_viability_diagnostic_t12_6a_bp35"
```

## Validation ciblée

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_future_viability_t12_6.py `
  tests\test_sage_t_future_viability_diagnostic_t12_6a.py

& $Py -m ruff check `
  theory\sage_t\causal\future_viability_diagnostic.py `
  theory\sage_t\causal\future_viability_diagnostic_protocol.py `
  theory\sage_t\causal\future_viability_diagnostic_experiment.py `
  theory\sage_t\causal\future_viability_diagnostic_cli.py `
  tests\test_sage_t_future_viability_diagnostic_t12_6a.py
```

## Gel

Le worktree doit être propre et versionné.

```powershell
& $Py -m theory.sage_t.causal.future_viability_diagnostic_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-compile-receipt "$Parent\compile\compile_receipt.json" `
  --manifest "$Root\manifest.json"
```

Le statut avant diagnostic doit autoriser uniquement `diagnostic`.

## Diagnostic hors ligne

```powershell
& $Py -m theory.sage_t.causal.future_viability_diagnostic_cli diagnose `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\diagnostic"
$DiagnosticCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_diagnostic_cli status `
  --manifest "$Root\manifest.json" `
  --diagnostic-receipt "$Root\diagnostic\diagnostic_receipt.json"
```

Même `PASS_T12_6A_DIAGNOSTIC_COMPLETE` ne rouvre pas le corpus d’évaluation et
n’autorise aucune suite automatiquement.
