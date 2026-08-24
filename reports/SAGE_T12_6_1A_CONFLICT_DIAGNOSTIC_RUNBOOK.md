# SAGE.T12.6.1a — Runbook du diagnostic de conflits

Cette phase est entièrement hors ligne. Elle n’expose aucune commande de
collecte ou d’évaluation confirmatoire.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Parent = ".\training\sage_t\future_viability_hierarchy_t12_6_1_bp35"
$Root = ".\training\sage_t\future_viability_conflict_diagnostic_t12_6_1a_bp35"
```

## Validation ciblée

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_future_viability_t12_6.py `
  tests\test_sage_t_future_viability_hierarchy_t12_6_1.py `
  tests\test_sage_t_future_viability_conflict_diagnostic_t12_6_1a.py

& $Py -m ruff check `
  theory\sage_t\causal\future_viability_conflict_diagnostic.py `
  theory\sage_t\causal\future_viability_conflict_diagnostic_protocol.py `
  theory\sage_t\causal\future_viability_conflict_diagnostic_experiment.py `
  theory\sage_t\causal\future_viability_conflict_diagnostic_cli.py `
  tests\test_sage_t_future_viability_conflict_diagnostic_t12_6_1a.py
```

## Gel

Le worktree doit être propre et versionné.

```powershell
& $Py -m theory.sage_t.causal.future_viability_conflict_diagnostic_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-evaluation-receipt "$Parent\evaluation\evaluation_receipt.json" `
  --manifest "$Root\manifest.json"
```

## Diagnostic

```powershell
& $Py -m theory.sage_t.causal.future_viability_conflict_diagnostic_cli diagnose `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\diagnostic"
$DiagnosticCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_conflict_diagnostic_cli status `
  --manifest "$Root\manifest.json" `
  --diagnostic-receipt "$Root\diagnostic\diagnostic_receipt.json"
```

Un exit code 3 indique un échec d’intégrité du diagnostic. Un exit code 0
indique seulement que l’analyse post-hoc est complète. Il n’autorise ni le
rerun confirmatoire sur 9201–9203, ni T12.6.2, ni une collecte physique.
