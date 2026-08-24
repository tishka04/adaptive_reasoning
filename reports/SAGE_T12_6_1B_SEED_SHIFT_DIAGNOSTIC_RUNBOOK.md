# SAGE.T12.6.1b — Runbook du diagnostic inter-seed

La CLI est exclusivement hors ligne et ne possède que `freeze`, `diagnose` et
`status`.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Hierarchy = ".\training\sage_t\future_viability_hierarchy_t12_6_1_bp35"
$Conflict = ".\training\sage_t\future_viability_conflict_diagnostic_t12_6_1a_bp35"
$Root = ".\training\sage_t\future_viability_seed_shift_diagnostic_t12_6_1b_bp35"
```

## Validation

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_future_viability_hierarchy_t12_6_1.py `
  tests\test_sage_t_future_viability_conflict_diagnostic_t12_6_1a.py `
  tests\test_sage_t_future_viability_seed_shift_diagnostic_t12_6_1b.py
```

## Gel

```powershell
& $Py -m theory.sage_t.causal.future_viability_seed_shift_diagnostic_cli freeze `
  --hierarchy-manifest "$Hierarchy\manifest.json" `
  --hierarchy-evaluation-receipt "$Hierarchy\evaluation\evaluation_receipt.json" `
  --conflict-manifest "$Conflict\manifest.json" `
  --conflict-diagnostic-receipt "$Conflict\diagnostic\diagnostic_receipt.json" `
  --manifest "$Root\manifest.json"
```

## Diagnostic

```powershell
& $Py -m theory.sage_t.causal.future_viability_seed_shift_diagnostic_cli diagnose `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\diagnostic"
$DiagnosticCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_seed_shift_diagnostic_cli status `
  --manifest "$Root\manifest.json" `
  --diagnostic-receipt "$Root\diagnostic\diagnostic_receipt.json"
```

Un exit code 0 signifie diagnostic complet, pas transfert établi. Un exit code
3 est un échec d’intégrité. Aucune commande n’autorise la collecte, un nouveau
holdout ou T12.6.2.
