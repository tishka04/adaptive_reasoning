# SAGE.T12.6.1c — Runbook source-train

La CLI ne propose que `freeze`, `compile` et `status`. Elle n’a pas de commande
`evaluate`, `collect` ou `run`.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Hierarchy = ".\training\sage_t\future_viability_hierarchy_t12_6_1_bp35"
$SeedShift = ".\training\sage_t\future_viability_seed_shift_diagnostic_t12_6_1b_bp35"
$Root = ".\training\sage_t\future_viability_reliability_t12_6_1c_bp35"
```

## Validation

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_future_viability_hierarchy_t12_6_1.py `
  tests\test_sage_t_future_viability_seed_shift_diagnostic_t12_6_1b.py `
  tests\test_sage_t_future_viability_reliability_hierarchy_t12_6_1c.py
```

## Gel

Le gel exige un worktree propre et lie le reçu compile T12.6.1, le reçu
diagnostic T12.6.1b, le code et les douze archives d’entraînement. Il n’importe
pas le registre des archives d’évaluation.

```powershell
& $Py -m theory.sage_t.causal.future_viability_reliability_hierarchy_cli freeze `
  --hierarchy-manifest "$Hierarchy\manifest.json" `
  --hierarchy-compile-receipt "$Hierarchy\compile\compile_receipt.json" `
  --seed-shift-diagnostic-receipt "$SeedShift\diagnostic\diagnostic_receipt.json" `
  --manifest "$Root\manifest.json"
```

## Compilation source-train

```powershell
& $Py -m theory.sage_t.causal.future_viability_reliability_hierarchy_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\compile"
$CompileCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_reliability_hierarchy_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json"
```

Un exit code 0 signifie uniquement que la règle source-train a franchi les
gates et qu’un protocole sur de nouvelles archives peut être gelé séparément.
Un exit code 3 est un gate miss scientifique ou d’intégrité. Aucune commande
ci-dessus ne lance de collecte physique.
