# SAGE.T12.6.1 — Runbook de viabilité hiérarchique

Cette expérience est entièrement hors ligne et ne possède aucune commande de
collecte physique.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Parent = ".\training\sage_t\future_viability_t12_6_bp35"
$Diagnostic = ".\training\sage_t\future_viability_diagnostic_t12_6a_bp35"
$Root = ".\training\sage_t\future_viability_hierarchy_t12_6_1_bp35"
```

## Validation ciblée

```powershell
& $Py -m pytest -q `
  tests\test_sage_t_future_viability_t12_6.py `
  tests\test_sage_t_future_viability_diagnostic_t12_6a.py `
  tests\test_sage_t_future_viability_hierarchy_t12_6_1.py

& $Py -m ruff check `
  theory\sage_t\causal\future_viability_hierarchy.py `
  theory\sage_t\causal\future_viability_hierarchy_protocol.py `
  theory\sage_t\causal\future_viability_hierarchy_experiment.py `
  theory\sage_t\causal\future_viability_hierarchy_cli.py `
  tests\test_sage_t_future_viability_hierarchy_t12_6_1.py
```

## Gel

Le worktree doit être propre et versionné.

```powershell
& $Py -m theory.sage_t.causal.future_viability_hierarchy_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-compile-receipt "$Parent\compile\compile_receipt.json" `
  --diagnostic-manifest "$Diagnostic\manifest.json" `
  --diagnostic-receipt "$Diagnostic\diagnostic\diagnostic_receipt.json" `
  --manifest "$Root\manifest.json"
```

## Compilation hors ligne

```powershell
& $Py -m theory.sage_t.causal.future_viability_hierarchy_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\compile"
$CompileCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_hierarchy_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json"
```

Un exit code 3 est un résultat scientifique négatif signé et interdit
l’évaluation.

## Évaluation scellée

Cette commande est valide uniquement après `PASS_T12_6_1_COMPILE_GATE`.

```powershell
& $Py -m theory.sage_t.causal.future_viability_hierarchy_cli evaluate `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --output-dir "$Root\evaluation"
$EvaluationCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_hierarchy_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --evaluation-receipt "$Root\evaluation\evaluation_receipt.json"
```

Un succès final autorise seulement un nouveau gel de protocole T12.6.2, jamais
une collecte physique automatique.
