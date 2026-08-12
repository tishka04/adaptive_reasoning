# SAGE.T12.2 — commandes PowerShell

Les commandes suivantes doivent être exécutées depuis un worktree propre,
après commit de l'implémentation. Elles ne modifient pas le run T12.1.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\graph_explore_t12_1_bp35"
$Root = ".\training\sage_t\burst_go_explore_t12_2_bp35"
```

Tests locaux :

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\burst_protocol.py `
  .\theory\sage_t\causal\burst_experiment.py `
  .\theory\sage_t\causal\burst_experiment_cli.py `
  .\tests\test_sage_t_burst_go_explore.py

& $Py -m pytest -q `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_causal_protocol_diagnostics.py
```

Gel scientifique lié au résultat négatif T12.1 :

```powershell
& $Py -m theory.sage_t.causal.burst_experiment_cli freeze `
  --game bp35 `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\archive\archive_receipt.json" `
  --manifest "$Root\manifest.json"
```

Vérifier que le freeze retourne `scientific_claims_authorized: true`, puis
lancer le run apparié :

```powershell
& $Py -m theory.sage_t.causal.burst_experiment_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

Enfin, vérifier le receipt :

```powershell
& $Py -m theory.sage_t.causal.burst_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\burst_receipt.json"
```

Si `passed` est faux ou `next_phase_authorized` vaut faux, arrêter. Ne pas
lancer une phase de bouclier, d'apprentissage neural ou d'option.

