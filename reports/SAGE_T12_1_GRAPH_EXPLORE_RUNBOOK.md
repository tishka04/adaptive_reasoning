# SAGE.T12.1 — runbook PowerShell

Ce runbook ne doit être exécuté qu'après commit, depuis un worktree propre. Le
code ne lance aucune collecte automatiquement.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Root = ".\training\sage_t\graph_explore_t12_1_bp35"
```

Vérification locale rapide :

```powershell
& $Py -m ruff check .\theory\sage_t\causal .\tests\test_sage_t_graph_explore_archive.py .\tests\test_sage_t_graph_explore_options.py .\tests\test_sage_t_graph_experiment_cli.py
& $Py -m pytest -q .\tests\test_sage_t_graph_explore_archive.py .\tests\test_sage_t_graph_explore_options.py .\tests\test_sage_t_graph_experiment_cli.py
```

Gel source-train, lié aux programmes causaux existants :

```powershell
& $Py -m theory.sage_t.causal.graph_experiment_cli freeze `
  --stage source_train `
  --games bp35 `
  --program-registry .\training\sage_t\causal_inputs\programs.sealed.json `
  --manifest "$Root\manifest.json"
```

Puis lancer une phase à la fois. Ne continuer que si la commande précédente
retourne `passed: true` :

```powershell
& $Py -m theory.sage_t.causal.graph_experiment_cli archive `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\archive" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.graph_experiment_cli shield `
  --manifest "$Root\manifest.json" `
  --archive-receipt "$Root\archive\archive_receipt.json" `
  --output-dir "$Root\shield" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.graph_experiment_cli train-novelty `
  --manifest "$Root\manifest.json" `
  --archive-receipt "$Root\archive\archive_receipt.json" `
  --shield-receipt "$Root\shield\shield_receipt.json" `
  --output-dir "$Root\novelty"

& $Py -m theory.sage_t.causal.graph_experiment_cli neural `
  --manifest "$Root\manifest.json" `
  --novelty-receipt "$Root\novelty\novelty_receipt.json" `
  --output-dir "$Root\neural" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.graph_experiment_cli extract-option `
  --manifest "$Root\manifest.json" `
  --archive-receipt "$Root\archive\archive_receipt.json" `
  --parent-receipt "$Root\neural\neural_ordering_receipt.json" `
  --output-dir "$Root\option" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.graph_experiment_cli compile-option `
  --manifest "$Root\manifest.json" `
  --option-receipt "$Root\option\option_receipt.json" `
  --output-dir "$Root\compiled_option"

& $Py -m theory.sage_t.causal.graph_experiment_cli transfer `
  --manifest "$Root\manifest.json" `
  --compilation-receipt "$Root\compiled_option\option_compilation_receipt.json" `
  --archive-receipt "$Root\archive\archive_receipt.json" `
  --novelty-receipt "$Root\novelty\novelty_receipt.json" `
  --output-dir "$Root\transfer" `
  --environments-dir .\environment_files
```

Audit final de la chaîne :

```powershell
& $Py -m theory.sage_t.causal.graph_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipts `
    "$Root\archive\archive_receipt.json" `
    "$Root\shield\shield_receipt.json" `
    "$Root\novelty\novelty_receipt.json" `
    "$Root\neural\neural_ordering_receipt.json" `
    "$Root\option\option_receipt.json" `
    "$Root\compiled_option\option_compilation_receipt.json" `
    "$Root\transfer\transfer_receipt.json"
```

Un manifeste `regression` peut exécuter le bras `archive` comme audit gelé,
mais le CLI refuse toutes les phases d'apprentissage ou d'adaptation sur ce
split. Le transfert inter-jeux vers `ft09` demandera donc un futur protocole
d'import explicite des artefacts source; il ne faut pas contourner ce firewall
en réentraînant sur `ft09`.
