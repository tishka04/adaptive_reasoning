# SAGE.T12.3c — commandes PowerShell

Exécuter depuis un worktree propre après avoir commit l'implémentation. Le run
physique n'est pas lancé par l'installation ou par les tests.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\terminal_shield_t12_3b_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\replay_lineage_t12_3c_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\lineage_archive.py `
  .\theory\sage_t\causal\lineage_protocol.py `
  .\theory\sage_t\causal\lineage_experiment.py `
  .\theory\sage_t\causal\lineage_experiment_cli.py `
  .\tests\test_sage_t_replay_lineage_t12_3c.py

& $Py -m pytest -q `
  .\tests\test_sage_t_replay_lineage_t12_3c.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py
```

## 2. Vérifier le résultat parent

```powershell
& $Py -m theory.sage_t.causal.shield_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\paired\shield_receipt.json"
```

La sortie attendue reste `FAIL_T12_3B_TERMINAL_SHIELD_GATE`. T12.3c exige que
ce soit un échec dû uniquement au replay exact.

## 3. Freeze T12.3c

```powershell
& $Py -m theory.sage_t.causal.lineage_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\paired\shield_receipt.json" `
  --audit-registry "$Root\replay_audit.sealed.json" `
  --manifest "$Root\manifest.json"
```

Ne pas employer `--allow-dirty` pour un résultat scientifique. Vérifier le
freeze :

```powershell
& $Py -m theory.sage_t.causal.lineage_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

La sortie doit contenir `PASS_T12_3C_FREEZE` et conserver tous les firewalls
aval à `false`.

## 4. Run physique borné

```powershell
& $Py -m theory.sage_t.causal.lineage_experiment_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

Le CLI refuse d'écraser un répertoire non vide. Le run est borné à 30 000
appels SDK et 3 Gio, sans sauvegarde des frames brutes.

## 5. Statut final

```powershell
& $Py -m theory.sage_t.causal.lineage_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\lineage_receipt.json"
```

Continuer vers un nouveau descendant du test du bouclier uniquement si la
sortie contient :

```text
receipt.passed: true
receipt.status: PASS_T12_3C_REPLAY_LINEAGE_GATE
firewall.t12_3b_child_rerun_authorized: true
```

Même en cas de passage, `neural_training_authorized`,
`option_extraction_authorized`, `source_validation_opened`, `holdout_opened` et
`terminal_shield_production_authority` doivent rester à `false`.

