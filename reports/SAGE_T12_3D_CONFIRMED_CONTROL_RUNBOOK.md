# SAGE.T12.3d — commandes PowerShell

Exécuter depuis un worktree propre après commit de l'implémentation.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\replay_lineage_t12_3c_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\confirmed_control_t12_3d_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\provenance_protocol.py `
  .\theory\sage_t\causal\provenance_experiment.py `
  .\theory\sage_t\causal\provenance_experiment_cli.py `
  .\tests\test_sage_t_confirmed_control_t12_3d.py

& $Py -m pytest -q `
  .\tests\test_sage_t_confirmed_control_t12_3d.py `
  .\tests\test_sage_t_replay_lineage_t12_3c.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py
```

## 2. Vérifier le parent négatif

```powershell
& $Py -m theory.sage_t.causal.lineage_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\paired\lineage_receipt.json"
```

La sortie attendue reste `FAIL_T12_3C_REPLAY_LINEAGE_GATE` avec
`next_phase_authorized: false`. Le freeze T12.3d vérifie lui-même que le
contrôle de provenance était le seul terme défaillant.

## 3. Freeze T12.3d

```powershell
& $Py -m theory.sage_t.causal.provenance_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\paired\lineage_receipt.json" `
  --control-registry "$Root\confirmed_controls.sealed.json" `
  --manifest "$Root\manifest.json"
```

Ne pas utiliser `--allow-dirty` pour un run scientifique. Vérifier le freeze :

```powershell
& $Py -m theory.sage_t.causal.provenance_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

La sortie doit contenir `PASS_T12_3D_FREEZE`.

## 4. Run physique borné

```powershell
& $Py -m theory.sage_t.causal.provenance_experiment_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

Le CLI refuse un répertoire de sortie non vide. Le run est borné à 30 000
appels SDK et 3 Gio sans frames brutes.

## 5. Statut final

```powershell
& $Py -m theory.sage_t.causal.provenance_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\provenance_receipt.json"
```

Continuer uniquement si la sortie contient :

```text
receipt.passed: true
receipt.status: PASS_T12_3D_CONFIRMED_CONTROL_GATE
firewall.t12_3b_child_rerun_authorized: true
```

Les autorisations neural, options, validation, holdout et production doivent
rester à `false`.

