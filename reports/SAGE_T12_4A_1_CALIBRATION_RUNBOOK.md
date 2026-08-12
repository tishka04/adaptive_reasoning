# SAGE.T12.4a.1 — commandes PowerShell

Exécuter depuis un worktree propre après commit. Codex ne lance ni la collecte
ARC prospective ni le fit scientifique.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\representation_t12_4a_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\calibration_t12_4a_1_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\calibration_protocol.py `
  .\theory\sage_t\causal\calibration_experiment.py `
  .\theory\sage_t\causal\calibration_experiment_cli.py `
  .\tests\test_sage_t_calibration_t12_4a_1.py

& $Py -m pytest -q `
  .\tests\test_sage_t_calibration_t12_4a_1.py `
  .\tests\test_sage_t_representation_t12_4a.py `
  .\tests\test_sage_t_neural_novelty_t12_4.py `
  .\tests\test_sage_t_lineage_shield_t12_3e.py `
  .\tests\test_sage_t_confirmed_control_t12_3d.py `
  .\tests\test_sage_t_replay_lineage_t12_3c.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py
```

## 2. Vérifier le parent négatif

```powershell
& $Py -m theory.sage_t.causal.representation_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\training\representation_receipt.json"
```

La sortie doit rester `FAIL_T12_4A_REPRESENTATION_GATE`, avec
`t12_4b_freeze_authorized: false`.

## 3. Freeze T12.4a.1

```powershell
& $Py -m theory.sage_t.causal.calibration_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\training\representation_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.calibration_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

Attendre `PASS_T12_4A_1_FREEZE`. Ne pas employer `--allow-dirty` pour un run
scientifique.

## 4. Collecte prospective bornée

```powershell
& $Py -m theory.sage_t.causal.calibration_experiment_cli collect `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\collection" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.calibration_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\collection\collection_receipt.json"
```

Continuer uniquement avec `PASS_T12_4A_1_COLLECTION_GATE` et
`calibration_training_authorized: true`. La collecte est limitée à 24 576
appels SDK sur un plafond de 26 000 et à 3 Gio.

## 5. Fit, calibration et confirmation offline

```powershell
& $Py -m theory.sage_t.causal.calibration_experiment_cli train `
  --manifest "$Root\manifest.json" `
  --collection-receipt "$Root\collection\collection_receipt.json" `
  --output-dir "$Root\training"

& $Py -m theory.sage_t.causal.calibration_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\training\calibration_receipt.json"
```

Le passage attendu est :

```text
receipt.status: PASS_T12_4A_1_CALIBRATION_GATE
firewall.t12_4b_freeze_authorized: true
```

Sur tout échec, ne pas réutiliser les seeds 8705–8706 pour retuner ou
recalibrer. Il n'existe volontairement aucune commande `evaluate` ou
`extract-option`. T12.4b, T12.5, holdout et production restent fermés.
