# SAGE.T12.4a.2 — commandes PowerShell

Exécuter depuis un worktree propre après commit. Codex ne lance pas les replays
physiques ARC.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\calibration_t12_4a_1_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\witness_reconfirmation_t12_4a_2_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\witness_reconfirmation_protocol.py `
  .\theory\sage_t\causal\witness_reconfirmation_experiment.py `
  .\theory\sage_t\causal\witness_reconfirmation_cli.py `
  .\tests\test_sage_t_witness_reconfirmation_t12_4a_2.py

& $Py -m pytest -q `
  .\tests\test_sage_t_witness_reconfirmation_t12_4a_2.py `
  .\tests\test_sage_t_calibration_t12_4a_1.py `
  .\tests\test_sage_t_representation_t12_4a.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_neural_novelty_t12_4.py `
  .\tests\test_sage_t_lineage_shield_t12_3e.py `
  .\tests\test_sage_t_confirmed_control_t12_3d.py `
  .\tests\test_sage_t_replay_lineage_t12_3c.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py
```

## 2. Vérifier les parents

```powershell
& $Py -m theory.sage_t.causal.calibration_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\training\calibration_receipt.json"

& $Py -m theory.sage_t.causal.calibration_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\collection\collection_receipt.json"
```

Les statuts doivent rester respectivement :

```text
FAIL_T12_4A_1_CALIBRATION_GATE
PASS_T12_4A_1_COLLECTION_GATE
```

## 3. Freeze T12.4a.2

```powershell
& $Py -m theory.sage_t.causal.witness_reconfirmation_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\training\calibration_receipt.json" `
  --collection-receipt "$Parent\collection\collection_receipt.json" `
  --witness-registry "$Root\witnesses.sealed.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.witness_reconfirmation_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

Attendre `PASS_T12_4A_2_FREEZE`. Ne pas employer `--allow-dirty` pour un run
scientifique.

## 4. Reconfirmation physique bornée

```powershell
& $Py -m theory.sage_t.causal.witness_reconfirmation_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\confirmation" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.witness_reconfirmation_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\confirmation\witness_receipt.json"
```

Le passage attendu est :

```text
receipt.status: PASS_T12_4A_2_WITNESS_GATE
firewall.t12_4a_3_option_freeze_authorized: true
```

Le run est plafonné à 2 048 appels SDK et 3 Gio. Sur tout échec, ne pas extraire
d'option et ne pas ouvrir T12.4b ou T12.5.
