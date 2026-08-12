# SAGE.T12.4 — commandes PowerShell

Exécuter depuis un worktree propre après commit de l'implémentation. Les
répertoires de résultats sont immuables et chaque run est limité à 3 Gio.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\lineage_shield_t12_3e_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\neural_novelty_t12_4_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\neural_novelty_protocol.py `
  .\theory\sage_t\causal\neural_novelty_experiment.py `
  .\theory\sage_t\causal\neural_novelty_experiment_cli.py `
  .\tests\test_sage_t_neural_novelty_t12_4.py

& $Py -m pytest -q `
  .\tests\test_sage_t_neural_novelty_t12_4.py `
  .\tests\test_sage_t_lineage_shield_t12_3e.py `
  .\tests\test_sage_t_confirmed_control_t12_3d.py `
  .\tests\test_sage_t_replay_lineage_t12_3c.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py
```

## 2. Vérifier le parent

```powershell
& $Py -m theory.sage_t.causal.lineage_shield_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\paired\lineage_shield_receipt.json"
```

La sortie doit contenir `PASS_T12_3E_LINEAGE_SHIELD_GATE` et
`t12_4_freeze_authorized: true`.

## 3. Freeze et audit du dataset

```powershell
& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\paired\lineage_shield_receipt.json" `
  --dataset "$Root\dataset.sealed.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

Attendre `PASS_T12_4_FREEZE`. Ne pas employer `--allow-dirty` pour un résultat
scientifique.

## 4. Fit offline

```powershell
& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli train `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\training"

& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\training\training_receipt.json"
```

Continuer uniquement si la sortie contient
`PASS_T12_4_NEURAL_FIT_GATE` et
`neural_active_evaluation_authorized: true`. Un échec offline ferme la suite.

## 5. Évaluation active bornée

```powershell
& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli evaluate `
  --manifest "$Root\manifest.json" `
  --training-receipt "$Root\training\training_receipt.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

Le run utilise au maximum 24 576 appels SDK sur une limite globale de 30 000,
et refuse un dossier de sortie non vide.

## 6. Statut final

```powershell
& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\active_receipt.json"
```

Le passage attendu est :

```text
receipt.status: PASS_T12_4_NEURAL_ACTIVE_GATE
firewall.t12_5_freeze_authorized: true
```

Les autorisations option, validation, holdout et production restent à `false`.
Ne pas ouvrir T12.5 sur un échec ou retoucher les seuils après observation.

