# SAGE.T12.4a — commandes PowerShell

Exécuter depuis un worktree propre après commit. Codex ne lance ni la collecte
ARC prospective ni le fit scientifique.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\neural_novelty_t12_4_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\representation_t12_4a_bp35"
```

## 1. Validation locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\relational_novelty.py `
  .\theory\sage_t\causal\representation_protocol.py `
  .\theory\sage_t\causal\representation_experiment.py `
  .\theory\sage_t\causal\representation_experiment_cli.py `
  .\tests\test_sage_t_representation_t12_4a.py

& $Py -m pytest -q `
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
& $Py -m theory.sage_t.causal.neural_novelty_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\training\training_receipt.json"
```

La sortie doit rester `FAIL_T12_4_NEURAL_FIT_GATE` avec
`next_phase_authorized: false`.

## 3. Freeze T12.4a

```powershell
& $Py -m theory.sage_t.causal.representation_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\training\training_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.representation_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

Attendre `PASS_T12_4A_FREEZE`. Ne pas employer `--allow-dirty` pour un run
scientifique.

## 4. Collecte prospective bornée

```powershell
& $Py -m theory.sage_t.causal.representation_experiment_cli collect `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\collection" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.representation_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\collection\collection_receipt.json"
```

Continuer uniquement avec `PASS_T12_4A_COLLECTION_GATE` et
`representation_training_authorized: true`. La collecte est limitée à 12 288
appels SDK sur un plafond de 15 000 et à 3 Gio.

## 5. Fit comparatif offline

```powershell
& $Py -m theory.sage_t.causal.representation_experiment_cli train `
  --manifest "$Root\manifest.json" `
  --collection-receipt "$Root\collection\collection_receipt.json" `
  --output-dir "$Root\training"

& $Py -m theory.sage_t.causal.representation_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\training\representation_receipt.json"
```

Le passage attendu est :

```text
receipt.status: PASS_T12_4A_REPRESENTATION_GATE
firewall.t12_4b_freeze_authorized: true
```

Il n'existe volontairement aucune commande `evaluate` ou `extract-option` dans
ce CLI. T12.4b, T12.5, holdout et production restent fermés sur tout échec.

