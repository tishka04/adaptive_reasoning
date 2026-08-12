# SAGE.T12.3a — commandes PowerShell

Exécuter ces commandes depuis un worktree propre, après avoir commité
l'implémentation. Elles lisent T12.2 sans modifier ses artefacts.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\burst_go_explore_t12_2_bp35"
$Root = ".\training\sage_t\progress_witness_t12_3a_bp35"
```

## 1. Vérification locale

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\witness_protocol.py `
  .\theory\sage_t\causal\witness_experiment.py `
  .\theory\sage_t\causal\witness_experiment_cli.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py

& $Py -m pytest -q `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py `
  .\tests\test_sage_t_causal_protocol_diagnostics.py
```

## 2. Freeze scientifique

Le freeze relit les archives T12.2, extrait les deux chemins positifs, vérifie
leur état initial, leur cible et leur suffixe commun, puis scelle le registre.

```powershell
& $Py -m theory.sage_t.causal.witness_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\paired\burst_receipt.json" `
  --witness-registry "$Root\witnesses.sealed.json" `
  --manifest "$Root\manifest.json"
```

Ne pas utiliser `--allow-dirty` pour une revendication scientifique. Vérifier
que la sortie contient `scientific_claims_authorized: true` et
`status: FROZEN_BEFORE_T12_3A_WITNESS_CONFIRMATION`.

## 3. Confirmation physique bornée

Cette commande lance 18 branches : trois routes complètes, trois suffixes et
trois contrôles par suppression pour chacun des deux témoins. Le budget dur est
de 2 048 appels SDK et 3 Gio d'artefacts.

```powershell
& $Py -m theory.sage_t.causal.witness_experiment_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\confirmation" `
  --environments-dir .\environment_files
```

Le répertoire de sortie est immuable : la commande refuse d'ajouter des
résultats à un run existant.

## 4. Vérification du receipt

```powershell
& $Py -m theory.sage_t.causal.witness_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\confirmation\witness_receipt.json"
```

Continuer uniquement si :

- `receipt.passed` vaut `true` ;
- `receipt.status` vaut `PASS_T12_3A_WITNESS_GATE` ;
- `next_phase_authorized` vaut `true`.

Sinon, arrêter. Ne pas lancer de bouclier, d'entraînement neural, d'extraction
d'option, de `source_validation` ou de holdout.

