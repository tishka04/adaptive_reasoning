# SAGE.T12.3b — commandes PowerShell

Exécuter depuis un worktree propre après avoir commité l'implémentation. Ne pas
réutiliser un ancien `$Parent` ou `$Root` dans la même console sans les
réassigner.

```powershell
Set-Location C:\Users\coudr\projects\adaptive_reasoning

$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = (Resolve-Path .\training\sage_t\progress_witness_t12_3a_bp35).Path
$Root = Join-Path (Get-Location) "training\sage_t\terminal_shield_t12_3b_bp35"
```

## 1. Tests locaux

```powershell
& $Py -m ruff check `
  .\theory\sage_t\causal\shield_model.py `
  .\theory\sage_t\causal\shield_protocol.py `
  .\theory\sage_t\causal\shield_experiment.py `
  .\theory\sage_t\causal\shield_experiment_cli.py `
  .\tests\test_sage_t_terminal_shield_t12_3b.py

& $Py -m pytest -q `
  .\tests\test_sage_t_terminal_shield_t12_3b.py `
  .\tests\test_sage_t_progress_witness_t12_3a.py `
  .\tests\test_sage_t_burst_go_explore.py `
  .\tests\test_sage_t_graph_explore_archive.py `
  .\tests\test_sage_t_causal_protocol_diagnostics.py
```

## 2. Vérifier le parent

```powershell
& $Py -m theory.sage_t.causal.witness_experiment_cli status `
  --manifest "$Parent\manifest.json" `
  --receipt "$Parent\confirmation\witness_receipt.json"
```

La sortie doit contenir `PASS_T12_3A_WITNESS_GATE` et
`next_phase_authorized: true`.

## 3. Freeze T12.3b

```powershell
& $Py -m theory.sage_t.causal.shield_experiment_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\confirmation\witness_receipt.json" `
  --terminal-registry "$Root\terminal_candidates.sealed.json" `
  --manifest "$Root\manifest.json"
```

Ne pas utiliser `--allow-dirty` pour un run scientifique. La sortie attendue
contient :

```text
status: FROZEN_BEFORE_T12_3B_TERMINAL_SHIELD
scientific_claims_authorized: true
```

Vérifier immédiatement le freeze :

```powershell
& $Py -m theory.sage_t.causal.shield_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\freeze_receipt.json"
```

## 4. Run physique borné

Cette commande confirme 12 traces terminales, rejoue six fois les témoins de
progression et exécute les six bras prospectifs. Les plafonds durs sont 30 000
appels SDK et 3 Gio d'artefacts.

```powershell
& $Py -m theory.sage_t.causal.shield_experiment_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

Le répertoire `paired` doit être neuf. Le CLI refuse d'écraser ou de compléter
un run précédent.

## 5. Vérification finale

```powershell
& $Py -m theory.sage_t.causal.shield_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\shield_receipt.json"
```

Continuer uniquement si les trois valeurs suivantes sont présentes :

```text
receipt.passed: true
receipt.status: PASS_T12_3B_TERMINAL_SHIELD_GATE
next_phase_authorized: true
```

Sinon, arrêter et conserver le résultat négatif. Ne pas lancer le prédicteur
neural, l'extraction d'option, `source_validation` ou le holdout.

