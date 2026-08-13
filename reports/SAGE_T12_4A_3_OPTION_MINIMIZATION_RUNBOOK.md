# Runbook PowerShell — SAGE.T12.4a.3

Exécuter depuis la racine de `adaptive_reasoning`, avec un worktree propre
avant le freeze.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\witness_reconfirmation_t12_4a_2_bp35"
$Root = ".\training\sage_t\option_minimization_t12_4a_3_bp35"
$Programs = ".\training\sage_t\causal_inputs\programs.sealed.json"
```

## 1. Freeze prospectif

```powershell
& $Py -m theory.sage_t.causal.option_minimization_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\confirmation\witness_receipt.json" `
  --program-registry $Programs `
  --manifest "$Root\manifest.json"
```

Le freeze doit afficher `FROZEN_BEFORE_T12_4A_3_OPTION_MINIMIZATION` et
`scientific_claims_authorized: true`. Ne pas utiliser `--allow-dirty` pour une
exécution scientifique.

## 2. Ablation physique exhaustive

Cette commande effectue les 390 branches ARC. Elle peut consommer jusqu'à
24 000 appels SDK, mais ne peut pas dépasser 3 Gio d'artefacts.

```powershell
& $Py -m theory.sage_t.causal.option_minimization_cli ablate `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\ablation" `
  --environments-dir .\environment_files
```

Vérifier le reçu avant toute suite :

```powershell
& $Py -m theory.sage_t.causal.option_minimization_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\ablation\option_ablation_receipt.json"
```

Si le statut n'est pas `PASS_T12_4A_3_OPTION_ABLATION_GATE`, arrêter ici et
conserver le résultat négatif.

## 3. Compilation dans le posterior, hors environnement

Cette commande ne collecte aucune donnée ARC et n'accorde aucune autorité de
contrôle :

```powershell
& $Py -m theory.sage_t.causal.option_minimization_cli compile-shadow `
  --manifest "$Root\manifest.json" `
  --ablation-receipt "$Root\ablation\option_ablation_receipt.json" `
  --output-dir "$Root\shadow_compile"
```

Statut final :

```powershell
& $Py -m theory.sage_t.causal.option_minimization_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\ablation\option_ablation_receipt.json" `
  --compile-receipt "$Root\shadow_compile\shadow_compile_receipt.json"
```

Seul `PASS_T12_4A_3_SHADOW_COMPILE_GATE` avec
`t12_4a_4_transfer_freeze_authorized: true` permet de préparer T12.4a.4. Les
champs `option_control_authorized`, `source_validation_opened`,
`holdout_opened` et `production_authority` doivent rester `false`.

