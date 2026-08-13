# Runbook PowerShell — SAGE.T12.4a.4

Exécuter depuis la racine de `adaptive_reasoning`. Le worktree doit être propre
avant le freeze scientifique.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\option_minimization_t12_4a_3r1_bp35"
$Root = ".\training\sage_t\option_transfer_t12_4a_4_bp35"
```

## 1. Freeze

```powershell
& $Py -m theory.sage_t.causal.option_transfer_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --ablation-receipt "$Parent\ablation\option_ablation_receipt.json" `
  --compile-receipt "$Parent\shadow_compile\shadow_compile_receipt.json" `
  --manifest "$Root\manifest.json"
```

Le résultat doit contenir :

```text
FROZEN_BEFORE_T12_4A_4_OPTION_TRANSFER
scientific_claims_authorized: true
```

Ne pas utiliser `--allow-dirty` pour une exécution scientifique.

## 2. Transfert physique apparié

Cette commande peut effectuer jusqu'à 60 branches et 4 500 appels SDK. Elle
ne persiste aucune frame brute et ne peut pas dépasser 3 Gio :

```powershell
& $Py -m theory.sage_t.causal.option_transfer_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\transfer" `
  --environments-dir .\environment_files
```

Le run s'arrête après le premier niveau non confirmé. Une progression complète
ouvre prospectivement le niveau suivant ; aucun résultat n'est retouché.

## 3. Statut

```powershell
& $Py -m theory.sage_t.causal.option_transfer_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\transfer\transfer_receipt.json"
```

Un pass valide doit afficher :

```text
PASS_T12_4A_4_OPTION_TRANSFER_GATE
t12_4a_5_option_control_freeze_authorized: true
option_control_authorized: false
```

Si le gate échoue, arrêter la lignée et conserver le reçu négatif. Ne pas
modifier les seuils, relancer avec d'autres branches ou ouvrir T12.4a.5.

