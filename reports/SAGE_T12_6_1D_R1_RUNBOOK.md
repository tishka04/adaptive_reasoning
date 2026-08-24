# SAGE.T12.6.1d-r1 — Runbook

Le dossier v1 ne doit être ni supprimé, ni modifié, ni réutilisé comme dossier
de sortie. Toutes les commandes partent de la racine du dépôt.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Parent = ".\training\sage_t\future_viability_confirmation_t12_6_1d_bp35"
$Root = ".\training\sage_t\future_viability_confirmation_t12_6_1d_r1_bp35"
$Reliable = ".\training\sage_t\future_viability_reliability_t12_6_1c_bp35"
$Hazard = ".\training\sage_t\hazard_diversity_t12_4a_4d_1_bp35"
```

## 1. Gel r1 et preflight hors ligne

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-preflight-receipt "$Parent\preflight\preflight_receipt.json" `
  --aborted-archive "$Parent\collection\pilot\bp35\9301\8701\local_archive_control.json" `
  --reliability-manifest "$Reliable\manifest.json" `
  --reliability-compile-receipt "$Reliable\compile\compile_receipt.json" `
  --hazard-manifest "$Hazard\manifest.json" `
  --hazard-compile-receipt "$Hazard\compile\compile_receipt.json" `
  --manifest "$Root\manifest.json"
$FreezeCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_prospective_cli preflight `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\preflight" `
  --environments-dir ".\environment_files"
$PreflightCode = $LASTEXITCODE
```

Continuer uniquement si les deux codes valent `0` et si le statut ouvre
seulement `pilot_collection_authorized`.

## 2. Nouveau lot pilote 9401 — collecte manuelle

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli collect-batch `
  --batch pilot `
  --manifest "$Root\manifest.json" `
  --preflight-receipt "$Root\preflight\preflight_receipt.json" `
  --output-dir "$Root\collection\pilot" `
  --environments-dir ".\environment_files"
$PilotCode = $LASTEXITCODE
```

Ne pas calculer de score après ce lot. Un code `2` conserve le dossier comme
exécution avortée ; un code `3` conserve le reçu comme échec du gate pilote.

## 3. Lot 9402–9403 — collecte manuelle

Cette commande exige `PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY`.

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli collect-batch `
  --batch completion `
  --manifest "$Root\manifest.json" `
  --preflight-receipt "$Root\preflight\preflight_receipt.json" `
  --pilot-receipt "$Root\collection\pilot\collection_receipt.json" `
  --output-dir "$Root\collection\completion" `
  --environments-dir ".\environment_files"
$CompletionCode = $LASTEXITCODE
```

## 4. Scellement, prédiction et adjudication

Les commandes sont identiques à v1, avec `$Root` pointant vers le dossier r1 :

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli seal-collection `
  --manifest "$Root\manifest.json" `
  --pilot-receipt "$Root\collection\pilot\collection_receipt.json" `
  --completion-receipt "$Root\collection\completion\collection_receipt.json" `
  --output-dir "$Root\collection\sealed"

& $Py -m theory.sage_t.causal.future_viability_prospective_cli predict `
  --manifest "$Root\manifest.json" `
  --collection-seal-receipt "$Root\collection\sealed\collection_seal_receipt.json" `
  --output-dir "$Root\prediction"

& $Py -m theory.sage_t.causal.future_viability_prospective_cli adjudicate `
  --manifest "$Root\manifest.json" `
  --collection-seal-receipt "$Root\collection\sealed\collection_seal_receipt.json" `
  --prediction-receipt "$Root\prediction\prediction_receipt.json" `
  --output-dir "$Root\adjudication"
```
