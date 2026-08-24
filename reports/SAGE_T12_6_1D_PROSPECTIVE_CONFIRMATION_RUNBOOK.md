# SAGE.T12.6.1d — Runbook

Toutes les commandes partent de la racine du dépôt. Le gel et le preflight sont
hors ligne. Les deux commandes `collect-batch` sont les seules qui déclenchent
la collecte physique ; elles doivent être lancées manuellement.

```powershell
$Py = ".\ARC-AGI-3-Agents\.venv\Scripts\python.exe"
$Root = ".\training\sage_t\future_viability_confirmation_t12_6_1d_bp35"
$Reliable = ".\training\sage_t\future_viability_reliability_t12_6_1c_bp35"
$Hazard = ".\training\sage_t\hazard_diversity_t12_4a_4d_1_bp35"
```

## 1. Gel et preflight hors ligne

Le gel exige un worktree propre et lie le commit courant. Le preflight ne fait
aucun appel SDK.

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli freeze `
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

Continuer seulement si les deux codes valent `0` et si le statut autorise
uniquement `pilot_collection_authorized`.

## 2. Lot pilote 9301 — collecte manuelle

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli collect-batch `
  --batch pilot `
  --manifest "$Root\manifest.json" `
  --preflight-receipt "$Root\preflight\preflight_receipt.json" `
  --output-dir "$Root\collection\pilot" `
  --environments-dir ".\environment_files"
$PilotCode = $LASTEXITCODE
```

Un code `3` est un miss scientifique : conserver les artefacts et arrêter. Un
code `2` est un échec fermé d’exécution. Aucun `predict` ou `adjudicate` ne doit
être lancé ici.

## 3. Lot 9302–9303 — collecte manuelle

Cette commande est autorisée uniquement par
`PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY`.

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

## 4. Scellement, engagement et adjudication

Les labels ne sont ouverts qu’à la dernière commande, après l’engagement signé
des prédictions.

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli seal-collection `
  --manifest "$Root\manifest.json" `
  --pilot-receipt "$Root\collection\pilot\collection_receipt.json" `
  --completion-receipt "$Root\collection\completion\collection_receipt.json" `
  --output-dir "$Root\collection\sealed"
$SealCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_prospective_cli predict `
  --manifest "$Root\manifest.json" `
  --collection-seal-receipt "$Root\collection\sealed\collection_seal_receipt.json" `
  --output-dir "$Root\prediction"
$PredictCode = $LASTEXITCODE

& $Py -m theory.sage_t.causal.future_viability_prospective_cli adjudicate `
  --manifest "$Root\manifest.json" `
  --collection-seal-receipt "$Root\collection\sealed\collection_seal_receipt.json" `
  --prediction-receipt "$Root\prediction\prediction_receipt.json" `
  --output-dir "$Root\adjudication"
$AdjudicationCode = $LASTEXITCODE
```

## 5. Statut vérifiable

```powershell
& $Py -m theory.sage_t.causal.future_viability_prospective_cli status `
  --manifest "$Root\manifest.json" `
  --preflight-receipt "$Root\preflight\preflight_receipt.json" `
  --pilot-receipt "$Root\collection\pilot\collection_receipt.json" `
  --completion-receipt "$Root\collection\completion\collection_receipt.json" `
  --collection-seal-receipt "$Root\collection\sealed\collection_seal_receipt.json" `
  --prediction-receipt "$Root\prediction\prediction_receipt.json" `
  --adjudication-receipt "$Root\adjudication\adjudication_receipt.json"
```

Chaque phase écrit dans un dossier neuf et immuable. Ne pas relancer dans un
dossier déjà rempli ; préserver le reçu négatif et choisir un nouveau protocole
si une correction scientifique est nécessaire.
