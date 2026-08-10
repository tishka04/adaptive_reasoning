# SAGE.T10.2.6 — runbook

Exécuter depuis la racine du dépôt avec l’environnement Python ARC-AGI-3.

## 1. Tests ciblés

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q tests\test_sage_t_t10_2_6_protocol.py tests\test_sage_t_t10_2_6_runtime.py
```

Le test de runtime ouvre un vrai processus `spawn` et vérifie que le seed frais
est décodable dans l’enfant avant toute action.

## 2. Gel du reçu et du manifeste

Cette commande ne lance aucune action dans un environnement. Elle doit être
exécutée une seule fois sur l’échec T10.2.5 authentifié.

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_6_protocol freeze --repo-root .
$LASTEXITCODE
```

## 3. Préflight

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_6_runtime status --repo-root .
$LASTEXITCODE
```

La sortie attendue est `READY_T10_2_6_SPAWN_RECOVERY`.

## 4. Collecte de la lane de remplacement

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_6_runtime collect --repo-root .
$LASTEXITCODE
```

Les huit resets parent et les trois échecs T10.2.5 ne sont pas rejoués. La
commande collecte uniquement une nouvelle lane de remplacement, avec au plus
trois tentatives déterministes.

Le succès terminal exact est :

```text
"status":"T10_2_6_SOURCE_COLLECTION_COMPLETE"
```

avec un code de sortie `0`. Un statut `FAIL_T10_2_6_RECOVERY` ou
`DATA_OR_PROVENANCE_INVALID` retourne le code `3`. Ne pas ouvrir la validation,
le holdout ou AR25 sans le statut positif exact.
