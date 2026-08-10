# SAGE.T10.2.5 — runbook

Exécuter depuis la racine du dépôt avec l'environnement Python ARC-AGI-3.

## 1. Vérification statique

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q tests\test_sage_t_t10_2_5_protocol.py tests\test_sage_t_t10_2_5_runtime.py
```

## 2. Gel du reçu et du manifeste

Cette commande ne lance aucune action dans un environnement. Elle doit être
exécutée une seule fois sur l'état orphelin authentifié.

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_5_protocol freeze --repo-root .
$LASTEXITCODE
```

## 3. Préflight

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_5_runtime status --repo-root .
$LASTEXITCODE
```

La sortie attendue avant reprise est `READY_T10_2_5_RECOVERY`.

## 4. Reprise et récupération

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_5_runtime collect --repo-root .
$LASTEXITCODE
```

La commande effectue dans cet ordre : réparation authentifiée du curseur,
fermeture sans action de la lane orpheline, achèvement des deux lanes parent
restantes, puis collecte d'une lane de remplacement. Elle est reprenable : une
interruption externe ne rejoue aucune action et fait passer la récupération à
la tentative déterministe suivante.

Le succès terminal exact est :

```text
"status":"T10_2_5_SOURCE_COLLECTION_COMPLETE"
```

avec un code de sortie `0`. Ne pas lancer les phases de compilation ou de
validation tant que ce statut et toutes les checks ne sont pas vérifiés.

