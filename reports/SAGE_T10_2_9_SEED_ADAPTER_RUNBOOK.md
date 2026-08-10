# SAGE.T10.2.9 — runbook

Depuis la racine du dépôt :

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests/test_sage_t_t10_2_9_protocol.py tests/test_sage_t_t10_2_9_runtime.py -q
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_9_protocol freeze --repo-root .
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_9_runtime status --repo-root .
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u -m theory.sage_t.t10_2_9_runtime compile --repo-root .
$LASTEXITCODE
```

Le code de sortie `0` signifie que toutes les gates de QA passent. Le code `3`
est un résultat scientifique négatif fail-closed, pas une panne du collecteur.
Le code `2` signale une dérive de protocole, de provenance ou d'artifact.

Le protocole n'a aucune commande de collecte ou d'entraînement.
