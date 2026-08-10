# Runbook SAGE.T10.3

Toutes les commandes partent de la racine du dépôt et utilisent le Python de `ARC-AGI-3-Agents` avec sortie non bufferisée.

## Préparation sans action physique

1. Figer une seule fois le manifeste et le reçu de handoff :

   `python -u -m theory.sage_t.t10_3_runtime freeze --repo-root .`

2. Authentifier T10.2.9 et documenter ses labels universels/projections incomplètes :

   `python -u -m theory.sage_t.t10_3_runtime audit --repo-root .`

3. Vérifier l’état :

   `python -u -m theory.sage_t.t10_3_runtime status --repo-root .`

## Acquisition et analyse

Exécuter chaque phase séparément et contrôler `$LASTEXITCODE` après chaque commande.

1. `collect` réalise uniquement les 48 resets du panel source.
2. `compile` backfill la cible par branche, écrit le ledger compact et applique la QA. Un code 3 est un résultat scientifique négatif et interdit `fit`.
3. `fit` effectue le leave-one-game-out et les contrôles causaux. Un code 3 interdit `confirm`.
4. `confirm` réalise uniquement les 12 resets source de confirmation.
5. `report` écrit le verdict exclusif terminal.

Codes de sortie : 0 phase réussie, 3 gate scientifique négatif, 2 dérive d’intégrité ou défaut d’exécution. Une reprise relit les reçus existants ; un intent dont l’issue est inconnue devient non résolu et n’est jamais rejoué.

Le répertoire `training/sage_t/t10_3_goal_progress_pilot` est write-once pour les intents, événements, reçus, QA, recette, confirmation et terminal. Ne pas modifier ces fichiers à la main et ne pas lancer deux collecteurs simultanément.

