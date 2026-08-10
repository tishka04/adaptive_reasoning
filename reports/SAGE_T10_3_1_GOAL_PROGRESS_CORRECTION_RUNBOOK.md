# Runbook SAGE.T10.3.1

Depuis la racine du dépôt, exécuter les phases séparément avec le Python de `ARC-AGI-3-Agents` et vérifier `$LASTEXITCODE`.

1. `freeze` écrit le manifeste et la recette de migration.
2. `audit` authentifie T10.3 et teste hors ligne le quotient transporté ; aucune action physique.
3. `status` doit afficher 0 action avant la collecte.
4. `collect` exécute les 48 resets du nouveau panel source.
5. `compile` écrit le ledger dérivé et applique la QA. Un code 3 interdit le fit.
6. `fit` exécute le leave-one-game-out et les contrôles causaux. Un code 3 interdit la confirmation.
7. `confirm` exécute les 12 resets source contrebalancés.
8. `report` écrit le verdict terminal exclusif.

Codes de sortie : 0 phase réussie, 3 gate scientifique négatif, 2 dérive d’intégrité. Après un code 3 de `compile`, `fit` ou `confirm`, lancer seulement `report`. Après un code 2, ne rien relancer avant diagnostic.

Le journal est write-once. Un intent sans événement devient non résolu et n’est jamais rejoué. Ne jamais lancer deux collecteurs simultanément et ne modifier aucun artifact sous `training/sage_t/t10_3_1_goal_progress_correction`.

