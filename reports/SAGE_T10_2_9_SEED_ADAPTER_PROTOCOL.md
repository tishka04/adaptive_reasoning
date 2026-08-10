# SAGE.T10.2.9 — correction append-only du registre de graines

## Objet

T10.2.8 s'est arrêté avant la QA scientifique sur une erreur d'adaptateur : le
validateur T10.2 historique reconnaissait les graines `0..5`, tandis que la
collecte durable T10.2.1 utilise les graines de découverte `101, 102, 103` et
de confirmation `111, 112, 113`. La lane de récupération T10.2.7 utilise en
plus la graine physique `3119945`.

T10.2.9 préserve le terminal T10.2.8, authentifie son arrêt fail-closed et
modifie uniquement le registre temporaire utilisé pendant la validation de
provenance. Les événements, les seuils et le calcul scientifique de QA restent
inchangés.

## Pare-feu

- aucune ouverture d'environnement ;
- aucune action physique et aucun replay ;
- aucun entraînement ou ajustement de modèle ;
- aucune validation source, AR25 ou holdout ;
- le terminal T10.2.8 et le ledger T10.2.7 sont en lecture seule ;
- un échec de lineage empêche la QA ;
- un échec de QA arrête le protocole avant tout fit.

## Gates

1. Authentifier le manifeste, le terminal, l'audit de lineage et l'abstention
   QA de T10.2.8.
2. Vérifier que T10.2.8 n'a exécuté aucune action, aucun fit et aucune
   validation active.
3. Valider les 1 370 événements et les 18 lanes avec les graines durables.
4. Appliquer sans modification les gates scientifiques gelés dans T10.2.8.
5. Produire un terminal write-once. Une QA négative interdit le fit et ne peut
   être contournée que par un nouveau protocole scientifique explicite.
