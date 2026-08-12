# SAGE.T12.3c — protocole gelé de lineage de replay

## Question scientifique

T12.3b a échoué uniquement sur le gate de replay exact : le minimum observé
était 0,9123 pour un seuil preregistré de 0,95. Les critères propres au
bouclier terminal, aux témoins de progression, à la couverture, au risque et au
budget avaient tous passé. T12.3c ne réinterprète pas ce résultat négatif et ne
modifie pas le bouclier. Il teste l'hypothèse étroite suivante :

> une transition observée au cours d'une excursion a parfois été rattachée au
> plus court préfixe représentant le même état visible, plutôt qu'au préfixe
> réellement exécuté ; préserver cette lineage doit restaurer la fiabilité du
> replay sans dégrader matériellement la couverture ou la progression.

## Parent admissible

Le freeze exige le manifeste et le receipt immuables de T12.3b. Le parent doit :

- être en `source_train` ;
- porter `FAIL_T12_3B_TERMINAL_SHIELD_GATE` ;
- avoir échoué sous le seuil de replay exact de 0,95 ;
- satisfaire séparément tous les autres termes du gate T12.3b.

Un échec du bouclier, des témoins, de la couverture, de la sécurité ou du budget
ne peut donc pas être reclassé comme problème de lineage.

## Registre d'audit scellé

Le freeze extrait des archives appariées T12.3b :

- toutes les variantes ayant au moins un échec de restauration, dans la limite
  d'une profondeur de 40 actions ;
- un contrôle sans échec, de profondeur aussi proche que possible et issu de
  la même cellule symbolique lorsqu'il existe ;
- les actions et le hash attendu après chaque étape du préfixe.

Le registre est signé, lié au checksum du receipt T12.3b et limité à 36 cas.
Au moins 12 cas défaillants et les deux bras parents doivent être représentés.
Chaque cas est rejoué trois fois. L'audit s'arrête à la première divergence et
enregistre son index, son type, le hash attendu et le hash observé.

## Comparaison appariée

Les deux bras utilisent le même Go-Explore symbolique, les mêmes seeds, le même
budget et le même calendrier d'excursions 4/8/16 :

- `shortest_prefix_control` conserve le rattachement historique au représentant
  le plus court de l'état visible ;
- `lineage_preserving` rattache chaque transition au préfixe et à la chaîne
  d'arêtes réellement exécutés pendant l'excursion.

Le traitement ne crée aucun réseau, ne modifie pas l'extracteur d'état et ne
change ni la sélection de cellule ni la sélection d'action.

Le seed 6803 est le seed de régression connu. Les seeds 7101 et 7102 sont
prospectifs et n'ont pas servi au diagnostic T12.3b.

## Valeurs gelées et budgets

- seeds : 6803, 7101, 7102 ;
- bras : contrôle historique et traitement préservant la lineage ;
- 3 500 appels SDK maximum par bras ;
- 30 000 appels SDK maximum pour l'audit et les six bras réunis ;
- 50 000 cellules maximum par archive ;
- 3 Gio maximum d'artefacts par run ;
- aucune frame brute persistée.

La borne preregistrée maximale est de 25 428 appels SDK : 4 428 pour 36 cas
d'audit de profondeur 40 répétés trois fois, puis 21 000 pour l'évaluation.

## Gate T12.3c

Le gate passe uniquement si toutes les conditions suivantes sont satisfaites :

1. au moins un échec parent est reproduit par l'audit pas-à-pas ;
2. les contrôles appariés ont un taux exact d'au moins 0,95 ;
3. le minimum du traitement sur les trois seeds atteint 0,95 ;
4. sur le seed de régression 6803, le gain de replay est d'au moins +0,02 ;
5. le ratio de couverture du traitement atteint au moins 0,80 sur chaque seed ;
6. le traitement régresse en progression sur au plus un seed ;
7. au moins une transition est rattachée à sa lineage réellement exécutée ;
8. au moins un rebasage historique est effectivement évité ;
9. aucune transition du traitement n'est rebasée ;
10. le plafond global de 30 000 appels SDK est respecté.

Le seuil de 0,95 de T12.3b n'est ni abaissé ni recalibré après observation.

## Firewalls et résultat négatif

Pendant T12.3c restent fermés : holdout, source-validation, autorité de
production, entraînement neural, extraction d'options et promotion du bouclier
terminal. Un passage du gate autorise seulement un nouveau descendant gelé de
T12.3b utilisant l'archive préservant la lineage. Il n'autorise pas T12.4.

Un échec est conservé tel quel. Il interdit le rerun du bouclier et conduit à
analyser la première divergence (état latent, non-déterminisme ou signature
d'état insuffisante) sans baisser les seuils ni relancer automatiquement les
mêmes seeds.

