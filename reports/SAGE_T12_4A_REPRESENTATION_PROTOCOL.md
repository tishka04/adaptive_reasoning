# SAGE.T12.4a — réparation relationnelle et contexte d'archive

## Résultat parent conservé

T12.4 reste un résultat négatif immuable. Son petit MLP a amélioré le Brier
moyen de 0,01258, mais a échoué sur les deux tests qui établissaient que l'état
était réellement utile :

- permutation de l'état : `−0,00317`, pour un seuil `≥ 0,01` ;
- ECE maximale : `0,15839`, pour un seuil `≤ 0,10`.

Le modèle était également moins bon que le prior action-only sur les deux têtes
dans le train. T12.4a ne modifie donc ni les seuils, ni la calibration, ni le
nombre d'époques. Il teste une nouvelle représentation correspondant au défaut
identifié.

Le seed 7703 a été ouvert durant le diagnostic T12.4. Il est explicitement
interdit pour le fit et la validation confirmatoire de T12.4a.

## Hypothèse

La représentation T12.4 ne reliait pas explicitement les coordonnées de
l'action aux entités perçues. Par ailleurs, la cible `novel` dépend de
l'historique de l'archive, absent des entrées du réseau.

T12.4a teste donc :

> un modèle toujours petit, mais recevant des relations action–objet et un
> contexte strictement pré-action de l'archive, doit battre le prior
> action-only et le modèle T12.4, et perdre sa performance lorsque ces deux
> sources sont détruites par contrôle.

## Nouvelle collecte prospective

La collecte utilise uniquement l'archive lineage-preserving avec le bouclier
terminal confirmé, sans réseau dans la politique :

- seeds 8401 et 8402 : train ;
- seed 8403 : validation confirmatoire ;
- 4 096 appels SDK par seed ;
- calendrier de bursts 4/8/16 ;
- 12 288 appels planifiés, plafond global 15 000 ;
- 3 Gio maximum et aucune frame brute.

Le gate de collecte exige un replay exact minimal de 0,95, au moins un veto du
bouclier, au moins 512 exemples train et 200 validation, huit actions uniques,
32 contextes d'archive distincts, 20 % d'exemples possédant des relations
action–objet et des prévalences de labels dans `[0,05 ; 0,95]`.

## Représentation enrichie

Le bloc relationnel contient notamment :

- point d'action ou déplacement projeté ;
- distance et décalage par rapport au joueur ;
- distance à l'entité la plus proche ;
- distances aux rôles target, hazardous et collectible ;
- alignement ligne/colonne ;
- rôle de l'entité la plus proche ;
- présence d'une entité de chaque rôle au point d'action.

Le contexte d'archive est construit avant l'action :

- visites et expansions de la cellule ;
- essais antérieurs de cette action ;
- fraction d'actions déjà testées ;
- taille de l'archive et nombre d'arêtes ;
- taux historiques de changement et de nouveauté pour l'action ;
- mêmes taux à l'échelle cellule/action.

Le compilateur reconstruit ces compteurs dans l'ordre des arêtes et vérifie que
la cible de nouveauté correspond bien à une cellule cible jamais observée avant
l'action. Toute divergence échoue fermée.

## Modèles et contrôles

Deux modèles sont entraînés avec les mêmes seed, largeur, nombre d'époques et
mini-batches :

- modèle legacy T12.4 ;
- modèle relationnel T12.4a, avec deux petites têtes indépendantes.

Les contrôles sont :

1. prior beta-binomial action-only ;
2. modèle legacy réentraîné sur les mêmes nouveaux exemples ;
3. permutation des états, action et contexte conservés ;
4. permutation du contexte d'archive, état et action conservés ;
5. ablation complète des relations action–objet.

## Gate T12.4a

Le gate de représentation passe uniquement si :

1. le modèle reste sous 15 000 paramètres ;
2. le gain Brier de la tête changement contre action-only atteint 0,01 ;
3. le gain Brier de la tête nouveauté contre action-only atteint 0,01 ;
4. le Brier moyen améliore le modèle legacy d'au moins 0,01 ;
5. permuter l'état dégrade la tête changement d'au moins 0,01 ;
6. permuter le contexte dégrade la tête nouveauté d'au moins 0,01 ;
7. retirer les relations dégrade le Brier moyen d'au moins 0,005 ;
8. l'ECE maximale ne dépasse pas 0,10.

Un bon Brier sans dépendance causale aux nouvelles entrées échoue. Un échec est
conservé sans retune.

## Firewalls

Le freeze autorise seulement la collecte prospective contrôle. Un gate de
collecte passé autorise seulement le fit offline. T12.4a n'expose aucune phase
d'évaluation active et n'extrait aucune option.

Seul un gate de représentation passé autorise le freeze de T12.4b, qui devra
preregister séparément l'évaluation active. Holdout, source-validation,
production, T12.5 et l'extraction d'option restent fermés.

