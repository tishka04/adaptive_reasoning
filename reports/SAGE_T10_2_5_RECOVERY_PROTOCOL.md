# SAGE.T10.2.5 — protocole de récupération après arrêt du watchdog

## Objet

T10.2.5 est une migration d'orchestration append-only de la collecte T10.2.4.
Elle ne modifie ni le noyau scientifique T10.2.2, ni les événements physiques
déjà scellés, ni les caches exacts T10.2.4. Elle traite exclusivement l'arrêt
brut survenu dans la lane de confirmation `su15`, seed `111`.

Le reçu de migration doit authentifier avant toute nouvelle action :

- 15 lanes et 60 resets complets ;
- le checkpoint complet et le curseur compact présents ;
- l'invocation parent `OPEN` sans reçu terminal ;
- deux resets complets dans la lane ouverte ;
- un reset orphelin `capacity_matched_independent` avec exactement 10 intentions,
  10 événements scellés et 10 mises à jour, sans intention inconnue ou non
  résolue ;
- le quatrième reset encore vierge ;
- les trois caches T10.2.4 finalisés.

## Réparation du curseur

Le crash est intervenu après la dernière écriture du curseur compact. Les
événements sont autoritatifs dans le journal ; le curseur n'est qu'un dérivé.
T10.2.5 enregistre dans son manifeste le seul curseur réparé admissible, calculé
par reconstruction complète du journal. La réparation avance la révision,
actualise les digests de comptabilité et ne modifie aucun enregistrement
physique.

## Fermeture de la lane orpheline

Le reset partiel reçoit un rapport `ABORTED` avec la raison
`interrupted_before_reset_commit`. Le reset suivant, qui n'a émis aucune action,
est fermé `ABORTED` avec la raison locale `parent_interrupted`. La lane entière
est exclue de l'ajustement scientifique. Ses événements restent conservés pour
l'audit et aucune action n'est rejouée.

## Remplacement borné

Trois seeds impairs de récupération sont dérivés de façon déterministe du
checkpoint, du curseur et du digest du reset orphelin. Chaque seed définit une
lane de confirmation complète de quatre resets avec l'ordre de contrôleurs
original : indépendant, appris, indépendant, appris.

Les tentatives utilisent un journal séparé et append-only. La première lane
ayant quatre resets complets remplace logiquement la lane exclue. Une tentative
incomplète reste auditée et n'entre jamais dans le modèle. Le plafond global est
de trois lanes, douze resets et 768 actions autorisées.

## Watchdog corrigé

Le garde dur externe est armé seulement après la création du worker. Sa cible
est le PID du worker et son arbre de descendants. Il lui est interdit de cibler
le PID du collecteur. Le collecteur reste ainsi disponible pour sceller un
timeout, écrire le rapport de reset et poursuivre avec une tentative suivante.

## Gate d'acquisition T10.2.5

Le rapport accepté exige :

- 18 lanes logiques et 72 resets complets ;
- les 17 lanes parent non orphelines complètes ;
- une lane de récupération complète ;
- exclusion totale des événements de la lane orpheline et des tentatives de
  récupération échouées ;
- comptabilité exacte de toutes les actions tentées, y compris les exclusions ;
- neuf unités de confirmation et une isolation stricte des deux donneurs ;
- zéro rejeu physique ;
- holdout, AR25, validation et autorité de production fermés.

La sortie positive exacte est `T10_2_5_SOURCE_COLLECTION_COMPLETE`. Toute autre
sortie reste négative et n'autorise aucune validation.

