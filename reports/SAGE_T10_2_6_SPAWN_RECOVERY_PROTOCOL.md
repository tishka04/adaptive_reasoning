# SAGE.T10.2.6 — protocole de récupération du registre Windows spawn

## Objet

T10.2.6 est une migration append-only de l’échec T10.2.5. Les trois workers
T10.2.5 ont quitté avant leur première intention parce que Windows `spawn`
recharge les modules dans un nouveau processus : le registre de seeds ajouté
uniquement dans le collecteur parent n’y existait pas au moment de décoder le
`ResetWorkSpec`.

Le reçu T10.2.6 authentifie exactement :

- le manifeste T10.2.5 figé ;
- le rapport terminal `FAIL_T10_2_5_RECOVERY` ;
- trois lanes avortées sur le reset zéro avec `worker_exited` ;
- zéro intention, événement, mise à jour, action non résolue ou rejeu dans ces
  trois tentatives ;
- le checkpoint parent final contenant 17 lanes complètes, l’unique lane
  orpheline exclue, 70 resets complets et deux resets avortés ;
- l’absence de ledger accepté ou de rapport de collecte T10.2.5.

## Correction de la frontière spawn

Le target de processus T10.2.6 est une fonction de module importable. Dans
l’enfant, elle lit le reçu signé porté par la factory, installe les seeds et la
lane registry T10.2.6, puis seulement appelle le worker scientifique gelé qui
désérialise le travail. Les bindings de cache gauge/factorisé T10.2.4 restent
inchangés.

Le watchdog dur cible exclusivement le PID du worker et ses descendants. Le
collecteur n’est jamais une cible autorisée.

## Remplacement borné

Trois nouveaux seeds impairs sont dérivés de façon déterministe du checkpoint
parent final, du rapport d’échec T10.2.5 et des trois tentatives échouées. Ils
sont distincts des seeds source et T10.2.5. Chaque tentative comporte au plus
quatre resets et 64 actions par reset, pour un plafond de 768 actions.

La première lane complète est acceptée. Toute tentative incomplète reste dans
le journal T10.2.6 mais n’entre jamais dans l’ajustement. Les journaux parent et
T10.2.5 sont strictement en lecture seule.

## Gate d’acquisition

Le succès exige :

- 18 lanes logiques et 72 resets complets ;
- 17 lanes parent complètes et une lane de remplacement complète ;
- exclusion intégrale de la lane parent orpheline ;
- exclusion attestée des trois tentatives T10.2.5 à zéro action ;
- comptabilité exacte des actions tentées et acceptées ;
- neuf unités de confirmation, isolation stricte des donneurs et aucun
  événement held-out utilisé avant fit ;
- zéro rejeu physique ;
- holdout, AR25, validation et autorité de production fermés.

La sortie positive exacte est `T10_2_6_SOURCE_COLLECTION_COMPLETE`. Le CLI
`collect` retourne désormais un code non nul pour tout autre statut.
