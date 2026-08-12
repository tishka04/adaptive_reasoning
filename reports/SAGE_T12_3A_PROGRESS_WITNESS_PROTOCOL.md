# SAGE.T12.3a — confirmation des témoins de progression

## Question scientifique

T12.2 a échoué son gate global, mais a produit deux observations positives sur
`bp35` : une route `one_step_archive` de seed 6502 et une route
`burst_archive` de seed 6503. Elles partent du même état exact, atteignent le
même état exact au niveau 1 et partagent le suffixe `ACTION3, ACTION3,
ACTION3`.

T12.3a teste uniquement les deux affirmations suivantes :

1. chaque route complète est reproductible depuis un reset exact ;
2. à préfixe rejoué exactement, le suffixe commun produit la progression alors
   que le même suffixe privé de sa dernière action ne la produit pas.

Ce test ne cherche pas à réhabiliter le gate T12.2 et ne permet pas d'attribuer
la progression au Go-Explore par bursts. Il isole un témoin de progression
minimal avant toute nouvelle induction.

## Entrées immuables

Le freeze T12.3a est lié par checksum :

- au manifeste T12.2 ;
- au receipt négatif `FAIL_T12_2_BURST_GATE` ;
- aux archives T12.2 qui contiennent les transitions positives ;
- au registre scellé des deux routes et de tous leurs hashes intermédiaires ;
- au code d'extraction et d'exécution ;
- au commit Git propre.

Le freeze échoue si les archives ne fournissent pas au moins deux routes
distinctes, si leurs états initiaux ou terminaux diffèrent, ou si leur suffixe
commun exact n'est pas `ACTION3 × 3`.

## Plan apparié

Pour chacun des deux témoins, T12.3a exécute trois répétitions indépendantes de
chaque condition :

| Condition | Exécution | Résultat attendu |
|---|---|---|
| `full_route` | reset puis route complète | état cible exact et niveau 1 |
| `common_suffix` | reset, préfixe exact, puis `ACTION3 × 3` | état cible exact et niveau 1 |
| `delete_last_suffix_action` | reset, même préfixe exact, puis `ACTION3 × 2` | état intermédiaire exact, aucune progression |

Chaque branche repart d'un nouvel environnement. Le reset, chaque source de
transition et chaque cible de transition sont comparés au hash enregistré
avant de poursuivre. Une divergence arrête immédiatement la branche concernée.
Les pixels bruts ne sont pas persistés.

## Gate preregistré

Le gate `PASS_T12_3A_WITNESS_GATE` exige simultanément :

- au moins 2 replays complets confirmés sur 3 pour chaque témoin ;
- au moins 2 suffixes complets confirmés sur 3 pour chaque témoin ;
- au moins 2 contrôles par suppression exacts et sans progression sur 3 pour
  chaque témoin ;
- au moins 2 répétitions strictement appariées sur 3 où les deux branches
  retrouvent le même hash de préfixe et confirment le contraste ;
- zéro progression dans tous les contrôles par suppression ;
- tous les resets exacts ;
- un taux global de comparaisons exactes d'au moins 0,99 ;
- au plus 2 048 appels SDK ;
- au plus 3 Gio d'artefacts pour le run.

Une vraisemblance favorable, une ressemblance visuelle ou un gain de couverture
ne remplacent aucun de ces critères.

## Pare-feu

T12.3a reste en `source_train`. Pendant cette phase :

- le holdout et `source_validation` restent fermés ;
- le bouclier terminal n'est ni appris ni appliqué ;
- aucun modèle neural n'est entraîné ;
- aucune option n'est extraite ou compilée ;
- aucune autorité de production n'est accordée.

Un succès autorise seulement la préparation d'un enfant T12.3b séparément
gelé. Un échec conserve le résultat négatif et arrête cette branche.

## Artefacts

Le freeze produit :

- `witnesses.sealed.json` ;
- `manifest.json` ;
- `freeze_receipt.json`.

Le run produit dans un répertoire neuf :

- `replay_trials.json` avec les hashes et décisions par étape ;
- `intervention_bundles.json` regroupant les branches appariées par préfixe ;
- `witness_report.json` ;
- `witness_receipt.json` lié aux checksums du manifeste et du protocole.
