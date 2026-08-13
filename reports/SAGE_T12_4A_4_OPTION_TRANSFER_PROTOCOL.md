# SAGE.T12.4a.4 — transfert multi-niveau de l'option causale

## Hypothèse

T12.4a.3 a identifié, par 390 branches exact-prefix, une option minimale commune
aux deux contextes confirmés de `bp35` :

```text
ACTION4, ACTION4, ACTION4, ACTION3, ACTION3
```

Cette option a ensuite été compilée dans les quatre programmes causaux complets
du posterior, avec une masse propriétaire pratiquement égale à 1. T12.4a.4
teste si la même séquence produit encore une transition de niveau depuis les
états suivants, sans réapprentissage, réparation ou recalibration.

Le résultat visé est un transfert causal local à plusieurs niveaux. Il ne vaut
ni généralisation inter-jeux, ni autorité de production.

## Ancestralité immuable

Le freeze lie par SHA-256 :

- le manifeste T12.4a.3r1 ;
- `PASS_T12_4A_3_OPTION_ABLATION_GATE` ;
- `PASS_T12_4A_3_SHADOW_COMPILE_GATE` et son lien vers le reçu d'ablation ;
- l'option minimale et son checksum ;
- les quatre programmes enfants et leur registre compilé ;
- le snapshot du posterior et sa masse propriétaire ;
- les deux routes confirmées issues des seeds `8701` et `8705` ;
- le code d'exécution et le commit Git propre.

Le run est limité à `source_train`. Holdout et source-validation restent
fermés.

## Entrée exacte et progression prospective

Les deux routes historiques, longues de 64 et 61 actions, conduisent au même
hash exact du niveau 1. Chaque branche T12.4a.4 repart de cet état par replay
exact de l'une de ces routes.

Lorsqu'une option complète progresse, sa trace action par action et son hash
cible deviennent le préfixe prospectif du niveau suivant. Les répétitions
suivantes doivent reproduire exactement cette trace avant qu'un nouveau niveau
soit testé. Aucune restauration interne de l'environnement n'est utilisée.

## Branches appariées

À chaque niveau, cinq branches sont exécutées quatre fois. La programmation des
lignées est toujours `8701, 8705, 8701, 8705` :

| Branche | Actions | Rôle |
|---|---|---|
| `option_full` | `4,4,4,3,3` | traitement |
| `delete_action4` | `4,4,3,3` | suppression d'un type `ACTION4` |
| `delete_action3` | `4,4,4,3` | suppression d'un type `ACTION3` |
| `reverse` | `3,3,4,4,4` | contrôle d'ordre et de multiensemble |
| `null` | aucune | stabilité du préfixe |

Un nouvel environnement est créé pour chaque branche. Le posterior reste
chargé uniquement comme preuve de propriété shadow ; il ne sélectionne aucune
action et n'est pas mis à jour.

## Gate par niveau

Un niveau est confirmé seulement si :

1. les 20 préfixes sont exacts ;
2. les deux lignées sont représentées deux fois dans chaque branche ;
3. l'option complète progresse quatre fois sur quatre ;
4. la progression est de exactement un niveau et survient sur la cinquième
   action ;
5. les quatre répétitions atteignent la même cible et la même trace exacte ;
6. aucun contrôle ne progresse ;
7. le contrôle nul conserve le hash d'entrée ;
8. aucune branche ne produit d'échec terminal.

Le niveau suivant n'est ouvert que si le niveau courant passe entièrement.

## Gate global

Le protocole tente au maximum trois transitions après le niveau 1. Le statut
`PASS_T12_4A_4_OPTION_TRANSFER_GATE` exige au moins deux niveaux transférés,
tous les préfixes exécutés exacts, zéro échec terminal, un appariement strict
des lignées et le respect du budget.

Le troisième niveau est une extension prospective. S'il ne passe pas après
deux niveaux confirmés, le gate global reste positif pour une portée démontrée
de deux niveaux, et la limite est enregistrée explicitement. Toute divergence
de préfixe ou tout échec terminal fait néanmoins échouer le gate global.

## Bornes

- au plus 60 branches physiques ;
- au plus 4 500 appels SDK ;
- au plus 3 Gio d'artefacts ;
- aucune frame brute persistée ;
- arrêt immédiat de la profondeur après le premier niveau non confirmé.

## Firewalls

T12.4a.4 n'accorde aucune autorité active à l'option. Même après un pass :

- `option_control_authorized` reste faux ;
- holdout, source-validation et production restent fermés ;
- T12.4b et T12.5 restent fermés ;
- aucun entraînement neural n'est autorisé.

Un pass autorise uniquement le freeze séparé d'un futur T12.4a.5 consacré à
une expérience appariée de contrôle par option dans le posterior commun.

