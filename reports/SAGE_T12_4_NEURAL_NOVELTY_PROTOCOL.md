# SAGE.T12.4 — prédicteur neural action → changement/nouveauté

## Question scientifique

T12.3e a validé l'archive lineage-preserving et le bouclier terminal sur le
protocole preregistré. T12.4 teste désormais une seule addition : un petit MLP
recevant l'état symbolique causal courant et l'action candidate peut-il mieux
prévoir le changement et la nouveauté, puis améliorer la couverture active sans
dégrader sécurité, progression ou replay ?

Le réseau n'est ni un extracteur visuel ni un world model. Il ne reçoit que les
faits, entités, relations et compteurs déjà produits par SAGE, plus le nom et
les paramètres de l'action. Le bouclier terminal reste lexicographiquement
prioritaire.

## Parent et données autorisées

Le parent direct doit être le receipt signé
`PASS_T12_3E_LINEAGE_SHIELD_GATE`. Le corpus provient uniquement du bras
`lineage_control` :

- seeds 7701 et 7702 pour l'entraînement ;
- seed 7703, jamais utilisé pour ajuster le modèle, pour la validation ;
- 1 232 exemples d'entraînement et 236 exemples de validation observés avant
  le freeze.

Les archives, le dataset compilé, le bouclier T12.3b, le parent, le protocole et
le code sont liés par checksum. Aucune frame brute n'est copiée.

## Amendement obligatoire des labels

L'audit preregistré a montré que l'ancien label `edge.changed`, défini par une
différence de hash pixel-exact, vaut 1 pour 1 468/1 468 transitions. Il est donc
universel, inutilisable pour apprendre et explicitement rejeté avant fit.

T12.4 le remplace par :

```text
semantic_changed =
    AbstractState.signature(before) != AbstractState.signature(after)
```

La cible `novel` reste : première observation de la cellule symbolique cible
dans l'archive courante. Les métriques du corpus gelé sont :

| Split | Exemples | semantic_changed | novel | Actions uniques |
|---|---:|---:|---:|---:|
| Train 7701–7702 | 1 232 | 0,4968 | 0,2825 | 33 |
| Validation 7703 | 236 | 0,7966 | 0,6102 | 19 |

Seuls 5 des 112 états sources de validation apparaissent dans l'entraînement
(4,46 %), et seulement 4 des 225 couples état/action de validation (1,78 %).
La validation mesure donc principalement une généralisation à de nouveaux
contextes du même jeu, pas la simple mémorisation des couples d'entraînement.

Le freeze échoue si l'une des deux cibles sort de `[0,05 ; 0,95]`, si le
support d'actions devient insuffisant ou si l'identité des exemples se répète.

## Modèle et entraînement

- MLP CPU à deux couches cachées de largeur 32 ;
- deux sorties logistiques : changement sémantique et nouveauté ;
- moins de 15 000 paramètres ;
- seed Torch 8124 ;
- 8 époques ;
- mini-batches déterministes de 32 ;
- AdamW, taux `1e-3`, clipping du gradient à 1 ;
- aucun update pendant l'évaluation active.

Le contrôle principal est un prior beta-binomial par action. Une ablation
supplémentaire décale les états de validation tout en conservant les actions.

## Gate offline

Le fit passe uniquement si :

1. les tailles, prévalences et supports gelés restent valides ;
2. le modèle contient au plus 15 000 paramètres ;
3. son Brier moyen améliore le prior action-only d'au moins 0,01 ;
4. permuter les états dégrade le Brier d'au moins 0,01, afin de montrer que le
   modèle utilise réellement l'état et pas seulement l'action ;
5. l'ECE maximale des deux têtes ne dépasse pas 0,10.

Un échec interdit l'évaluation active. Les seuils ne sont pas retouchés.

## Évaluation active appariée

Sur les seeds prospectifs 8101, 8102 et 8103 :

- `lineage_shield_control` utilise l'archive corrigée et le bouclier ;
- `lineage_shield_neural` utilise exactement les mêmes composants et le modèle
  gelé pour départager uniquement les actions autorisées par le bouclier.

Chaque bras reçoit 4 096 appels SDK et le calendrier 4/8/16. Le plafond total
est 30 000 appels SDK et 3 Gio sans frames brutes.

Le gate actif exige :

1. au moins 10 % de couverture agrégée supplémentaire ;
2. au moins 80 % de la couverture du contrôle sur chaque seed ;
3. aucune hausse du taux terminal, agrégée ou par seed ;
4. aucune perte de progression, agrégée ou par seed ;
5. un replay exact minimal de 0,95 ;
6. au moins une décision réellement modifiée par le score neural ;
7. une latence p95 au plus égale à 50 ms ;
8. au moins un veto du bouclier et le respect des budgets.

Un bon score offline sans gain de couverture ne passe pas.

## Firewalls

Le freeze autorise seulement le fit offline. Le fit passé autorise seulement
l'évaluation active source-train. Holdout, source-validation, production,
promotion du bouclier et extraction d'option restent fermés. Seul le passage du
gate actif autorise le freeze de T12.5, consacré à l'extraction de l'option
minimale ; il ne l'extrait pas automatiquement.
