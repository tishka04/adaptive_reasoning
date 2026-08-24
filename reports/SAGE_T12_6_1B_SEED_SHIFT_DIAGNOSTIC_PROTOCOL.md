# SAGE.T12.6.1b — Diagnostic du décrochage inter-seed

## Statut scientifique

T12.6.1a a montré qu’aucune des six consolidations enregistrées ne sauve le
transfert T12.6.1. Le contraste restant est fortement localisé : avec la règle
`parent_order`, le modèle atteint 0,8402 sur 9201, 0,3913 sur 9202 et 0,5810
sur 9203.

T12.6.1b est une autopsie post-hoc de 9202. Les trois seeds d’évaluation sont
déjà ouverts. Le diagnostic ne répare aucun reçu et ne fournit aucune preuve
confirmatoire.

## Question

Le décrochage 9202 provient-il principalement :

- d’un manque de support dans la hiérarchie exacte/composition/famille ;
- d’une hétérogénéité des labels dans le support d’entraînement ;
- d’une inversion stable entre le rang appris et la portée productive ;
- d’une dépendance à un seed d’entraînement, une lignée ou un bras d’archive ?

## Axes gelés

Avant inspection des erreurs individuelles 9202, six axes sont fixés :

1. attribution du niveau de support sélectionné ;
2. hétérogénéité des labels du support d’entraînement ;
3. concordance paire-à-paire entre scores et portées observées ;
4. contraste de 9202 avec le pool de référence 9201+9203 ;
5. sensibilité leave-one-training-seed-out sur 9101, 9102 et 9103 ;
6. stratification par lignée et bras d’archive.

Le modèle futur, le rayon sept, le support minimal deux, les signatures et la
règle de consolidation `parent_order` sont inchangés. Les trois sensibilités
leave-one-out réutilisent la même architecture et servent uniquement à mesurer
la dépendance aux données ; elles ne sélectionnent pas un nouveau modèle.

## Attribution des erreurs

Pour chaque groupe éligible, le diagnostic conserve l’action choisie, la
meilleure action observée, leur marge de score, le regret de portée, le niveau
de support et la distribution des labels d’entraînement. Une erreur est classée
comme égalité, mauvais classement entre niveaux, hétérogénéité exacte ou de
composition, inversion stable exacte ou de composition, ou échec du repli
familial/global.

La classe dominante est descriptive. Aucun seuil de dominance ne porte une
autorité scientifique.

## Portail d’intégrité

Le diagnostic est complet seulement si :

- les 12 archives d’entraînement et 18 archives d’évaluation sont présentes ;
- les seeds et lignées enregistrés sont complets ;
- l’entraînement contient zéro conflit et les 37 conflits d’évaluation sont
  reproduits ;
- les résultats parent-order globaux par seed sont reproduits ;
- 9202 reproduit exactement 92 groupes et 36 hits ;
- les six axes correspondent au gel ;
- le bundle modèle signé est réutilisé ;
- zéro appel SDK, moins de 600 secondes et moins de 512 Mio.

Un pass signifie seulement que l’attribution post-hoc est complète.

## Frontière

T12.6.1b ne peut autoriser ni changement de descripteur, ni T12.6.2, ni
collecte. Une éventuelle expérience suivante devra être gelée séparément et
utiliser de nouvelles archives pour toute confirmation.
