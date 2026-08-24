# SAGE.T12.6.1c — Hiérarchie conditionnée par la fiabilité

## Statut scientifique

T12.6.1b attribue le décrochage 9202 principalement aux signatures exactes
hétérogènes. Cette attribution est post-hoc : elle motive T12.6.1c, mais les
archives 9201–9203 ne peuvent ni sélectionner la nouvelle règle ni la confirmer.

T12.6.1c est donc une phase de développement exclusivement source-train. Elle
réutilise les douze archives 9101–9103, ne charge aucun payload d’évaluation et
n’offre aucune commande d’évaluation ou de collecte.

## Hypothèse opératoire

Une signature exacte n’est utilisée que si son support :

1. contient au moins deux observations ;
2. couvre au moins deux seeds d’entraînement distincts ;
3. présente une étendue de labels productive bornée.

Si l’exact est rejeté, le score revient à la composition locale typée, puis à
la famille d’action et enfin à la moyenne globale. Le descripteur, le rayon
sept et le label de portée productive à horizon quatre restent ceux de
T12.6.1.

## Sélection source-train gelée

Trois candidats sont enregistrés avant la compilation :

- `exact_span2_range0` ;
- `exact_span2_range1` ;
- `exact_span2_range2`.

Chacun est évalué par leave-one-search-seed-out sur 9101, 9102 et 9103. La
sélection est lexicographique : précision de la pire fold, précision micro,
gain sur la hiérarchie exact-first, couverture, puis ordre de priorité gelé.
Cet ordre privilégie la règle la plus stricte en cas d’égalité.

L’exploration source-train ayant précédé le gel a trouvé les trois candidats
ex aequo : 209/270 hits, soit 0,7741, et une pire fold à 0,7634. La règle stricte
est donc sélectionnée sans revendiquer un gain sur T12.6.1. Son intérêt mesuré
à ce stade est une substitution de support fragile sans perte source-train.

## Contrôles et portail

Le gate exige notamment :

- au moins 240 groupes éligibles ;
- précision micro et pire fold au moins 0,75 ;
- précision de chaque lignée dans chaque fold au moins 0,75 ;
- gain au moins 0,10 sur le contrôle immédiat et 0,30 sur le binding-swap ;
- non-infériorité stricte face à la hiérarchie exact-first ;
- couverture hiérarchique au moins 0,80, taux de sommet unique au moins 0,90
  et couverture de recommandation au moins 0,75 ;
- exercice du rejet exact dans au moins 50 % des groupes ;
- conservation d’au moins un support exact fiable dans le modèle full-fit ;
- zéro conflit d’action dans les données d’entraînement, zéro appel SDK, moins
  de 600 secondes et moins de 512 Mio.

Un pass qualifie seulement la règle pour le gel séparé d’un protocole prospectif
sur de nouvelles archives. Un échec conserve les artefacts et interdit cette
étape.

## Frontière d’autorité

Même en cas de pass, T12.6.1c n’établit ni transfert, ni progrès ARC-AGI, ni
contrôle environnemental. Il n’autorise pas la collecte physique. Toute
confirmation doit utiliser de nouvelles archives choisies et gelées dans une
phase ultérieure ; 9201–9203 restent définitivement post-hoc pour cette règle.
