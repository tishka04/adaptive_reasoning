# SAGE.T12.6.1 — Viabilité future hiérarchique

## Statut scientifique

T12.6 reste un résultat négatif signé. T12.6a a montré que la signature locale
exacte est correcte dans 16 groupes sur 17 lorsqu’elle est sélectionnée, mais
que le repli par famille d’action ne réussit que 14 groupes sur 26. Sept des
treize erreurs focales sont des égalités entre différentes coordonnées
`ACTION6`; six sont des mauvais classements impliquant le repli.

T12.6.1 est une nouvelle version, pas une modification du reçu T12.6. Son
développement est post-hoc sur 9101–9103. Seule une évaluation ultérieure sur
les archives chronologiquement postérieures et toujours scellées 9201–9203
peut fournir une preuve confirmatoire de transfert.

## Hypothèse

> Une hiérarchie qui utilise d’abord la géométrie locale exacte, puis une
> composition locale typée sans offsets, et seulement ensuite la famille
> d’action, augmente la couverture transportable et réduit le pire fold sans
> perdre matériellement la précision moyenne.

## Représentation gelée

Le label de portée productive, l’horizon quatre, le rayon sept et le support
minimal deux restent exactement ceux de T12.6.

La hiérarchie contient trois niveaux :

1. **signature locale exacte** de T12.6 : attributs, rôles et offsets relatifs
   exacts autour de la cible ;
2. **composition locale typée** : multiensemble trié des couples
   `(area, aspect, roles)` présents dans le même rayon, sans offsets ni
   coordonnées absolues ;
3. **repli familial** : nom d’action et présence de coordonnées, identique au
   repli de T12.6.

Le premier niveau supporté est utilisé. Les identifiants, hashes, coordonnées
absolues, seed, lignée, bras, résultats et labels sont exclus des signatures.

## Contrôles

Quatre classements sont évalués sur les mêmes groupes et labels :

- modèle hiérarchique de portée productive ;
- modèle hiérarchique de l’effet immédiat, de capacité identique ;
- permutation circulaire des scores du modèle futur, qui casse le binding ;
- incumbent T12.6, limité à signature exacte puis famille d’action.

Une recommandation est dite couverte lorsque le score maximal est unique et
que l’action sélectionnée est supportée par la signature exacte ou la
composition locale. Cette mesure est descriptive ; le top-1 principal compte
tous les groupes et conserve le départage déterministe pour comparabilité.

## Exploration de développement avant gel

Les seules archives 9101–9103 ont servi à comparer des abstractions fixées a
priori : secteurs directionnels, bandes radiales, profils par axe et
composition typée. La hiérarchie avec composition typée a produit :

- top-1 micro : 0,7741 contre 0,7852 pour l’incumbent ;
- couverture hiérarchique : 0,8444 contre 0,5519 ;
- taux de maximum unique : 0,9333 ;
- précision du pire fold : 0,7634 contre 0,7093 ;
- chaque couple fold/lignée : au moins 0,7609 ;
- gain micro sur le contrôle immédiat : 0,1815 ;
- gain micro sur la permutation : 0,5519.

Ces valeurs ont fixé les portails de développement. Elles ne sont pas une
preuve indépendante. Aucun contenu des archives 9201–9203 n’a été chargé.

## Portail de compilation

L’intégrité exige les 12 conditions d’archive, les trois seeds, les deux
lignées, zéro conflit, zéro SDK, moins d’une heure et moins de 3 Gio.

Les critères scientifiques sont :

- au moins 240 groupes éligibles ;
- top-1 futur au moins 0,75 ;
- gains d’au moins 0,15 sur l’immédiat et 0,30 sur la permutation ;
- couverture hiérarchique au moins 0,80 et gain de couverture d’au moins 0,25
  sur l’incumbent ;
- maximum unique dans au moins 0,90 des groupes ;
- couverture des recommandations au moins 0,75 ;
- précision au moins 0,75 dans chaque couple fold/lignée ;
- avantage strict sur les deux contrôles dans chaque fold ;
- non-infériorité micro à l’incumbent avec marge -0,02 ;
- gain d’au moins 0,04 sur la précision du pire fold de l’incumbent.

Un échec interdit l’ouverture de 9201–9203.

## Portail d’évaluation scellé

Les modèles entraînés sur les douze archives 9101–9103 sont chargés sans mise
à jour. L’évaluation exige :

- au moins 250 groupes éligibles ;
- top-1 futur au moins 0,70 ;
- gains d’au moins 0,10 sur l’immédiat, 0,25 sur la permutation et 0,02 sur
  l’incumbent ;
- couverture hiérarchique au moins 0,70 et gain de couverture d’au moins 0,15
  sur l’incumbent ;
- maximum unique au moins 0,85 et couverture de recommandation au moins 0,60 ;
- précision d’au moins 0,65 dans chaque lignée ;
- avantage strict sur l’immédiat et la permutation dans chaque seed ;
- aucune perte par seed contre l’incumbent et victoire stricte sur au moins
  deux des trois seeds.

## Résultats exclusifs et autorité

Les statuts sont séparés entre échec d’intégrité, support insuffisant,
identification/transfert insuffisant et succès. Tout échec est signé et interdit
le rerun de même version, le changement de seuil, le retrait d’archive ou
l’ouverture manuelle de la phase suivante.

Même un succès final établirait seulement le transfert chronologique d’un
classement de portée productive dans les archives `bp35`. Il pourrait
autoriser uniquement la préparation d’un protocole T12.6.2 séparé. Collecte
physique, validation, holdout, contrôle autonome, réseau neuronal et production
restent fermés.
