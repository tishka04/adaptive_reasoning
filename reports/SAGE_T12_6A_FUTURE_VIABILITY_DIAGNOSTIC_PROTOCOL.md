# SAGE.T12.6a — Diagnostic du défaut de viabilité future

## Statut scientifique

T12.6 est un résultat négatif signé. Son modèle de viabilité future atteint
0,7852 en micro-moyenne et bat les deux contrôles, mais la lignée 8701 du fold
9103 obtient 30 succès sur 43, soit 0,6977, sous le seuil gelé de 0,70. Le seul
check en échec est `every_compile_lineage_accuracy_sufficient`.

T12.6a ne modifie ni le modèle, ni les seuils, ni le reçu T12.6. Il s’agit
d’un diagnostic post-hoc explicatif sur les douze archives d’entraînement
9101–9103. Il ne produit aucune nouvelle preuve confirmatoire.

## Question

> Les treize erreurs du fold 9103/lignée 8701 sont-elles principalement dues
> à un manque de support, à des égalités de score, à l’hétérogénéité du
> descripteur local, ou à une dépendance de lignée ou de politique d’archive ?

## Entrées gelées

Le gel exige exactement le reçu parent
`FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE`, avec un unique check faux :
`every_compile_lineage_accuracy_sufficient`. Le manifeste, le reçu, le
cross-fit, les modèles, le rapport et les douze archives d’entraînement sont
liés par leurs empreintes.

Les dix-huit archives 9201–9203 ne figurent pas dans les entrées de T12.6a et
leurs contenus ne sont jamais chargés. Le corpus d’évaluation T12.6 reste
scellé.

## Axes définis avant inspection des erreurs

Pour chacun des 43 groupes éligibles du fold focal, l’audit enregistre :

- le niveau de support sélectionné : signature locale, famille d’action ou
  repli global ;
- la présence d’une égalité au meilleur score et la présence éventuelle d’une
  action optimale dans cette égalité ;
- l’hétérogénéité des labels historiques portés par les signatures de
  l’action sélectionnée et de la meilleure action observée ;
- la composition du support par lignée et par bras d’archive ;
- le regret en portée productive et les familles d’actions concernées.

Chaque erreur reçoit exactement une catégorie, dans cet ordre logique :

1. égalité interne à une même signature ;
2. égalité entre signatures distinctes ;
3. erreur impliquant un repli faute de signature supportée ;
4. mauvais classement entre signatures exactes aux labels hétérogènes ;
5. mauvais classement entre signatures exactes historiquement stables.

La catégorie la plus fréquente est rapportée comme diagnostic dominant. Ce
classement est descriptif et ne constitue pas un portail de performance.

## Sensibilités exploratoires gelées

Trois modèles contrefactuels utilisent le même descripteur, le même label et
le même support minimal que T12.6 :

- entraînement sur la seule lignée 8701 ;
- entraînement sur la seule lignée de référence 8705 ;
- entraînement séparé par bras d’archive.

Une borne supérieure mesure également combien d’erreurs disparaîtraient si
une action optimale était choisie à l’intérieur d’une égalité de meilleur
score. Cette borne utilise le label observé et n’est donc pas une politique
exécutable. Aucune de ces sensibilités ne répare ou ne revalide T12.6.

## Portail d’intégrité

Le diagnostic est complet uniquement si :

- les douze conditions d’archive, les trois seeds et les deux lignées sont
  présentes sans conflit d’action ;
- le résultat focal 30/43 est reproduit exactement ;
- les axes calculés correspondent aux axes gelés ;
- zéro appel SDK et zéro lecture de contenu d’archive d’évaluation sont
  effectués ;
- la durée reste sous dix minutes et les nouveaux artefacts sous 512 Mio.

Un succès d’intégrité produit `PASS_T12_6A_DIAGNOSTIC_COMPLETE`. Il signifie
uniquement que l’autopsie est reproductible. Un échec produit
`FAIL_T12_6A_DIAGNOSTIC_INTEGRITY_GATE`.

## Autorité et arrêt

T12.6a ne peut autoriser ni l’évaluation T12.6, ni une collecte physique, ni
un nouveau gel expérimental. Il ne change aucune conclusion ARC-AGI. Toute
hypothèse issue du diagnostic devra faire l’objet d’un protocole futur séparé,
préenregistré et approuvé.
