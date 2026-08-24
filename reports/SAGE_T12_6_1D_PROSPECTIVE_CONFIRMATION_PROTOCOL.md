# SAGE.T12.6.1d — Confirmation prospective de la hiérarchie fiable

## Question scientifique

T12.6.1d demande si la règle gelée `exact_span2_range0`, sélectionnée sur les
seules archives source-train 9101–9103, conserve un avantage de classement sur
de nouvelles archives `bp35`. Les archives déjà ouvertes 9201–9203 sont
interdites, tout comme le réentraînement, le recalibrage et la modification des
descripteurs.

Un PASS confirme uniquement une supériorité prospective sur deux lignées
source connues. Il ne démontre ni généralisation vers un autre jeu, ni progrès
ARC-AGI générique, ni capacité de contrôle autonome.

## Liaison et matrice gelées

Le manifeste lie par checksum :

- le manifeste, le reçu de compilation et le bundle de modèles T12.6.1c ;
- le manifeste et le reçu de compilation du collecteur T12.4a.4d.1 ;
- tout le code de collecte, d’extraction, de prédiction et d’adjudication ;
- les seeds prospectives 9301, 9302 et 9303 ;
- les lignées 8701 et 8705 ;
- les bras `local_archive_control`, `diversity_control` et
  `abstract_hazard_diversity`.

La matrice contient 18 archives. Le premier lot contient les six conditions de
la seed 9301. Le second contient les douze conditions 9302–9303 et ne peut être
lancé qu’après un PASS d’intégrité du premier lot. Aucun score, label de portée
future ou classement ne peut être calculé entre les lots.

## Bornes opérationnelles

Chaque archive est limitée à 2 048 appels SDK, 64 excursions et 10 000 cellules.
La matrice entière est limitée à 38 000 appels, 1 Gio d’artefacts, et quatre
heures par lot. Le preflight et chaque phase hors ligne sont limités à 30
minutes. Les frames brutes ne sont pas persistées.

La collecte physique reste manuelle. Le preflight vérifie seulement les
artefacts liés, le bundle strict, les deux témoins de lignée et la présence de
l’environnement ; il effectue zéro appel SDK.

## Engagement aveugle aux labels

Les 18 archives doivent être scellées avant le classement. L’extraction de
prédiction construit alors les unités par `source_exact_hash`, sans calculer la
portée productive. Elle enregistre, pour chaque action :

- le score de `exact_span2_range0` et son niveau de support ;
- le score de la hiérarchie T12.6.1 exact-first ;
- le contrôle d’effet immédiat ;
- le contrôle de permutation de binding.

Cet engagement est signé avant l’ouverture des labels. Toute présence d’un
champ oracle, d’un label de portée ou d’un résultat de hit invalide la phase.

## Oracle exact et intégrité

L’unité de décision est un état exact rejoué possédant au moins deux actions
observées. La portée future est la longueur maximale d’un chemin changé et non
terminal dans le graphe `source_exact_hash → target_exact_hash`, à horizon
quatre après l’action candidate.

Le champ `novel` est ignoré : il dépend de l’ordre d’insertion dans l’archive.
Deux répétitions du même couple état exact/action sont compatibles si leur
transition exacte est identique, même si `novel` ou l’identifiant abstrait de
cellule diffèrent. Des `target_exact_hash` ou résultats de transition
différents constituent un échec d’intégrité.

Les archives de contenu identique dans une même condition seed/lignée sont
scorées une seule fois, tout en conservant la liste de leurs bras d’origine.
Le support global exige par ailleurs au moins douze checksums de contenu
distincts sur les 18 archives.

## Gates préenregistrés

L’adjudication exige simultanément :

- 18 archives complètes, au moins 12 contenus distincts et zéro conflit exact ;
- au moins 250 groupes éligibles ;
- une précision top-1 prospective d’au moins 0,70 ;
- des gains d’au moins 0,10 sur le contrôle immédiat, 0,25 sur le binding-swap
  et 0,02 sur la hiérarchie exact-first ;
- aucune seed sous l’exact-first et au moins deux seeds strictement meilleures ;
- une précision d’au moins 0,65 sur chacune des lignées ;
- une couverture hiérarchique d’au moins 0,70, un sommet unique dans au moins
  0,85 des groupes et une recommandation dans au moins 0,60 des groupes ;
- l’exercice du rejet d’un exact fragile dans au moins 0,25 des groupes ;
- une borne basse à 90 % non négative pour le gain sur exact-first, obtenue par
  10 000 réplications d’un bootstrap bloqué par seed.

Les seuils sont inclusifs : un gain exact de 0,020 passe ce critère, 0,019
échoue.

## Classification et autorité

Les verdicts sont ordonnés et exclusifs : échec d’intégrité, support
insuffisant, absence de supériorité, ou confirmation prospective. Tout échec
conserve les artefacts et ferme T12.6.2. Un PASS autorise seulement le gel d’un
protocole T12.6.2 séparé. Validation source, holdout, contrôle environnemental,
entraînement neuronal et production restent fermés.
