# SAGE.T12.6.1d-r1 — Amendement d’intégrité d’exécution

## Motif et classification

La première commande physique T12.6.1d-v1 a exécuté le bras
`9301/8701/local_archive_control`, écrit son archive, puis échoué avec
`KeyError:'cells'`. Le collecteur canonique publie `symbolic_cells` dans
`GoExploreArchive.metrics()` ; l’adaptateur v1 attendait à tort `cells`.

Il s’agit d’un échec d’instrumentation, pas d’un miss scientifique : aucun reçu
pilote, score de modèle, label de portée future ou classement n’a été produit.
Le dossier v1 et son archive partielle restent immuables.

## Artefact avorté lié

Le gel r1 vérifie et lie l’unique archive v1 :

- seed 9301, lignée 8701, bras `local_archive_control` ;
- budget consommé : 2 048 appels SDK ;
- 62 cellules symboliques et 139 transitions ;
- 23 replays exacts sur 23 ;
- aucun `collection_receipt.json`.

Ces informations sont exclusivement des contrôles opérationnels autorisés
avant le second lot. L’archive avortée est exclue de tout scoring, oracle,
support, déduplication et décompte des 18 archives r1.

## Modification autorisée

R1 apporte exactement deux changements :

1. la métrique de budget de cellules est lue sous son nom canonique
   `symbolic_cells` ;
2. les seeds 9301–9303 sont retirées et remplacées par les seeds vierges
   9401–9403, avec 9401 comme nouveau lot pilote.

Le test n’utilise plus un faux dictionnaire contenant `cells`. Il projette les
métriques réelles d’un `GoExploreArchive`, puis la simulation complète vérifie
la matrice 3 × 2 × 3 avec `symbolic_cells`.

## Éléments inchangés

Le bundle `exact_span2_range0`, les deux contrôles, les descripteurs, le rayon,
l’horizon, les lignées, les trois bras, le budget propre à la nouvelle matrice
et tous les seuils scientifiques restent bit-à-bit ou valeur-à-valeur
inchangés. R1 interdit :

- le réentraînement ou recalibrage ;
- l’emploi de 9201–9203 ou 9301–9303 ;
- la consultation scientifique de l’archive avortée ;
- tout score entre les deux lots ;
- l’ouverture du second lot sans reçu pilote r1 passé.

## Nouvelle matrice et autorité

Le lot pilote contient six archives sur 9401. Le lot de complétion contient
douze archives sur 9402–9403. Les mêmes gates de scellement, engagement
label-blind et adjudication s’appliquent ensuite aux 18 nouvelles archives.

Un preflight r1 passé autorise uniquement la collecte manuelle de 9401. Un PASS
final autorise uniquement le gel séparé de T12.6.2. Holdout, validation source,
contrôle environnemental autonome, entraînement neuronal et production restent
fermés.

## Ledger cumulatif

Le budget r1 reste limité à 38 000 appels SDK. Les 2 048 appels déjà consommés
par l’archive v1 avortée sont conservés séparément dans chaque reçu. Le plafond
cumulatif explicite est donc 40 048 appels. Avec la matrice fixe de 18 archives,
la consommation maximale effective serait 38 912 appels, soit 2 048 appels v1
et 36 864 appels r1.

Le plafond r1 d'artefacts reste 1 Gio (1 073 741 824 octets). Les 20 911 530
octets de l'archive v1 sont liés séparément, portant le plafond cumulatif
explicite à 1 094 653 354 octets. Les reçus de lot suivent à la fois les octets
des archives r1 et leur somme avec l'archive avortée ; le scellement recalcule
ces tailles depuis les fichiers liés.
