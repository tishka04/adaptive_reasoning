# SAGE.T12.4a.1 — transport prospectif de la calibration

## Résultat parent conservé

Le reçu T12.4a reste immuable et négatif. La représentation relationnelle a
passé tous les critères prédictifs et causaux préenregistrés : gains Brier sur
les deux têtes, amélioration contre le modèle legacy, permutation de l'état,
permutation du contexte et ablation des relations. Le seul échec est la
calibration : ECE changement `0,13026`, ECE nouveauté `0,14587`, pour un seuil
maximal de `0,10`.

T12.4a.1 ne modifie donc ni les labels, ni la représentation, ni le bouclier,
ni la politique de collecte. Il teste prospectivement si une séparation
stricte entre apprentissage, calibration et confirmation rend les probabilités
transportables entre trajectoires.

Les seeds déjà ouvertes `7701–7703` et `8401–8403` sont interdites pour tout
fit, calibrage ou test confirmatoire T12.4a.1.

## Hypothèse

Le biais principal T12.4a vient d'un décalage inter-seeds : la prévalence de
nouveauté était de `0,2683` dans le train, de `0,1543` en validation, tandis que
la probabilité moyenne prédite atteignait `0,3002`. Une seed de validation
unique ne pouvait simultanément servir à diagnostiquer ce décalage et rester
confirmatoire.

Hypothèse préenregistrée :

> un modèle relationnel appris sur plusieurs trajectoires, suivi d'un petit
> calibrateur monotone ajusté sur une trajectoire distincte, doit conserver ses
> contrôles causaux et obtenir des probabilités calibrées sur deux nouvelles
> trajectoires confirmatoires.

## Split prospectif

Six nouvelles seeds source-train sont collectées avec le même contrôle
lineage-preserving et le même bouclier terminal, sans réseau dans la politique :

- apprentissage de la représentation : `8701`, `8702`, `8703` ;
- calibration seulement : `8704` ;
- confirmation seulement : `8705`, `8706`.

Chaque seed reçoit 4 096 appels SDK. Le total planifié est 24 576, avec un
plafond dur de 26 000. Chaque run est limité à 3 Gio et ne persiste aucune frame
brute.

Le gate de collecte exige, sur chaque split, des labels non dégénérés, au moins
huit actions, 32 contextes d'archive, 20 % de couverture relationnelle et les
volumes minimaux suivants : 768 exemples train, 256 calibration et 400
confirmation. Le replay exact minimal reste 0,95 et au moins un veto réel du
bouclier doit être observé.

## Modèle et calibrateur

Le modèle T12.4a est réappris sans changement d'architecture : deux têtes MLP
indépendantes et moins de 15 000 paramètres. Il ne voit que les seeds train.

Le calibrateur applique séparément à chaque logit :

```text
calibrated_logit_h = positive_scale_h * raw_logit_h + bias_h
```

Les deux échelles sont contraintes positives afin de préserver l'ordre des
actions. Le calibrateur contient exactement quatre paramètres et ne voit que
la seed 8704. Aucun gradient du modèle de base ni du calibrateur ne dépend des
seeds 8705–8706.

Le modèle legacy reçoit son propre calibrateur de même forme et la même seed,
afin que la comparaison de représentation reste équitable.

## Contrôles conservés

Sur les deux seeds confirmatoires, T12.4a.1 compare :

1. prior beta-binomial action-only appris sur les seeds train ;
2. modèle legacy T12.4 réappris et calibré sur le même split ;
3. modèle relationnel non calibré ;
4. modèle relationnel calibré ;
5. permutation de l'état avant le modèle calibré ;
6. permutation du contexte d'archive avant le modèle calibré ;
7. ablation des relations action–objet avant le modèle calibré.

Les métriques sont calculées en pool et séparément pour chaque seed
confirmatoire. Les prédictions confirmatoires sont scellées avec leur seed,
leur cible et les comparateurs.

## Gate T12.4a.1

Le gate passe uniquement si :

1. les trois seeds train, la seed calibration et les deux seeds de validation
   sont toutes présentes dans leur split exclusif ;
2. modèle et calibrateur totalisent au plus 15 000 paramètres, dont quatre
   paramètres de calibration au maximum ;
3. les gains Brier changement et nouveauté contre action-only sont chacun au
   moins `0,01` ;
4. le Brier moyen bat le modèle legacy calibré d'au moins `0,01` ;
5. le shuffle état dégrade le Brier changement d'au moins `0,01` ;
6. le shuffle contexte dégrade le Brier nouveauté d'au moins `0,01` ;
7. l'ablation relationnelle dégrade le Brier moyen d'au moins `0,005` ;
8. le calibrage améliore l'ECE maximale d'au moins `0,02` ;
9. il ne dégrade pas le Brier moyen de plus de `0,005` ;
10. l'ECE poolée maximale est au plus `0,10` ;
11. l'ECE maximale de chaque seed confirmatoire est au plus `0,15`.

Un ECE favorable sans les contrôles causaux échoue. Un résultat favorable sur
une seule seed échoue. Tout échec est conservé sans nouvelle calibration sur
les seeds confirmatoires.

## Firewalls

Le freeze autorise uniquement la collecte prospective contrôle. Un gate de
collecte passé autorise uniquement le fit offline et le calibrage scellé.
T12.4a.1 n'expose aucune phase d'évaluation active et aucune extraction
d'option.

Seul `PASS_T12_4A_1_CALIBRATION_GATE` peut autoriser le freeze séparé de
T12.4b. Holdout, source-validation, production, autorité du bouclier, T12.5 et
extraction d'option restent fermés.
