# SAGE.T12.2 — Burst Go-Explore symbolique

## Motivation et filiation

T12.1 a été exécuté proprement sur `bp35` mais a échoué son gate : la
couverture symbolique est 5,85 fois celle de la baseline, les restaurations
exact-prefix sont toutes exactes, mais aucune progression n'a été observée.
Seulement 350 à 538 actions exploratoires ont été exécutées par seed pour
8 192 appels SDK, car chaque restauration n'était suivie que d'une action.

T12.2 est une expérience corrective distincte. Son manifeste doit être lié au
manifeste T12.1 et au receipt signé `FAIL_ARCHIVE_GATE`. T12.1 et ses artefacts
restent immuables.

## Changement testé

Le contrôle réexécute l'archive monoétape T12.1 avec de nouveaux seeds. Le
traitement restaure le même type de cellule exact-prefix, puis exécute une
excursion de 4, 8 ou 16 actions selon un cycle déterministe. Chaque état
intermédiaire et chaque transition sont archivés. L'excursion s'arrête
immédiatement sur :

- une progression ou victoire;
- une terminaison;
- une divergence de replay;
- une action indisponible;
- la limite d'appels SDK.

Aucun réseau, bouclier, programme causal ajouté ou option n'intervient dans
cette comparaison.

## Appariement préenregistré

- Jeu : `bp35`, split `source_train` uniquement.
- Seeds : 6501, 6502 et 6503.
- Bras : `one_step_archive` et `burst_archive`.
- Budget : 8 192 appels SDK par seed et par bras.
- Total maximal du run : 49 152 appels SDK.
- Artefacts : 3 Gio maximum, contrôlés avant chaque écriture.
- Frames brutes : non persistées.
- Archive : 50 000 cellules maximum.

Chaque bras repart d'un environnement et d'une mémoire vides. Les mêmes seeds
et budgets sont utilisés dans chaque paire. Le traitement utilise exactement
le cycle 4/8/16; aucun horizon ne sera sélectionné après observation des
résultats.

## Gate primaire

Le gate T12.2 passe seulement si toutes les conditions suivantes sont vraies :

1. Le ratio agrégé des fractions `actions exploratoires / appels SDK` entre
   burst et monoétape est supérieur ou égal à 2,0.
2. Ce ratio est strictement supérieur à 1 pour chacun des trois seeds.
3. Le gain agrégé de cellules symboliques par 1 000 appels est au moins 25 %.
4. Le bras burst observe au moins une progression.
5. Le taux de terminaisons en échec par action exploratoire du bras burst ne
   dépasse pas 10 %.
6. Toutes les restaurations exact-prefix des deux bras sont exactes.

Une amélioration d'efficacité ou de couverture sans progression reste donc un
résultat négatif. Un échec interdit le bouclier, le MLP et l'extraction
d'option. Un passage autorise seulement la conception de la phase de bouclier
suivante; il n'ouvre ni source-validation, ni holdout, ni autorité active.

## Artefacts et diagnostics

Le run produit :

- une archive par seed et par bras;
- les excursions burst avec horizon demandé, actions exécutées, coût de
  restauration et motif d'arrêt;
- les bundles exact-prefix ayant au moins deux branches observées;
- un rapport apparié;
- un receipt checksummé lié au parent T12.1.

