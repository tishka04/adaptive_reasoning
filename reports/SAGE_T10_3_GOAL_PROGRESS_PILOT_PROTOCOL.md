# SAGE.T10.3 — pilot causal de progression orienté objectif

## Statut et portée

T10.3 est un protocole source borné, indépendant des artefacts T10.2.* figés. Il part du terminal T10.2.9 `a94d09c5…`, de sa QA `eff6b2d6…` et du rapport positif T10.0b `a72bff60…`. Les 1 370 événements historiques sont authentifiés et audités en lecture seule, mais ne sont ni relabellisés ni admis dans le fit.

Le protocole s’arrête avant toute validation T10.1, AR25, holdout ou autorité de production. Un résultat positif autorise seulement la préparation d’un protocole de validation séparé.

## Matrice physique gelée

- Source : `bp35`, `lp85`, `su15`.
- Panel : graines 3101–3104, quatre bras indépendants, au plus 16 actions par reset, soit 48 resets et 768 actions au maximum.
- Confirmation : graines 3111–3112, contrôleurs learned et capacity-matched-independent, ordre contrebalancé, soit 12 resets et 192 actions au maximum.
- Budget terminal : 60 resets et 960 actions au maximum.

Les bras `lp85` et `su15` sont : option canonique re-groundée, binding swap, intervention d’option et contrôle indépendant à capacité identique. `bp35` utilise exploration équilibrée, smoke actor-root, permutation de binding disponible et contrôle indépendant. Une impossibilité de grounding produit `CONTROL_GROUNDING_MISS` ou `OPTION_GROUNDING_MISS` ; elle n’autorise ni padding, ni remplacement de graine, ni substitution adaptative.

## Journal durable

Chaque action suit une transaction irréversible : intent write-once avant l’appel physique, événement structurel scellé immédiatement après, puis reçu de branche write-once à la fermeture du reset. Un intent sans événement est marqué explicitement non résolu et n’est jamais rejoué. Le checkpoint vérifie l’équation `intents = événements scellés + intents non résolus`, le plafond de 960 actions et zéro replay physique.

Le label `goal_reachable_within_option` est ajouté seulement lors de `compile`, à partir du reçu de la même branche. Aucun label ne traverse un reset. Les événements physiques ne sont jamais réécrits.

## Représentation

L’ancrage dépend de la famille d’action. `ACTION1–4` choisit d’abord l’acteur/joueur unique. Une action paramétrée exige un binding explicite ou une ancre spatiale transitoire unique. Les autres interactions acceptent seulement `selected`, `action_root` ou une cible structurelle unique. La racine après action est reconnue uniquement par correspondance exacte et unique de la signature structurelle pré-action ; les effets observés ne servent jamais à l’inférer.

Les fichiers persistants ne gardent que la méthode de binding, un hash de signature structurelle et la preuve d’unicité. Coordonnées, couleurs, grilles, identifiants d’entités et actions groundées historiques sont interdits.

## Gates

Le fit est interdit avant passage de la QA : correspondance confiante ≥ 90 %, ambiguïté < 10 %, au moins 50 % de préfixes non terminaux avec deux frames complètes et un transport exact non-identitaire, transports comparables commutatifs et round-trip exacts, prévalence de la cible entre 0,5 % et 95 %, au moins 32 positifs dans deux jeux, et reproduction sans incident des options positives sur les quatre graines de `lp85` et `su15`.

Le fit leave-one-game-out utilise uniquement des features structurelles. Il matérialise des `JointProgramHypothesis`, des automates mixtes et cinq marginales factorisées. Les gates sont AUROC ≥ 0,75, amélioration Brier positive, rang top 8 et médian ≤ 4, marges positives contre binding swap et intervention, dégradation sans transport et probe d’identité limité.

La confirmation exige au moins un niveau sur `lp85` et `su15`, aucun recul sur `bp35`, un avantage total d’au moins un niveau, zéro erreur/action illégale et un taux de `GAME_OVER` non supérieur.

Les verdicts sont exclusifs : `PROVENANCE_INVALID`, `ROOTING_MISS`, `WITNESS_REPRODUCTION_MISS`, `QA_MISS`, `CAUSAL_SEMANTICS_MISS`, `OPTION_INDUCTION_MISS`, `SOURCE_CONFIRMATION_MISS` ou `PASS_T10_3_SOURCE_PILOT`. Aucun échec ne déclenche un retuning.

