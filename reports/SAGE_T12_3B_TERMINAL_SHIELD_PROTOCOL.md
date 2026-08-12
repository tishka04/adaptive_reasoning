# SAGE.T12.3b — bouclier terminal multiétape

## Question scientifique

T12.3a a confirmé deux routes de progression sur `bp35`, ainsi que la nécessité
locale de leur dernier `ACTION3`. T12.3b teste maintenant une seule extension :
un bouclier symbolique peut-il réduire les échecs terminaux pendant
l'exploration par bursts sans détruire la couverture ni les progressions déjà
observées ?

Cette phase n'entraîne aucun réseau et n'extrait aucune option.

## Ascendance et données

Le freeze T12.3b est lié par checksum :

- au manifeste T12.3a ;
- au receipt `PASS_T12_3A_WITNESS_GATE` ;
- au registre des deux témoins ;
- au manifeste et au receipt T12.2 utilisés comme source des traces
  terminales ;
- au code du protocole, du bouclier et de l'expérience ;
- au commit Git propre.

Le run reste limité à `source_train` sur `bp35`.

## Registre terminal preregistré

Pour chaque couple parmi trois seeds T12.2 et deux types d'archive
(`one_step_archive`, `burst_archive`), le freeze sélectionne déterministement
les deux routes terminales les plus courtes ayant une paire état/action finale
distincte et une longueur maximale de 64 actions.

Le registre contient donc exactement 12 candidats équilibrés :

- 2 candidats × 3 seeds × 2 types d'archive ;
- tous les hashes intermédiaires ;
- toutes les actions concrètes ;
- la provenance et le checksum de l'archive.

Chaque candidat est rejoué physiquement une fois. L'observation T12.2 et le
replay exact constituent les deux observations requises par le bouclier. Seules
les traces dont tous les hashes et la terminalité échouée sont reproduits sont
utilisées.

Le gate exige au moins 8 confirmations sur 12, un taux de confirmation d'au
moins 2/3 et au moins une confirmation dans chacun des six groupes source.

## Protection des progressions

Les deux routes T12.3a sont recompilées en 99 couples symboliques
`(cell_id, action_key)` protégés. Une action appartenant à une route de
progression confirmée reste toujours autorisée, même si une trace terminale
propagée lui attribue aussi un risque.

Après construction du bouclier, chaque témoin est rejoué trois fois avec le
bouclier actif. Les six replays doivent :

- être exacts jusqu'au hash cible ;
- augmenter le niveau ;
- retrouver chaque action dans la liste protégée ;
- ne subir aucun veto.

## Évaluation prospective appariée

Trois seeds neufs et preregistrés sont utilisés : `6801`, `6802`, `6803`.
Chaque seed exécute deux bras indépendants avec le même budget :

| Bras | Politique |
|---|---|
| `burst_control` | archive symbolique par bursts sans bouclier |
| `burst_terminal_shield` | même archive et même schedule avec bouclier gelé |

Le schedule reste `4/8/16`. Chaque bras reçoit au plus 4 096 appels SDK.
L'ordre de sélection reste déterministe par seed ; le bouclier ne fait que
retirer les actions confirmées dangereuses.

## Gate preregistré

`PASS_T12_3B_TERMINAL_SHIELD_GATE` exige simultanément :

- les confirmations terminales décrites plus haut ;
- au moins un risque multiétape de distance supérieure à 1 ;
- au moins une action confirmée dangereuse et au moins un veto prospectif ;
- 6/6 replays témoins exacts, progressifs, protégés et non vetoés ;
- un taux terminal agrégé du traitement inférieur ou égal à 90 % de celui du
  contrôle ;
- au plus une seed présentant une régression terminale ;
- zéro seed présentant une régression du nombre de progressions ;
- un nombre agrégé de progressions au moins égal au contrôle ;
- une couverture symbolique par appel SDK au moins égale à 80 % du contrôle,
  agrégée et sur chaque seed ;
- un taux de replay exact d'au moins 0,95 dans tous les bras ;
- au plus 30 000 appels SDK au total ;
- au plus 3 Gio d'artefacts.

Un gate manqué produit un résultat négatif et ferme la suite. Les seuils ne
sont pas retouchés après le run.

## Pare-feu

Pendant T12.3b :

- holdout et `source_validation` restent fermés ;
- le bouclier n'a aucune autorité de production ;
- l'entraînement neural reste interdit ;
- l'extraction et la compilation d'options restent interdites.

Un succès autorise uniquement la préparation preregistrée de T12.4, le petit
prédicteur neural `action → changement/nouveauté`. Il ne constitue pas une
preuve de transfert inter-niveaux.

## Artefacts

Le freeze produit :

- `terminal_candidates.sealed.json` ;
- `manifest.json` ;
- `freeze_receipt.json`.

Le run produit :

- `terminal_confirmations.json` ;
- `terminal_shield.json` ;
- `witness_non_regression.json` ;
- `paired_evaluation.json` ;
- les archives et excursions de chaque bras ;
- `shield_report.json` ;
- `shield_receipt.json`.

Le status vérifie aussi les checksums imbriqués des archives prospectives.

