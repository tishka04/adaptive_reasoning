# SAGE.T11.1 — posterior action-local et plafond dur de 3 Gio

Statut : préenregistré avant l’exécution SAGE.T11.1.

## Motivation et résultat précédent

SAGE.T11 v1 a passé le gate d’exact-prefix replay mais a échoué au gate actif
sur `bp35` : zéro progression et aucune supériorité du posterior complet. Les
quatre particules ont convergé après la première preuve vers deux hypothèses de
persistance indépendantes de l’action. A40 a en outre écrit environ 11,4 Go en
répétant les états relationnels complets.

Ce résultat est un échec scientifique de v1 et sert uniquement de donnée de
source-train pour définir l’itération suivante. Il n’autorise ni validation,
ni holdout, ni autorité de production.

## Hypothèse SAGE.T11.1

Sur `bp35`, un posterior de programmes complets dont la dynamique de la
position du rôle `player` dépend explicitement de `ACTION3`, `ACTION4` et des
coordonnées de `ACTION6` doit :

1. produire des prédictions interventionnelles différentes avant la branche ;
2. réduire son entropie après exact-prefix replay ;
3. conserver et recharger cette croyance via A40 ;
4. modifier le choix en autorité `bounded` sans préempter une route protégée ;
5. obtenir une progression mesurée supérieure à l’ablation sans update.

Les mécanismes rivaux préenregistrés comprennent : persistance, colonnes
source-train, colonnes inversées, déplacements source-train, déplacements
inversés, grounding aux coordonnées du clic et variantes de sous-but gauche ou
droit. Chaque particule contient conjointement la dynamique, le prédicat de
progression et le but final `levels_completed >= 1`.

## Correction de représentation et vraisemblance

Le runtime causal utilise un graphe objet-centrique compact : entités, rôles,
agrégats par rôle et relations locales au joueur. Les relations spatiales
globales toutes-paires ne sont pas consommées par ce pilote.

La composante `variables` de la vraisemblance compare uniquement les variables
déclarées par le programme. Seuls les canaux déclarés par son modèle
d’observation sont scorés. Une particule n’est donc plus pénalisée par des
dizaines de milliers de relations qu’elle ne prétend pas expliquer.

## Plafond de stockage

Chaque invocation `replay` et `run` possède un plafond signé et fail-closed de
`3 * 1024^3 = 3 221 225 472` octets. Toute écriture est réservée avant d’être
effectuée. Le dépassement potentiel lève `ArtifactBudgetExceeded` et invalide
le run.

A40 persiste seulement les variables déclarées par les particules, les poids,
les lignages et une preuve compacte. Les entités et relations complètes ne sont
pas répétées dans `posterior.jsonl`.

## Design apparié gelé

- étape : `source_train` ;
- jeu : `bp35` uniquement ;
- seeds : `4101, 4102, 4103` ;
- resets : `2` par condition ;
- budget : `48` actions par reset ;
- autorité : `bounded`, protégée par le gate replay ;
- bras : `baseline`, `posterior_full`, `no_posterior_update`,
  `no_information_gain`, `no_a40_memory`, `no_mdl_prior` ;
- environnement, seed, reset et budget strictement appariés ;
- mémoire distincte par jeu, seed et bras ;
- aucun holdout ouvert.

## Gates

Le gate replay exige des hashes de préfixe exacts, au moins deux branches,
des prédictions enregistrées avant l’exécution et une réduction totale
d’entropie strictement positive.

Le gate actif source-train exige simultanément : intégrité complète, zéro
erreur contrôleur/environnement/action illégale, zéro régression de sécurité,
au moins une progression `bp35`, et avantage strict du posterior complet sur
`no_posterior_update` (niveau, puis efficacité si niveau positif égal).

Un échec arrête l’itération : aucune ouverture de source-validation ou de
holdout et aucun retuning post-hoc présenté comme confirmatoire.
