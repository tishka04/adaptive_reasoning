# SAGE.T12.6.1a — Diagnostic des transitions conflictuelles

## Statut scientifique

T12.6.1 reste un échec d’intégrité signé et immuable. Son évaluation a trouvé
37 répétitions d’un même couple état–action avec des résultats enregistrés
différemment. Le présent travail commence après l’ouverture de 9201–9203 et
après inspection de ces conflits : il est donc exclusivement post-hoc.

T12.6.1a ne répare pas le reçu parent, ne transforme pas les mêmes archives en
nouveau holdout et ne peut produire aucune preuve confirmatoire.

## Question diagnostique

Deux mécanismes sont séparés :

1. la nouveauté d’archive, dépendante de l’ordre de parcours, peut changer alors
   que la transition abstraite reste identique ;
2. une répétition peut réellement changer la terminaison, la cible atteinte et
   la portée productive future.

Le diagnostic demande si le verdict de transfert observé par T12.6.1 dépend du
choix arbitraire d’une répétition conflictuelle.

## Comptages gelés avant exécution

L’audit agrégé ayant motivé T12.6.1a a fixé :

- 37 conflits au sens exact du parent ;
- 29 différences `novel` seulement ;
- 2 différences `novel+target_cell_id` ;
- 6 différences `terminal+novel+target_cell_id` ;
- 6 conflits changeant le label de viabilité future ;
- 37 conflits changeant le label immédiat observé ;
- 12 conditions d’archive et 8 payloads uniques concernés.

Ces valeurs sont des contrôles de reproduction, pas des résultats découverts
par la nouvelle exécution.

## Politiques de consolidation gelées

Les modèles, signatures, seuils et contrôles de T12.6.1 restent inchangés. Le
graphe brut de chaque archive reste également inchangé afin d’isoler le choix
du résultat directement associé au couple état–action. Six sensibilités sont
calculées :

1. `parent_order` reproduit exactement la règle du parent : dernière répétition
   identique, première issue conservée face à un conflit ;
2. `archive_last` conserve la dernière répétition ;
3. `modal_future_label` conserve le label futur modal, avec départage
   canonique ;
4. `minimum_future_label` fournit l’enveloppe pessimiste observée ;
5. `maximum_future_label` fournit l’enveloppe optimiste observée ;
6. `drop_conflicted_groups` retire entièrement toute décision contenant un
   conflit.

Les enveloppes minimale et maximale utilisent le label observé et ne sont pas
des politiques déployables. Aucune politique ne réentraîne le modèle.

## Portail d’intégrité diagnostique

Le diagnostic est complet seulement si :

- les 18 conditions, les seeds 9201–9203 et les lignées 8701/8705 sont présents ;
- les comptages gelés ci-dessus sont reproduits exactement ;
- `parent_order` reproduit les métriques globales, par seed et par lignée du
  reçu T12.6.1 à une tolérance de `1e-12` ;
- les six politiques sont toutes exécutées ;
- aucun appel SDK n’est effectué ;
- le temps reste inférieur à 600 secondes et les artefacts à 512 Mio.

Le succès de ce portail signifie uniquement « diagnostic reproductible et
complet ».

## Interprétation

Si au moins une consolidation satisfait tous les anciens portails
scientifiques, le verdict est classé sensible à la politique. Sinon, l’échec
est robuste uniquement dans la famille de consolidations enregistrée. Dans les
deux cas, il reste post-hoc.

Une confirmation ultérieure exigerait de nouvelles archives inédites et une
sémantique de transitions multirésultats figée avant leur ouverture. T12.6.2,
collecte physique, validation, contrôle autonome, entraînement neuronal et
production restent fermés.
