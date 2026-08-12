# SAGE.T12.3d — protocole gelé de contrôles de replay confirmés

## Question scientifique

T12.3c a confirmé le mécanisme de lineage sur trois seeds : le traitement a
atteint 100 % de replay exact, a évité 233 rebasages et n'a perdu ni couverture
ni progression. Le gate a néanmoins échoué parce que son contrôle positif
reposait sur `replay_failures == 0`, qui signifie « aucun échec enregistré » et
non « replay exact confirmé ». Neuf lignes de contrôle ne représentaient en
outre que quatre préfixes uniques.

T12.3d conserve ce résultat négatif et teste une correction de provenance :

> des contrôles checksum-uniques, déjà confirmés exactement par T12.3a, doivent
> repasser avant qu'une comparaison sur de nouveaux seeds puisse autoriser un
> nouveau descendant du bouclier terminal.

## Parents et sources admissibles

Le parent direct est le receipt T12.3c portant
`FAIL_T12_3C_REPLAY_LINEAGE_GATE`. Le freeze recalcule chaque terme du gate et
n'accepte le parent que si le contrôle positif est le seul terme défaillant.

La source des contrôles est le parent T12.3a de T12.3b. Elle doit porter
`PASS_T12_3A_WITNESS_GATE`, un taux exact pas-à-pas de 1,0 et trois
confirmations de route pour chaque witness.

## Contrôles confirmés

Les deux routes T12.3a sont scellées avec :

- leur checksum de route ;
- leur witness d'origine ;
- leurs 36 ou 63 actions ;
- le hash exact attendu au reset et après chaque action ;
- le nombre de confirmations exactes antérieures.

La déduplication porte sur le checksum de route. Le freeze exige exactement
deux routes et deux checksums de contrôle distincts. Aucun état issu de
`replay_failures == 0` n'est admissible.

Chaque contrôle est rejoué trois fois avec comparaison après chaque action.

## Comparaison prospective

Les seeds 7401, 7402 et 7403 n'ont servi ni au diagnostic T12.3b ni à
l'expérience T12.3c. Ils comparent :

- `shortest_prefix_control`, comportement historique ;
- `lineage_preserving`, rattachement au préfixe réellement exécuté.

Les deux bras utilisent le même Go-Explore symbolique, le calendrier 4/8/16,
le même budget et la même sélection d'actions. Aucun réseau, bouclier ou
extracteur d'état n'est modifié.

## Valeurs et budgets gelés

- 2 contrôles uniques ;
- 3 répétitions par contrôle ;
- seeds 7401, 7402, 7403 ;
- 3 500 appels SDK maximum par bras ;
- 30 000 appels SDK maximum au total ;
- borne preregistrée maximale : 21 774 appels SDK ;
- 50 000 cellules maximum par archive ;
- 3 Gio maximum d'artefacts ;
- aucune frame brute persistée.

## Gate T12.3d

Le gate passe uniquement si :

1. les deux contrôles et leurs checksums de route sont uniques ;
2. les six répétitions prévues sont présentes ;
3. chaque contrôle possède au moins trois confirmations antérieures T12.3a ;
4. le taux exact des contrôles confirmés atteint au moins 0,95 ;
5. le minimum de replay exact du traitement atteint au moins 0,95 ;
6. le traitement ne régresse sur aucun seed en replay exact ;
7. le ratio de couverture du traitement atteint au moins 0,80 sur chaque seed ;
8. le traitement ne régresse sur aucun seed en progression ;
9. au moins un rebasage est évité et aucune transition n'est rebasée ;
10. les plafonds SDK et stockage sont respectés.

Les seuils de T12.3c ne sont pas abaissés. Le seed 6803 et les résultats
T12.3c ne sont pas réutilisés dans la comparaison prospective.

## Firewalls

Restent fermés : holdout, source-validation, production, entraînement neural,
extraction d'options et promotion du bouclier. Un passage autorise seulement le
freeze d'un nouveau descendant T12.3b utilisant la lineage corrigée. Un échec
est conservé et interdit cette relance.

