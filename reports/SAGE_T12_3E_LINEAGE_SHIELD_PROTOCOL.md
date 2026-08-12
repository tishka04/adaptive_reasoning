# SAGE.T12.3e — retest gelé du bouclier sur lineage corrigée

## Question scientifique

T12.3b a réduit le taux terminal agrégé tout en conservant couverture et
progression, mais a échoué parce que le replay exact est descendu à 0,9123.
T12.3c a isolé la cause : l'archive pouvait rattacher une transition au plus
court représentant d'un état visible, même si ce préfixe n'avait pas été
exécuté durant l'excursion. T12.3d a ensuite confirmé deux contrôles de replay
checksum-uniques et validé l'archive qui conserve la lineage réellement
exécutée.

T12.3e teste donc une seule hypothèse :

> le bouclier terminal T12.3b produit-il encore son effet lorsque les deux bras
> utilisent l'archive lineage-preserving validée par T12.3d ?

Il ne réapprend pas le bouclier, ne retouche pas ses seuils et ne modifie pas
l'extracteur d'état.

## Parents et preuves scellées

Le parent direct doit être le receipt
`PASS_T12_3D_CONFIRMED_CONTROL_GATE`. Le freeze remonte ensuite la chaîne
signée et exige :

- le receipt T12.3b original `FAIL_T12_3B_TERMINAL_SHIELD_GATE`, avec le replay
  exact comme unique classe d'échec ;
- les 12 traces terminales confirmées de T12.3b ;
- les 99 paires état/action protégées par les routes de progression ;
- les deux witnesses T12.3a, chacun déjà confirmé exactement ;
- le payload original du bouclier, sans réestimation.

Les checksums du parent, du registre, du bouclier, des confirmations, des
witnesses, du protocole et du code sont liés dans le manifeste T12.3e.

## Comparaison appariée

Les seeds prospectifs 7701, 7702 et 7703 comparent :

- `lineage_control` : archive lineage-preserving sans bouclier ;
- `lineage_terminal_shield` : même archive et même budget, avec le bouclier
  T12.3b confirmé.

Les deux bras utilisent le calendrier de bursts 4/8/16. Chaque transition est
rattachée au préfixe effectivement exécuté ; aucune transition ne peut être
rebasée vers un représentant plus court.

Avant cette comparaison, les deux routes de progression sont rejouées trois
fois chacune sous le bouclier. Chaque étape doit conserver son hash exact,
rester protégée, ne subir aucun veto et atteindre la progression attendue.

## Budgets gelés

- 3 seeds et 2 bras ;
- 4 096 appels SDK maximum par bras ;
- 2 witnesses répétés 3 fois, avec 128 actions maximum ;
- borne preregistrée : 25 350 appels SDK ;
- plafond global : 30 000 appels SDK ;
- 50 000 cellules maximum par archive ;
- 3 Gio maximum d'artefacts par run ;
- aucune frame brute persistée.

## Gate T12.3e

Le gate passe uniquement si :

1. les 12 traces terminales et les 99 protections scellées sont rechargées ;
2. le bouclier contient une preuve terminale multiétape et au moins une action
   confirmée dangereuse ;
3. les six replays de witness sont exacts, progressent et ne sont jamais
   bloqués par le bouclier ;
4. le traitement exerce au moins un veto durant l'évaluation ;
5. son taux terminal agrégé est au plus égal à 90 % de celui du contrôle ;
6. au plus un seed régresse individuellement sur le taux terminal ;
7. la couverture agrégée et celle de chaque seed restent au moins à 80 % du
   contrôle ;
8. aucune progression agrégée ou par seed n'est perdue ;
9. le minimum de replay exact des six bras atteint 0,95 ;
10. la lineage est effectivement utilisée, au moins un rebasage est évité et
    aucune transition n'est rebasée ;
11. les plafonds SDK et stockage sont respectés.

Si le contrôle n'observe aucune terminaison, une absence de terminaison dans le
traitement ne suffit pas à prouver un effet : le ratio vaut alors 1 et le gate
échoue. Le résultat négatif doit être conservé sans retune post-hoc.

## Firewalls

Le freeze et le run T12.3e gardent fermés le holdout, source-validation,
production, promotion du bouclier, entraînement neural et extraction d'option.
Un passage autorise uniquement le freeze de T12.4, destiné au petit prédicteur
neural action → changement/nouveauté. Il n'autorise pas encore cet entraînement.

