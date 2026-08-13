# SAGE.T12.4a.2 — reconfirmation exacte de deux témoins de progression

## Résultats parents conservés

T12.4a.1 reste un résultat négatif immuable : son calibrateur global améliore
la seed de calibration mais dégrade les deux seeds confirmatoires en agrégat.
Ce résultat ne peut autoriser ni T12.4b, ni une évaluation neurale active.

La collecte contrôle T12.4a.1 a toutefois produit deux progressions qui ne
dépendent pas du réseau :

- seed `8701`, route de 64 actions ;
- seed `8705`, route de 61 actions.

Chaque archive contient exactement une transition de niveau 0 vers le niveau
1. Les routes partent du même hash exact, arrivent au même hash exact et
partagent le suffixe :

```text
ACTION3, ACTION4, ACTION4, ACTION4, ACTION3, ACTION3
```

T12.4a.2 teste seulement la reproductibilité et la causalité locale de ces
deux témoins. Il ne réhabilite pas le calibrateur et n'attribue pas le succès au
modèle neural.

## Entrées immuables

Le freeze est lié par checksum :

- au manifeste T12.4a.1 ;
- au reçu négatif `FAIL_T12_4A_1_CALIBRATION_GATE` ;
- au reçu positif `PASS_T12_4A_1_COLLECTION_GATE` utilisé par le fit ;
- aux six archives de la collecte et particulièrement celles de `8701` et
  `8705` ;
- aux deux routes, à chaque action et à chaque hash intermédiaire ;
- au code d'exécution et au commit Git propre.

Le freeze échoue si une autre seed possède une progression, si une source
possède plusieurs progressions, si les longueurs diffèrent de 64 et 61, si les
hashes initiaux ou cibles diffèrent, ou si le suffixe commun n'est pas
exactement celui préenregistré.

## Expérience appariée

Pour chaque témoin, trois répétitions indépendantes sont exécutées pour chaque
condition :

| Condition | Exécution | Résultat attendu |
|---|---|---|
| `full_route` | reset puis route complète | cible exacte, niveau 1 |
| `common_suffix` | reset, préfixe exact, suffixe de six actions | cible exacte, niveau 1 |
| `delete_last_suffix_action` | même préfixe, suffixe sans la dernière `ACTION3` | état intermédiaire exact, aucune progression |

Chaque branche utilise un nouvel environnement. Le reset et chaque source et
cible de transition sont comparés au hash scellé. Une divergence arrête la
branche. Les pixels bruts ne sont jamais persistés.

Les branches suffixe et suppression forment un exact-prefix intervention
bundle seulement si elles retrouvent le même hash de préfixe avant de diverger
par la dernière action.

## Gate T12.4a.2

`PASS_T12_4A_2_WITNESS_GATE` exige simultanément, pour chacun des deux témoins :

1. trois replays complets exacts sur trois ;
2. trois replays du suffixe complet exacts et progressifs sur trois ;
3. trois contrôles par suppression exacts et sans progression sur trois ;
4. trois contrastes appariés confirmés sur trois ;
5. zéro progression dans les contrôles par suppression ;
6. tous les resets exacts ;
7. un taux global de comparaison exacte de `1,0` ;
8. au plus 2 048 appels SDK ;
9. au plus 3 Gio d'artefacts.

Aucune similarité approximative, aucun score neural et aucune amélioration de
couverture ne remplace ces critères.

## Firewalls

Pendant T12.4a.2 :

- le calibrateur et le réseau sont inactifs ;
- aucun entraînement ou recalibrage n'est autorisé ;
- aucune option n'est extraite ou compilée ;
- holdout, source-validation, T12.4b, T12.5 et production restent fermés ;
- le bouclier n'acquiert aucune autorité de production.

Un succès autorise seulement le freeze séparé d'une future T12.4a.3 dédiée à
l'extraction et à l'ablation d'une option minimale. Il n'autorise pas cette
extraction dans le run T12.4a.2.

## Artefacts

Le freeze produit `witnesses.sealed.json`, `manifest.json` et
`freeze_receipt.json`. Le run produit `replay_trials.json`,
`intervention_bundles.json`, `witness_report.json` et `witness_receipt.json`.
