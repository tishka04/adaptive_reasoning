# SAGE.T12.1 — protocole Go-Explore symbolique et options causales

## Question scientifique

Le protocole teste séquentiellement cinq affirmations falsifiables :

1. Une archive Go-Explore symbolique couvre au moins 25 % d'états de plus que
   la baseline historique, à budget SDK identique, et observe une progression.
2. Un bouclier terminal à horizon 64, appris uniquement à partir de terminaisons
   reproduites par exact-prefix replay, réduit les échecs sans réduire la
   progression.
3. Un MLP symbolique `état + action -> changement, nouveauté` bat un prior
   action-only hors échantillon, dépend réellement de l'état et reste calibré.
4. L'ordre d'exploration produit par ce MLP augmente encore la couverture d'au
   moins 10 %, sans régression de sécurité ni de progression.
5. La première progression peut être réduite à une option minimale, compilée
   dans chaque particule causale complète, puis transférée sur au moins deux des
   trois changements de niveau suivants.

Chaque affirmation a son propre receipt signé. Un échec ferme la phase
suivante. Un résultat négatif est conservé tel quel; il n'autorise ni retuning,
ni source-validation, ni holdout.

## Appariement et indépendance

- Une seule partie par manifeste. Les parties de validation utilisent donc des
  manifestes séparés, ce qui force la remise à zéro de l'archive, du bouclier,
  du modèle, du posterior et de toute mémoire.
- Chaque paire utilise le même jeu, le même seed, le même budget de 8 192 appels
  SDK et le même état restauré par préfixe exact.
- Les branches appariées doivent partager le même `entry_hash`.
- L'archive regroupe les transpositions sur une signature symbolique, mais
  conserve séparément chaque hash pixel-exact et son préfixe.
- Les bundles d'intervention sont émis seulement lorsqu'au moins deux actions
  distinctes ont réellement été exécutées depuis le même hash exact.

## Bras

| Phase | Contrôle | Traitement |
|---|---|---|
| Archive | baseline historique, SAGE.T off | Go-Explore symbolique pur |
| Sécurité | archive symbolique | archive + bouclier terminal |
| Ordonnancement | archive + bouclier | archive + bouclier + MLP gelé |
| Transfert | archive sans option; option brute; option causale sans update | option causale + posterior mis à jour |

Le MLP ne reçoit ni pixels, ni label terminal, et ne peut pas contourner le
bouclier. Il contient moins de 15 000 paramètres. Les mécanismes de l'option
sont des automates finis ajoutés aux programmes causaux complets; l'option n'est
proposée que si la masse postérieure de ses particules propriétaires atteint
0,80.

## Gates préenregistrés

- Archive : gain moyen de couverture >= 25 %, gain positif pour chaque seed,
  au moins une progression et 100 % des restaurations exactes.
- Bouclier : danger multiétape observé, au moins un veto, aucune régression de
  progression ou de sécurité.
- MLP : >= 512 exemples d'entraînement; prévalences entre 5 % et 95 %; gain de
  Brier >= 0,01 contre action-only; dégradation après permutation de l'état
  >= 0,01; ECE maximale <= 0,10.
- Ordonnancement neural : gain moyen >= 10 %, positif sur au moins deux seeds,
  aucune régression de progression/sécurité et replay exact.
- Option : longueur 1–32, deux reproductions exactes, aucune suppression d'une
  action ne conserve le succès.
- Compilation : l'automate accepte la séquence, refuse les contrôles par
  suppression et par inversion lorsqu'ils sont discriminants, et la masse
  propriétaire vaut au moins 0,80.
- Transfert : entrées strictement appariées, progression causale complète sur
  au moins deux changements de niveau, pas moins bonne que les trois contrôles,
  aucune régression terminale.

## Bornes et firewall

- 3 Gio maximum par run, vérifiés avant chaque écriture.
- 8 192 appels SDK maximum par seed et par bras.
- 50 000 cellules d'archive maximum.
- Aucune frame brute persistée.
- Source-validation et holdout restent fermés.
- Un manifeste gelé sur un worktree sale est `DIRTY_SMOKE_ONLY` et ne peut pas
  soutenir une conclusion scientifique.
