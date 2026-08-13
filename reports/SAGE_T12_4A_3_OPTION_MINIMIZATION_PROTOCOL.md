# SAGE.T12.4a.3 — option minimale causale à deux contextes

## Question scientifique

T12.4a.2 a confirmé deux routes indépendantes vers le même état exact du
niveau 1. Les routes issues des seeds `8701` et `8705` partagent les six
dernières actions :

```text
ACTION3, ACTION4, ACTION4, ACTION4, ACTION3, ACTION3
```

T12.4a.3 teste prospectivement si une sous-séquence ordonnée unique de ce
suffixe suffit à reproduire exactement la progression depuis chacun des deux
préfixes. Il ne teste ni le réseau T12.4, ni son calibrateur, ni une politique
active.

## Entrées scellées

Le manifeste lie par SHA-256 :

- le manifeste et le reçu positif T12.4a.2 ;
- le registre des deux témoins confirmés ;
- les archives sources `8701` et `8705` ;
- le registre des quatre programmes causaux rivaux de `bp35` ;
- les fichiers de code impliqués et le commit Git propre ;
- le split SAGE11 et le protocole ci-dessous.

Le freeze refuse un parent autre que `PASS_T12_4A_2_WITNESS_GATE`, une route
modifiée, un autre suffixe ou un worktree sale.

## Ablation exhaustive

Les 64 sous-séquences du suffixe de six actions sont évaluées. Une
sous-séquence conserve l'ordre original et peut être vide. Pour chaque seed et
chaque sous-séquence :

1. un nouvel environnement est initialisé ;
2. le préfixe de 58 ou 55 actions est rejoué ;
3. chaque source et cible du préfixe est vérifiée par hash exact ;
4. la sous-séquence est exécutée comme intervention ;
5. le hash final et le compteur de niveau sont enregistrés.

Chaque branche est répétée trois fois. Le suffixe inversé est aussi exécuté
trois fois dans chaque contexte. Cela donne exactement 390 branches et, avec
les longueurs scellées, 23 613 appels SDK au maximum utile sous un plafond dur
de 24 000. Aucun frame brut n'est persisté et le run est limité à 3 Gio.

## Gate d'ablation

`PASS_T12_4A_3_OPTION_ABLATION_GATE` exige simultanément :

1. 390 branches exécutées et toutes leurs actions disponibles ;
2. 390 resets/préfixes exacts ;
3. trois reproductions exactes de la cible dans chaque contexte pour le
   suffixe complet ;
4. une seule sous-séquence de longueur minimale commune aux deux contextes ;
5. l'échec de toutes ses sous-séquences propres ;
6. aucune progression vers un autre état ;
7. aucune progression du contrôle inversé ;
8. le respect des plafonds de 24 000 appels SDK et 3 Gio.

Le mot « minimale » porte donc sur l'ensemble exhaustif des 64 sous-séquences,
pas sur une suppression gloutonne.

## Compilation fantôme

Une ablation positive produit un `MinimalCausalOption` lié aux deux contextes,
aux deux témoins et à la matrice d'ablation. Une phase séparée compile cette
option dans chacun des programmes causaux rivaux complets de `bp35`, puis
construit un posterior commun sur les programmes enfants.

La compilation vérifie hors environnement :

- que chaque programme parent possède un enfant dynamique + but + option ;
- que la masse postérieure attribuée au fournisseur d'option est au moins
  `0.80` ;
- que l'automate termine sur la séquence positive ;
- qu'il ne termine après aucune suppression d'une action ;
- qu'il ne termine pas sur la séquence inversée.

Cette phase est strictement `shadow`: elle n'appelle pas le SDK et ne peut pas
choisir ni exécuter une action réelle.

## Firewalls et décision suivante

Pendant tout T12.4a.3, holdout, source-validation, contrôle actif, production,
entraînement neural, T12.4b et T12.5 restent fermés. Un gate d'ablation négatif
interdit la compilation. Un gate de compilation négatif interdit tout test de
transfert.

Seul `PASS_T12_4A_3_SHADOW_COMPILE_GATE` autorise le freeze séparé d'un futur
T12.4a.4 consacré au transfert apparié sur les niveaux suivants. Il ne donne
pas directement d'autorité à l'option.

