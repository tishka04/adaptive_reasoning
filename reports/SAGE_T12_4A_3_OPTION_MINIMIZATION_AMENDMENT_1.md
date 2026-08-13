# SAGE.T12.4a.3 — amendement infrastructure 1

Le premier appel de la phase `ablate` s'est arrêté avant tout appel ARC avec :

```text
KeyError:'candidate'
```

La cause était constituée de deux références résiduelles à
`manifest["candidate"]["length"]` et `manifest["source_seeds"]` dans le
chargeur de contextes. Le manifeste T12.4a.3 stocke correctement ces valeurs
sous `manifest["protocol"]["candidate_action_count"]` et
`manifest["protocol"]["source_seeds"]`, avec les valeurs scellées `6` et
`[8701, 8705]`.

Le même audit a corrigé l'accès au champ du contrat `ProgressWitness` : la seed
est exposée par `source_seed`, pas par un attribut `seed`. Ce défaut aurait été
le prochain arrêt avant reset ; il est couvert par le même test de régression.

La correction remplace uniquement la source de ce paramètre. Elle ne modifie
ni les deux seeds, ni les routes, ni le suffixe, ni les 64 sous-séquences, ni
les trois répétitions, ni les gates, ni les limites de 24 000 appels SDK et
3 Gio. Un test de régression charge désormais la longueur exclusivement depuis
le protocole gelé.

Le run interrompu n'a produit ni dossier d'ablation, ni reçu, ni observation
ARC. Il ne constitue donc pas un résultat scientifique. Comme le manifeste
initial lie le checksum de l'ancien code, il reste immuable et ne doit pas être
réutilisé avec le correctif. Un nouveau manifeste doit être gelé dans un
nouveau répertoire `option_minimization_t12_4a_3r1_bp35` avant l'ablation.
