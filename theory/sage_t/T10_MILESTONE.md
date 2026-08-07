# SAGE.T10 - Progress Witness Induction

T10 remplace l'exploration locale sans but par une recherche active du premier
changement de niveau. Les actions sont d'abord regroupees par effet causal,
puis deux macros structurelles sont instanciees : repeter une cible distincte,
ou suivre les successeurs d'une chaine vers son extremite incluse. Les
coordonnees ne figurent que dans la preuve grounded ; le programme canonique
reste sans identifiant de jeu, couleur brute ou position absolue.

## T10.0b source-train

Le gate passe :

- `lp85` progresse apres cinq applications repetees ;
- `su15` progresse apres dix successeurs relationnels ;
- les deux programmes sont au rang 1 du posterior ;
- les deux folds leave-one-game-out progressent ;
- aucune action illegale, erreur ou mort n'est observee.

Le rapport auditable est
`training/sage_t/progress_witness_v10_0b/report.json`.

## T10.1 source-validation

La validation source gelee unique sur `re86`, `ls20` et `sc25` echoue fermee :
182 actions, zero erreur, zero action illegale et zero `GAME_OVER`, mais aucun
niveau. Les trois diagnostics sont `SEQUENCE_MISS`.

`re86` et `ls20` produisent plusieurs effets causaux distincts mais aucune
repetition monolithique ne progresse. `sc25` ne produit que des no-op au
premier pas. Le prochain axe scientifique est donc l'induction de sequences
mixtes et d'automates d'amorcage, et non une nouvelle calibration du posterior.

Ni le holdout, ni `ar25`, ni l'autorite de production ne sont ouverts.
