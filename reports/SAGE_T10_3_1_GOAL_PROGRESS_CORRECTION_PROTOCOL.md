# SAGE.T10.3.1 — correction du pilot goal-progress

## Portée

T10.3.1 est une continuation append-only du terminal T10.3 `ROOTING_MISS`. Le manifeste, les 540 événements et le rapport terminal T10.3 restent immuables. Ils authentifient le diagnostic et les huit reproductions positives de l’option canonique, mais aucun événement T10.3 n’est relabellisé ou admis dans le fit T10.3.1.

La correction ne constitue pas un retuning scientifique. Elle répare quatre contrats d’implémentation : re-grounding depuis l’inventaire légal courant à chaque étape, distinction entre binding causal pré-action et continuité après action, transport exact sur quotient commun déclaré, et séparation de la comptabilité des intents de la complétude des labels de branche.

## Recollecte gelée

Le panel conserve les jeux `bp35`, `lp85` et `su15`, les quatre bras et le plafond de 16 actions par reset. Il utilise les nouvelles graines 3121–3124. La confirmation utilise 3131–3132. Le budget demeure 48 + 12 resets et 960 actions au maximum. Aucun padding, remplacement de graine ou contrôle de substitution n’est autorisé.

Chaque choix d’action est refait depuis l’état et l’inventaire légal courants. Pour `repeat_target`, le binding frais du reset peut être maintenu seulement tant que la même action reste légale ; ce token grounded est transitoire et n’est jamais sérialisé. Une impossibilité produit `OPTION_GROUNDING_MISS` ou `CONTROL_GROUNDING_MISS`.

## Rooting corrigé

La correspondance évaluée par la QA est la preuve d’unicité du root pré-action. La continuité après action suit seulement des règles déclarées avant observation de l’effet : même identifiant local de branche, acteur unique pour un mouvement, même binding explicite/ancre transitoire, puis signature structurelle exacte et unique. Aucun delta d’effet n’est consulté. La disparition de la racine lors d’un changement de niveau reste visible, mais n’annule pas rétroactivement un binding pré-action valide.

## Transport corrigé

Le seul transport non-identitaire comparable relie `allocentric_object_relative` et `action_aligned_relational`. Son domaine est le quotient structurel commun qui retire les prédicats d’orientation propres à chaque gauge et conserve rôles, faits non directionnels, compteurs, registres et topologie. Le certificat est exact seulement si les quotients source et cible sont identiques avant et après l’action. Les transports depuis `root_only` vers des frames plus riches restent visibles et explicitement non comparables.

L’audit hors ligne doit démontrer la faisabilité de cette définition sur tous les couples complets T10.3, sans transformer ces événements en données d’entraînement.

## Gates et arrêt

Les seuils T10.3 sont conservés : correspondance ≥ 90 %, ambiguïté < 10 %, cohérence multiframe ≥ 50 %, transport comparable commutatif et round-trip exact, cible entre 0,5 % et 95 % avec au moins 32 positifs dans deux jeux, et reproduction des options positives. L’équation des intents et l’absence de labels inconnus sont deux gates distincts.

Le fit leave-one-game-out ajoute au modèle les métriques du quotient transporté (`changed`, taille du delta), tout en maintenant les jeux, graines, coordonnées et identités hors features. Le contrôle no-transport supprime réellement ces métriques. Les contrats `JointProgramHypothesis`, automates mixtes et posterior factorisé restent obligatoires.

Validation, AR25, holdout et production restent fermés. Un PASS autorise uniquement la préparation d’un protocole séparé.

