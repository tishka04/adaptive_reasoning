# SAGE.T — posterior unifié de programmes du monde

SAGE.T est une voie parallèle, fail-closed, du contrôleur cognitif. Son unité de
croyance est un programme complet : schéma de rôles, sémantique locale des
actions, dynamique, progression, terminal et but. Les anciens modules
proposent des fragments ou des gardes ; ils ne fournissent jamais de preuve au
posterior.

## SAGE.T11/T12 - programmes causaux complets

La cible canonique se trouve dans `theory/sage_t/causal/`. Elle ajoute un SCM
dynamique a deux tranches, un executeur avec semantique `do`, un posterior
robuste A38T/A39T, une memoire checksummée A40T, les bundles d'intervention par
replay exact et des mecanismes neuronaux partages masques par leurs parents.

`CausalRuntime` possede l'unique executeur, le registre, le posterior et
l'arbitre. `CausalSageTController` constitue le controleur cible sans belief
store historique parallele. Il reste off/shadow par defaut et peut etre injecte
via le point d'injection `sage_t_controller` existant.

Le manifeste et les gates sont documentes dans
`reports/SAGE_T11_CAUSAL_PROGRAM_POSTERIOR_PROTOCOL.md`. Aucun gate scientifique
actif ni holdout n'est ouvert par l'implementation seule.

Le runner expérimental est disponible via :

```bash
python -m theory.sage_t.causal.experiment_cli --help
```

Il scelle séparément les programmes rivaux et les plans de branches, gèle une
matrice appariée, exécute le replay exact avant toute autorité bornée, puis
compare baseline, posterior complet et ablations avec des receipts liés au
checksum du protocole. Le runbook complet se trouve dans
`reports/SAGE_T11_CAUSAL_EXPERIMENT_RUNBOOK.md`.

## Composants

- `contracts.py` définit l’état abstrait, la DSL typée et sérialisable, les
  programmes complets, l’alpha-normalisation et les paquets de prédiction
  partielle.
- `compiler.py` convertit les `TransitionRecord` réels avec le scene graph
  SAGE12 et les événements morpho-topologiques SAGE-MT.
- `executor.py` est l’unique interpréteur pur et déterministe. Les rollouts
  ordinaires sont bornés à trois actions ; les macros mémoire à huit.
- `synthesis.py` adapte les croyances existantes en fragments `support=0`,
  synthétise une grammaire déterministe et assemble au plus 64 programmes
  complets.
- `posterior.py` maintient les particules en log-espace, applique le prior MDL,
  la pénalité de couverture et rejoue tout l’historique lors d’une réparation.
- `decision.py` construit la matrice contrefactuelle, mesure les désaccords
  observationnel, causal, téléologique et planificationnel, puis applique
  l’utilité bayésienne spécifiée.
- `controller.py` fournit les modes `off`, `shadow`, `bounded` et `active`,
  leurs gates, les budgets d’intervention et le journal JSONL.
- `evaluation.py` fournit le protocole same-prestate source-only, la limite de
  cinq révélations et le gate contrefactuel explicite.

## Autorité

Le comportement par défaut reste strictement inchangé :

```python
UnifiedCognitiveConfig(sage_t_authority_mode="off")
```

Le mode `bounded` est automatiquement dégradé en `shadow` tant que
`sage_t_counterfactual_gate_passed` est faux. Le mode `active` est dégradé en
`bounded`, puis en `shadow`, tant que ses gates respectifs ne sont pas passés.
En mode borné, SAGE.T ne peut intervenir qu’une fois par contexte abstrait
inconnu, cinq fois par reset, avec un risque marginal maximal de `0.05`.

Les protections historiques restent prioritaires : action illégale, danger
observé et route protégée empêchent toute intervention. Une seule action est
exécutée avant reconstruction de l’état et replanification.

## Configuration minimale de shadow

```python
UnifiedCognitiveConfig(
    sage_t_authority_mode="shadow",
    sage_t_trace_path="reports/sage_t_shadow.jsonl",
)
```

Chaque décision trace le posterior, les séquences, les prédictions de chaque
programme, les désaccords, l’utilité, les veto et l’action retenue. Chaque
transition réelle trace le posterior avant/après et la surprise observée.

## Promotion

Le code ne marque aucun gate scientifique comme passé. La promotion exige
d’abord le replay contrefactuel, puis la validation active appariée sur `re86`,
`ls20` et `sc25`. Le holdout SAGE11 reste fermé jusqu’à ce passage.

## Replay scientifique T7

`replay_gate.py` groupe les panels V4.3 par racine vérifiée, mesure les quatre
conditions après 1, 3 et 5 observations, calcule les intervalles appariés et
diagnostique séparément la couverture du générateur et la sélection du
posterior. `sage_t7_frozen_manifest.json` verrouille la grammaire, l’exécuteur,
les coefficients, les budgets et les hashes du code avant le replay.

```bash
python -m theory.sage_t.replay_gate all
```

Les lignes auditables et les rapports agrégés sont écrits dans
`training/sage_t/replay_scientific_v1/`. Si source-train ne passe pas ses
comparaisons appariées, ou si les shards de validation manquent,
source-validation reste fermée et l’autorité active demeure interdite.

La « bonne famille » est un oracle d’évaluation relatif à la grammaire gelée :
les bras cachés servent uniquement à désigner la famille qui prédit le mieux.
Ils ne sont jamais fournis au générateur, au posterior, à la réparation ou au
choix d’action.

## Autopsie de sélection T7.1

`selection_autopsy.py` est une expérience source-train indépendante qui laisse
le replay T7 gelé intact. Elle sépare le score utilisé pour mettre à jour chaque
posterior du score commun utilisé pour comparer les ablations. Elle mesure
également le rang, la masse, l’élagage, le prior et l’évidence de la meilleure
famille générée.

```bash
python -m theory.sage_t.selection_autopsy --workers 4
```

Le manifeste `sage_t7_1_frozen_manifest.json` est lié au checksum du manifeste
T7 et au hash du code d’autopsie. Le rapport peut être reconstruit à partir des
lignes brutes sans rejouer les programmes :

```bash
python -m theory.sage_t.selection_autopsy --rebuild-report
```

Le gate reste fail-closed si les signaux de progression et de but sont trop
rares. Un résultat favorable sur un score commun ne suffit donc ni à ouvrir
source-validation, ni à donner une autorité active à SAGE.T.

## Pilote réel shadow T8

`live_shadow_pilot.py` exécute des trajectoires ARC réelles appariées avec
SAGE.T successivement désactivé puis en shadow. Il mesure la calibration, la
surprise, la réduction d'entropie, les signaux de progression/but/terminal, la
sécurité, la latence, les réparations et la stabilité du posterior. Chaque
protocole est gelé par un manifeste checksummé avant exécution.

Le SDK peut n'énumérer que quelques clics représentatifs alors que le
contrôleur historique matérialise un autre clic paramétré. Le challenger T8.5
ajoute uniquement cette action déjà sélectionnée à l'ensemble contrefactuel et
lui réserve une séquence. Il reste strictement en shadow :

```bash
.sage12_cache/v4_18/runtime/Scripts/python.exe \
  -m theory.sage_t.live_shadow_pilot_v5
```

La répétition longue gelée exécute 25 actions par jeu :

```bash
.sage12_cache/v4_18/runtime/Scripts/python.exe \
  -m theory.sage_t.live_shadow_pilot_v5 \
  --manifest theory/sage_t/sage_t8_5_long_frozen_manifest.json \
  --output-dir training/sage_t/live_shadow_pilot_v1_t8_5_long
```

Les sorties auditables sont écrites dans
`training/sage_t/live_shadow_pilot_v1_t8_5/`. Le gate d'intégration et le gate
scientifique sont distincts : une exécution peut être sûre, complète et assez
rapide tout en restant non identifiable si elle n'observe aucun événement de
progression, de but ou terminal. Dans ce cas source-validation et toute
autorité SAGE.T restent fermées.

Le rapport long de référence se trouve dans
`training/sage_t/live_shadow_pilot_v1_t8_5_long/report.json`. Il ne constitue
pas un passage de gate : le statut reste fail-closed tant que les canaux
téléologiques ne contiennent pas de positifs observés.

## Autorité active T9.5–T9.6

T9.5 a évalué sans retouche la politique T9.4 sur quinze paires
source-validation de 1 008 actions. Le gate a échoué fermé : aucun niveau,
10 749 interventions sans progrès et une hausse des `GAME_OVER` sur `sc25`.
Le holdout reste donc fermé.

T9.6 ajoute uniquement une abstention d'autorité : après cinq interventions
sans changement de niveau dans une branche, SAGE.T rend la main à la baseline.
Sur le protocole source-train apparié de T9.4, les trois niveaux sont conservés,
les neuf interventions utiles restent présentes et le nombre total
d'interventions passe de 114 à 45, sans mort ni erreur. Ce succès autorise un
futur retest source-validation gelé, pas l'ouverture du holdout.

## SAGE.T12.1 graph exploration and causal options

`causal/graph_experiment_cli.py` implements the gated sequence: pure symbolic
Go-Explore, a replay-confirmed 64-step terminal shield, a small symbolic
state/action change-novelty predictor, minimal successful-option extraction,
compilation into complete causal-program posterior particles, and paired
three-level transfer. Every phase emits an immutable checksummed receipt and
refuses a failed upstream gate. Each run is capped at 3 GiB and raw frames are
not persisted. Use one manifest per game so no archive, model, posterior or
memory can leak between validation games.

The scientific protocol and PowerShell commands are documented in
`reports/SAGE_T12_1_GRAPH_EXPLORE_PROTOCOL.md` and
`reports/SAGE_T12_1_GRAPH_EXPLORE_RUNBOOK.md`.

T12.1 ultimately failed its archive gate despite a large coverage gain because
it observed no progress and spent most SDK calls restoring a prefix before a
single exploratory action. T12.2 keeps that result immutable and tests a
strictly symbolic correction: deterministic 4/8/16-action excursions after
each exact restoration. The child manifest must bind the failed T12.1 receipt;
the new gate still requires real progression, not coverage alone. See
`reports/SAGE_T12_2_BURST_GO_EXPLORE_PROTOCOL.md` and
`reports/SAGE_T12_2_BURST_GO_EXPLORE_RUNBOOK.md`.

T12.2 also failed its aggregate gate, but left two progress witnesses: distinct
36- and 63-action routes from the same exact initial state to the same exact
level-1 state, with a common `ACTION3 x3` suffix. T12.3a does not retune search.
It seals those routes, replays each one three times with per-step exact hashes,
and pairs the common suffix with a deletion control. Shield learning, neural
training, option extraction and all validation remain closed until this narrow
gate passes. See `reports/SAGE_T12_3A_PROGRESS_WITNESS_PROTOCOL.md` and
`reports/SAGE_T12_3A_PROGRESS_WITNESS_RUNBOOK.md`.

T12.3a passed with 903/903 exact step comparisons and six of six paired causal
contrasts. T12.3b is its checksummed child: it confirms a balanced set of 12
terminal traces from the immutable T12.2 archives, propagates risk for up to 64
steps, and protects all 99 state/action pairs in the two successful routes. A
three-seed paired burst experiment must reduce terminal failures without losing
coverage or any known/prospective progress. Neural training, option extraction,
validation and holdout remain closed. See
`reports/SAGE_T12_3B_TERMINAL_SHIELD_PROTOCOL.md` and
`reports/SAGE_T12_3B_TERMINAL_SHIELD_RUNBOOK.md`.

T12.3b reduced the aggregate terminal rate and preserved coverage/progress, but
failed closed because the minimum exact-replay rate was 0.9123 instead of 0.95.
T12.3c keeps that negative result immutable. It seals the failing prefixes,
locates the first divergent action with per-step hashes, and compares the
historical shortest-prefix attachment against a treatment that attaches every
transition to the lineage actually executed inside the burst. Seed 6803 is the
known regression case; 7101 and 7102 are prospective. Passing T12.3c authorizes
only a new child terminal-shield experiment, never neural training, option
extraction, validation or production authority. See
`reports/SAGE_T12_3C_REPLAY_LINEAGE_PROTOCOL.md` and
`reports/SAGE_T12_3C_REPLAY_LINEAGE_RUNBOOK.md`.

T12.3c reached 100% exact replay in every lineage-preserving arm, but its gate
failed because controls selected only by `replay_failures == 0` were neither
confirmed nor unique. T12.3d corrects only that provenance error. It seals the
two checksum-distinct T12.3a progress routes already confirmed three times,
replays each three more times, and evaluates shortest-prefix versus
lineage-preserving archives on fresh seeds 7401–7403. A pass authorizes only a
new child terminal-shield experiment. See
`reports/SAGE_T12_3D_CONFIRMED_CONTROL_PROTOCOL.md` and
`reports/SAGE_T12_3D_CONFIRMED_CONTROL_RUNBOOK.md`.

T12.3d passed its confirmed-control gate and authorizes exactly one corrected
terminal-shield child. T12.3e reloads the original 12 confirmed terminal traces,
99 progress-protected state/action pairs and two T12.3a witnesses without
relearning them. On fresh seeds 7701–7703 it compares two lineage-preserving
archives, differing only by the terminal shield. The paired gate requires a
terminal-rate reduction, per-seed coverage and progress non-regression, exact
replay, real shield vetoes and zero lineage rebasing. A pass authorizes only the
freeze of T12.4; neural training, option extraction, validation and production
remain closed. See `reports/SAGE_T12_3E_LINEAGE_SHIELD_PROTOCOL.md` and
`reports/SAGE_T12_3E_LINEAGE_SHIELD_RUNBOOK.md`.

T12.3e passed narrowly and authorizes the preregistered T12.4 freeze. T12.4
trains a sub-15k-parameter CPU MLP on symbolic state/action inputs only. Dataset
QA rejected the legacy pixel-hash `changed` target because it was positive for
all 1,468 source transitions; the sealed replacement is an abstract-state
signature change, paired with first-cell novelty. Seeds 7701–7702 train, 7703
validates, and prospective seeds 8101–8103 compare the same lineage archive and
terminal shield with and without fixed neural reranking. Offline prediction
cannot pass without state dependence and calibration, and active authority
requires a 10% coverage gain with no replay, progress or terminal regression.
Only an active pass may freeze T12.5 option extraction. See
`reports/SAGE_T12_4_NEURAL_NOVELTY_PROTOCOL.md` and
`reports/SAGE_T12_4_NEURAL_NOVELTY_RUNBOOK.md`.

T12.4 failed its offline fit gate: its aggregate Brier gain cleared the frozen
threshold, but state shuffling improved rather than degraded predictions and
maximum ECE reached 0.158. T12.4a preserves that result and repairs the tested
mechanism, not the thresholds. It excludes the opened seed 7703, collects new
shielded lineage-control data on seeds 8401–8403, adds explicit action-to-object
relations and strictly pre-action archive history, and compares against the
action-only prior, a freshly trained T12.4 legacy model, state/context shuffles
and relation ablation. T12.4a has no active or option-extraction phase. A pass
may only freeze T12.4b. See
`reports/SAGE_T12_4A_REPRESENTATION_PROTOCOL.md` and
`reports/SAGE_T12_4A_REPRESENTATION_RUNBOOK.md`.

T12.4a passed every registered predictive and causal representation control,
but remained a negative result because its maximum ECE was 0.1459. T12.4a.1
tests calibration transport without changing the representation or policy. It
uses three unopened training seeds, one calibration-only seed and two untouched
confirmation seeds. A four-parameter monotone Platt layer must improve ECE
without losing Brier score, while all state/context/relation controls must pass
again. The CLI exposes no active-evaluation or option-extraction phase; a pass
may only freeze T12.4b. See
`reports/SAGE_T12_4A_1_CALIBRATION_PROTOCOL.md` and
`reports/SAGE_T12_4A_1_CALIBRATION_RUNBOOK.md`.
