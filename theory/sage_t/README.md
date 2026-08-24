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

T12.4a.1 failed because a global four-parameter calibrator transported from a
short high-prevalence archive to one confirmation seed but strongly degraded a
longer archive. Its control collection nevertheless found two neural-free
level-1 progress routes on seeds 8701 and 8705. T12.4a.2 seals their 64/61
actions, common initial and target hashes, and six-action common suffix. It
requires three exact full-route replays plus three paired suffix/deletion
contrasts per witness. Neural fitting, active evaluation and option extraction
remain disabled; a pass may only freeze T12.4a.3 option extraction. See
`reports/SAGE_T12_4A_2_WITNESS_RECONFIRMATION_PROTOCOL.md` and
`reports/SAGE_T12_4A_2_WITNESS_RECONFIRMATION_RUNBOOK.md`.

T12.4a.2 then passed with 1,137/1,137 exact transition comparisons and all
route, suffix, deletion and paired-contrast confirmations. T12.4a.3 seals its
two contexts and exhaustively evaluates all 64 ordered subsequences of the
shared six-action suffix, three times per context, plus reversed controls. A
unique minimal sequence must reach the same exact target in both contexts,
with no off-target progression. Only a passed ablation may be compiled into
the four complete causal-program particles, and that compilation remains
strictly shadow. The 390-branch run is capped at 24,000 SDK calls and 3 GiB. A
full shadow-compile pass may only freeze a separate level-transfer experiment;
it grants no active, validation, holdout or production authority. See
`reports/SAGE_T12_4A_3_OPTION_MINIMIZATION_PROTOCOL.md` and
`reports/SAGE_T12_4A_3_OPTION_MINIMIZATION_RUNBOOK.md`.

The first T12.4a.3 ablation invocation stopped before any ARC call because its
context loader read candidate length and source seeds from obsolete top-level
manifest fields. Amendment 1 moves those reads to the sealed protocol and uses
the canonical `ProgressWitness.source_seed` contract. The original manifest is
retained; scientific execution must freeze a new `T12.4a.3r1` manifest. See
`reports/SAGE_T12_4A_3_OPTION_MINIMIZATION_AMENDMENT_1.md`.

The corrected T12.4a.3r1 run passed both gates. All 390 exact-prefix branches
were available and exact; exhaustive minimization removed the leading
`ACTION3`, yielding `ACTION4 x3, ACTION3 x2`. The option was compiled into all
four posterior particles with owner mass effectively 1. T12.4a.4 now tests
that frozen five-action option prospectively from subsequent levels. At each
level, full, two typed-deletion, reversed and null branches are repeated across
both exact route lineages. At least two transferred levels are required, with
three attempted at most, 4,500 SDK calls and 3 GiB. A pass may only freeze a
separate paired option-control experiment; active, validation, holdout and
production authority remain closed. See
`reports/SAGE_T12_4A_4_OPTION_TRANSFER_PROTOCOL.md` and
`reports/SAGE_T12_4A_4_OPTION_TRANSFER_RUNBOOK.md`.

T12.4a.4 then produced a clean negative transfer result. All 20 level-1
prefixes were exact, every branch was available and deterministic, and no
terminal failure occurred, but the complete option progressed in 0/4 trials.
T12.4a.4b diagnoses that miss without retuning or adding a network. It compares
object-centric transition deltas from the two sealed level-0 success contexts
against the common level-1 failure context, using two lineages, null controls
and 16 exact-prefix trials. Its mutually exclusive result separates initiation,
dynamics, goal/termination and representation failures. A passed diagnostic
authorizes only the matching child freeze; policy, validation, holdout, neural
training and production authority remain closed. See
`reports/SAGE_T12_4A_4B_OPTION_APPLICABILITY_PROTOCOL.md` and
`reports/SAGE_T12_4A_4B_OPTION_APPLICABILITY_RUNBOOK.md`.

T12.4a.4b passed and classified the miss as
`INITIATION_AND_DYNAMICS_SHIFT`: the anchor structures differ, the first
mechanism delta already diverges and none of the 20 paired step deltas match.
T12.4a.4c therefore compiles the sequence as a guarded option rather than a
universal one. It retains up to six sparse initiation hypotheses as rival
particles, adds one typed effect contract per step, and crosses them with the
four complete dynamics-plus-goal programs under a 24-particle bound. The phase
is fully offline, forbids hashes, level identity, pixels and grounded entities
inside the new contracts, preserves aggregate parent posterior mass, and tests
guard/effect, deletion, reverse and shuffle ablations. A pass only authorizes a
separate target-local re-grounding/search freeze; all active and external
authority remains closed. See
`reports/SAGE_T12_4A_4C_OPTION_CONTRACT_PROTOCOL.md` and
`reports/SAGE_T12_4A_4C_OPTION_CONTRACT_RUNBOOK.md`.

T12.4a.4c passed its fully offline contract gate and authorizes the bounded
T12.4a.4d source-train search. From each of the two exact level-1 route
lineages, fresh seeds 9101–9103 compare generic symbolic Go-Explore against the
same archive reranked only by guard mismatch and target-local object roles.
Both arms share their grounded candidates, 4/8/16 burst schedule, terminal
shield and 2,048-call budget; the entire run is capped at 26,000 SDK calls and
3 GiB. The old option must remain blocked. A new level-1-to-level-2 suffix must
be confirmed twice from both lineages before a separate option-extraction
freeze can open. Generic discovery and causal-guidance advantage are reported
as distinct claims. See
`reports/SAGE_T12_4A_4D_TARGET_REGROUNDING_PROTOCOL.md` and
`reports/SAGE_T12_4A_4D_TARGET_REGROUNDING_RUNBOOK.md`.

T12.4a.4d completed as an integrity-clean negative result: all exact replay,
catalogue and budget checks passed, but neither arm found progress and terminal
rates were 18.10% for the local control and 16.75% for contract re-grounding.
The treatment also collapsed onto `ACTION6` for all 615 explored transitions,
while its exact-cell shield had no support for the target-local hazards.
T12.4a.4d.1 therefore remains symbolic and separates two repairs. It first
cross-fits a translation-invariant seven-cell hazard signature on the sealed
4d archives. Only a passed offline gate may run a prospective three-arm design
on fresh seeds 9201–9203: unchanged local archive, diversity-only control, and
diversity plus abstract hazard vetoes. The run is capped at 38,000 SDK calls
and 3 GiB; T12.4a.4e, validation, holdout, neural and production authority stay
closed until an exact cross-lineage witness passes. See
`reports/SAGE_T12_4A_4D_1_HAZARD_DIVERSITY_PROTOCOL.md` and
`reports/SAGE_T12_4A_4D_1_HAZARD_DIVERSITY_RUNBOOK.md`.

The next scientific bifurcation is T12.5: recognize whether a trajectory is
closer to its causal goal, rather than add another archive or hazard heuristic.
It binds the five typed T12.4a.4c effects to four rival progress programs:
terminal-only, change count, unordered milestones and ordered milestones. The
programs are crossed by reference with all 24 complete dynamics-plus-goal
owners, producing a common 96-particle joint posterior. Ordering is induced on
lineage 8701, measured on lineage 8705 before any update, then consolidated.
The gate requires perfect replication, ordered posterior mass at least 0.95,
strictly increasing successful-prefix value, a flat same-action failed trace,
action-label invariance, correct next-effect ranking and superiority over
terminal, novelty/count, unordered, action-only and state-only baselines. The
phase is fully offline, capped at 3 GiB and grants only a future shadow-ranking
freeze. See `reports/SAGE_T12_5_CAUSAL_PROGRESS_PROTOCOL.md` and
`reports/SAGE_T12_5_CAUSAL_PROGRESS_RUNBOOK.md`.

T12.5 passed its offline causal-progress gate but retained one explicit evidence
gap: the deletion branches established ordering without persisting an observed
typed delta for every executable candidate. T12.5b closes only that gap. At
each of five known progress stages it executes `ACTION3`, `ACTION4` and
`ACTION6` twice from both exact route lineages. `ACTION7` is advertised in the
frame signature but was found absent from the SDK executable-action set at all
five anchors; integrity amendment r1 excludes it without treating
unavailability as a causal effect. An empirical stage/action-to-effect table
is fitted on 8701, frozen before 8705 confirmation, and scored through the
unchanged T12.5 posterior. The causal ranking must place the known next effect
first at every stage and beat change-only, effect-magnitude and lexicographic
baselines. All 60 branches are fixed independently of the ranking, capped at
5,000 SDK calls and 3 GiB. A pass authorizes only a separately frozen paired
control experiment; live policy, validation, holdout, neural and production
authority remain closed. See `reports/SAGE_T12_5B_PROGRESS_SHADOW_PROTOCOL.md`
and `reports/SAGE_T12_5B_PROGRESS_SHADOW_RUNBOOK.md`.

T12.5b-r1 completed as a sealed scientific miss: causal progress ranked the
known next action perfectly, but effect magnitude did too, and `ACTION6` was
executable only on lineage 8701. T12.5b.2 is the zero-SDK offline diagnostic
for that exact failure class. It reconstructs the executable candidate set
inside each exact context, treats unavailable actions as missing interventions
rather than zero effects, and binds progress affordances across lineages using
only the milestone signature. Its hard-contrast gate requires an observed
same-prefix distractor with larger magnitude and lower progress in both
lineages. If the sealed data contain no such contrast, T12.5b.2 authorizes only
a future T12.5b.3 collection freeze; T12.5c and all production authority remain
closed. See `reports/SAGE_T12_5B_2_PROGRESS_DISCRIMINATION_PROTOCOL.md` and
`reports/SAGE_T12_5B_2_PROGRESS_DISCRIMINATION_RUNBOOK.md`.

T12.5b.2 confirmed that the sealed corpus contains zero hard contrasts and
authorized only a separately frozen prospective collection. T12.5b.3 targets
the unique nearest magnitude contest at stage 3 without consulting the causal
score: from both exact lineages it executes fixed `ACTION4` detours of depth
1–3, then branches `ACTION3`, `ACTION4`, and `ACTION6` twice from every exact
detour context. Unavailable actions remain missing interventions and progress
affordances bind across lineages only by `(stage, milestone_signature)`. The
36-trial source-train matrix is capped at 3,500 SDK calls, two hours, and 3 GiB.
A pass may authorize only a separate T12.5c control freeze; collection remains
manual, and control, validation, holdout, neural, and production authority stay
closed. See `reports/SAGE_T12_5B_3_PROGRESS_CONTRAST_PROTOCOL.md` and
`reports/SAGE_T12_5B_3_PROGRESS_CONTRAST_RUNBOOK.md`.

T12.5b.3 completed its fixed schedule but failed the preregistered zero-terminal
integrity gate; it also produced no transported progress affordance and no hard
one-step contrast. T12.5b.4 preserved that result and tested every length-2/3
program over `ACTION3/ACTION4/ACTION6` after the shallowest common neutral
detour. Its 72-trial calibration was exact, deterministic and within budget,
but 64 trials ended in `GAME_OVER`, the remaining eight became incomplete, and
no safe progress program existed. Evaluation therefore remained closed. See
`reports/SAGE_T12_5B_4_LOCAL_PROGRAM_UTILITY_PROTOCOL.md` and
`reports/SAGE_T12_5B_4_LOCAL_PROGRAM_UTILITY_RUNBOOK.md`.

T12.5b.5 tests the mechanism exposed by that miss: milestone neutrality did not
preserve future goal viability. From the exact pre-detour stage-3 state, it
branches `ACTION3`, `ACTION4`, and calibration-only `ACTION6`, then re-grounds
the remaining confirmed `ACTION3>ACTION3` continuation from the live legal
inventory. The cursor-advance branch and milestone-neutral cursor-mismatch
control are labelled only by observed level progress and terminal risk. Six
lineage-8701 calibration trials can seal a two-branch registry; only a passed
receipt authorizes four lineage-8705 evaluation trials. The total bound is
1,750 SDK calls and 3 GiB per phase. A final pass may authorize only preparation
of a separate T12.5c control freeze. See
`reports/SAGE_T12_5B_5_GOAL_VIABILITY_PROTOCOL.md` and
`reports/SAGE_T12_5B_5_GOAL_VIABILITY_RUNBOOK.md`.

T12.5b.5 then passed on both confirmed route lineages: the cursor-advance
`ACTION3>ACTION3` continuation produced safe level progress, while the
milestone-neutral `ACTION4>ACTION3>ACTION3` mismatch did not. T12.5c is the
separately frozen causal control for that result. From the same exact stage-3
anchors it compares `ACTION3>ACTION3` with a two-slot binding swap
`ACTION4>ACTION3`; both arms have identical maximum horizon, live-action
grounding, resets and repetitions. The eight-trial order is counterbalanced
across lineages 8701 and 8705 and capped at 1,000 SDK calls and 3 GiB. A pass
would show only that the correct local goal-cursor binding is necessary for
the observed source-context progress advantage, and may authorize only a
separate T12.6 freeze. See
`reports/SAGE_T12_5C_GOAL_CURSOR_CONTROL_PROTOCOL.md` and
`reports/SAGE_T12_5C_GOAL_CURSOR_CONTROL_RUNBOOK.md`.

T12.5c passed its equal-capacity causal control: all four correctly bound
two-step programs progressed, all four binding swaps did not, and neither arm
terminated. T12.6 asks whether the same binding principle exposes a useful
target-local signal at the already difficult level-1 anchor without spending
another SDK action. It cross-fits a four-step future productive-reach table on
the signed T12.4a.4d archives from seeds 9101–9103, then evaluates the frozen
table on the later T12.4a.4d.1 archives from seeds 9201–9203. An
immediate-effect table and a score-preserving binding permutation are fixed
controls. The CLI has only freeze, compile, evaluate and status; a pass may
authorize only a separate T12.6b prospective-control freeze. See
`reports/SAGE_T12_6_FUTURE_VIABILITY_PROTOCOL.md` and
`reports/SAGE_T12_6_FUTURE_VIABILITY_RUNBOOK.md`.
