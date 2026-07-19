# SAGE milestones - closed-loop integration

Derniere mise a jour : 2026-07-19

SAGE orchestre les briques M1/M2/M3/P1 dans une boucle agentique. SAGE ne
confirme pas une mecanique, ne refute rien, et ne transforme jamais un resultat
de policy en support scientifique.

## Principes fixes

- `offline_trace_context` sert a l'observation, au diagnostic et au grounding.
- Les actions alternatives contrefactuelles ne sont autorisees que depuis un
  `live_env_context`.
- Une grille offline `grid_t` n'est jamais traitee comme un etat env complet.
- Les resultats LLM, WM, M3 et P1 restent candidate-only.
- A32/A33 restent les seuls lieux de revision/verdict scientifique.
- Tous les artefacts SAGE gardent `support=0`, `revision_status=CANDIDATE_ONLY`
  et `wrong_confirmations=0`.

## Etat

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| SAGE.0 - Known-game closed-loop scaffold | Fait | `theory/sage/known_game_closed_loop_scaffold.py`, `tests/test_sage_known_game_closed_loop_scaffold.py`, `diagnostics/sage/sage0_known_game_closed_loop_scaffold.json` | Boucle observation live -> contexte M2/M3/P1 -> action candidate -> env step -> log scientifique; offline counterfactual interdit; active counterfactual live autorise; support=0 |
| SAGE.1 - Known-game technical live run | Fait | `theory/sage/known_game_closed_loop_runner.py`, `theory/sage/live_prefix_counterfactual_collector.py`, `tests/test_sage_known_game_closed_loop_runner.py`, `tests/test_sage_live_prefix_counterfactual_collector.py`, `diagnostics/sage/sage1_known_game_closed_loop_results.json` | Runner bp35 reel budget 20; actions disponibles depuis `real_env_live_api`; aucune action illegale; replay prefix live exact pour ACTION3 alternatif; support=0 |
| SAGE.1b - Known-game policy loop guard | Fait | `theory/sage/policy_loop_guard.py`, `tests/test_sage_policy_loop_guard.py`, `diagnostics/sage/sage1b_policy_loop_guard_results.json` | Detecte le fallback repetitif `ACTION6 {"x":18,"y":0}`; interrompt apres 2 repetitions consecutives; switch legal `ACTION3`; support=0 |
| SAGE.2 - Known-game closed-loop policy probe | Fait | `theory/sage/known_game_policy_probe.py`, `tests/test_sage_known_game_policy_probe.py`, `diagnostics/sage/sage2_known_game_policy_probe_results.json` | Compare 4 policies sur le meme RESET bp35 (random legal, repeat ACTION6 fallback, SAGE.1 sans guard, SAGE.1b avec guard); metriques comparatives candidate-only; `outcome_status=SAGE_1B_IMPROVES_LOOP_DISCIPLINE_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.3 - Subgoal switch after success-like exhaustion | Fait | `theory/sage/subgoal_switcher.py`, `tests/test_sage_subgoal_switcher.py`, `diagnostics/sage/sage3_subgoal_switch_results.json` | Apres epuisement des cibles success-like, choisit un nouveau sous-but legal non repetitif parmi reposition / explore ACTION6 / contrefactuel actif / rerun M2-M3 / safe-hold; `outcome_status=SAGE_SWITCHES_TO_NEW_LEGAL_SUBGOAL_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.4 - Long-horizon transfer probe | Fait - gate echouee candidate-only | `theory/sage/long_horizon_transfer.py`, `tests/test_sage_long_horizon_transfer.py`, `diagnostics/sage/sage4_long_horizon_transfer_results.json` | Teste le transfert SAGE.3 vers ar25 sur budgets 50/150/300; 0/3 gates passed; repetition collapse, aucun subgoal switch; `outcome_status=SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.4b - Progress-stall trigger repair | Fait | `theory/sage/progress_stall_trigger.py`, `tests/test_sage_progress_stall_trigger.py`, `diagnostics/sage/sage4b_progress_stall_trigger_results.json` | Ajoute un trigger long-horizon generique sur repetition/stall; ar25 budget 150; `progress_stall_detected=true`, `subgoal_switches=49`, repetition post-trigger 0.41844, gate passed; support=0; aucun write A32/A33 |
| SAGE.4c - Long-horizon transfer rerun with progress-stall trigger | Fait | `theory/sage/long_horizon_progress_stall_transfer.py`, `tests/test_sage_long_horizon_progress_stall_transfer.py`, `diagnostics/sage/sage4c_long_horizon_progress_stall_transfer_results.json` | Relance ar25 budgets 50/150/300 avec trigger progress-stall; 3/3 gates passed; repetition reduite vs SAGE.4 sur tous les budgets; `outcome_status=SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.5 - First unknown-game bounded closed-loop probe | Fait | `theory/sage/unknown_game_bounded_probe.py`, `tests/test_sage_unknown_game_bounded_probe.py`, `diagnostics/sage/sage5_unknown_game_bounded_probe_results.json` | Premier probe public_unseen sur `sb26-7fbdac44`; budgets 50/150/300; 3/3 gates passed; unknown hygiene true; actions legales; repetition 0.0; terminal sous seuil 0.05; support=0; aucun write A32/A33 |
| SAGE.5b - Switch attribution and placeholder audit | Fait - dependance placeholder elevee | `theory/sage/switch_attribution_placeholder_audit.py`, `tests/test_sage_switch_attribution_placeholder_audit.py`, `diagnostics/sage/sage5b_switch_attribution_placeholder_audit.json` | Attribue 308 switches SAGE.5 : 306 success-like/loop-guard, 0 progress-stall, 2 terminal-guard; 297 placeholders `rerun_m2_m3`, 0 requests effectives; `outcome_status=SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.5c - Live mini-frontier generation from unknown state | Fait | `theory/sage/live_mini_frontier_generation.py`, `tests/test_sage_live_mini_frontier_generation.py`, `diagnostics/sage/sage5c_live_mini_frontier_results.json` | Convertit 20 placeholders `rerun_m2_m3` en 20 hypotheses + 20 requests M3 live-prefix candidate-only; effective_request_ratio=0.06734; residual placeholder ratio 0.905229; support=0; aucun write A32/A33 |
| SAGE.5d - Live-prefix mini-frontier M3 execution | Fait | `theory/sage/live_mini_frontier_m3_executor.py`, `tests/test_sage_live_mini_frontier_m3_executor.py`, `diagnostics/sage/sage5d_live_mini_frontier_m3_results.json` | Execute 8 requests SAGE.5c stratifiees en `LIVE_PREFIX_REPLAY_CONTEXT`; 8/8 replay exact + hash verifie; ACTION5/ACTION6 et object/local couverts; support_events=8, contradiction_events=0, mais support=0 et aucun write A32/A33 |
| SAGE.5e - Distributed live mini-frontier generation | Fait | `theory/sage/distributed_live_mini_frontier_generation.py`, `tests/test_sage_distributed_live_mini_frontier_generation.py`, `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json` | Distribue 8 requests par budget 50/150/300 au lieu de consommer tout le cap au budget 50; dedup par contexte/action/args/diff; execute 2 requests par budget; 6/6 replay exact + hash verifie; support_events=6, contradiction_events=0, mais support=0 et aucun write A32/A33 |
| SAGE.5f - Mini-frontier event consolidation | Fait | `theory/sage/mini_frontier_event_consolidation.py`, `tests/test_sage_mini_frontier_event_consolidation.py`, `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json` | Consolide les 6 events SAGE.5e en 3 clusters candidate-only; 2 clusters robustes multi-budget; 2 frontieres `ready_for_A32_review` non-verdict; support=0; aucun write A32/A33 |
| SAGE.5g - A32 review handoff compiler | Fait | `theory/sage/a32_review_handoff.py`, `tests/test_sage_a32_review_handoff.py`, `diagnostics/sage/sage5g_a32_review_handoff.json` | Compile 2 dossiers A32-review candidate-only et 4 followups; `ACTION6` requiert une diversite de controle; `ACTION5` requiert support + diversite de controle et remesure croisee du cluster lie non fusionne; support=0; aucune execution ni write A32/A33 |
| SAGE.5h - Controlled follow-up acquisition | Fait - partiel, surface de controle epuisee | `theory/sage/controlled_followup_acquisition.py`, `tests/test_sage_controlled_followup_acquisition.py`, `diagnostics/sage/sage5h_controlled_followup_acquisition.json` | Resout les 4 followups : 2 acquis, 2 bloques car seuls ACTION5/ACTION6 sont legaux; ACTION5 atteint 3 evenements comparables; remesure `local_patch` alignee mais `object_delta` divergente entre clusters 002/003; support=0; aucun write A32/A33 |
| SAGE.5i - Control-surface expansion | Fait - epuisement action-distinct borne | `theory/sage/control_surface_expansion.py`, `tests/test_sage_control_surface_expansion.py`, `diagnostics/sage/sage5i_control_surface_expansion.json` | Audite les 24 contextes candidats SAGE.5e en replay exact; tous exposent seulement ACTION5/ACTION6; aucune troisieme famille d'action; 7 options parametrees conservees sans requalification; proposition de decision protocolaire A32; support=0; aucun write A32/A33 |
| SAGE.5j - Pre-registered parameterized control acquisition | Fait - resultat mixte candidate-only | `theory/sage/parameterized_control_acquisition.py`, `tests/test_sage_parameterized_control_acquisition.py`, `diagnostics/sage/sage5j_parameterized_control_acquisition.json` | Execute exactement les 8 experiences A32.4 sans substitution; ACTION6 : 4/4 effets non discriminants (5 vs 5); ACTION5 : 4/4 effets discriminants (21 vs 4); 0 contradiction; 2 dossiers prets pour revue A32; support=0; aucun write A32/A33 |
| SAGE.6 - Second unknown-game bounded transfer | Fait - 3/3 gates passes sur wa30 | `theory/sage/second_unknown_game_transfer.py`, `tests/test_sage_second_unknown_game_transfer.py`, `diagnostics/sage/sage6_second_unknown_game_transfer_results.json` | Selectionne wa30 par ordre public_unseen fixe avant execution; exclut les jeux connus et sb26 source; budgets 50/150/300 passes; progress-stall detecte sur les 3 budgets; 98 switches; quarantaine A33.2 respectee; support=0; aucun write A32/A33 |
| SAGE.6a - Second-game switch attribution and live mini-frontier | Fait - 20 requests M3 candidate-only | `theory/sage/second_unknown_game_switch_frontier.py`, `tests/test_sage_second_unknown_game_switch_frontier.py`, `diagnostics/sage/sage6a_switch_attribution_mini_frontier.json` | Reproduit et attribue les 98 switches wa30 : 32 contrefactuels actifs, 34 repositionnements, 32 placeholders; distingue 1 garde terminale hors compteur source; convertit 20 placeholders en hypotheses + requests M3 live-prefix, reparties 4/12/4; support=0; aucun write A32/A33 |
| SAGE.6b - Stratified second-game M3 execution | Fait - 6/6 replays exacts | `theory/sage/second_unknown_game_m3_execution.py`, `tests/test_sage_second_unknown_game_m3_execution.py`, `diagnostics/sage/sage6b_second_unknown_game_m3_execution.json` | Selectionne 2 requests wa30 par budget avec 6 hashes distincts; execute ACTION2 contre ACTION1; 6/6 replay exact + hash verifie, 0 blocage; 5 effets bruts positifs et 1 neutre, mais support=0 et aucun write A32/A33 |
| SAGE.6c - Context-preserving event consolidation | Fait - motif +32 stable, exception neutre conservee | `theory/sage/second_unknown_game_event_consolidation.py`, `tests/test_sage_second_unknown_game_event_consolidation.py`, `diagnostics/sage/sage6c_second_unknown_game_event_consolidation.json` | Conserve 6 clusters singleton sans fusion; groupe 5 effets +32 sur les 3 budgets et 1 contexte neutre separe; frontiere compilable pour handoff mais non prete A32 car un seul controle distinct; support=0; aucun write A32/A33 |
| SAGE.6d - Second-game handoff compiler | Fait - 4 followups pre-enregistres | `theory/sage/second_unknown_game_handoff_compiler.py`, `tests/test_sage_second_unknown_game_handoff_compiler.py`, `diagnostics/sage/sage6d_second_unknown_game_handoff.json` | Compile 1 handoff candidate-only; pre-enregistre ACTION2/ACTION3 sur un contexte stable par budget et une replication exacte ACTION2/ACTION1 du contexte neutre; conserve les 6 frontieres de contexte; aucune execution, support=0, aucun write A32/A33 |
| SAGE.6e - Exact pre-registered followup execution | Fait - effet depend du controle candidate-only | `theory/sage/second_unknown_game_followup_execution.py`, `tests/test_sage_second_unknown_game_followup_execution.py`, `diagnostics/sage/sage6e_second_unknown_game_followup_execution.json` | Execute les 4 protocoles sans substitution; 4/4 replays exacts; ACTION2-ACTION3=0 sur les 3 budgets alors que ACTION2-ACTION1 valait +32; replication neutre ACTION2-ACTION1=0; 3 deviations pre-enregistrees, 1 condition satisfaite; support=0; aucun write A32/A33 |
| SAGE.6f - Control-dependence consolidation and A32 eligibility | Fait - 1 frontier A32 non-verdict eligible | `theory/sage/second_unknown_game_control_dependence_consolidation.py`, `tests/test_sage_second_unknown_game_control_dependence_consolidation.py`, `diagnostics/sage/sage6f_second_unknown_game_control_dependence_consolidation.json` | Consolide 10 observations dans 6 contextes non fusionnes; garde ACTION1/ACTION3 separes; 3 paires multi-budget avec gap stable 32; exception neutre repliquee; 14/14 criteres d'eligibilite; reformule le candidat comme contraste dependant du controle; support=0; aucun write A32/A33 |
| SAGE.7 - Third unknown-game bounded transfer | Fait - 3/3 gates passes, mini-frontiere parametree requise | `theory/sage/third_unknown_game_transfer.py`, `tests/test_sage_third_unknown_game_transfer.py`, `diagnostics/sage/sage7_third_unknown_game_transfer_results.json` | Selectionne tn36 par ordre public_unseen fixe apres exclusion de sb26/wa30; budgets 50/150/300 passes; 172 switches, 0 repetition d'argument; quarantaine A33.2/A33.3 respectee; surface live=11 variantes ACTION6 d'une seule famille; 145 placeholders M2/M3 non materialises; support=0; aucun write A32/A33 |

## SAGE.0 - Known-game closed-loop scaffold

Objectif :

- Verifier que la boucle SAGE minimale tient sur jeu connu, sans benchmark.
- Charger le contexte candidate-only M2.15, M3.7e, M3.7f et P1.
- Selectionner une action candidate depuis un contexte live.
- Executer un pas env dans un scaffold deterministe.
- Logger explicitement la frontiere offline-vs-live pour les contrefactuels.

Ajouts :

- `theory/sage/known_game_closed_loop_scaffold.py`
- `theory/sage/__init__.py`
- `tests/test_sage_known_game_closed_loop_scaffold.py`
- `diagnostics/sage/sage0_known_game_closed_loop_scaffold.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 2
- inputs read = `M2.15`, `M3.7e`, `M3.7f`, `P1`
- benchmark_run = false
- scaffold_only = true
- live_observations = 3
- env_steps_executed = 2
- actions selectionnees :
  - step 0 : `ACTION4`
  - step 1 : `ACTION6 {"x":12,"y":0}`
- m2_ready_for_m3 = 3
- m3_fused_requests_executed = 3
- blocked_capability_frontiers_logged = true
- offline_counterfactual_allowed = false
- active_counterfactual_collection_allowed = true
- env_state_restore_available = false
- offline_trace_context_is_observation_only = true
- live_env_context_authorizes_alternative_actions = true
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_0`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Logger contract :

```json
{
  "offline_counterfactual_allowed": false,
  "active_counterfactual_collection_allowed": true,
  "env_state_restore_available": false,
  "blocked_capability_frontiers_logged": true
}
```

Lecture :

SAGE.0 valide l'integration minimale : la boucle sait lire les hypotheses et
observations candidate-only, choisir une action candidate depuis un contexte
live, executer un pas env, puis produire un log scientifique disciplined. Il ne
s'agit pas d'un benchmark de performance et il n'y a aucun verdict A32/A33.

M3.7g offline counterfactual executor reste `NO-GO` tant qu'aucun etat env
complet/restorable n'est disponible. Les alternatives doivent etre collectees
activement en live.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_scaffold `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage0_known_game_closed_loop_scaffold.json `
  --budget 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_m3_offline_frame_counterfactual_probe.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.1 - Known-game technical live run

Objectif :

- Passer du scaffold SAGE.0 a un runner connu utilisant le vrai wrapper env.
- Verifier que les actions disponibles viennent de l'API live reelle.
- Selectionner uniquement des actions legales.
- Collecter une alternative contrefactuelle active par replay de prefixe live,
  jamais depuis une frame offline visuelle.
- Garder le run technique : pas de benchmark, pas de verdict, pas de support.

Ajouts :

- `theory/sage/known_game_closed_loop_runner.py`
- `theory/sage/live_prefix_counterfactual_collector.py`
- `tests/test_sage_known_game_closed_loop_runner.py`
- `tests/test_sage_live_prefix_counterfactual_collector.py`
- `diagnostics/sage/sage1_known_game_closed_loop_results.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 20
- technical_live_run = true
- benchmark_run = false
- inputs read = `M2.15`, `M3.7e`, `M3.7f`, `P1`
- available_actions_source = `real_env_live_api`
- real_env_available_actions_used = true
- synthetic_available_actions_used = false
- actions disponibles observees via env reel =
  `ACTION3`, `ACTION4`, `ACTION6`, `ACTION7`
- selected_action_always_legal = true
- invalid_action_selected = false
- live_steps_executed = 20
- env_actions = 20
- active_counterfactual_collection_attempted = true
- live_prefix_replay_exact = true
- live_prefix_replay_exact_count = 1
- counterfactual_collections = 1
- alternative collectee = `ACTION3`
- offline_counterfactual_allowed = false
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_1`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Details importants :

- Le point SAGE.0 a ete corrige : les actions disponibles de SAGE.1 ne viennent
  plus d'un scaffold permissif. Elles viennent du wrapper env bp35 reel.
- Le collecteur contrefactuel ne restaure pas un etat env depuis `grid_t`.
  Il rejoue `RESET -> prefix live`, verifie la signature, puis execute l'action
  alternative seulement si le replay est exact.
- Le premier replay prefix exact a collecte `ACTION3` apres le prefix
  `ACTION4`, avec changement d'etat observe.

Lecture :

SAGE.1 valide la capacite technique minimale pour une boucle connue reelle :
lecture des contextes candidate-only, action selection depuis API live,
execution env, et collecte contrefactuelle active par replay de prefixe. Ce run
ne prouve pas que SAGE joue bien, generalise ou decouvre une mecanique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_runner `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage1_known_game_closed_loop_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.1b - Known-game policy loop guard

Objectif :

- Corriger le signal negatif observe dans SAGE.1 : apres epuisement des cibles
  success-like, le fallback repetait `ACTION6 {"x":18,"y":0}`.
- Detecter les repetitions consecutives du meme couple `(action, args)`.
- Interrompre le fallback cible apres un seuil court.
- Selectionner un switch legal depuis `real_env_live_api`, en priorite la
  frontiere active `ACTION3`.
- Garder le statut technique : pas de benchmark, pas de verdict, pas de support.

Ajouts :

- `theory/sage/policy_loop_guard.py`
- integration optionnelle dans `theory/sage/known_game_closed_loop_runner.py`
- `tests/test_sage_policy_loop_guard.py`
- `diagnostics/sage/sage1b_policy_loop_guard_results.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 20
- technical_live_run = true
- benchmark_run = false
- loop_guard_enabled = true
- loop_guard_max_repeats = 2
- repeated_same_action_args_detected = true
- fallback_loop_interrupted = true
- switch_action_selected_after_exhaustion = true
- loop_guard_switches = 4
- max_same_action_arg_repeats = 2
- blocked fallback = `ACTION6 {"x":18,"y":0}`
- switch action = `ACTION3`
- selected_action_always_legal = true
- invalid_action_selected = false
- synthetic_available_actions_used = false
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_1B`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Lecture :

SAGE.1b ne rend pas la policy performante. Il retire seulement un attracteur
evident du runner connu : le meme fallback `ACTION6 {"x":18,"y":0}` ne peut
plus etre choisi indefiniment. Le switch reste un choix technique legal, pas une
confirmation scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_runner `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage1b_policy_loop_guard_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940 `
  --enable-loop-guard `
  --loop-guard-max-repeats 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.2 - Known-game closed-loop policy probe

Objectif :

- Passer de "le runner tourne legalement" (SAGE.1/1b) a une premiere mesure
  *comparative* sur jeu connu : SAGE produit-il une trajectoire meilleure qu'un
  baseline simple, sous budget court/moyen ?
- Rejouer le meme RESET bp35 reel pour chaque policy et comparer des metriques
  comportementales, sans benchmark de score humain et sans verdict scientifique.
- Garder la discipline : `support=0`, `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_2`, aucun write A32/A33.

Ajouts :

- `theory/sage/known_game_policy_probe.py`
- `tests/test_sage_known_game_policy_probe.py`
- `diagnostics/sage/sage2_known_game_policy_probe_results.json`

Baselines comparees (toutes depuis le meme RESET, API live reelle) :

- `random_legal` - action legale aleatoire (deterministe par seed)
- `repeat_action6_fallback` - repete le fallback cible `ACTION6`
- `sage1_without_loop_guard` - selecteur SAGE.1 sans loop guard
- `sage1b_with_loop_guard` - selecteur SAGE.1b avec loop guard

Metriques par policy :

- `levels_completed`
- `terminal_rate`
- `state_changed_rate`
- `unique_state_signatures`
- `repeated_action_arg_rate`
- `max_same_action_arg_repeats`
- `loop_guard_switches`
- `active_counterfactual_collections` (collecte active live, SAGE seulement)
- `policy_result_counted_as_confirmation = false`
- `support = 0`

Run du 2026-06-30 (jeu `bp35-0a0ad940`, budget 20, seed 0, guard max repeats 2) :

| policy | levels | terminal_rate | state_changed_rate | unique_sigs | repeated_arg_rate | max_repeats | loop_guard_switches | counterfactuals |
|---|---|---|---|---|---|---|---|---|
| random_legal | 0 | 0.0 | 1.0 | 21 | 0.20 | 2 | 0 | 0 |
| repeat_action6_fallback | 0 | 0.0 | 1.0 | 21 | 0.95 | 20 | 0 | 0 |
| sage1_without_loop_guard | 0 | 0.0 | 1.0 | 21 | 0.60 | 13 | 0 | 1 |
| sage1b_with_loop_guard | 0 | 0.0 | 1.0 | 21 | 0.20 | 2 | 4 | 1 |

- `sage1b_loop_discipline_improved_vs_sage1 = true`
- `sage1b_explores_at_least_as_much_as_repeat_fallback = true`
- `outcome_status = SAGE_1B_IMPROVES_LOOP_DISCIPLINE_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- champ nomme `outcome_status` (jamais `verdict`) pour eviter toute ambiguite avec A32/A33
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_2`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

Sur bp35, chaque action change l'etat (`state_changed_rate = 1.0` partout), donc
le discriminant n'est pas l'exploration brute mais la *discipline de boucle*. Le
loop guard fait chuter `repeated_action_arg_rate` de 0.95 (repeat) et 0.60
(SAGE.1) a 0.20, et plafonne `max_same_action_arg_repeats` a 2, tout en restant
au moins aussi exploratoire que le repeat fallback. SAGE.1b est donc le seul des
quatre a combiner faible repetition ET collecte contrefactuelle active.

C'est un premier signal comportemental candidate-only, pas une preuve de
competence agentique : aucun policy ne progresse de niveau (`levels_completed = 0`),
le switch reste technique, et le probe ne convertit aucun resultat en support
scientifique. L'`outcome_status` est candidate-only et n'ecrit ni A32 ni A33.

Limites encore ouvertes (non resolues par SAGE.2) :

- Apres epuisement des cibles success-like, le switch ne choisit pas encore un
  nouveau sous-but intelligent (repositionnement, contrefactuel cible, relance
  M2/M3, mode exploration).
- Jeu inconnu : pas encore evalue.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_policy_probe `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage2_known_game_policy_probe_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940 `
  --seed 0 `
  --loop-guard-max-repeats 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Prochaine etape : SAGE.3 (ci-dessous) - apres epuisement des cibles
success-like, choisir un nouveau sous-but au lieu de seulement eviter la boucle.

## SAGE.3 - Subgoal switch after success-like exhaustion

Objectif :

- SAGE.1b retire l'attracteur degenere, mais apres epuisement des cibles
  ACTION6 success-like (ou declenchement du guard) SAGE ne sait pas encore
  *quoi chercher ensuite*. SAGE.3 est un module de transition de mode, pas un
  meilleur guard.
- Quand l'epuisement/loop est detecte, choisir un nouveau sous-but legal et non
  repetitif parmi 5 modes :
  1. `active_counterfactual_collection` - collecter un contrefactuel live
  2. `repositioning` - repositionner via l'action de repositionnement
  3. `explore_new_candidate_action6_target` - essayer une cible ACTION6 inutilisee
  4. `rerun_m2_m3` - flag de re-derivation + action legale neutre
  5. `stop_safe_hold` - tenir si un risque terminal est present
- Critere minimal : PAS gagner, mais produire apres epuisement un comportement
  nouveau, legal, non trivial, non repetitif et mesurable.
- Discipline : `support=0`, `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_3`, champ `outcome_status` (jamais
  `verdict`), aucun write A32/A33.

Ajouts :

- `theory/sage/subgoal_switcher.py`
- `tests/test_sage_subgoal_switcher.py`
- `diagnostics/sage/sage3_subgoal_switch_results.json`

Detection du declencheur :

- L'epuisement est detecte quand `select_runner_action` (sans loop guard) tombe
  sur `decision_reason == candidate_policy_live_target_fallback`.
- `trigger_reason = success_like_targets_exhausted_or_loop_guard` (canonique).
- La rotation round-robin sur les modes *faisables* garantit la diversite et la
  non-repetition; chaque action candidate est verifiee != action bloquee et !=
  action precedente.

Contrat (par evenement de switch) :

```json
{
  "subgoal_switch_triggered": true,
  "trigger_reason": "success_like_targets_exhausted_or_loop_guard",
  "selected_subgoal": "active_counterfactual_collection_or_repositioning",
  "selected_action_legal": true,
  "policy_result_counted_as_confirmation": false,
  "support": 0,
  "truth_status": "NOT_EVALUATED_BY_SAGE_3",
  "a32_write_performed": false,
  "a33_write_performed": false
}
```

Metriques SAGE.3 (en plus des metriques SAGE.2) :

- `subgoal_switches`
- `subgoal_switch_success_rate` (legal ET state_changed ET non repetitif)
- `subgoals_used`
- `new_candidate_targets_discovered`
- `active_counterfactuals_after_exhaustion`
- `post_switch_state_changed_rate`
- `post_switch_repeat_rate`
- `post_switch_terminal_rate`
- `levels_completed`

Run du 2026-06-30 (jeu `bp35-0a0ad940`, budget 20) :

- `env_steps = 20`
- `subgoal_switches = 13`
- `subgoal_switch_success_rate = 1.0`
- `subgoals_used = [active_counterfactual_collection, repositioning, rerun_m2_m3]`
- `new_candidate_targets_discovered = 0`
- `active_counterfactuals_after_exhaustion = 4`
- `post_switch_state_changed_rate = 1.0`
- `post_switch_repeat_rate = 0.0`
- `post_switch_terminal_rate = 0.0`
- `placeholder_actions_used = 4`
- `placeholder_action_counted_as_subgoal_success = false`
- `rerun_m2_m3_requested = 4`
- `rerun_m2_m3_effective_requests_generated = 0`
- `levels_completed = 0`
- `selected_action_always_legal = true`
- `outcome_status = SAGE_SWITCHES_TO_NEW_LEGAL_SUBGOAL_CANDIDATE_ONLY`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_3`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

Apres epuisement, SAGE ne repete plus : il alterne reposition, contrefactuel
actif et rerun M2-M3, chaque switch produisant un changement d'etat sans
repetition consecutive (`post_switch_repeat_rate = 0.0`,
`subgoal_switch_success_rate = 1.0`) et sans entree terminale. Le critere minimal
SAGE.3 est atteint : un comportement nouveau, legal, non trivial et mesurable
apres epuisement.

Limite honnete observee sur bp35 : `explore_new_candidate_action6_target` n'a pas
ete faisable aux frames d'epuisement (l'unique cible ACTION6 live par frame
etait deja utilisee), donc `new_candidate_targets_discovered = 0`. Le module
degrade proprement vers les 3 autres modes au lieu d'inventer une cible. Aucun
niveau n'est franchi (`levels_completed = 0`) : SAGE sait desormais reformuler
un sous-but, mais ce n'est pas encore une competence de jeu.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.subgoal_switcher `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage3_subgoal_switch_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4 a maintenant ete execute sur ar25 connu. Le resultat ci-dessous remplace
l'ancien ordre conseille : le transfert long-horizon echoue proprement, donc le
jeu inconnu reste bloque tant que le trigger de switch ne transfere pas au moins
partiellement.

## SAGE.4 - Long-horizon transfer probe

Objectif :

- Tester si la discipline de boucle apprise sur le cas court bp35
  (SAGE.1b + SAGE.3) se transfere a un jeu connu plus long-horizon :
  `ar25-e3c63847`.
- Evaluer plusieurs budgets (`50`, `150`, `300`) sans demander de resoudre ar25.
- Gate minimal : actions toujours legales, pas de collapse repetitif,
  sous-buts effectivement declenches, et au moins une capacite de sous-but utile.
- Garder le statut candidate-only : `support=0`,
  `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_4`, aucun write A32/A33.

Ajouts :

- `theory/sage/long_horizon_transfer.py`
- `tests/test_sage_long_horizon_transfer.py`
- `diagnostics/sage/sage4_long_horizon_transfer_results.json`

Gate SAGE.4 :

```json
{
  "selected_action_always_legal": true,
  "no_repeat_collapse": true,
  "post_switch_repeat_discipline": true,
  "subgoal_switches_happened": true,
  "useful_subgoal_capability": true,
  "gate_passed": true
}
```

Run du 2026-06-30 (jeu `ar25-e3c63847`, budgets `50`, `150`, `300`) :

| budget | gate_passed | repeated_arg_rate | unique_sigs | subgoal_switches | subgoal_success | active_counterfactuals | new_targets | levels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | false | 0.96 | 7 | 0 | 0.0 | 0 | 0 | 0 |
| 150 | false | 0.986667 | 7 | 0 | 0.0 | 0 | 0 | 0 |
| 300 | false | 0.993333 | 7 | 0 | 0.0 | 0 | 0 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 0`
- `budgets_total = 3`
- `any_budget_gate_passed = false`
- `all_budgets_gate_passed = false`
- `discovered_new_candidate_targets_beyond_bp35 = false`
- `outcome_status = SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4 est un resultat negatif utile. La legalite tient
(`selected_action_always_legal = true` sur tous les budgets), mais la discipline
de boucle ne transfere pas a ar25 : le taux de repetition monte de `0.96` a
`0.993333`, aucun switch de sous-but n'est declenche, aucune collecte
contrefactuelle active n'a lieu, et aucune nouvelle cible candidate n'est
decouverte. Les gates echouent donc sur `no_repeat_collapse`,
`subgoal_switches_happened` et `useful_subgoal_capability`.

Interpretation :

SAGE.3 est valide localement sur bp35, mais son trigger d'epuisement ne se
generalise pas tel quel a ar25. SAGE.4 ne dit pas que SAGE est faux ou inutile ;
il dit que le mecanisme actuel de switch est trop attache a la dynamique bp35 et
ne produit pas encore une strategie long-horizon transferable. Ce resultat ne
devient pas support scientifique et ne modifie aucune hypothese A32/A33.

Point technique :

L'artefact SAGE.4 n'a pas de champ `summary` top-level ; le resume utile est
dans `comparison`, et les mesures detaillees sont dans `per_budget_results`.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.long_horizon_transfer `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage4_long_horizon_transfer_results.json `
  --game-id ar25-e3c63847 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4b a maintenant ete execute : le trigger long-horizon repare le defaut
principal de SAGE.4 en declenchant des switches sur ar25 sans attendre le
fallback bp35-style.

## SAGE.4b - Progress-stall trigger repair

Objectif :

- Ajouter un declencheur generique de stall long-horizon, independant de
  `decision_reason == candidate_policy_live_target_fallback`.
- Declencher le switch sur repetition/stall observable :
  - meme couple `(action, args)` repete au moins `K`
  - pas de progression de niveau dans la fenetre
  - faible nouveaute d'etats ou taux de repetition local eleve
- Verifier sur ar25 que le switcher est appele, que la repetition baisse, et que
  les actions restent legales.
- Garder le statut candidate-only : `support=0`, aucun verdict, aucun write
  A32/A33.

Ajouts :

- `theory/sage/progress_stall_trigger.py`
- integration optionnelle dans `theory/sage/subgoal_switcher.py`
- `tests/test_sage_progress_stall_trigger.py`
- `diagnostics/sage/sage4b_progress_stall_trigger_results.json`

Trigger :

```json
{
  "trigger_reason": "progress_stall_or_repeat_collapse",
  "same_action_args_repeated": true,
  "low_state_novelty": true,
  "no_level_progress": true,
  "switch_required": true,
  "support": 0
}
```

Gate SAGE.4b :

- `selected_action_always_legal = true`
- `progress_stall_detected = true`
- `subgoal_switches > 0`
- `repeat_collapse_interrupted = true`
- `repeated_action_arg_rate_after_trigger < repeated_action_arg_rate_before_trigger`
- `repeated_action_arg_rate_after_trigger < 0.9`
- `active_counterfactual_collection_attempted OR rerun_m2_m3_requested`
- `support = 0`
- aucun write A32/A33

Run du 2026-06-30 (jeu `ar25-e3c63847`, budget 150) :

- `progress_stall_trigger_enabled = true`
- `progress_stall_detected = true`
- `first_progress_stall_trigger_step = 4`
- `subgoal_switches = 49`
- `progress_stall_switches = 49`
- `repeat_collapse_interrupted = true`
- `repeated_action_arg_rate_before_trigger = 0.5`
- `repeated_action_arg_rate_after_trigger = 0.41844`
- `post_trigger_repeat_failure_threshold = 0.9`
- `active_counterfactual_collection_attempted = true`
- `active_counterfactuals_after_exhaustion = 8`
- `rerun_m2_m3_requested = 9`
- `rerun_m2_m3_effective_requests_generated = 0`
- `new_candidate_targets_discovered = 0`
- `unique_state_signatures = 65`
- `levels_completed = 0`
- `selected_action_always_legal = true`
- `invalid_action_selected = false`
- `gate_passed = true`
- `outcome_status = SAGE_PROGRESS_STALL_TRIGGER_REPAIRED_CANDIDATE_ONLY`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4B`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4b repare le verrou precis observe dans SAGE.4 : le switcher est maintenant
appele sur ar25 meme sans le signal bp35 `candidate_policy_live_target_fallback`.
Le collapse repetitif est interrompu, la legalite est preservee, et les sous-buts
`repositioning`, `active_counterfactual_collection` et `rerun_m2_m3` sont utilises
en contexte long-horizon.

Ce n'est toujours pas une competence de jeu : `levels_completed = 0`, aucune
nouvelle cible candidate n'est decouverte, et `rerun_m2_m3` reste un placeholder
non effectif (`rerun_m2_m3_effective_requests_generated = 0`). Le resultat est
un gate de capacite candidate-only, pas une preuve de performance.

Point technique :

Le runner stoppe maintenant proprement en `stop_safe_hold` lorsqu'un etat terminal
est atteint, au lieu d'essayer une action live apres `GAME_OVER`.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.progress_stall_trigger `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage4b_progress_stall_trigger_results.json `
  --game-id ar25-e3c63847 `
  --budget 150 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4c a maintenant ete execute. Le trigger progress-stall ne se contente plus
de passer un smoke test a budget 150 : il repare le transfert long-horizon sur
les trois budgets ar25 testes, en restant candidate-only.

## SAGE.4c - Long-horizon transfer rerun with progress-stall trigger

Objectif :

- Rejouer le protocole SAGE.4 sur `ar25-e3c63847` avec le trigger
  progress-stall active.
- Evaluer les budgets `50`, `150` et `300` contre le baseline SAGE.4.
- Verifier que la repetition baisse nettement, que les switches de sous-but se
  declenchent, et que les actions restent legales.
- Garder le statut candidate-only : `support=0`,
  `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_4C`, aucun write A32/A33.

Ajouts :

- `theory/sage/long_horizon_progress_stall_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_long_horizon_progress_stall_transfer.py`
- `diagnostics/sage/sage4c_long_horizon_progress_stall_transfer_results.json`

Gate SAGE.4c :

```json
{
  "selected_action_always_legal": true,
  "progress_stall_detected": true,
  "subgoal_switches_happened": true,
  "no_repeat_collapse": true,
  "post_switch_repeat_discipline": true,
  "repeated_action_arg_rate_lower_than_sage4": true,
  "active_counterfactual_or_rerun_requested": true,
  "support": 0
}
```

Run du 2026-06-30 (jeu `ar25-e3c63847`, budgets `50`, `150`, `300`) :

| budget | gate_passed | repeated_arg_rate | SAGE4_repeat | subgoal_switches | active_counterfactuals | rerun_m2_m3 | unique_sigs | terminal_rate | levels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | true | 0.44 | 0.96 | 15 | 5 | 3 | 25 | 0.0 | 0 |
| 150 | true | 0.42069 | 0.986667 | 49 | 8 | 9 | 65 | 0.006897 | 0 |
| 300 | true | 0.42069 | 0.993333 | 49 | 8 | 9 | 65 | 0.006897 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 3`
- `budgets_total = 3`
- `any_budget_gate_passed = true`
- `all_budgets_gate_passed = true`
- `budgets_with_progress_stall_detected = [50, 150, 300]`
- `budgets_with_repetition_improved_vs_sage4 = [50, 150, 300]`
- `budgets_with_subgoal_switches = [50, 150, 300]`
- `baseline_sage4_outcome_status = SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`
- `outcome_status = SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4C`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4c ferme le verrou ouvert par SAGE.4 : avec le trigger progress-stall, le
transfert de discipline de boucle fonctionne sur tous les budgets ar25 testes.
Le collapse repetitif de SAGE.4 est reduit de `0.96/0.986667/0.993333` a
`0.44/0.42069/0.42069`, les switches se declenchent, et des contrefactuels
actifs sont collectes.

Ce n'est toujours pas une competence de jeu ni un benchmark de performance :
`levels_completed = 0` sur tous les budgets, `new_candidate_targets_discovered = 0`,
et `rerun_m2_m3_effective_requests_generated = 0`. Le taux terminal non nul sur
les budgets `150` et `300` (`0.006897`) doit rester surveille. SAGE.4c valide
une capacite comportementale candidate-only : detecter un stall long-horizon et
sortir de la repetition, pas resoudre ar25.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.long_horizon_progress_stall_transfer `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --baseline-sage4 diagnostics\sage\sage4_long_horizon_transfer_results.json `
  --out diagnostics\sage\sage4c_long_horizon_progress_stall_transfer_results.json `
  --game-id ar25-e3c63847 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5 a maintenant ete execute sur un seul jeu public_unseen. Le resultat
ci-dessous remplace le plan prospectif : le probe inconnu borne passe ses gates,
mais reste strictement candidate-only et ne valide pas une competence de jeu.

## SAGE.5 - First unknown-game bounded closed-loop probe

Objectif :

- Executer une premiere boucle SAGE sur un jeu inconnu du point de vue M2.14 :
  `sb26-7fbdac44`.
- Garder le run borne : budgets `50`, `150`, `300`, sans objectif de benchmark.
- Comparer des baselines minimales :
  - `random_legal`
  - `neutral_legal_fallback`
  - contexte source `SAGE.4c`
- Verifier seulement les proprietes techniques :
  - hygiene unknown-game
  - actions legales
  - detecteur progress-stall actif
  - au moins un switch de sous-but sur au moins un budget
  - pas de collapse repetitif catastrophique
  - garde terminal sous seuil
  - `support=0`, aucun write A32/A33

Ajouts :

- `theory/sage/unknown_game_bounded_probe.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_unknown_game_bounded_probe.py`
- `diagnostics/sage/sage5_unknown_game_bounded_probe_results.json`

Gate SAGE.5 :

```json
{
  "unknown_game": true,
  "no_human_trace_for_game": true,
  "no_game_specific_prior": true,
  "selected_action_always_legal": true,
  "progress_stall_detector_runs": true,
  "subgoal_switches_happened": true,
  "no_catastrophic_repeat_collapse": true,
  "terminal_rate_guarded": true,
  "terminal_rate_under_threshold": true,
  "offline_counterfactual_allowed": false,
  "support": 0
}
```

Run du 2026-07-05 :

- `game_id = sb26-7fbdac44`
- `short_game_id = sb26`
- `split = unseen`
- `unknown_game = true`
- `no_human_trace_for_game = true`
- `no_m2_arc_lewm_trace_for_game = true`
- `no_game_specific_prior = true`
- `terminal_rate_threshold = 0.05`
- `source_sage4c_all_budgets_gate_passed = true`

Resultats SAGE.5 :

| budget | gate_passed | switches | progress_stall_detected | repeated_arg_rate | terminal_rate | active_cf | rerun_m2_m3 | new_targets | unique_sigs | levels |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | true | 50 | false | 0.0 | 0.0 | 0 | 47 | 3 | 51 | 0 |
| 150 | true | 128 | false | 0.0 | 0.007812 | 0 | 125 | 3 | 129 | 0 |
| 300 | true | 128 | false | 0.0 | 0.007812 | 0 | 125 | 3 | 129 | 0 |

Baselines observees :

| budget | baseline | repeated_arg_rate | terminal_rate | legal | unique_sigs | levels |
|---|---|---:|---:|---|---:|---:|
| 50 | neutral_legal_fallback | 0.98 | 0.0 | true | 51 | 0 |
| 50 | random_legal | 0.14 | 0.0 | true | 51 | 0 |
| 150 | neutral_legal_fallback | 0.984375 | 0.015625 | true | 65 | 0 |
| 150 | random_legal | 0.119658 | 0.008547 | true | 118 | 0 |
| 300 | neutral_legal_fallback | 0.984375 | 0.015625 | true | 65 | 0 |
| 300 | random_legal | 0.119658 | 0.008547 | true | 118 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 3`
- `budgets_total = 3`
- `all_budgets_gate_passed = true`
- `any_budget_gate_passed = true`
- `budgets_with_subgoal_switches = [50, 150, 300]`
- `budgets_without_catastrophic_repeat_collapse = [50, 150, 300]`
- `budgets_terminal_rate_under_threshold = [50, 150, 300]`
- `budgets_with_progress_stall_detected = []`
- `progress_stall_detector_runs = true`
- `offline_counterfactual_allowed = false`
- `active_counterfactual_collection_allowed = true`
- `outcome_status = SAGE_UNKNOWN_BOUNDED_PROBE_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.5 valide le premier passage technique sur jeu inconnu borne : le jeu est
bien hors traces M2.14/human traces, les actions restent legales, les switches
de sous-but se produisent sur les trois budgets, la repetition tombe a `0.0`,
et le taux terminal reste sous le seuil `0.05`.

Point important : le detecteur progress-stall etait actif, mais il n'a pas ete
declenche (`budgets_with_progress_stall_detected = []`). Sur `sb26`, le chemin
`success_like_targets_exhausted_or_loop_guard` suffisait a produire des
switches. SAGE.5 valide donc une boucle inconnue bornee et disciplined, pas une
generalisation du trigger progress-stall lui-meme.

Limites :

- `levels_completed = 0` sur tous les budgets.
- `active_counterfactuals_after_exhaustion = 0`.
- `rerun_m2_m3_effective_requests_generated = 0`.
- Les `new_candidate_targets_discovered = 3` sont des cibles candidates live,
  pas des preuves scientifiques.
- Le resultat ne confirme aucune mecanique et n'autorise aucun write A32/A33.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.unknown_game_bounded_probe `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage4c diagnostics\sage\sage4c_long_horizon_progress_stall_transfer_results.json `
  --m2-dataset-manifest diagnostics\m2\arc_lewm_dataset_manifest.json `
  --human-traces-dir human_traces `
  --out diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --game-id sb26-7fbdac44 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --terminal-rate-threshold 0.05
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5b a maintenant ete execute. Il confirme que SAGE.5 est une boucle inconnue
safe et non repetitive, mais que la majorite des switches reste un placeholder
`rerun_m2_m3` non effectif.

## SAGE.5b - Switch attribution and placeholder audit

Objectif :

- Attribuer chaque switch SAGE.5 a une cause explicite :
  - `success_like_targets_exhausted_or_loop_guard`
  - `progress_stall_or_repeat_collapse`
  - `terminal_guard`
- Mesurer les switches qui produisent une action exploratoire reelle.
- Mesurer les switches qui tombent sur le placeholder `rerun_m2_m3`.
- Calculer `rerun_m2_m3_effective_requests_generated / rerun_m2_m3_requested`.
- Garder le statut candidate-only : `support=0`, aucun verdict, aucun write
  A32/A33.

Ajouts :

- `theory/sage/switch_attribution_placeholder_audit.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_switch_attribution_placeholder_audit.py`
- `diagnostics/sage/sage5b_switch_attribution_placeholder_audit.json`

Run du 2026-07-05 :

| budget | switches | success-like / loop-guard | progress-stall | terminal-guard | exploratory | placeholder | placeholder_ratio | rerun_requested | effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50 | 50 | 0 | 0 | 3 | 47 | 0.94 | 47 | 0 |
| 150 | 129 | 128 | 0 | 1 | 3 | 125 | 0.968992 | 125 | 0 |
| 300 | 129 | 128 | 0 | 1 | 3 | 125 | 0.968992 | 125 | 0 |

Totals :

- `switches_total = 308`
- `switches_due_to_success_like_targets_exhausted_or_loop_guard = 306`
- `switches_due_to_progress_stall_or_repeat_collapse = 0`
- `switches_due_to_terminal_guard = 2`
- `true_exploratory_switches = 9`
- `active_counterfactual_switches_total = 0`
- `active_counterfactuals_after_exhaustion_total = 0`
- `placeholder_rerun_m2_m3_switches = 297`
- `placeholder_switch_ratio = 0.964286`
- `placeholder_dependency_threshold = 0.5`
- `placeholder_dependency_under_threshold = false`
- `rerun_m2_m3_requested = 297`
- `rerun_m2_m3_effective_requests_generated = 0`
- `effective_request_ratio = 0.0`
- `levels_completed_max = 0`
- `outcome_status = SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5B`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.5b confirme le point subtil de SAGE.5 : le detecteur progress-stall tourne,
mais ne cause aucun switch sur `sb26`. Les switches viennent presque tous du
chemin `success_like_targets_exhausted_or_loop_guard`, avec deux safe holds lies
au terminal guard. La boucle reste disciplined, mais elle n'est pas encore
apprenante : `rerun_m2_m3` est demande 297 fois et ne genere aucune request
effective.

Les 9 switches exploratoires correspondent aux nouvelles cibles candidates
locales, mais le reste du comportement est domine par le placeholder. SAGE.5b
n'est donc pas un gate de progression ; c'est une fermeture propre du diagnostic
placeholder.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.switch_attribution_placeholder_audit `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --out diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --placeholder-dependency-threshold 0.5
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5c a maintenant ete execute. Il transforme une partie du placeholder
`rerun_m2_m3` en hypotheses et requests M3 candidate-only effectives, sans
pretendre que ces requests sont deja support ou evidence.

## SAGE.5c - Live mini-frontier generation from unknown state

Objectif :

- Remplacer une partie des placeholders `rerun_m2_m3` par une mini-frontier live.
- Capturer `frame_before`, action, `frame_after` par replay prefix live.
- Calculer un diff deterministe :
  - cellules changees
  - bbox du changement
  - transitions de couleurs
  - couleurs ajoutees/supprimees
  - composantes creees/supprimees
  - terminal/non-terminal
- Generer une hypothese candidate simple :
  - `local_patch_change_candidate`
  - `object_delta_candidate`
  - `terminal_risk_candidate`
  - `no_effect_or_stale_candidate`
- Compiler une request M3 minimale en `LIVE_PREFIX_REPLAY_CONTEXT` quand un
  controle legal est disponible.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_live_mini_frontier_generation.py`
- `diagnostics/sage/sage5c_live_mini_frontier_results.json`

Run du 2026-07-05 :

- `game_id = sb26-7fbdac44`
- budgets = `[50, 150, 300]`
- `max_generated_requests = 20`
- `rerun_m2_m3_requested = 297`
- `effective_requests_generated = 20`
- `effective_request_ratio = 0.06734`
- `mini_frontier_hypotheses_generated = 20`
- `mini_frontier_m3_requests = 20`
- `unresolved_placeholder_switches_after_generation = 277`
- `source_placeholder_switch_ratio = 0.970588`
- `residual_placeholder_switch_ratio = 0.905229`
- `true_exploratory_or_scientific_switches = 29`
- `levels_completed_max = 0`
- `outcome_status = SAGE_LIVE_MINI_FRONTIER_GENERATED_CANDIDATE_ONLY`
- `generated_requests_counted_as_support = false`
- `mini_frontier_counted_as_evidence = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5C`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Par budget :

| budget | placeholders | generated | effective_ratio | source_placeholder_ratio | residual_placeholder_ratio | exploratory_or_scientific | generation_budget_exhausted |
|---|---:|---:|---:|---:|---:|---:|---|
| 50 | 47 | 20 | 0.425532 | 0.94 | 0.54 | 23 | true |
| 150 | 125 | 0 | 0.0 | 0.976562 | 0.976562 | 3 | true |
| 300 | 125 | 0 | 0.0 | 0.976562 | 0.976562 | 3 | true |

Familles generees :

- `local_patch_change_candidate = 19`
- `object_delta_candidate = 1`

Exemple de request generee :

- `request_id = m2m3::sage5c::live_mini_frontier::050::0001`
- `game_id = sb26-7fbdac44`
- `context_state_origin = sage5_live_prefix_frame_before`
- `replayability = LIVE_PREFIX_REPLAY_CONTEXT`
- `target_action = ACTION5`
- `metric = local_patch_before_after`
- `status = READY_FOR_M3`
- `support = 0`

Lecture :

SAGE.5c est le premier pas ou le placeholder `rerun_m2_m3` devient partiellement
effectif : 20 transitions live inconnues produisent des hypotheses et requests M3
chargeables, au lieu de rester de simples demandes non executees. Le ratio
`20/297 = 0.06734` est encore faible, mais il suffit a transformer la boucle de
survie disciplined en debut de boucle d'apprentissage active.

Limites :

- La generation est capee a 20 requests, donc 277 placeholders restent non
  transformes.
- Les budgets 150/300 ne generent rien apres epuisement du cap global.
- Les requests ne sont pas encore executees par M3.
- `levels_completed_max = 0`.
- Les hypotheses generees ne comptent pas comme support/evidence.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.live_mini_frontier_generation `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-sage5b diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --out diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --max-generated-requests 20
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5d a maintenant ete execute. Il ferme la chaine unknown-game minimale :
jeu inconnu -> placeholder SAGE -> mini-frontier SAGE.5c -> request M3
`LIVE_PREFIX_REPLAY_CONTEXT` -> replay exact -> mesure candidate-only.

## SAGE.5d - Live-prefix mini-frontier M3 execution

Objectif :

- Lire `diagnostics/sage/sage5c_live_mini_frontier_results.json`.
- Selectionner 4 a 8 requests SAGE.5c stratifiees :
  - au moins une `object_delta_candidate`
  - plusieurs `local_patch_change_candidate`
  - au moins une `target_action=ACTION5`
  - au moins une `target_action=ACTION6`
- Rejouer le prefixe live exact depuis RESET.
- Verifier `context_snapshot_hash` avant toute action cible ou controle.
- Executer l'action target dans un env replaye.
- Executer le controle dynamique dans un env replaye separe.
- Mesurer `local_patch_before_after`.
- Produire seulement des observations candidate-only :
  `support_events`, `contradiction_events`, `blocked_replay_events`,
  `execution_failures`.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/live_mini_frontier_m3_executor.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_live_mini_frontier_m3_executor.py`
- `diagnostics/sage/sage5d_live_mini_frontier_m3_results.json`

Run du 2026-07-05 :

- source = `diagnostics/sage/sage5c_live_mini_frontier_results.json`
- `source_sage5c_effective_requests_generated = 20`
- `selected_requests = 8`
- `requests_executed = 8`
- `requests_blocked = 0`
- `live_prefix_replay_exact_events = 8`
- `context_snapshot_hash_verified_events = 8`
- `families_executed = {"object_delta_candidate":1, "local_patch_change_candidate":7}`
- `target_actions_executed = {"ACTION5":5, "ACTION6":3}`
- `support_events = 8`
- `contradiction_events = 0`
- `neutral_events = 0`
- `blocked_replay_events = 0`
- `execution_failures = 0`
- `outcome_status = SAGE_LIVE_MINI_FRONTIER_M3_EXECUTED_CANDIDATE_ONLY`
- `support_events_counted_as_support = false`
- `mini_frontier_execution_counted_as_evidence = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5D`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Requests executees :

| request | family | target | control | target_signal | control_signal | support_events | contradiction_events |
|---|---|---|---|---:|---:|---:|---:|
| `...::0001` | `object_delta_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0003` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0005` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0006` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::0007` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0008` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::0009` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0010` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |

Note de mesure :

- Les requests `local_patch_before_after` sans coordonnees target explicites
  gardent la metrique demandee, mais l'execution annote un fallback de signal
  `changed_pixels_fallback_for_unparameterized_local_patch`.
- Ce fallback evite de transformer une action non parametree comme `ACTION5`
  en faux neutre technique.
- Le fallback est une mesure candidate-only, pas une preuve.

Lecture :

SAGE.5d valide la premiere boucle scientifique unknown-game end-to-end :
SAGE peut partir d'un etat live inconnu, produire une request M3 via SAGE.5c,
rejouer exactement le prefixe, verifier le hash de contexte, executer target et
controle dynamique, puis mesurer un effet local. Les 8 support_events indiquent
que les effets target sont superieurs aux controles dans ces contextes, mais ils
ne sont pas convertis en support scientifique.

Limites :

- Seulement 8/20 requests SAGE.5c sont executees.
- Les 20 requests SAGE.5c viennent encore du budget 50, car le cap SAGE.5c est
  epuise avant les budgets 150/300.
- `levels_completed` reste absent de la validation.
- La competence de jeu n'est pas validee.
- A32/A33 restent inchanges.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.live_mini_frontier_m3_executor `
  --source-sage5c diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5d_live_mini_frontier_m3_results.json `
  --min-requests 4 `
  --max-requests 8
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5e a maintenant ete execute. Il corrige la limite principale de SAGE.5c :
la generation n'est plus concentree sur le budget 50. Le signal candidate-only
est distribue sur plusieurs profondeurs de trajectoire, avec dedup strict et
execution M3 live-prefix sur chaque budget.

## SAGE.5e - Distributed live mini-frontier generation

Objectif :

- Lire `SAGE.5`, `SAGE.5b` et `SAGE.5c` comme contexte candidate-only.
- Generer une mini-frontier live inconnue avec quota par budget.
- Deduplicater par `(context_hash, target_action, target_args, diff_signature)`.
- Prioriser les contextes non redondants.
- Executer un petit nombre de requests par budget en `LIVE_PREFIX_REPLAY_CONTEXT`.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/distributed_live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_distributed_live_mini_frontier_generation.py`
- `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`

Run du 2026-07-18 :

- `game_id = sb26-7fbdac44`
- budgets = `[50, 150, 300]`
- `requests_per_budget = 8`
- `execute_requests_per_budget = 2`
- `distributed_generation = true`
- `dedup_policy = context_hash,target_action,target_args,diff_signature`
- `rerun_m2_m3_requested = 297`
- `effective_requests_generated = 24`
- `effective_request_ratio = 0.080808`
- `source_sage5c_effective_requests_generated = 20`
- `mini_frontier_hypotheses_generated = 24`
- `mini_frontier_m3_requests = 24`
- `dedup_key_count = 24`
- `duplicate_candidates_skipped = 24`
- `selected_execution_requests = 6`
- `requests_executed = 6`
- `requests_blocked = 0`
- `live_prefix_replay_exact_events = 6`
- `context_snapshot_hash_verified_events = 6`
- `support_events = 6`
- `contradiction_events = 0`
- `neutral_events = 0`
- `blocked_replay_events = 0`
- `execution_failures = 0`
- `outcome_status = SAGE_DISTRIBUTED_LIVE_MINI_FRONTIER_GENERATED_AND_EXECUTED_CANDIDATE_ONLY`
- `gate_passed = true`
- `generated_requests_counted_as_support = false`
- `mini_frontier_counted_as_evidence = false`
- `mini_frontier_execution_counted_as_evidence = false`
- `support_events_counted_as_support = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5E`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Par budget :

| budget | placeholders | generated | effective_ratio | duplicates_skipped | execution_selected | executed |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 47 | 8 | 0.170213 | 0 | 2 | 2 |
| 150 | 125 | 8 | 0.064 | 8 | 2 | 2 |
| 300 | 125 | 8 | 0.064 | 16 | 2 | 2 |

Familles :

- `families_generated = {"local_patch_change_candidate":23, "object_delta_candidate":1}`
- `families_executed = {"local_patch_change_candidate":5, "object_delta_candidate":1}`
- `target_actions_generated = {"ACTION5":13, "ACTION6":11}`
- `target_actions_executed = {"ACTION5":3, "ACTION6":3}`

Requests executees :

| request | family | budget | target | control | target_signal | control_signal | support_events | contradiction_events |
|---|---|---:|---|---|---:|---:|---:|---:|
| `...::050::0001` | `object_delta_candidate` | 50 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::050::0006` | `local_patch_change_candidate` | 50 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::150::0011` | `local_patch_change_candidate` | 150 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::150::0012` | `local_patch_change_candidate` | 150 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::300::0019` | `local_patch_change_candidate` | 300 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::300::0020` | `local_patch_change_candidate` | 300 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |

Lecture :

SAGE.5e valide la repartition de la mini-frontier live sur plusieurs profondeurs
de trajectoire inconnue. La chaine reste strictement candidate-only : les
support_events locaux montrent seulement que les mesures target depassent les
controles dynamiques dans ces contextes replayes, mais ils ne sont pas convertis
en support scientifique.

La correction principale par rapport a SAGE.5c est nette : SAGE.5c generait
20/20 requests sur le budget 50, alors que SAGE.5e genere 8 requests sur chacun
des budgets 50, 150 et 300, puis execute 2 requests par budget avec replay exact
et hash de contexte verifie.

Limites :

- Le test reste sur un seul jeu inconnu : `sb26-7fbdac44`.
- `levels_completed` reste hors validation ; aucune competence de jeu n'est
  confirmee.
- Les familles restent dominees par `local_patch_change_candidate` avec une
  seule `object_delta_candidate`.
- Les 6 support_events sont locaux, potentiellement redondants, et ne comptent
  ni comme evidence ni comme support.
- Des placeholders restent non transformes : la boucle apprenante existe, mais
  elle n'est pas encore continue.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.distributed_live_mini_frontier_generation `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-sage5b diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --source-sage5c diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --requests-per-budget 8 `
  --execute-requests-per-budget 2 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5f a maintenant ete execute. Il ne genere pas de nouvelles observations et
ne rejoue pas l'environnement : il consolide les observations SAGE.5e en motifs
candidate-only pour distinguer support local, redondance, robustesse multi-budget
et frontieres candidates A32-review sans write.

## SAGE.5f - Mini-frontier event consolidation

Objectif :

- Lire `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`.
- Verifier que la source SAGE.5e reste candidate-only.
- Construire des event records normalises depuis les executions M3 live-prefix.
- Regrouper les observations par regularite :
  - famille d'hypothese
  - action cible et arguments
  - transitions de couleurs
  - taille de patch / `changed_cells`
  - deltas de composantes
  - terminal / level delta
- Mesurer la stabilite par budget, action et contexte.
- Emettre seulement des clusters candidate-only, jamais un verdict.
- Garder `support=0`, aucun write A32/A33.

Ajouts :

- `theory/sage/mini_frontier_event_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_mini_frontier_event_consolidation.py`
- `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json`

Run du 2026-07-18 :

- source = `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`
- `source_requests_generated = 24`
- `source_requests_executed = 6`
- `event_records = 6`
- `candidate_mechanism_clusters = 3`
- `multi_budget_clusters = 2`
- `robust_multi_budget_clusters = 2`
- `ready_for_A32_review_candidates = 2`
- `unique_contexts = 6`
- `unique_dedup_keys = 6`
- `raw_support_events = 6`
- `raw_contradiction_events = 0`
- `raw_neutral_events = 0`
- `actions_covered = {"ACTION5":3, "ACTION6":3}`
- `families_covered = {"local_patch_change_candidate":5, "object_delta_candidate":1}`
- `candidate_status_counts = {"LOCAL_SUPPORT_CANDIDATE_ONLY":1, "ROBUST_MULTI_BUDGET_CANDIDATE_ONLY":2}`
- `outcome_status = SAGE_MINI_FRONTIER_EVENTS_CLUSTERED_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5F`
- `revision_status = CANDIDATE_ONLY`
- `clustered_support_events_counted_as_support = false`
- `cluster_status_counted_as_scientific_verdict = false`
- `candidate_a32_frontier_counted_as_revision = false`
- `ready_for_A32_review_is_not_verdict = true`
- `a32_write_performed = false`
- `a33_write_performed = false`

Clusters :

| cluster | family | action | budgets | status | raw_support | contradictions | pattern |
|---|---|---|---|---|---:|---:|---|
| `001` | `local_patch_change_candidate` | `ACTION6 {"x":26,"y":57}` | 50/150/300 | `ROBUST_MULTI_BUDGET_CANDIDATE_ONLY` | 3 | 0 | `4->0` x20, `changed_cells=20`, effect size 4 |
| `002` | `object_delta_candidate` | `ACTION5` | 50 | `LOCAL_SUPPORT_CANDIDATE_ONLY` | 1 | 0 | `0->4` x20 + `2->3` x1, effect size 17 |
| `003` | `local_patch_change_candidate` | `ACTION5` | 150/300 | `ROBUST_MULTI_BUDGET_CANDIDATE_ONLY` | 2 | 0 | `0->4` x20 + `2->3` x1, `changed_cells=21`, effect size 17 |

Frontieres candidates A32-review :

- `sage5f::candidate_a32_frontier::001` derive du cluster `001`.
- `sage5f::candidate_a32_frontier::002` derive du cluster `003`.
- Ces frontieres sont marquees `ready_for_A32_review=true`, mais
  `ready_for_A32_review_is_not_verdict=true` et
  `candidate_a32_frontier_counted_as_revision=false`.
- Aucun fichier A32/A33 n'est modifie.

Lecture :

SAGE.5f valide la premiere memoire experimentale candidate-only sur jeu inconnu :
les observations locales SAGE.5e ne restent plus une liste plate de support_events.
Elles sont regroupees en motifs d'effet, avec une distinction entre support local
single-budget et robustesse multi-budget.

Deux motifs sont assez stables pour etre proposes a une revue A32 candidate-only :
`ACTION6` produit un motif local `4->0` sur les budgets 50/150/300, et `ACTION5`
produit un motif `0->4` + `2->3` sur les budgets 150/300. Cela ne confirme pas
une mecanique : c'est seulement une consolidation structurante pour une future
revue.

Limites :

- Le test reste sur `sb26-7fbdac44` uniquement.
- Les clusters viennent de 6 executions seulement.
- Les 2 frontieres A32-review candidates ne sont pas des decisions A32.
- `levels_completed` reste absent : aucune competence de jeu n'est validee.
- Les effets sont locaux et peuvent encore etre redondants ou contextuels.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.mini_frontier_event_consolidation `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5f_mini_frontier_event_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5g a maintenant ete compile. Il transforme les deux frontieres robustes de
SAGE.5f en dossiers auditables pour A32, sans executer de nouveau test, sans
produire de verdict et sans ecrire dans A32/A33. Les lacunes historiques A32
sont explicites et converties en demandes de suivi controlees.

## SAGE.5g - A32 review handoff compiler

Objectif :

- Lire `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json`.
- Verifier que la source SAGE.5f reste candidate-only et `support=0`.
- Compiler chaque cluster robuste en dossier A32-review auditable.
- Conserver les contextes, budgets, interventions cibles, controles et motifs
  d'effet observes.
- Comparer chaque dossier aux seuils historiques A32 :
  - au moins 3 support events bruts ;
  - au moins 2 contextes replay-exacts ;
  - zero contradiction ;
  - au moins 2 actions de controle distinctes.
- Lier les clusters proches sans les fusionner automatiquement.
- Produire des followups precis, sans execution ni revision.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/a32_review_handoff.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_a32_review_handoff.py`
- `diagnostics/sage/sage5g_a32_review_handoff.json`

Run du 2026-07-18 :

- `source_candidate_mechanism_clusters = 3`
- `source_ready_for_A32_review_candidates = 2`
- `handoff_items = 2`
- `items_ready_for_A32_review = 2`
- `items_without_followup_requirements = 0`
- `followup_requests = 4`
- `raw_support_events_in_handoff = 5`
- `raw_contradiction_events_in_handoff = 0`
- `independent_context_events_in_handoff = 5`
- `related_nonmerged_cluster_links = 1`
- `outcome_status = SAGE_A32_REVIEW_HANDOFF_COMPILED_CANDIDATE_ONLY`
- `gate_passed = true`
- `candidate_review_item_counted_as_revision = false`
- `independent_context_events_counted_as_scientific_support = false`
- `execution_performed = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5G`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Dossiers :

| candidat | support brut | contextes | controles distincts | recommandation |
|---|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | 3 | 3 | 1 (`ACTION5`) | `FOLLOWUP_REQUIRED_CONTROL_DIVERSITY` |
| `ACTION5` | 2 | 2 | 1 (`ACTION6`) | `FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY` |

Followups :

- `ACTION6` : retester contre une action de controle autre que `ACTION5`, dans
  au moins deux contextes replay-exacts.
- `ACTION5` : obtenir une troisieme observation comparable.
- `ACTION5` : retester contre une action de controle autre que `ACTION6`.
- `ACTION5` : remesurer les clusters `002` et `003` avec les lectures
  `local_patch` et `object_delta`, en les gardant non fusionnes avant revue.

Lecture :

SAGE.5g ne confirme ni `ACTION6` ni `ACTION5`. Il ferme le contrat de handoff :
un motif robuste devient un dossier falsifiable, ses insuffisances deviennent
des demandes d'experience explicites, et A32 reste le premier composant autorise
a produire une decision scientifique.

Les cinq contextes cumules ne sont pas transformes en support scientifique. Ils
servent uniquement a documenter la diversite des observations. De meme, les
budgets 50/150/300 ne sont jamais comptes comme des actions de controle
distinctes.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.a32_review_handoff `
  --source-sage5f diagnostics\sage\sage5f_mini_frontier_event_consolidation.json `
  --out diagnostics\sage\sage5g_a32_review_handoff.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5h a maintenant execute les quatre followups SAGE.5g. Deux acquisitions
sont obtenues et deux sont bloquees avec une raison structurelle reproductible :
aux contextes replayes, les seules familles d'action legales sont `ACTION5` et
`ACTION6`. Apres exclusion de l'action cible et du controle deja utilise, aucune
troisieme action distincte n'est disponible.

## SAGE.5h - Controlled follow-up acquisition

Objectif :

- Lire les dossiers et followups SAGE.5g.
- Retrouver les requests replayables SAGE.5e et les clusters SAGE.5f.
- Auditer la surface d'actions legales avant toute acquisition de nouveau
  controle.
- Exiger un replay exact et un hash de contexte identique.
- Acquerir un troisieme evenement comparable pour `ACTION5`.
- Remesurer les clusters lies `002` et `003` avec :
  - `local_patch_before_after` ;
  - `object_counts_before_after` comme lecture object-delta.
- Garder les clusters non fusionnes en cas de divergence de mesure.
- Resoudre chaque followup comme acquis ou bloque avec une raison auditable.
- Garder `support=0`, aucune revision, aucun write A32/A33.

Ajouts :

- `theory/sage/controlled_followup_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_controlled_followup_acquisition.py`
- `diagnostics/sage/sage5h_controlled_followup_acquisition.json`

Run du 2026-07-18 :

- `followup_requests_consumed = 4`
- `followup_outcomes = 4`
- `followups_completed = 2`
- `followups_blocked = 2`
- `control_diversity_followups_completed = 0`
- `control_diversity_followups_blocked = 2`
- `support_followups_completed = 1`
- `cross_measurement_followups_completed = 1`
- `control_surface_contexts_audited = 3`
- `replay_exact_control_surface_audits = 3`
- `control_surface_exhausted_audits = 3`
- `controlled_experiments_executed = 4`
- `controlled_experiments_blocked = 0`
- `live_prefix_replay_exact_experiments = 4`
- `raw_support_events = 2`
- `raw_contradiction_events = 0`
- `raw_neutral_events = 2`
- `comparable_support_events_acquired = 1`
- `cross_measurement_alignments = 0`
- `cross_measurement_divergences = 1`
- `candidates_ready_for_A32_intake = 0`
- `all_followups_resolved = true`
- `all_requested_followups_completed = false`
- `outcome_status = SAGE_CONTROLLED_FOLLOWUP_ACQUISITION_PARTIAL_CONTROL_SURFACE_LIMIT_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5H`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Resolution des followups :

| candidat | followup | resolution |
|---|---|---|
| `ACTION6 {"x":26,"y":57}` | controle distinct dans 2 contextes | `BLOCKED_NO_DISTINCT_LEGAL_CONTROL_ACTION` |
| `ACTION5` | troisieme support comparable | `ACQUIRED_COMPARABLE_SUPPORT_CANDIDATE_ONLY` |
| `ACTION5` | controle distinct | `BLOCKED_NO_DISTINCT_LEGAL_CONTROL_ACTION` |
| `ACTION5` | remesure clusters 002/003 | `ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE_CANDIDATE_ONLY` |

Surface de controle :

- Pour `ACTION6`, les contextes audites exposent uniquement `ACTION5` et
  plusieurs variantes parametrees de `ACTION6`.
- Pour `ACTION5`, le contexte audite expose uniquement `ACTION5` et plusieurs
  variantes parametrees de `ACTION6`.
- `RESET` reste exclu des controles scientifiques.
- Les variantes d'arguments d'`ACTION6` ne sont pas comptees comme de nouvelles
  actions de controle distinctes par le contrat A32 historique.

Remesure croisee `ACTION5` :

- Lecture locale alignee sur les deux clusters :
  - `changed_pixels = 21` ;
  - `local_patch_available = false` ;
  - `local_changed_pixels = 0`.
- Lecture object-delta divergente :
  - cluster `002` : `object_count_delta = 0`, deltas couleur
    `{"0":-1,"3":1}` ;
  - cluster `003` : `object_count_delta = -1`, delta couleur `{"0":-1}`.
- Les clusters restent donc explicitement lies mais non fusionnes.
- Cette divergence n'est ni une contradiction scientifique ni une refutation :
  elle montre que l'effet object-centric depend du contexte.

Mise a jour des dossiers :

| candidat | support brut avant | nouveau support comparable | support brut apres | contextes | controles distincts | statut |
|---|---:|---:|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | 3 | 0 | 3 | 3 | 1 | controle distinct indisponible |
| `ACTION5` | 2 | 1 | 3 | 3 | 1 | support atteint, controle distinct indisponible |

Les deux dossiers restent `CANDIDATE_ONLY` et ne sont pas prets pour une intake
A32 confirmatoire avec les seuils historiques.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.controlled_followup_acquisition `
  --source-sage5g diagnostics\sage\sage5g_a32_review_handoff.json `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --source-sage5f diagnostics\sage\sage5f_mini_frontier_event_consolidation.json `
  --out diagnostics\sage\sage5h_controlled_followup_acquisition.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_controlled_followup_acquisition.py `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5i a maintenant parcouru toute la frontiere candidate SAGE.5e rejouable,
et non plus seulement les trois contextes necessaires aux followups SAGE.5h.
Les 24 contextes ont ete reproduits exactement. Aucun ne rend legale une
troisieme famille d'action : la surface action-distincte est donc epuisee dans
ce perimetre borne, sans que cet epuisement constitue une refutation globale.

## SAGE.5i - Control-surface expansion

Objectif :

- Lire les artefacts candidate-only SAGE.5e, SAGE.5g et SAGE.5h.
- Selectionner toutes les requests SAGE.5e qui correspondent exactement aux
  deux signatures candidates SAGE.5g.
- Rejouer chaque contexte et exiger un hash identique avant d'auditer les
  actions legales.
- Executer un nouveau controle dans au moins deux contextes si une troisieme
  famille d'action est disponible.
- Sinon, produire une preuve d'epuisement limitee a cette frontiere rejouable.
- Inventorier separement les variantes parametrees sans les compter comme des
  familles d'action distinctes.
- Proposer les choix protocolaires a A32 sans modifier A32/A33 ni produire de
  verdict automatique.
- Garder `support=0`, aucune revision et aucun write A32/A33.

Ajouts :

- `theory/sage/control_surface_expansion.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_control_surface_expansion.py`
- `diagnostics/sage/sage5i_control_surface_expansion.json`

Run du 2026-07-18 :

- `source_requests_available = 24`
- `candidates_evaluated = 2`
- `candidate_contexts_considered = 24`
- `unique_candidate_contexts = 24`
- `context_surface_audits = 24`
- `replay_exact_context_surface_audits = 24`
- `action_surface_signature_counts = {"ACTION5,ACTION6":24}`
- `candidate_context_counts = {"ACTION5":13,"ACTION6":11}`
- `contexts_with_third_action_family = 0`
- `candidates_with_action_distinct_surface_exhausted = 2`
- `candidates_with_new_distinct_control = 0`
- `control_expansion_experiments_executed = 0`
- `parameterized_control_option_counts = {"ACTION5":4,"ACTION6":3}`
- `candidates_ready_for_A32_intake = 0`
- `bounded_action_distinct_exhaustion_proven = true`
- `bounded_scope_only = true`
- `all_candidate_contexts_audited_exact = true`
- `a32_parameterized_protocol_proposal_generated = true`
- `outcome_status = SAGE_CONTROL_SURFACE_EXPANSION_ACTION_DISTINCT_EXHAUSTED_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5I`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Surface auditee :

| candidat | contextes exacts | familles legales | troisieme famille | options parametrees |
|---|---:|---|---:|---:|
| `ACTION6 {"x":26,"y":57}` | 11 | `ACTION5`, `ACTION6` | 0 | 3 |
| `ACTION5` | 13 | `ACTION5`, `ACTION6` | 0 | 4 |

Les trois options du candidat `ACTION6` sont les autres positions
`ACTION6 {"x":18|34|42,"y":57}`. Les quatre options du candidat `ACTION5`
sont `ACTION6 {"x":21|27|33|39,"y":28}`. Elles sont conservees comme
interventions parametrees potentielles, jamais comme nouvelles familles
d'action.

Conclusion bornee :

- SAGE.5i prouve uniquement qu'aucune troisieme famille d'action n'est
  disponible dans les 24 contextes candidats rejouables de SAGE.5e.
- Cette preuve ne couvre ni tous les etats atteignables de `sb26`, ni un autre
  jeu, et ne compte pas comme refutation.
- Les deux candidats conservent 3 observations brutes, 3 contextes independants,
  zero contradiction et une seule famille de controle distincte.
- Le seuil historique A32 de deux familles de controle reste donc non satisfait.
- Aucune variante parametree n'est requalifiee automatiquement et aucune intake
  confirmatoire A32 n'est declenchee.

Proposition protocolaire A32 :

- Conserver le seuil action-distinct et garder les candidats non resolus ; ou
- autoriser explicitement un protocole pre-enregistre de controles parametres ;
  ou
- rejeter les candidats comme non identifiables dans la surface actuelle.
- En cas d'autorisation parametree : arguments distincts de la cible, replay
  exact dans des contextes distincts, au moins deux variantes et zero
  contradiction, sans les renommer en familles d'action distinctes.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.control_surface_expansion `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --source-sage5g diagnostics\sage\sage5g_a32_review_handoff.json `
  --source-sage5h diagnostics\sage\sage5h_controlled_followup_acquisition.json `
  --out diagnostics\sage\sage5i_control_surface_expansion.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_control_surface_expansion.py `
  tests\test_sage_controlled_followup_acquisition.py `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Decision A32.4 du 2026-07-18 :

- A32.4 choisit
  `AUTHORIZE_PRE_REGISTERED_PARAMETERIZED_CONTROL_PROTOCOL` pour les deux
  candidats.
- L'exigence action-distincte reste la regle par defaut ; l'autorisation est une
  exception candidate-specifique et bornee.
- Deux variantes extremes sont pre-enregistrees par candidat :
  - cible `ACTION6 {"x":26,"y":57}` : controles `x=18` et `x=42` ;
  - cible `ACTION5` : controles `ACTION6 x=21` et `ACTION6 x=39`, `y=28`.
- Chaque variante doit etre executee dans deux contextes replay-exacts, budgets
  50 et 300 : 8 experiences demandees au total.
- Aucune experience n'est executee par A32.4 ; les deux hypotheses restent
  unresolved, `support=0`, non A33-ready.
- Artefact :
  `diagnostics/a32/unknown_game_control_protocol_decisions.json`.

Transition :

SAGE.5j a execute les huit experiences A32.4 exactement comme pre-enregistrees.
Aucune variante, aucun contexte, aucun argument ni aucune mesure n'a ete
substitue. Les deux candidats donnent des resultats opposes mais parfaitement
stables sur leurs quatre repetitions : `ACTION6` est non discriminant face aux
autres positions `ACTION6`, tandis qu'`ACTION5` reste discriminant face aux
positions `ACTION6` selectionnees.

## SAGE.5j - Pre-registered parameterized control acquisition

Objectif :

- Lire la decision A32.4 et les requests replayables SAGE.5e.
- Verifier que les deux hypotheses A32.4 sont toujours unresolved, `support=0`,
  sans confirmation, refutation ni write A33.
- Consommer uniquement les huit experiences `PRE_REGISTERED_NOT_EXECUTED`.
- Faire correspondre chaque experience a sa request SAGE.5e par identifiant,
  cible, arguments, contexte, budget et mesure.
- Rejouer independamment les bras cible et controle depuis le meme contexte.
- Exiger les arguments exacts des controles `18/42` et `21/39`.
- Interdire toute substitution post-hoc de variante, contexte ou mesure.
- Classer chaque paire comme discriminante, non discriminante ou controle
  superieur, sans transformer ce resultat en verdict scientifique.
- Garder `support=0`, aucune revision, aucun write A32/A33.

Ajouts :

- `theory/sage/parameterized_control_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_parameterized_control_acquisition.py`
- `diagnostics/sage/sage5j_parameterized_control_acquisition.json`

Run du 2026-07-18 :

- `source_protocol_decisions=2`
- `pre_registered_experiments_consumed=8`
- `experiments_executed=8`
- `experiments_blocked=0`
- `live_prefix_replay_exact_experiments=8`
- `protocol_exact_match_experiments=8`
- `protocol_substitutions_detected=0`
- `target_control_pairs_executed=8`
- `parameterized_variants_executed=4`
- `variant_replications_completed=4`
- `raw_support_events=4`
- `raw_contradiction_events=0`
- `raw_neutral_events=4`
- `candidates_evaluated=2`
- `candidate_protocols_complete=2`
- `candidates_with_discriminating_parameterized_controls=1`
- `candidates_with_non_discriminating_parameterized_controls=1`
- `candidates_with_control_exceeding_target=0`
- `candidates_ready_for_A32_protocol_result_review=2`
- `all_pre_registered_experiments_executed_exactly=true`
- `gate_passed=true`
- `outcome_status=SAGE_PARAMETERIZED_CONTROL_ACQUISITION_MIXED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_5J`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats par candidat :

| candidat | controles parametres | paires exactes | signal cible | signal controle | resultat |
|---|---|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | `ACTION6 x=18/42,y=57` | 4 | 5 | 5 | `PARAMETERIZED_CONTROLS_NON_DISCRIMINATING_CANDIDATE_ONLY` |
| `ACTION5` | `ACTION6 x=21/39,y=28` | 4 | 21 | 4 | `PARAMETERIZED_CONTROLS_DISCRIMINATING_CANDIDATE_ONLY` |

Lecture scientifique :

- Les quatre controles parametres du candidat `ACTION6` reproduisent exactement
  le meme signal local que la cible. Le protocole n'identifie donc pas un effet
  specifique a `x=26`.
- Les quatre controles du candidat `ACTION5` restent nettement differents de la
  cible, avec un delta apparie constant de `17`.
- Les evenements bruts `4 support / 4 neutral / 0 contradiction` ne sont pas
  comptes comme support scientifique par SAGE.
- Les evenements neutres `ACTION6` ne sont pas comptes comme refutation.
- Les variantes parametrees ne sont toujours pas renommees en familles d'action
  distinctes.
- Les deux dossiers sont complets et prets pour une revue A32 du resultat du
  protocole ; A32 reste seul autorise a decider.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.parameterized_control_acquisition `
  --source-a32-4 diagnostics\a32\unknown_game_control_protocol_decisions.json `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5j_parameterized_control_acquisition.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_parameterized_control_acquisition.py `
  tests\test_a32_unknown_game_control_protocol_decisions.py `
  tests\test_sage_control_surface_expansion.py `
  tests\test_sage_controlled_followup_acquisition.py -q
```

Retour A32.5 du 2026-07-18 :

- A32.5 confirme ACTION5 uniquement dans le scope borne de `sb26-7fbdac44` :
  quatre paires exactes discriminantes, support scientifique `4`, aucune
  contradiction.
- A32.5 conserve `ACTION6 {"x":26,"y":57}` unresolved et non identifiable :
  les quatre controles ACTION6 parametres reproduisent le signal cible `5`.
- Cette non-discrimination n'est pas une refutation d'un eventuel effet ACTION6
  position-invariant.
- Un seul dossier, ACTION5, est pret pour revue du registre A33 ; A32.5
  n'effectue aucun write A33.
- Artefact :
  `diagnostics/a32/unknown_game_parameterized_control_revision_decisions.json`.

Retour A33.2 du 2026-07-18 :

- ACTION5 est enregistree dans
  `diagnostics/a33/scoped_unknown_game_registry.json` avec son jeu, son
  candidat, ses quatre contextes, sa mesure, ses budgets et ses deux controles
  verrouilles.
- Le support `4` provient exclusivement du verdict A32.5 ; A33.2 ne reconfirme
  pas la mecanique et ne generalise pas son scope.
- ACTION6 unresolved est exclue explicitement du registre confirme.
- Le registre A33.1 et ses consommateurs A34-A39 ne sont pas modifies.

## SAGE.6 - Second unknown-game bounded transfer

Objectif :

- Fermer explicitement la boucle source SAGE.5j -> A32.5 -> A33.2 avant de
  changer de jeu.
- Auditer les jeux `public_unseen` dans leur ordre fixe, sans lire de metrique
  d'outcome pour choisir le second jeu.
- Exiger pour le jeu selectionne : aucune trace humaine, aucune trace M2, aucun
  prior specifique et une identite differente de `sb26`.
- Rejouer la discipline bornee SAGE.5 sur budgets `50`, `150`, `300`.
- Verifier la legalite des actions, les switches, le trigger progress-stall, la
  repetition, le taux terminal et la nouveaute d'etat.
- Lire A33.2 uniquement comme garde de quarantaine : ne reutiliser ni ACTION5
  confirmee sur `sb26`, ni le candidat ACTION6 unresolved.
- Garder le resultat candidate-only, `support=0`, sans write A32/A33.

Selection pre-execution :

| rang fixe | jeu | hygiene unknown | decision |
|---:|---|---|---|
| 1 | `wa30-ee6fef47` | oui | selectionne |
| 2 | `tn36-ab4f63cc` | oui | eligible, non selectionne |
| 3 | `ft09-0d8bbf25` | non, traces humaines et M2 | exclu connu |
| 4 | `cn04-65d47d14` | non, traces humaines et M2 | exclu connu |
| 5 | `sb26-7fbdac44` | oui | exclu car jeu source |

La selection de `wa30` provient uniquement de
`PUBLIC_UNSEEN_FIXED_ORDER_BEFORE_EXECUTION`. Les resultats de `wa30` et
`tn36` n'interviennent jamais dans ce choix.

Ajouts :

- `theory/sage/second_unknown_game_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_transfer.py`
- `diagnostics/sage/sage6_second_unknown_game_transfer_results.json`

Run du 2026-07-18 :

- `source_game_id=sb26-7fbdac44`
- `selected_second_game_id=wa30-ee6fef47`
- `candidate_games_audited=5`
- `eligible_unknown_games=2`
- `known_or_seen_candidates_excluded=2`
- `source_or_registry_scope_candidates_excluded=1`
- `budgets_evaluated=[50,150,300]`
- `budgets_gate_passed=3`
- `budgets_total=3`
- `all_budgets_gate_passed=true`
- `budgets_with_progress_stall_detected=[50,150,300]`
- `subgoal_switches_total=98`
- `max_repeated_action_arg_rate=0.5`
- `max_terminal_rate=0.005`
- `min_unique_state_signatures=37`
- `max_unique_state_signatures=132`
- `levels_completed_max=0`
- `source_a33_2_entries_quarantined=1`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`
- `wrong_confirmations=0`

Resultats par budget :

| budget | gate | switches | progress-stall | repetition | terminal | signatures uniques | niveaux |
|---:|---|---:|---|---:|---:|---:|---:|
| 50 | passe | 12 | oui, 12 switches | 0.48 | 0.0 | 37 | 0 |
| 150 | passe | 37 | oui, 37 switches | 0.493333 | 0.0 | 101 | 0 |
| 300 | passe | 49 | oui, 49 switches | 0.5 | 0.005 | 132 | 0 |

Lecture :

- SAGE.6 etend le probe technique borne a un deuxieme jeu inconnu reel. Il ne
  constitue toujours ni un benchmark ni une preuve de competence de jeu.
- Contrairement a `sb26`, le trigger progress-stall est effectivement detecte
  sur les trois budgets de `wa30`. SAGE.6 fournit donc une seconde validation
  de transfert de la discipline de boucle, sur une famille warehouse/navigation.
- La repetition reste sous le seuil catastrophique `0.9` et le taux terminal
  sous `0.05`, mais `levels_completed=0` : aucune reussite ludique n'est
  revendiquee.
- L'entree ACTION5 A33.2 reste verrouillee a `sb26`; aucune action, mecanique ou
  evidence de cette memoire n'est appliquee a `wa30`.
- Les observations `wa30` restent candidate-only et ne sont pas du support
  scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_transfer `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-a33-2 diagnostics\a33\scoped_unknown_game_registry.json `
  --m2-dataset-manifest diagnostics\m2\arc_lewm_dataset_manifest.json `
  --human-traces-dir human_traces `
  --budgets 50 150 300 `
  --out diagnostics\sage\sage6_second_unknown_game_transfer_results.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_a33_scoped_unknown_game_registry.py -q
```

## SAGE.6a - Second-game switch attribution and live mini-frontier

Objectif :

- Reproduire exactement les 98 switches `wa30` de SAGE.6 sur les budgets
  pre-enregistres `50`, `150`, `300`.
- Attribuer chaque switch a son trigger et a son sous-but, sans confondre une
  garde terminale avec le compteur de switches source.
- Mesurer la dependance au placeholder `rerun_m2_m3`.
- Rejouer les prefixes live correspondants et convertir un sous-ensemble borne
  de ces placeholders en hypotheses et requests M3 testables.
- Preserver la quarantaine A33.2 de `sb26`, `support=0` et l'absence de write
  A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_switch_frontier.py`
- generalisation parametree, retrocompatible SAGE.5c, de
  `theory/sage/live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_switch_frontier.py`
- `diagnostics/sage/sage6a_switch_attribution_mini_frontier.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets_evaluated=[50,150,300]`
- `source_switches_expected=98`
- `switches_reproduced=98`
- `source_switch_count_reproduced_exactly=true`
- `switches_due_to_progress_stall_or_repeat_collapse=98`
- `true_exploratory_switches=66`
- `active_counterfactual_switches=32`
- `reposition_switches=34`
- `placeholder_rerun_m2_m3_switches=32`
- `source_placeholder_switch_ratio=0.326531`
- `source_placeholder_dependency_under_threshold=true`
- `total_switch_events_observed=99`
- `terminal_guard_events_outside_source_switch_count=1`
- `max_generated_requests=20`
- `mini_frontier_hypotheses_generated=20`
- `effective_requests_generated=20`
- `effective_request_ratio=0.625`
- `unresolved_placeholder_switches_after_generation=12`
- `residual_placeholder_switch_ratio=0.122449`
- `unique_context_snapshot_hashes=12`
- `all_requests_ready_for_m3=true`
- `all_requests_live_prefix_replayable=true`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_SWITCH_FRONTIER_GENERATED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6A`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Attribution et generation par budget :

| budget | switches source | evenements observes | contrefactuels | repositionnements | placeholders | requests generees | placeholders residuels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 12 | 12 | 4 | 4 | 4 | 4 | 0 |
| 150 | 37 | 37 | 12 | 13 | 12 | 12 | 0 |
| 300 | 49 | 50 | 16 | 17 | 16 | 4 | 12 |

Le cinquantieme evenement du budget 300 est un `stop_safe_hold` terminal. Il
est conserve dans l'audit mais reste hors des 49 switches de sous-objectif
comptes par SAGE.6. Cette separation explique les 99 evenements observes sans
alterer la reproduction exacte des 98 switches source.

Les 20 hypotheses appartiennent a la famille candidate
`local_patch_change_candidate` et ciblent `ACTION2`. Leur distribution suit
l'ordre fixe des budgets sous un plafond global : 4 au budget 50, 12 au budget
150 et 4 au budget 300. Chaque request porte un prefixe live rejouable, un hash
de contexte et le statut `READY_FOR_M3`. Les 20 generations ne sont ni du
support, ni une evidence, ni une confirmation.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_switch_frontier `
  --source-sage6 diagnostics\sage\sage6_second_unknown_game_transfer_results.json `
  --budgets 50 150 300 `
  --max-generated-requests 20 `
  --out diagnostics\sage\sage6a_switch_attribution_mini_frontier.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_switch_frontier.py `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py -q
```

## SAGE.6b - Stratified second-game M3 execution

Objectif :

- Selectionner exactement deux requests SAGE.6a par budget `50`, `150`, `300`.
- Maximiser la diversite globale des hashes de contexte, puis la dispersion des
  steps, sans consulter les resultats d'execution.
- Rejouer le meme prefixe live pour la cible et le controle dynamique.
- Exiger la verification du hash avant chaque action cible ou controle.
- Executer `ACTION2` contre le premier controle distinct pre-enregistre,
  `ACTION1`, et conserver les deltas comme evenements bruts candidate-only.
- Preserver la quarantaine A33.2, `support=0` et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_m3_execution.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_m3_execution.py`
- `diagnostics/sage/sage6b_second_unknown_game_m3_execution.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets_available=[50,150,300]`
- `requests_available=20`
- `requests_per_budget=2`
- `requests_selected=6`
- `requests_selected_by_budget={50:2,150:2,300:2}`
- `unique_selected_context_snapshot_hashes=6`
- `selected_source_steps_by_budget={50:[12,48],150:[132,144],300:[24,36]}`
- `requests_executed=6`
- `requests_executed_by_budget={50:2,150:2,300:2}`
- `requests_blocked=0`
- `live_prefix_replay_exact_events=6`
- `context_snapshot_hash_verified_events=6`
- `target_actions_executed={ACTION2:6}`
- `control_actions_executed={ACTION1:6}`
- `hypothesis_families_executed={local_patch_change_candidate:6}`
- `target_signal_total=194`
- `control_signal_total=34`
- `controlled_effect_sizes=[0,32,32,32,32,32]`
- `positive_effect_events=5`
- `zero_effect_events=1`
- `negative_effect_events=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_M3_EXECUTION_COMPLETED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6B`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats par budget :

| budget | steps | execution | replay/hash | signal cible | signal controle | effets bruts |
|---:|---|---:|---:|---:|---:|---|
| 50 | 12, 48 | 2/2 | 2/2 | 65 | 33 | 1 positif, 1 neutre |
| 150 | 132, 144 | 2/2 | 2/2 | 65 | 1 | 2 positifs |
| 300 | 24, 36 | 2/2 | 2/2 | 64 | 0 | 2 positifs |

La selection est pre-execution. Les budgets disposant du moins de contextes
sont alloues en premier pour que les budgets 50 et 300 se partagent leurs
quatre hashes communs. Le budget 150 fournit ensuite deux contexts tardifs
encore distincts. Les six hashes selectionnes sont donc uniques.

Les cinq `raw_support_events` indiquent seulement que, dans ces contextes live,
le signal local mesure apres ACTION2 depasse celui mesure apres ACTION1. Le
sixieme contexte donne un delta nul. SAGE.6b ne compte aucun de ces evenements
comme support scientifique, ne traite pas l'evenement neutre comme refutation
et ne produit aucun verdict.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_m3_execution `
  --source-sage6a diagnostics\sage\sage6a_switch_attribution_mini_frontier.json `
  --requests-per-budget 2 `
  --out diagnostics\sage\sage6b_second_unknown_game_m3_execution.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_m3_execution.py `
  tests\test_sage_second_unknown_game_switch_frontier.py `
  tests\test_sage_live_mini_frontier_m3_executor.py -q
```

## SAGE.6c - Context-preserving event consolidation

Objectif :

- Transformer les six executions SAGE.6b en enregistrements comparables sans
  rejouer l'environnement.
- Conserver exactement un cluster par hash de contexte live.
- Interdire toute fusion inter-contexte et utiliser les groupes d'effet comme
  index de comparaison uniquement.
- Evaluer la stabilite du delta ACTION2/ACTION1 entre budgets avec des seuils
  explicites de contextes, budgets et dispersion d'effet.
- Conserver l'evenement neutre comme exception contextuelle, jamais comme
  contradiction ou refutation.
- Bloquer la revue A32 tant que la diversite de controle reste insuffisante.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_event_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_event_consolidation.py`
- `diagnostics/sage/sage6c_second_unknown_game_event_consolidation.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `event_records=6`
- `context_clusters=6`
- `singleton_context_clusters=6`
- `cross_context_merges_performed=0`
- `all_contexts_preserved_without_merge=true`
- `effect_signature_groups=2`
- `stable_positive_multi_budget_groups=1`
- `neutral_effect_groups=1`
- `stable_positive_contexts=5`
- `stable_positive_events=5`
- `stable_positive_budgets=[50,150,300]`
- `stable_positive_effect_size=32`
- `stable_positive_effect_spread=0`
- `stable_positive_across_all_budgets=true`
- `neutral_context_exceptions=1`
- `negative_context_exceptions=0`
- `context_sensitive_exception_detected=true`
- `distinct_control_actions=1`
- `control_diversity_sufficient_for_a32_review=false`
- `candidate_handoff_frontiers=1`
- `ready_for_A32_handoff_compilation=1`
- `ready_for_A32_review=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_EVENTS_CONSOLIDATED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6C`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Clusters de contexte :

| cluster | budget | step | effet ACTION2-ACTION1 | statut candidate-only |
|---|---:|---:|---:|---|
| `001` | 50 | 12 | 0 | `NEUTRAL_CONTEXT_CANDIDATE_ONLY` |
| `002` | 50 | 48 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `003` | 150 | 132 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `004` | 150 | 144 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `005` | 300 | 24 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `006` | 300 | 36 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |

Groupes de comparaison :

| groupe | contexts | budgets | effet | statut |
|---|---:|---|---:|---|
| `001` | 1 | 50 | 0 | `LOCAL_NEUTRAL_CANDIDATE_ONLY` |
| `002` | 5 | 50/150/300 | +32, dispersion 0 | `STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY` |

Le groupe `002` indexe cinq clusters distincts mais ne les fusionne pas. Le
cluster neutre `001` reste une exception explicite du meme protocole
ACTION2/ACTION1. Le statut global est donc
`STABLE_POSITIVE_MULTI_BUDGET_WITH_NEUTRAL_EXCEPTION_CANDIDATE_ONLY`.

La frontiere est prete pour compilation d'un handoff, pas pour revue A32. Un
seul controle distinct, ACTION1, a ete execute alors que le seuil est fixe a
deux. Les followups requis sont : ajouter un controle distinct par budget,
repliquer le contexte neutre et conserver les frontieres entre clusters.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_event_consolidation `
  --source-sage6b diagnostics\sage\sage6b_second_unknown_game_m3_execution.json `
  --min-stable-contexts 3 `
  --min-stable-budgets 2 `
  --max-positive-effect-spread 0 `
  --min-distinct-controls-for-a32-review 2 `
  --out diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py `
  tests\test_sage_mini_frontier_event_consolidation.py -q
```

La compilation candidate-only demandee ici est realisee par SAGE.6d ci-dessous.

SAGE.6c autorise maintenant a dire : SAGE sait distinguer un motif controle
stable multi-budget d'une exception contextuelle tout en preservant les
frontieres entre contexts. Il ne faut pas dire : SAGE a prouve une mecanique
ACTION2, que l'effet est universel sur `wa30`, ou qu'un verdict A32 est pret.

## SAGE.6d - Second-game handoff compiler

Objectif :

- Compiler la frontiere SAGE.6c en un dossier candidate-only executable lors
  d'une iteration ulterieure, sans rejouer l'environnement.
- Choisir de maniere deterministe le premier controle suggere qui soit distinct
  de la cible ACTION2 et du controle deja execute ACTION1.
- Pre-enregistrer une experience ACTION2/ACTION3 sur un contexte positif stable
  de chacun des budgets 50, 150 et 300.
- Pre-enregistrer une replication exacte ACTION2/ACTION1 du contexte neutre du
  budget 50.
- Embarquer pour chaque protocole le replay live complet, ses arguments, son
  hash de contexte et ses conditions d'interpretation fixees avant execution.
- Conserver les six clusters SAGE.6c dans un manifeste non fusionne.
- Bloquer la revue A32 jusqu'a l'execution des quatre followups.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_handoff_compiler.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_handoff_compiler.py`
- `diagnostics/sage/sage6d_second_unknown_game_handoff.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `source_candidate_handoff_frontiers=1`
- `handoff_items=1`
- `pre_registered_followup_protocols=4`
- `control_diversity_protocols=3`
- `control_diversity_budgets=[50,150,300]`
- `neutral_context_replication_protocols=1`
- `pre_registered_new_control_actions=[ACTION3]`
- `executed_distinct_control_actions=1`
- `projected_distinct_control_actions_after_execution=2`
- `context_clusters_preserved=6`
- `protocol_contexts=4`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `ready_for_followup_execution=1`
- `ready_for_A32_review=0`
- `execution_performed=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_HANDOFF_COMPILED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6D`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Protocoles pre-enregistres :

| type | budget | cluster | step | comparaison | effet precedent |
|---|---:|---|---:|---|---:|
| diversite de controle | 50 | `002` | 48 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| diversite de controle | 150 | `003` | 132 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| diversite de controle | 300 | `005` | 24 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| replication neutre | 50 | `001` | 12 | ACTION2 - ACTION1 | 0 |

ACTION3 est le premier controle suggere par les quatre requests sources qui ne
soit ni ACTION2 ni le controle ACTION1 deja execute. Son emploi commun aux trois
budgets isole la variation de contexte et de budget. Le protocole ne suppose pas
que le delta restera +32 : pour les controles ACTION3, un delta strictement
positif est pre-enregistre comme coherence candidate-only et un delta nul ou
negatif comme deviation candidate-only. Pour la replication neutre, zero est la
condition de replication et tout delta non nul est une deviation. Aucun de ces
outcomes ne produit automatiquement support, confirmation, refutation ou write.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_handoff_compiler `
  --source-sage6c diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json `
  --source-sage6b diagnostics\sage\sage6b_second_unknown_game_m3_execution.json `
  --out diagnostics\sage\sage6d_second_unknown_game_handoff.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py -q
```

L'execution exacte demandee ici est realisee par SAGE.6e ci-dessous.

SAGE.6d autorise maintenant a dire : SAGE sait transformer un motif stable avec
exception en plan de falsification multi-budget auditable et fixe avant
execution. Il ne faut pas dire : ACTION3 a confirme le motif, le contexte neutre
a ete replique, ou le dossier est deja pret pour une revue A32.

## SAGE.6e - Exact pre-registered followup execution

Objectif :

- Executer les quatre protocoles SAGE.6d dans leur ordre source, sans selection
  fondee sur les outcomes.
- Rejouer exactement chaque prefixe live pour les bras cible et controle, puis
  verifier les deux hashes de contexte.
- Interdire toute substitution de protocole, contexte, budget, cible ou
  controle.
- Executer ACTION2 contre ACTION3 dans un contexte positif stable de chacun des
  budgets 50, 150 et 300.
- Repliquer ACTION2 contre ACTION1 dans le contexte neutre du budget 50.
- Appliquer les conditions d'interpretation pre-enregistrees sans transformer
  coherence en confirmation ni deviation en refutation.
- Produire une assessment candidate-only de la dependance au controle avant
  toute consolidation ou revue A32.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_followup_execution.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_followup_execution.py`
- `diagnostics/sage/sage6e_second_unknown_game_followup_execution.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `protocols_available=4`
- `protocols_selected=4`
- `protocols_executed=4`
- `protocols_blocked=0`
- `protocols_executed_by_budget={50:2,150:1,300:1}`
- `control_diversity_protocols_executed=3`
- `neutral_replication_protocols_executed=1`
- `live_prefix_replay_exact_events=4`
- `context_snapshot_hash_verified_events=4`
- `protocol_execution_exact_events=4`
- `target_actions_executed={ACTION2:4}`
- `control_actions_executed={ACTION1:1,ACTION3:3}`
- `target_signal_total=130`
- `control_signal_total=130`
- `controlled_effect_sizes=[0,0,0,0]`
- `positive_effect_events=0`
- `negative_effect_events=0`
- `zero_effect_events=4`
- `raw_support_events=0`
- `raw_contradiction_events=0`
- `raw_neutral_events=4`
- `pre_registered_conditions_met=1`
- `pre_registered_deviations=3`
- `distinct_control_matches_target_across_all_budgets=true`
- `neutral_context_replication_matched=true`
- `ready_for_post_execution_consolidation=true`
- `ready_for_A32_review=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_FOLLOWUPS_EXECUTED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6E`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats pre-enregistres :

| type | budget | cluster | comparaison | cible | controle | delta | interpretation |
|---|---:|---|---|---:|---:|---:|---|
| diversite | 50 | `002` | ACTION2 - ACTION3 | 33 | 33 | 0 | deviation candidate-only |
| diversite | 150 | `003` | ACTION2 - ACTION3 | 33 | 33 | 0 | deviation candidate-only |
| diversite | 300 | `005` | ACTION2 - ACTION3 | 32 | 32 | 0 | deviation candidate-only |
| replication | 50 | `001` | ACTION2 - ACTION1 | 32 | 32 | 0 | condition satisfaite candidate-only |

Le controle ACTION3 egale ACTION2 sur les trois budgets et annule ainsi les
deltas +32 observes contre ACTION1 dans les memes familles de contextes. Cela
indique un motif candidat sensible a l'identite du controle, pas une mecanique
ACTION2 isolee. La replication exacte du contexte neutre reste a zero. Les trois
deltas nuls ACTION2/ACTION3 sont des deviations par rapport a la condition
positive pre-enregistree; ils ne sont ni contradictions scientifiques ni
refutations. La replication neutre satisfaite n'est pas une confirmation.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_followup_execution `
  --source-sage6d diagnostics\sage\sage6d_second_unknown_game_handoff.json `
  --out diagnostics\sage\sage6e_second_unknown_game_followup_execution.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_followup_execution.py `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py -q
```

La consolidation et la reevaluation demandees ici sont realisees par SAGE.6f
ci-dessous.

SAGE.6e autorise maintenant a dire : SAGE sait executer un plan de falsification
pre-enregistre et detecter qu'un effet apparent depend du controle choisi. Il ne
faut pas dire : ACTION2 est sans effet, ACTION1 ou ACTION3 definit la bonne
baseline scientifique, ou le dossier est deja eligible a une revision A32.

## SAGE.6f - Control-dependence consolidation and A32 eligibility

Objectif :

- Consolider les six observations ACTION2/ACTION1 de SAGE.6c avec les quatre
  followups SAGE.6e, sans nouvelle execution.
- Conserver exactement les six clusters de contexte SAGE.6c et rattacher la
  replication neutre a son contexte d'origine sans la compter comme contexte
  independant.
- Garder les identites ACTION1 et ACTION3 separees dans les groupes d'effet.
- Construire trois comparaisons appairees ACTION1/ACTION3, une par budget.
- Reformuler le candidat comme contraste local dependant du controle, jamais
  comme effet ACTION2 autonome et inconditionnel.
- Evaluer l'eligibilite a une revue A32 avec des criteres explicites : volume
  brut, deux controles, trois contextes appaires, couverture multi-budget,
  replication neutre, absence d'effet negatif et preservation des contextes.
- Emettre un frontier A32 candidate-only si tous les criteres passent, sans
  demander l'intake ni ecrire A32/A33.
- Garder `support=0`, la quarantaine A33.2 et l'absence de verdict SAGE.

Ajouts :

- `theory/sage/second_unknown_game_control_dependence_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_control_dependence_consolidation.py`
- `diagnostics/sage/sage6f_second_unknown_game_control_dependence_consolidation.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `observation_records=10`
- `source_sage6c_observations=6`
- `source_sage6e_observations=4`
- `context_clusters=6`
- `context_clusters_preserved=6`
- `cross_context_merges_performed=0`
- `control_actions=[ACTION1,ACTION3]`
- `distinct_control_actions=2`
- `control_conditioned_effect_groups=3`
- `paired_control_contexts=3`
- `paired_control_budgets=[50,150,300]`
- `paired_control_effect_gaps=[32,32,32]`
- `paired_control_effect_gap_spread=0`
- `action1_positive_contexts=5`
- `action1_neutral_observations=2`
- `action3_neutral_contexts=3`
- `replicated_neutral_contexts=1`
- `negative_effect_events=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=5`
- `eligibility_requirements_passed=14`
- `eligibility_requirements_total=14`
- `missing_eligibility_requirements=[]`
- `candidate_a32_review_frontiers=1`
- `ready_for_A32_review=1`
- `ready_for_A32_review_is_not_verdict=true`
- `a32_intake_requested=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_CONSOLIDATED_A32_REVIEW_ELIGIBLE_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6F`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Groupes conditionnes par controle :

| controle | direction | observations | contextes independants | budgets | effets |
|---|---|---:|---:|---|---|
| ACTION1 | positive | 5 | 5 | 50/150/300 | 32/32/32/32/32 |
| ACTION1 | neutre | 2 | 1 | 50 | 0/0 |
| ACTION3 | neutre | 3 | 3 | 50/150/300 | 0/0/0 |

Clusters consolides :

| cluster | budget | step | observations | controles | statut candidate-only |
|---|---:|---:|---:|---|---|
| `001` | 50 | 12 | 2 | ACTION1 | neutre replique |
| `002` | 50 | 48 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `003` | 150 | 132 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `004` | 150 | 144 | 1 | ACTION1 | positif non apparie |
| `005` | 300 | 24 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `006` | 300 | 36 | 1 | ACTION1 | positif non apparie |

Dans chacun des trois contextes appaires, le signal cible est reproduit entre
les deux experiences, le delta ACTION2/ACTION1 vaut 32 et le delta
ACTION2/ACTION3 vaut 0. Le gap conditionne par controle vaut donc 32 avec une
dispersion nulle sur les trois budgets. La replication du cluster `001` reste
neutre et n'ajoute pas artificiellement un contexte independant.

Les quatorze criteres d'eligibilite passent. Le frontier demande a A32 de revoir
la proposition suivante : le contraste local ACTION2 est positif contre ACTION1
mais neutre contre ACTION3 dans les contextes wa30 appaires. Il interdit une
revue sous la forme `STANDALONE_UNCONDITIONAL_ACTION2_EFFECT`. Cette eligibilite
n'est ni une confirmation, ni une revision, ni une demande d'intake automatique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_control_dependence_consolidation `
  --source-sage6c diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json `
  --source-sage6e diagnostics\sage\sage6e_second_unknown_game_followup_execution.json `
  --out diagnostics\sage\sage6f_second_unknown_game_control_dependence_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_control_dependence_consolidation.py `
  tests\test_sage_second_unknown_game_followup_execution.py `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py -q
```

Suite conseillee apres SAGE.6f :

1. A32.6 - effectuer la revue scientifique du candidat dependant du controle,
   en decidant notamment si ACTION1 est un comparateur valide et si
   l'equivalence ACTION3 rend la revendication ACTION2 autonome non identifiable.
2. Ne soumettre a A33 qu'une eventuelle decision confirmee et strictement limitee
   a `wa30-ee6fef47` et aux conditions de controle/contextes revues.
3. Conserver `tn36` comme troisieme jeu inconnu pre-enregistre, sans le choisir
   sur la base de son outcome.

SAGE.6f autorise maintenant a dire : SAGE sait consolider des observations
multi-controles sans confondre baseline, contexte et support scientifique, puis
preparer un dossier A32 correctement reformule. Il ne faut pas dire : ACTION2 a
une mecanique confirmee, ACTION1 est la baseline correcte, ou le frontier A32 est
deja une decision scientifique.

## SAGE.7 - Third unknown-game bounded transfer

Objectif :

- Fermer la boucle SAGE.6f -> A32.6 -> A33.3 avant de changer de jeu.
- Reprendre l'ordre `public_unseen` fixe sans lire aucune metrique d'outcome.
- Exclure les deux jeux inconnus deja utilises et les scopes enregistres :
  `sb26-7fbdac44` et `wa30-ee6fef47`.
- Selectionner `tn36-ab4f63cc`, prochain jeu inconnu eligible pre-enregistre.
- Rejouer la discipline bornee sur budgets 50, 150 et 300.
- Lire A33.2 et A33.3 uniquement comme gardes de quarantaine ; ne reutiliser ni
  ACTION5/sb26, ni la relation ACTION2/ACTION1/ACTION3 de wa30.
- Auditer la surface d'action live au RESET et conserver les variantes
  parametrees ACTION6 comme variantes d'une seule famille d'action.
- Garder le resultat candidate-only, `support=0`, sans intake ni write A32/A33.

Ajouts :

- `theory/sage/third_unknown_game_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_transfer.py`
- `diagnostics/sage/sage7_third_unknown_game_transfer_results.json`

Run du 2026-07-19 :

- `first_unknown_game_id=sb26-7fbdac44`
- `second_unknown_game_id=wa30-ee6fef47`
- `selected_third_game_id=tn36-ab4f63cc`
- `candidate_games_audited=5`
- `eligible_unknown_games=1`
- `known_or_seen_candidates_excluded=2`
- `prior_or_registry_scope_candidates_excluded=2`
- `outcome_metrics_read_for_selection=false`
- `source_a33_2_entries_quarantined=1`
- `source_a33_3_entries_quarantined=1`
- `budgets_evaluated=[50,150,300]`
- `budgets_gate_passed=3`
- `all_budgets_gate_passed=true`
- `env_steps_total=172`
- `subgoal_switches_total=172`
- `new_candidate_targets_discovered_total=27`
- `rerun_m2_m3_requested_total=145`
- `rerun_m2_m3_effective_requests_generated_total=0`
- `budgets_with_progress_stall_detected=[]`
- `max_terminal_rate=0.016393`
- `max_repeated_action_arg_rate=0`
- `min_unique_state_signatures=51`
- `max_unique_state_signatures=62`
- `levels_completed_max=0`
- `action_families=[ACTION6]`
- `distinct_action_families=1`
- `legal_action_options_count=11`
- `parameterized_action_options_count=11`
- `parameterized_action_variants_counted_as_distinct_actions=false`
- `ready_for_parameterized_mini_frontier=true`
- `required_next_step=PARAMETERIZED_ACTION6_MINI_FRONTIER_REQUIRED_CANDIDATE_ONLY`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Selection pre-execution :

| rang fixe | jeu | decision |
|---:|---|---|
| 1 | `wa30-ee6fef47` | exclu, jeu inconnu deja utilise et scope A33.3 |
| 2 | `tn36-ab4f63cc` | eligible, selectionne avant execution |
| 3 | `ft09-0d8bbf25` | exclu connu |
| 4 | `cn04-65d47d14` | exclu connu |
| 5 | `sb26-7fbdac44` | exclu, jeu inconnu deja utilise et scope A33.2 |

Surface d'action live au RESET :

- une seule famille : `ACTION6` ;
- onze options legales parametrees par `x/y` ;
- aucune variante n'est recomptee comme famille d'action distincte ;
- cette surface autorise une mini-frontiere parametree, pas une conclusion sur
  la mecanique du jeu.

Lecture :

- La discipline de boucle transfere techniquement sur un troisieme jeu inconnu :
  toutes les actions sont legales, aucune repetition catastrophique n'apparait
  et les taux terminaux restent sous le seuil.
- Le trigger progress-stall ne se declenche pas : chaque action change l'etat et
  la boucle alterne entre exploration de cibles ACTION6 et placeholders M2/M3.
- Les 145 demandes `rerun_m2_m3` restent des placeholders, avec zero requete
  effective. SAGE.7 n'est donc pas une boucle d'hypotheses complete.
- Les onze options ACTION6 montrent qu'une diversite parametree est disponible
  des le RESET. SAGE.7a doit la transformer en mini-frontieres live-prefix
  multi-budget sans feindre une diversite de familles d'action.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_transfer `
  --source-sage6 diagnostics\sage\sage6_second_unknown_game_transfer_results.json `
  --source-a33-2 diagnostics\a33\scoped_unknown_game_registry.json `
  --source-a33-3 diagnostics\a33\control_dependent_relational_registry.json `
  --out diagnostics\sage\sage7_third_unknown_game_transfer_results.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_transfer.py `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_a33_control_dependent_relational_registry.py -q
```

Suite requise apres SAGE.7 :

1. SAGE.7a - materialiser une mini-frontiere ACTION6 parametree a partir des
   placeholders, avec contextes live-prefix et budgets distribues.
2. Ne compter aucune variante parametree comme action distincte et conserver
   `support=0`.
3. N'ouvrir A32.7/A33.4 que si les executions controlees ulterieures produisent
   un dossier eligible selon des criteres fixes avant verdict.

SAGE.7 autorise maintenant a dire : SAGE transfere sa discipline bornee sur
trois jeux inconnus successifs et sait detecter qu'une surface mono-action exige
des controles parametres. Il ne faut pas dire : ACTION6 a une mecanique
confirmee, les onze coordonnees constituent onze actions distinctes, ou tn36 est
deja eligible a une revue A32.
