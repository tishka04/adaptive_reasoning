# M3 milestones - journal de mise en place

Derniere mise a jour : 2026-06-29

Ce fichier sert de relais de communication avec l'agent de reflexion /
validation. Il doit etre mis a jour a chaque etape M3 implementee et apres
chaque run experimental.

## Principe fixe

M3 ne confirme jamais une hypothese. M3 planifie et execute des tests
controles, puis ajoute uniquement des evenements experimentaux :
`support_events`, `independent_support_events`,
`reused_control_support_events`, `contradiction_events`.

Le ledger garde toujours :

- `status=UNRESOLVED`
- `revision_status=CANDIDATE_ONLY`
- `support=0`
- `contradictions=0`
- `controlled_test_required=true`
- `wrong_confirmations=0`

Toute confirmation ou refutation reste reservee a A15-A31 apres revision
scientifique explicite.

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| M3.1 - Scientific planner state | Fait | `theory/m3/scientific_planner_state.py`, `tests/test_m3_scientific_planner_state.py` | Agregation sans double counting; support brut / independant / reutilise separes |
| M3.2 - Next controlled test selector | Fait | `theory/m3/next_experiment_selector.py`, `tests/test_m3_next_experiment_selector.py` | Controles dynamiques depuis actions live; controles preferes tries; controle reutilise marque |
| M3.3 - Controlled experiment executor loop | Fait | `theory/m3/scientific_planning_loop.py`, `tests/test_m3_scientific_planning_loop.py`, `diagnostics/m3/scientific_planning_bp35.json` | Budget total 3 sur bp35; 3 experiences controlees totalisees |
| M3.4 - Stop / escalation rules | Fait | `diagnostics/m3/scientific_planning_bp35.json` | Readiness candidate-only basee sur support brut >= 3, support independant >= 2, contradiction = 0 |
| M3.5 - Premier run cible bp35 | Fait | `diagnostics/m3/scientific_planning_bp35.json` | ACTION6 position effect teste contre ACTION3, ACTION4, puis ACTION3 reutilise |
| M3.6 - A15 revision queue bridge | Fait | `theory/m3/a15_revision_queue.py`, `tests/test_m3_a15_revision_queue.py`, `diagnostics/m3/a15_revision_queue.json` | 1 candidat A15 queue, candidate-only, support=0, wrong_confirmations=0 |
| M3.7 - Execute M2 candidate experiment queue | Fait | `theory/m3/m2_candidate_experiment_runner.py`, `tests/test_m3_m2_candidate_experiment_runner.py`, `diagnostics/m3/m2_candidate_experiment_results.json` | 4 requests M2 executees; replay causal exact; metriques secondaires si signal local neutre; no verdict |
| M3.7b - Secondary metric grounding guard | Fait | `theory/m3/m2_candidate_experiment_runner.py`, `tests/test_m3_m2_candidate_experiment_runner.py`, `diagnostics/m3/m2_candidate_experiment_results.json` | `changed_pixels` raw-grounded; `topology_before_after` diagnostic-only tant qu'ungrounded; support=0 preserve |
| M3.7c - ARC-LeWM offline trace context executor | Fait | `theory/m3/m2_candidate_experiment_runner.py`, `tests/test_m3_m2_candidate_experiment_runner.py`, `diagnostics/m3/arc_lewm_m2_candidate_experiment_results.json` | 1 request ARC-LeWM executee depuis frame-before trace; ACTION6 terminal_rate 1.0 vs ACTION3 controls 0.0; support_events=1; support=0 |
| M3.7d - ARC-LeWM terminal-risk replication | Fait | `theory/m3/arc_lewm_terminal_risk_replication.py`, `tests/test_m3_arc_lewm_terminal_risk_replication.py`, `diagnostics/m3/arc_lewm_terminal_risk_replication_requests.json`, `diagnostics/m3/arc_lewm_terminal_risk_replication_results.json` | 14 requests terminal-risk executees offline; 14 sources/episodes, 4 jeux, 6 actions; controles same-game/same-actions; support_events=14; contradiction_events=0; support=0 |
| M3.7e - Fused LLM+WM request runner | Fait | `theory/m3/fused_llm_wm_experiment_runner.py`, `tests/test_m3_fused_llm_wm_experiment_runner.py`, `diagnostics/m3/fused_llm_wm_experiment_results.json` | 3 requests M2.15 executees; 1 request contrefactuelle skippee; support_events=3 tous reutilises depuis M3.7d; independent_source_support_events=0; support=0 |
| M3.7f - Offline-frame counterfactual feasibility probe | Fait | `theory/m3/offline_frame_counterfactual_probe.py`, `tests/test_m3_offline_frame_counterfactual_probe.py`, `diagnostics/m3/offline_frame_counterfactual_feasibility.json` | 1 request `terminal_safe_alternative_action` inspectee; ACTION3 disponible mais aucun env state complet/restorable; frontier recommandee `NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT`; support=0 |
| M3.8 - Observation-to-hypothesis refinement | Fait | `theory/m3/m2_observation_refinement.py`, `tests/test_m3_m2_observation_refinement.py`, `diagnostics/m3/refined_candidate_hypotheses_from_m2.json` | 4 input support_events -> 2 signatures positives uniques -> 1 hypothese raffinee candidate-only |
| M3.9 - Refined follow-up planner | Fait | `theory/m3/refined_followup_planner.py`, `tests/test_m3_refined_followup_planner.py`, `diagnostics/m3/refined_followup_experiment_requests.json` | 1 test de reactivation ACTION6 apres ACTION6/ACTION3/ACTION4; no execution; support=0 |
| M3.10 - Refined follow-up executor | Fait | `theory/m3/refined_followup_executor.py`, `tests/test_m3_refined_followup_executor.py`, `diagnostics/m3/refined_followup_experiment_results.json` | ACTION6 reteste avec args resolus `{x:18,y:0}`; 8 experiences; 0 support_events; 4 contradictions; support=0 |
| M3.11 - Dynamic retarget after repositioning planner | Fait | `theory/m3/dynamic_retarget_followup_planner.py`, `tests/test_m3_dynamic_retarget_followup_planner.py`, `diagnostics/m3/dynamic_retarget_followup_requests.json` | 5 nouveaux args ACTION6 generes apres repositionnement; `{x:18,y:0}` exclu; no execution; support=0 |
| M3.12 - Dynamic retarget follow-up executor | Fait | `theory/m3/dynamic_retarget_followup_executor.py`, `tests/test_m3_dynamic_retarget_followup_executor.py`, `diagnostics/m3/dynamic_retarget_followup_results.json` | 5 retargets executes; 4 args avec support grounded; 16 supports arg-level -> 1 mechanism event; support=0 |
| M3.13 - Dynamic retarget mechanism consolidation | Fait | `theory/m3/dynamic_retarget_mechanism_consolidation.py`, `tests/test_m3_dynamic_retarget_mechanism_consolidation.py`, `diagnostics/m3/dynamic_retarget_mechanism_candidates.json` | 4 retargets reussis + 1 echec -> 1 candidat mecanistique; changed_pixels classe non decisif; support=0 |
| M3.14 - Retarget selection rule induction | Fait | `theory/m3/dynamic_retarget_selection_rule_induction.py`, `tests/test_m3_dynamic_retarget_selection_rule_induction.py`, `diagnostics/m3/dynamic_retarget_selection_rules.json` | 1 set de regles; 3 familles candidates falsifiables; x/y seuls bloques; support=0 |
| M3.15 - Selection rule follow-up planner | Fait | `theory/m3/dynamic_retarget_selection_followup_planner.py`, `tests/test_m3_dynamic_retarget_selection_followup_planner.py`, `diagnostics/m3/dynamic_retarget_selection_followup_requests.json` | 8 requests discriminantes; 6 args explicites + 2 probes patch; no execution; support=0 |
| M3.16 - Selection rule follow-up executor | Fait | `theory/m3/dynamic_retarget_selection_followup_executor.py`, `tests/test_m3_dynamic_retarget_selection_followup_executor.py`, `diagnostics/m3/dynamic_retarget_selection_followup_results.json` | 8 requests consommees; 2 probes dynamiques executees; 6 explicites bloques/exclus; 1 arg unique resolu; support=0 |
| M3.17 - Selection rule evidence consolidation | Fait | `theory/m3/dynamic_retarget_selection_rule_consolidation.py`, `tests/test_m3_dynamic_retarget_selection_rule_consolidation.py`, `diagnostics/m3/dynamic_retarget_selection_rule_consolidation.json` | `local_patch_transformability` meilleure candidate actuelle; `row_or_band` non teste directement; support=0 |
| M3.18 - Patch-similarity expansion planner | Fait | `theory/m3/dynamic_retarget_patch_similarity_expansion_planner.py`, `tests/test_m3_dynamic_retarget_patch_similarity_expansion_planner.py`, `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_requests.json` | 7 args exclus; 1 nouvel arg live `{x:48,y:12}`; 3 requests convergentes; no execution; support=0 |
| M3.19 - Patch-similarity expansion executor | Fait | `theory/m3/dynamic_retarget_patch_similarity_expansion_executor.py`, `tests/test_m3_dynamic_retarget_patch_similarity_expansion_executor.py`, `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_results.json` | 3 rationales -> 1 signature executee; 8 experiences; 4 supports success-metric; support=0 |
| M3.20 - Patch-similarity generativity consolidation | Fait | `theory/m3/dynamic_retarget_patch_similarity_generativity_consolidation.py`, `tests/test_m3_dynamic_retarget_patch_similarity_generativity_consolidation.py`, `diagnostics/m3/dynamic_retarget_patch_similarity_generativity_consolidation.json` | 6 args reussis total; `{x:42,y:12}` + `{x:48,y:12}` comme expansion candidate; ready A32 queue sans verdict; support=0 |
| M3.21 - Patch-similarity A32 revision queue bridge | Fait | `theory/m3/patch_similarity_a32_revision_queue.py`, `tests/test_m3_patch_similarity_a32_revision_queue.py`, `diagnostics/m3/patch_similarity_a32_revision_queue.json` | 1 dossier M3.20 transforme en queue item A32-ready; changed_pixels reste diagnostic; no A32/A33 write; support=0 |
| M3.22 - A32 requested patch-similarity follow-up planner | Fait | `theory/m3/a32_requested_patch_similarity_followup_planner.py`, `tests/test_m3_a32_requested_patch_similarity_followup_planner.py`, `diagnostics/m3/a32_requested_patch_similarity_followup_requests.json` | 2 demandes A32.3 transformees en requests M3; outside-region + alternate-context; no execution; support=0 |
| M3.23 - A32 requested patch-similarity follow-up executor | Fait | `theory/m3/a32_requested_patch_similarity_followup_executor.py`, `tests/test_m3_a32_requested_patch_similarity_followup_executor.py`, `diagnostics/m3/a32_requested_patch_similarity_followup_results.json` | 4 signatures executees; outside-region borne le scope; alternate-context soutient l'expansion; support=0 |
| M3.24 - A32 requested scope follow-up consolidation | Fait | `theory/m3/a32_requested_patch_similarity_scope_consolidation.py`, `tests/test_m3_a32_requested_patch_similarity_scope_consolidation.py`, `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json` | `SCOPE_EXPANDED_CANDIDATE_ONLY`; A33 non pret; policy probe candidate-only pret; support=0 |
| M3.O1 - Objective stop/switch experiment planner | Fait | `theory/m3/objective_stop_switch_experiment_planner.py`, `tests/test_m3_objective_stop_switch_experiment_planner.py`, `diagnostics/m3/objective_stop_switch_experiment_requests.json` | 4 hypotheses M2.O1 -> 4 protocoles objectif; 120 cellules continuer/stop/switch; no execution; support=0 |
| M3.O2 - Objective stop/switch experiment executor | Fait | `theory/m3/objective_stop_switch_experiment_executor.py`, `tests/test_m3_objective_stop_switch_experiment_executor.py`, `diagnostics/m3/objective_stop_switch_experiment_results.json` | 120 liens -> 30 cellules uniques; 16 executees; 8 controles bloques; 6 early-terminal; support=0 |
| M3.O3 - Objective threshold consolidation | Fait | `theory/m3/objective_threshold_consolidation.py`, `tests/test_m3_objective_threshold_consolidation.py`, `diagnostics/m3/objective_threshold_consolidation.json` | Fenetre pre-terminale candidate `[49,64]`; safe max 48; early-terminal min 64; M3.O4 recommande |
| M3.O4 - Refined objective window executor | Fait | `theory/m3/objective_refined_window_executor.py`, `tests/test_m3_objective_refined_window_executor.py`, `diagnostics/m3/objective_refined_window_results.json` | 24 cellules raffinees; prefix 63 continue GAME_OVER; stop reste non terminal; support=0 |
| A32 - M3 revision intake | Fait | `theory/a32_m3_revision_intake.py`, `tests/test_a32_m3_revision_intake.py`, `diagnostics/a32/m3_revision_intake.json` | 1 candidat accepte comme `HypothesisRecord` unresolved, 0 confirmation/refutation |
| M3.G1.1 - Objective-conversion experiment planner | Fait | `theory/m3/objective_conversion_experiment_planner.py`, `tests/test_m3_objective_conversion_experiment_planner.py`, `diagnostics/m3/objective_conversion_experiment_requests.json` | 12 hypotheses M2.G1 -> 12 requests; 22 conditions candidates; 2 controles (hold + relation horizon-matched); delta-vs-hold decisif; support=0 |
| M3.G1.2 - Stop-state objective-conversion experiment executor | Fait | `theory/m3/objective_conversion_experiment_executor.py`, `tests/test_m3_objective_conversion_experiment_executor.py`, `diagnostics/m3/objective_conversion_experiment_results.json` | Safe-stop P3.G1 lambda_0 rejoue (prefix 15), 46 cellules -> 10 uniques, 10 replay-exact, 0 bloque; measurements seulement; support=0 |
| M3.G1.3 - Objective-conversion candidate-only consolidation | Fait | `theory/m3/objective_conversion_experiment_consolidation.py`, `tests/test_m3_objective_conversion_experiment_consolidation.py`, `diagnostics/m3/objective_conversion_experiment_consolidation.json` | `MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY`; 3 signaux de conversion; best `ACTION6,ACTION3` delta +15; support=0; aucun verdict |
| M3.G2 - Multi-safe-stop objective-conversion validation | Fait | `theory/m3/objective_conversion_multi_safe_stop_validation.py`, `tests/test_m3_objective_conversion_multi_safe_stop_validation.py`, `diagnostics/m3/objective_conversion_multi_safe_stop_validation.json` | 8 captures planifiees -> 1 safe-stop unique; signal ACTION6-led reproduit localement mais non multi-safe-stop; `LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY`; support=0; aucun verdict |
| M3.G3 - Safe-stop diversity acquisition / anti-attractor sampler | Fait | `theory/m3/objective_conversion_safe_stop_diversity_sampler.py`, `tests/test_m3_objective_conversion_safe_stop_diversity_sampler.py`, `diagnostics/m3/objective_conversion_safe_stop_diversity_sampler.json` | 26 candidats executes; 13 safe-stops divers acceptes; `SUFFICIENT_FOR_M3_G4`; aucune sequence objective testee; support=0; aucun verdict |
| M3.G4 - Multi-diverse-safe-stop objective-conversion validation | Fait | `theory/m3/objective_conversion_diverse_safe_stop_validation.py`, `tests/test_m3_objective_conversion_diverse_safe_stop_validation.py`, `diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json` | 13 safe-stops diversifies consommes; weak signal 13/13, medium 1/13, terminal risk 1/13; `MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY`; support=0; aucun verdict |
| M3.G5 - Objective-readiness / commit-action experiment planner | Fait | `theory/m3/risk_aware_objective_completion_experiment_planner.py`, `tests/test_m3_risk_aware_objective_completion_experiment_planner.py`, `diagnostics/m3/risk_aware_objective_completion_experiment_requests.json` | 15 hypotheses M2.G2 -> 15 requests controlees readiness/commit/goal/discriminator/selector-gap; `OBJECTIVE_READINESS_EXPERIMENT_REQUESTS_COMPILED_CANDIDATE_ONLY`; no execution; support=0 |
| M3.G6 - Objective-completion experiment executor | Fait | `theory/m3/risk_aware_objective_completion_experiment_executor.py`, `tests/test_m3_risk_aware_objective_completion_experiment_executor.py`, `diagnostics/m3/risk_aware_objective_completion_experiment_results.json` | 336 cellules M3.G5 -> 72 dedupliquees; 66 mesurees, 6 commits bloques; no completion; `PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY`; support=0 |

## M3.1 - Scientific planner state

- Ajout de `ScientificPlanningState`.
- Entrees : ledger M1.3k et artefacts de resultats controles M1.3l/M3.
- Les `controlled_experiments` detailles sont la source principale.
- Les `updated_ledger_entries` ne sont lus qu'en fallback si aucun detail
  d'experience n'est present.
- KPI exposes :
  `open_hypotheses`, `tested_hypotheses`, `untested_hypotheses`,
  `support_events_total`, `independent_support_events_total`,
  `reused_control_support_events_total`, `contradiction_events_total`.
- Discipline : les evenements ne modifient pas `support` ni `contradictions`
  du ledger.

## M3.2 - Next controlled test selector

- Ajout de `PlannedControlledExperiment`.
- Score V1 :
  `2.0 * is_untested + 1.0 * has_support_but_unresolved + 1.0 * has_competing_controls - 0.5 * already_tested_count`.
- Les controles sont construits dynamiquement :
  `live_available_actions - {target_action, RESET}`.
- Tri des controles :
  `ACTION3`, `ACTION4`, `ACTION1`, `ACTION2`, puis autres actions live
  disponibles.
- Les controles preferes indisponibles sont journalises dans
  `skipped_controls`, pas dans `open_questions`.
- Si tous les controles distincts disponibles ont deja ete utilises, le
  controle le moins utilise est repete et marque avec
  `control_reuse_reason=no_unused_distinct_control_available`.
- `open_questions` reste reserve aux questions scientifiques :
  `insufficient_distinct_controls`, `all_controls_support_hypothesis`,
  `contradictory_control_observed`.

## M3.3 - Controlled experiment executor loop

- Ajout de `run_scientific_planning_loop`.
- Defaut :
  - jeu : `bp35-0a0ad940`
  - budget total : `3`
  - entree ledger : `diagnostics/m1/scientific_integration_pretest.json`
  - entree controlee initiale : `diagnostics/m1/controlled_experiment_results.json`
  - sortie : `diagnostics/m3/scientific_planning_bp35.json`
- L'experience M1.3l existante compte dans le budget total.
- La boucle M3 execute seulement les experiences manquantes pour atteindre le
  budget.
- Execution via la machinerie M1.3l : target vs control depuis reset, meme
  metrique, support/contradiction par delta controle.
- Les experiences importees sont normalisees dans l'artefact M3 avec
  `independent_support_events` et `reused_control_support_events`.

## M3.4 - Stop / escalation rules

- `propose_ready_for_A15_revision=true` seulement si :
  - `independent_support_events >= 2`
  - `support_events >= 3`
  - `contradiction_events == 0`
- Meme quand ce flag est vrai, l'artefact garde :
  - `revision_status=CANDIDATE_ONLY`
  - `support=0`
  - `controlled_test_required=true`
- `propose_followup_disambiguation=true` si `contradiction_events >= 1`.

## M3.5 - Premier run cible bp35

Hypothese :

- `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`

Objectif :

- Tester si le signal local associe a `ACTION6` reste specifique face a des
  controles disponibles.

Actions live au reset bp35 :

- `ACTION3`, `ACTION4`, `ACTION6`

Controles preferes indisponibles :

- `ACTION1` : `action_not_available_at_reset`
- `ACTION2` : `action_not_available_at_reset`

Experiences documentees :

| # | Origine | Cible | Controle | Metrique | Baseline | Perturbation | Support brut | Support independant | Support reutilise | Contradiction | Statut |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | M1.3l importee | ACTION6 | ACTION3 | local_patch_before_after | 0.0 | 1.0 | 1 | 1 | 0 | 0 | UNRESOLVED |
| 2 | M3 planifiee | ACTION6 | ACTION4 | local_patch_before_after | 0.0 | 1.0 | 1 | 1 | 0 | 0 | UNRESOLVED |
| 3 | M3 planifiee | ACTION6 | ACTION3 | local_patch_before_after | 0.0 | 1.0 | 1 | 0 | 1 | 0 | UNRESOLVED |

Note experience 3 :

- `control_reuse_reason=no_unused_distinct_control_available`
- Le troisieme support est donc moins independant que les deux premiers.

KPI du run bp35 :

- `controlled_experiments_run=3`
- `support_events=3`
- `independent_support_events=2`
- `reused_control_support_events=1`
- `contradiction_events=0`
- `propose_ready_for_A15_revision=true`
- `revision_status=CANDIDATE_ONLY`
- `support=0`
- `wrong_confirmations=0`

Lecture :

- M3 prouve la sequence minimale :
  ledger d'hypothese -> choix du prochain controle -> execution -> evenement
  brut/independant -> choix suivant.
- Le signal `ACTION6` reste supporte face a deux controles distincts bp35.
- Le troisieme support est utile mais explicitement moins independant, car il
  reutilise `ACTION3`.
- Aucune integration M2, A30/A31, ni planner long horizon n'a ete ajoutee.

## M3.6 - A15 revision queue bridge

Objectif :

- Transformer la sortie M3 en file candidate pour A15-A31.
- Ne pas reviser.
- Ne pas produire de verdict.
- Verifier qu'une hypothese M3 prete peut devenir une entree candidate
  consommable sous forme de `HypothesisRecord`.

Ajouts :

- `theory/m3/a15_revision_queue.py`
- `A15RevisionQueueItem`
- `run_a15_revision_queue_generation`
- `diagnostics/m3/a15_revision_queue.json`

Regle d'entree dans la file :

- `propose_ready_for_A15_revision=true` dans l'artefact M3.
- Entree ledger candidate-only :
  - `status=UNRESOLVED`
  - `revision_status=CANDIDATE_ONLY`
  - `support=0`
  - `controlled_test_required=true`
- Evidence minimale :
  - `independent_support_events >= 2`
  - `support_events >= 3`
  - `contradiction_events == 0`

Run du 2026-06-19 :

- entree : `diagnostics/m3/scientific_planning_bp35.json`
- sortie : `diagnostics/m3/a15_revision_queue.json`
- `queue_items=1`
- `candidate_records=1`
- `ready_for_a15_revision_candidates=1`
- `support_events=3`
- `independent_support_events=2`
- `reused_control_support_events=1`
- `contradiction_events=0`
- `revision_status=CANDIDATE_ONLY`
- `support=0`
- `wrong_confirmations=0`

Item mis en file :

- `m3_6::0001::mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`

Record candidat A15 :

- `key=mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- `description=ACTION6 position_effect_candidate via local_patch_before_after`
- `status=unresolved`
- `support=0`
- `contradictions=0`
- `experiments_spent=3`

Lecture :

- M3.6 ne rajoute pas d'intelligence au planner.
- M3.6 agit comme douane : evidence experimentale M3 -> file candidate A15.
- La file expose assez d'evidence pour que A15-A31 puisse decider ensuite entre
  revision, demande de test supplementaire ou rejet, mais elle ne prend pas
  cette decision elle-meme.

## M3.7 - Execute M2 candidate experiment queue

Objectif :

- Consommer `diagnostics/m2/m3_candidate_experiment_requests.json`.
- Rejouer exactement le contexte causal encode par M2.
- Executer target vs controle dynamique M3.
- Ajouter des evenements experimentaux :
  `support_events`, `contradiction_events`, `neutral_events`.
- Escalader vers des metriques secondaires quand la metrique primaire locale
  est neutre mais que `changed_pixels` indique un effet global.
- Ne pas reviser.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Ajouts :

- `theory/m3/m2_candidate_experiment_runner.py`
- `run_m2_candidate_experiment_queue`
- `diagnostics/m3/m2_candidate_experiment_results.json`

Metriques secondaires V1 :

- `changed_pixels`
- `object_positions_before_after`
- `contact_graph_before_after`
- `topology_before_after`

Run du 2026-06-19 :

- entree : `diagnostics/m2/m3_candidate_experiment_requests.json`
- sortie : `diagnostics/m3/m2_candidate_experiment_results.json`
- `m2_requests_ready_for_m3=4`
- `m2_requests_executed=4`
- `controlled_experiments_run=12`
- `primary_metric_experiments=4`
- `secondary_metric_experiments=8`
- `metric_escalations=2`
- `support_events=4`
- `contradiction_events=0`
- `neutral_events=8`
- `diagnostic_only_experiments=2`
- `grounded_metric_experiments=10`
- `raw_changed_pixels_experiments=2`
- `grounding_suppressed_support_events=2`
- `grounding_suppressed_contradiction_events=0`
- `blocked_experiments=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M2`
- `support=0`
- `wrong_confirmations=0`
- `a32_remains_only_verdict_location=true`

Replay critique verifie :

- frontier : `after_ACTION3_live_after_ACTION6`
- replay : `["ACTION6", "ACTION3"]`
- replay args : `[{"x": 18, "y": 0}, {}]`
- target : `ACTION4`
- control dynamique : `ACTION3`
- signature cible :
  `(64, 64):b33aabd922c7b268:0:NOT_FINISHED`

## M3.7b - Secondary metric grounding guard

Objectif :

- Empecher une metrique secondaire non-grounded de produire un
  `support_event`.
- Garder l'observation brute comme diagnostic.
- Autoriser explicitement `changed_pixels` comme signal brut si
  `signal_source=raw_changed_pixels`.

Regle :

- Une metrique peut produire un `support_event` seulement si :
  - `observed_baseline.measurable=true`
  - `observed_perturbation.measurable=true`
  - `observed_baseline.metric == metric`
  - `observed_perturbation.metric == metric`
- Exception :
  - `changed_pixels` peut compter si les deux observations exposent
    `changed_pixels`, et M3 ecrit alors :
    `metric=changed_pixels`, `measurable=true`,
    `signal_source=raw_changed_pixels`.
- Toute autre metrique non-grounded devient :
  - `diagnostic_only=true`
  - `support_events=0`
  - `contradiction_events=0`
  - `raw_support_events` conserve ce qui aurait ete compte sans guard.

Lecture corrigee pour ACTION4 apres `["ACTION6", "ACTION3"]` :

- `local_patch_before_after` : neutre.
- `changed_pixels` : support candidate-only, source brute explicite.
- `object_positions_before_after` : support candidate-only via extracteur.
- `contact_graph_before_after` : neutre.
- `topology_before_after` : diagnostic-only tant qu'aucun extracteur
  topologique reel ne grounde la mesure.

Lecture :

- La metrique primaire `local_patch_before_after` reste neutre pour ACTION4
  dans le contexte live apres ACTION6 puis ACTION3.
- Les metriques secondaires grounded montrent cependant un effet global :
  `changed_pixels` et `object_positions_before_after` produisent des
  support_events candidate-only.
- `topology_before_after` conserve son delta brut, mais ne produit plus de
  support_event tant que sa mesure reste `metric=unknown` /
  `measurable=false`.
- Ces evenements augmentent l'information experimentale disponible, jamais le
  champ `support`.
- A32 reste le seul lieu autorise a produire un verdict scientifique.

## M3.7c - ARC-LeWM offline trace context executor

Objectif :

- Consommer uniquement les requests M2.14e v2 marquees
  `OFFLINE_TRACE_CONTEXT_ONLY`.
- Charger le contexte source depuis `training/m2_arc_lewm_transitions.jsonl`
  via `source_transition_id`.
- Mesurer la transition target observee dans la trace source.
- Comparer contre des controles dynamiques apparies dans le meme jeu et avec
  le meme set d'actions disponibles.
- Ne pas executer les requests `BLOCKED_NOT_TESTABLE`.
- Ne pas reviser, ne pas confirmer, ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/m2_candidate_experiment_runner.py` :
  chemin `offline_trace_context`.
- `tests/test_m3_m2_candidate_experiment_runner.py` :
  fixture `OFFLINE_TRACE_CONTEXT_ONLY`.
- `theory/m1/controlled_followup_experiment.py` :
  `metric_signal` sait lire `terminal_state_after_rollout`.
- `diagnostics/m3/arc_lewm_m2_candidate_experiment_results.json`

Run du 2026-06-29 :

- entree : `diagnostics/m2/arc_lewm_m3_candidate_requests_v2.json`
- request executee :
  `m2m3::m2_14_lewm::terminal_like_latent_neighborhoods::004`
- source :
  `m2_14d::dc22-4c9bff3e::3c719b36eb4d::0239`
- target : `ACTION6 {"x":45,"y":30}`
- controle dynamique : `ACTION3`
- control policy : `same_game_same_available_actions`
- matched_control_samples = 195
- observed_perturbation.terminal_rate = 1.0
- observed_baseline.terminal_rate = 0.0
- delta.effect_size = 1.0
- support_events = 1
- contradiction_events = 0
- support = 0
- truth_status = `NOT_EVALUATED_BY_M2`
- revision_status = `CANDIDATE_ONLY`
- a32_remains_only_verdict_location = true

## M3.7d - ARC-LeWM terminal-risk replication

Objectif :

- Reprendre les signaux `terminal_like_latent_neighborhoods` produits par
  M2.14d.
- Generer plusieurs requests `OFFLINE_TRACE_CONTEXT_ONLY`, non-RESET et non
  `unknown_game`, une par `source_transition_id` distinct.
- Executer ces requests depuis `human_trace_frame_before`.
- Comparer chaque target observee a des controles dynamiques apparies
  `same_game_same_available_actions`.
- Separer les supports par source distincte, les contextes reutilises et les
  doublons de source.
- Compter explicitement les contradictions.
- Garder `support=0`, `revision_status=CANDIDATE_ONLY` et aucun write A32/A33.

Ajouts :

- `theory/m3/arc_lewm_terminal_risk_replication.py`
- `tests/test_m3_arc_lewm_terminal_risk_replication.py`
- `diagnostics/m3/arc_lewm_terminal_risk_replication_requests.json`
- `diagnostics/m3/arc_lewm_terminal_risk_replication_results.json`

Run du 2026-06-29 :

- entree signal :
  `diagnostics/m2/arc_lewm_signal_report.json`
- entree trace :
  `training/m2_arc_lewm_transitions.jsonl`
- requests READY generees = 14
- requests executees = 14
- blocked_experiments = 0
- controlled_experiments_run = 14
- supporting_source_transition_ids = 14
- supporting_source_episodes = 14
- supporting_games = 4
- supporting_target_actions = 6
- target_trace_samples_total = 14
- matched_control_samples_total = 3357
- same_game_same_available_actions_controls = 14
- all_controls_same_game_same_available_actions = true
- support_events = 14
- contradiction_events = 0
- neutral_events = 0
- independent_source_support_events = 14
- duplicate_source_support_events = 0
- unique_context_support_events = 7
- context_reused_support_events = 7
- replication_breadth_low = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_M2`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Lecture :

M3.7d transforme le signal ponctuel M3.7c en replication candidate-only plus
large : les 14 transitions terminal-risk extraites par ARC-LeWM sont
trace-grounded, executables depuis le frame-before source, et montrent toutes
un terminal_rate target superieur aux controles apparies. Le signal couvre 14
episodes distincts, 4 jeux et 6 actions cibles.

Ce n'est toujours pas un verdict A32/A33. Chaque target reste une observation
offline de trace, donc on peut dire que le signal terminal-risk ARC-LeWM se
reproduit dans les traces apparies. On ne dit pas encore que le world model
predirait le terminal en general.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.arc_lewm_terminal_risk_replication `
  --signal-report diagnostics\m2\arc_lewm_signal_report.json `
  --offline-trace-dataset training\m2_arc_lewm_transitions.jsonl `
  --requests-out diagnostics\m3\arc_lewm_terminal_risk_replication_requests.json `
  --out diagnostics\m3\arc_lewm_terminal_risk_replication_results.json `
  --max-requests 14
```

## M3.7e - Fused LLM+WM request runner

Objectif :

- Executer uniquement les requests M2.15 `READY_FOR_M3`.
- Ignorer la request bloquee
  `BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME`.
- Reutiliser l'executor `offline_trace_context` existant.
- Comparer chaque `source_transition_id` aux sources deja executees en M3.7d.
- Marquer les supports issus de sources deja vues comme validation de routage,
  pas comme nouvelle evidence independante.
- Garder `support=0`, `revision_status=CANDIDATE_ONLY` et aucun write A32/A33.

Ajouts :

- `theory/m3/fused_llm_wm_experiment_runner.py`
- `tests/test_m3_fused_llm_wm_experiment_runner.py`
- `diagnostics/m3/fused_llm_wm_experiment_results.json`

Run du 2026-06-29 :

- entree :
  `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`
- sortie :
  `diagnostics/m3/fused_llm_wm_experiment_results.json`
- fused_requests_executed = 3
- fused_requests_skipped_blocked = 1
- blocked_request_ids =
  `m2m3::m2_15_fused::terminal_safe_alternative_action::002`
- controlled_experiments_run = 3
- target_trace_samples_total = 3
- matched_control_samples_total = 784
- all_controls_same_game_same_available_actions = true
- support_events = 3
- contradiction_events = 0
- neutral_events = 0
- source_transition_reused_from_m3_7d = true
- source_transition_reused_from_m3_7d_count = 3
- reused_source_support_events = 3
- independent_source_support_events = 0
- new_independent_terminal_risk_evidence = false
- fusion_hypothesis_routing_validated = true
- support = 0
- truth_status = `NOT_EVALUATED_BY_M2`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Requests executees :

- `m2m3::m2_15_fused::terminal_risk_precondition::001` :
  source `m2_14d::ar25-e3c63847::be48a96d70f0::0295`,
  target `ACTION1`, control `ACTION3`, matched controls = 232.
- `m2m3::m2_15_fused::wm_llm_disagreement_frontier::003` :
  source `m2_14d::bp35-0a0ad940::73dd8a6e40e1::0086`,
  target `ACTION3`, control `ACTION4`, matched controls = 298.
- `m2m3::m2_15_fused::objective_completion_vs_terminal_risk_tradeoff::004` :
  source `m2_14d::bp35-0a0ad940::182270820f60::0023`,
  target `ACTION6 {"x":22,"y":33}`, control `ACTION3`,
  matched controls = 254.

Lecture :

M3.7e valide que les hypotheses M2.15 sont routables et coherentes avec
l'executor offline-trace-context. Les trois `support_events` ne sont pas de
nouvelles observations independantes de terminal-risk : les trois sources
etaient deja presentes dans M3.7d. Le resultat confirme donc la testabilite et
le routage de la fusion LLM+WM, pas une nouvelle preuve scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.fused_llm_wm_experiment_runner `
  --m2-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-7d-results diagnostics\m3\arc_lewm_terminal_risk_replication_results.json `
  --offline-trace-dataset training\m2_arc_lewm_transitions.jsonl `
  --out diagnostics\m3\fused_llm_wm_experiment_results.json
```

## M3.7f - Offline-frame counterfactual feasibility probe

Objectif :

- Inspecter la request M2.15 bloquee
  `terminal_safe_alternative_action::002`.
- Verifier si le `frame_before` offline contient un etat env complet
  restaurable, pas seulement une grille visuelle.
- Ne jamais simuler une action alternative depuis `grid_t` seule.
- Si l'etat restaurable manque, produire une frontier de capacite candidate,
  sans ecrire A40/P2/A32/A33.

Ajouts :

- `theory/m3/offline_frame_counterfactual_probe.py`
- `tests/test_m3_offline_frame_counterfactual_probe.py`
- `diagnostics/m3/offline_frame_counterfactual_feasibility.json`

Run du 2026-06-29 :

- entree :
  `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`
- trace :
  `training/m2_arc_lewm_transitions.jsonl`
- sortie :
  `diagnostics/m3/offline_frame_counterfactual_feasibility.json`
- counterfactual_requests_seen = 1
- source_transitions_found = 1
- source_transition_id =
  `m2_14d::ar25-e3c63847::be48a96d70f0::0295`
- target alternative = `ACTION3`
- observed trace action = `ACTION1`
- target_action_available_in_frame = true
- frame_before_grid_present = true
- frame_after_grid_present = true
- full_env_state_payloads_present = 0
- restore_contracts_detected = 0
- can_reconstruct_env_state = false
- can_replay_alternative_action = false
- replay_exact_hashable = false
- counterfactual_feasibility =
  `BLOCKED_REQUIRES_ENV_STATE_RESTORE_OR_ACTIVE_COLLECTION`
- recommended_frontier_type =
  `NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT`
- support = 0
- truth_status = `NOT_EVALUATED_BY_M2`
- revision_status = `CANDIDATE_ONLY`
- frontier_write_performed = false
- a32_write_performed = false
- a33_write_performed = false

Lecture :

M3.7f ferme proprement la question contrefactuelle : l'action alternative est
bien disponible dans le contexte source, mais le dataset offline ne transporte
pas d'etat env complet restaurable. La bonne suite n'est donc pas M3.7g tout de
suite ; c'est une frontier de capacite : restaurer un etat complet depuis
`frame_before` ou collecter activement la transition alternative depuis le meme
contexte.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.offline_frame_counterfactual_probe `
  --m2-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --offline-trace-dataset training\m2_arc_lewm_transitions.jsonl `
  --out diagnostics\m3\offline_frame_counterfactual_feasibility.json
```

## M3.8 - Observation-to-hypothesis refinement

Objectif :

- Consommer `diagnostics/m3/m2_candidate_experiment_results.json`.
- Grouper les observations par signature experimentale :
  `game_id`, `context_replay`, `context_replay_args`, `target_action`,
  `control_action`, `metric`, `signal_source`.
- Grouper ensuite par signature mecanistique :
  `game_id`, `context_replay`, `target_action`, `observed_effect_family`.
- Fusionner les hypotheses M2 proches si elles menent au meme effet observe.
- Produire des hypotheses raffinees candidate-only.
- Ne jamais transformer `input_support_events` en `support`.

Ajouts :

- `theory/m3/m2_observation_refinement.py`
- `ExperimentalObservationGroup`
- `RefinedCandidateHypothesis`
- `run_m2_observation_refinement`
- `diagnostics/m3/refined_candidate_hypotheses_from_m2.json`

Run du 2026-06-19 :

- entree : `diagnostics/m3/m2_candidate_experiment_results.json`
- sortie : `diagnostics/m3/refined_candidate_hypotheses_from_m2.json`
- `source_experiments=12`
- `source_hypotheses_consumed=4`
- `experimental_signature_groups=7`
- `unique_grounded_positive_signatures=2`
- `unique_neutral_signatures=4`
- `unique_diagnostic_only_signatures=1`
- `input_support_events=4`
- `input_support_events_after_signature_dedup=2`
- `refined_candidate_hypotheses=1`
- `mechanistic_signature_groups=1`
- `duplicate_source_hypotheses_merged=true`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `input_support_events_counted_as_support=false`

Hypothese raffinee produite :

- `refined_hypothesis_id=m3_8::bp35::A6_A3::ACTION4::global_motion`
- sources fusionnees :
  - `m2::after_ACTION3_live_after_ACTION6::h001`
  - `m2::after_ACTION3_live_after_ACTION6::h002`
- `candidate_mechanic=global_object_repositioning_after_consumption`
- contexte : `["ACTION6", "ACTION3"]`
- contexte args : `[{"x": 18, "y": 0}, {}]`
- target : `ACTION4`
- control : `ACTION3`
- observations positives :
  - `changed_pixels_support`
  - `object_positions_before_after_support`
- observations neutres :
  - `local_patch_before_after_neutral`
  - `contact_graph_before_after_neutral`
- observations diagnostic-only :
  - `topology_before_after`

Lecture :

- M3.8 transforme l'information M3.7b en hypothese plus structuree :
  `ACTION4` apres `ACTION6 -> ACTION3` ressemble a un operateur de
  repositionnement global, pas a un operateur `local_patch`.
- Les deux hypotheses M2 initiales h001/h002 ne deviennent pas deux
  decouvertes independantes : elles sont fusionnees par signature
  mecanistique.
- Le raffinement est derive depuis les observations, jamais presente comme un
  verdict.

## M3.9 - Refined follow-up planner

Objectif :

- Consommer `diagnostics/m3/refined_candidate_hypotheses_from_m2.json`.
- Pour chaque hypothese
  `global_object_repositioning_after_consumption`, produire un test
  discriminant de reactivation.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas modifier M2, M3.8, A32 ou A33.

Question scientifique :

- Apres `ACTION6 -> ACTION3 -> ACTION4`, est-ce que `ACTION6` redevient
  utile ?

Ajouts :

- `theory/m3/refined_followup_planner.py`
- `RefinedFollowupExperimentRequest`
- `run_refined_followup_planning`
- `diagnostics/m3/refined_followup_experiment_requests.json`

Run du 2026-06-19 :

- entree : `diagnostics/m3/refined_candidate_hypotheses_from_m2.json`
- sortie : `diagnostics/m3/refined_followup_experiment_requests.json`
- `refined_hypotheses_consumed=1`
- `followup_requests_generated=1`
- `reactivation_tests_generated=1`
- `multi_metric_requests=1`
- `skipped_refined_hypotheses=0`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Request produite :

- `request_id=m3_9::m3_8_bp35_A6_A3_ACTION4_global_motion::retest_ACTION6`
- `source_refined_hypothesis_id=m3_8::bp35::A6_A3::ACTION4::global_motion`
- `hypothesis_tested=ACTION4 is a global repositioning/reset operator that re-enables ACTION6 after consumption`
- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- target : `ACTION6`
- controles suggeres : `["ACTION3", "ACTION4"]`
- `control_policy=m3_dynamic_available_controls`
- metriques :
  - `local_patch_before_after`
  - `changed_pixels`
  - `object_positions_before_after`
  - `contact_graph_before_after`

Lecture :

- M3.9 transforme l'hypothese descriptive de repositionnement global en test
  utile pour la policy : verifier si le repositionnement rend le skill consomme
  a nouveau exploitable.
- La request ne compte pas comme evidence et n'ajoute aucun `support`.
- M3.10 pourra executer ce follow-up en target vs controles dynamiques.

## M3.10 - Refined follow-up executor

Objectif :

- Consommer `diagnostics/m3/refined_followup_experiment_requests.json`.
- Rejouer exactement `["ACTION6", "ACTION3", "ACTION4"]`.
- Resoudre explicitement les args de `target_action=ACTION6`.
- Executer `ACTION6` vs controles dynamiques `ACTION3` et `ACTION4`.
- Mesurer les metriques grounded :
  - `local_patch_before_after`
  - `changed_pixels`
  - `object_positions_before_after`
  - `contact_graph_before_after`
- Ajouter des `support_events`, `contradiction_events`, `neutral_events`.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Regle d'args M3.10 :

- Si `target_action_args=null` et si `target_action` apparait deja dans le
  replay, M3.10 applique :
  `target_action_arg_policy=same_args_as_previous_skill_occurrence`.
- Pour le run bp35 :
  - occurrence precedente : `ACTION6 {"x": 18, "y": 0}`
  - target resolu : `ACTION6 {"x": 18, "y": 0}`

Run du 2026-06-19 :

- entree : `diagnostics/m3/refined_followup_experiment_requests.json`
- sortie : `diagnostics/m3/refined_followup_experiment_results.json`
- `followup_requests_consumed=1`
- `followup_requests_executed=1`
- `controlled_experiments_run=8`
- `metrics_executed=4`
- `controls_executed=2`
- `target_action_args_resolved=true`
- `target_action_arg_policy=same_args_as_previous_skill_occurrence`
- `support_events=0`
- `contradiction_events=4`
- `neutral_events=4`
- `blocked_experiments=0`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Lecture des metriques :

- `local_patch_before_after` :
  - ACTION6 apres ACTION4 reste neutre contre ACTION3 et ACTION4.
- `changed_pixels` :
  - target ACTION6 : signal 1
  - controles ACTION3/ACTION4 : signal 47
  - contradiction candidate-only face aux deux controles.
- `object_positions_before_after` :
  - target ACTION6 : signal 2
  - controles ACTION3/ACTION4 : signal 5
  - contradiction candidate-only face aux deux controles.
- `contact_graph_before_after` :
  - neutre.

Lecture :

- Le follow-up ne supporte pas l'hypothese stricte :
  `ACTION4 reactive la meme application de ACTION6`.
- Il suggere plutot que `ACTION4` reste un operateur de mouvement global, sans
  reactivation evidente de `ACTION6 {"x":18,"y":0}`.
- Cette lecture reste candidate-only : contradiction_events ne sont pas une
  refutation A32/A33.
- La suite naturelle est de tester une politique
  `dynamic_retarget_after_repositioning` pour verifier si `ACTION4` cree une
  nouvelle cible ACTION6 ailleurs.

## M3.11 - Dynamic retarget after repositioning planner

Objectif :

- Consommer `diagnostics/m3/refined_followup_experiment_results.json`.
- Utiliser l'echec strict M3.10 comme signal pour chercher de nouveaux args
  `ACTION6(x,y)` apres `ACTION6 -> ACTION3 -> ACTION4`.
- Exclure l'application deja testee :
  `ACTION6 {"x": 18, "y": 0}`.
- Generer une liste bornee de requests retarget.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Politique de generation :

- Rejouer le contexte pour lire les `ACTION6` disponibles apres
  repositionnement.
- Dedoublonner les args disponibles.
- Exclure les args deja testes.
- Scorer les candidats avec les offsets de mouvement observes dans
  `object_positions_before_after`.
- Ajouter un indice `changed_pixels_effect_present` quand le run M3.10 a
  montre un effet global.
- Limiter a `max_candidate_args=5`.

Run du 2026-06-20 :

- entree : `diagnostics/m3/refined_followup_experiment_results.json`
- sortie : `diagnostics/m3/dynamic_retarget_followup_requests.json`
- `followup_results_consumed=1`
- `source_experiments_consumed=8`
- `retarget_groups=1`
- `candidate_args_generated=5`
- `followup_requests_generated=5`
- `excluded_args_count=1`
- `max_candidate_args=5`
- `skipped_retarget_groups=0`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Args generes :

| Rang | Args ACTION6 | Score | Sources |
|---:|---|---:|---|
| 1 | `{"x": 12, "y": 0}` | 0.0 | live action, motion offset, changed pixels |
| 2 | `{"x": 24, "y": 0}` | 0.0 | live action, motion offset, changed pixels |
| 3 | `{"x": 30, "y": 0}` | 4.0 | live action, motion offset, changed pixels |
| 4 | `{"x": 30, "y": 12}` | 16.0 | live action, changed pixels |
| 5 | `{"x": 36, "y": 12}` | 22.0 | live action, changed pixels |

Lecture :

- M3.11 transforme l'echec strict de reactivation en recherche de nouvelle
  cible.
- Les requests testent l'hypothese ouverte :
  `ACTION4` deplace le probleme et cree une application `ACTION6` pertinente
  ailleurs.
- Les requests ne sont pas des preuves : elles gardent `support=0` et
  `followup_request_counted_as_support=false`.
- M3.12 pourra executer ces retargets en target vs controles dynamiques.

## M3.12 - Dynamic retarget follow-up executor

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_followup_requests.json`.
- Executer les 5 retargets ACTION6 generes par M3.11.
- Rejouer exactement `["ACTION6", "ACTION3", "ACTION4"]`.
- Tester chaque `ACTION6(x,y)` contre les controles dynamiques `ACTION3` et
  `ACTION4`.
- Mesurer les metriques grounded :
  - `local_patch_before_after`
  - `changed_pixels`
  - `object_positions_before_after`
  - `contact_graph_before_after`
- Produire deux niveaux de lecture :
  - `per_arg_results`
  - `mechanism_summary`
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_followup_requests.json`
- sortie : `diagnostics/m3/dynamic_retarget_followup_results.json`
- `retarget_requests_consumed=5`
- `retarget_requests_executed=5`
- `controlled_experiments_run=40`
- `tested_candidate_args=5`
- `args_with_grounded_support=4`
- `arg_level_support_events=16`
- `arg_level_support_events_counted_as_mechanism_support=false`
- `mechanism_support_events=1`
- `contradiction_events=12`
- `neutral_events=12`
- `blocked_experiments=0`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Resultats par args :

| Args ACTION6 | Support events | Contradictions | Neutres | Metriques positives grounded |
|---|---:|---:|---:|---|
| `{"x": 12, "y": 0}` | 4 | 2 | 2 | `local_patch_before_after`, `object_positions_before_after` |
| `{"x": 24, "y": 0}` | 4 | 2 | 2 | `local_patch_before_after`, `object_positions_before_after` |
| `{"x": 30, "y": 0}` | 0 | 4 | 4 | aucune |
| `{"x": 30, "y": 12}` | 4 | 2 | 2 | `local_patch_before_after`, `object_positions_before_after` |
| `{"x": 36, "y": 12}` | 4 | 2 | 2 | `local_patch_before_after`, `object_positions_before_after` |

Best arg :

- `{"x": 12, "y": 0}`
- `best_arg_support_events=4`
- `best_arg_grounded_support_metrics=[local_patch_before_after, object_positions_before_after]`

Lecture :

- M3.12 supporte candidate-only l'hypothese ouverte par M3.11 :
  `ACTION4` ne reactive pas l'ancienne cible, mais rend plusieurs nouvelles
  cibles `ACTION6` pertinentes.
- Le resultat est un cas C : plusieurs retargets marchent.
- Les 16 supports arg-level ne deviennent pas 16 preuves mecanistiques.
  Ils sont compresses en `mechanism_support_events=1`, et le ledger reste
  `support=0`.
- La suite naturelle est d'apprendre une regle de selection des retargets, ou
  de transmettre ce mecanisme candidate-only a A32/A15-A31 pour revision.

## M3.13 - Dynamic retarget mechanism consolidation

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_followup_results.json`.
- Compresser les resultats M3.12 en candidats mecanistiques candidate-only.
- Garder deux niveaux de lecture :
  - observations par arg retarget.
  - candidat mecanistique consolide.
- Distinguer les metriques de succes des metriques non decisives ou negatives.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Ajouts :

- `theory/m3/dynamic_retarget_mechanism_consolidation.py`
- `DynamicRetargetMechanismCandidate`
- `run_dynamic_retarget_mechanism_consolidation`
- `diagnostics/m3/dynamic_retarget_mechanism_candidates.json`

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_followup_results.json`
- sortie : `diagnostics/m3/dynamic_retarget_mechanism_candidates.json`
- `source_retarget_args=5`
- `source_controlled_experiments=40`
- `candidate_mechanisms=1`
- `successful_retargets=4`
- `failed_retargets=1`
- `positive_metrics=[local_patch_before_after, object_positions_before_after]`
- `non_decisive_or_negative_metrics=[changed_pixels, contact_graph_before_after]`
- `arg_level_support_events=16`
- `arg_level_support_events_counted_as_mechanism_support=false`
- `mechanism_support_events=1`
- `mechanism_support_events_counted_as_support=false`
- `selection_problem_candidates=1`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Candidat produit :

- `mechanism_candidate_id=m3_13::bp35::A6_A3_A4::ACTION6::retarget_region`
- `candidate_mechanic=repositioning_opens_new_action6_target_region`
- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- `initial_consumed_args={"x": 18, "y": 0}`
- `repositioning_action=ACTION4`
- target : `ACTION6`
- `target_action_arg_policy=dynamic_retarget_after_repositioning`

Retargets reussis :

- `{"x": 12, "y": 0}`
- `{"x": 24, "y": 0}`
- `{"x": 30, "y": 12}`
- `{"x": 36, "y": 12}`

Retarget echoue :

- `{"x": 30, "y": 0}`

Lecture des metriques :

- Metriques positives :
  - `local_patch_before_after`
  - `object_positions_before_after`
- Metriques non decisives ou negatives :
  - `changed_pixels`
  - `contact_graph_before_after`
- `changed_pixels_role=effect_radar_not_retarget_success_metric`

Lecture :

- M3.13 consolide la lecture scientifique :
  `ACTION4` ne reactive pas l'ancienne cible `ACTION6`, mais ouvre une region
  de nouvelles cibles `ACTION6`.
- Le cas `{"x": 30, "y": 0}` reste dans l'artefact comme echec explicite.
- La question suivante n'est plus seulement de savoir si retarget fonctionne,
  mais quelle regle distingue les retargets valides des retargets invalides.
- M3.14 pourra donc induire une regle de selection d'affordance a partir de
  cette separation succes/echec.
- La consolidation n'ajoute aucun `support` et ne transforme aucun evenement
  en verdict.

## M3.14 - Retarget selection rule induction

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_mechanism_candidates.json`.
- Induire des regles candidates qui distinguent les retargets reussis des
  retargets echoues.
- Produire uniquement des regles falsifiables candidate-only.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Ajouts :

- `theory/m3/dynamic_retarget_selection_rule_induction.py`
- `RetargetSelectionRuleSet`
- `run_dynamic_retarget_selection_rule_induction`
- `diagnostics/m3/dynamic_retarget_selection_rules.json`

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_mechanism_candidates.json`
- sortie : `diagnostics/m3/dynamic_retarget_selection_rules.json`
- `mechanism_candidates_consumed=1`
- `selection_rule_sets=1`
- `candidate_rules=3`
- `rules_with_falsification=3`
- `selection_problem_sets=1`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Set de regles produit :

- `selection_rule_candidate_id=m3_14::bp35::ACTION4_ACTION6::retarget_selection_rule`
- source :
  `m3_13::bp35::A6_A3_A4::ACTION6::retarget_region`
- mecanisme source :
  `repositioning_opens_new_action6_target_region`
- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- target : `ACTION6`

Retargets reussis :

- `{"x": 12, "y": 0}`
- `{"x": 24, "y": 0}`
- `{"x": 30, "y": 12}`
- `{"x": 36, "y": 12}`

Retarget echoue :

- `{"x": 30, "y": 0}`

Contrastes observes :

- Meme `x=30` :
  - succes sur `y=12`
  - echec sur `y=0`
- Meme `y=0` :
  - succes sur `x=12`, `x=24`
  - echec sur `x=30`
- Donc :
  - `pure_x_rule_blocked=true`
  - `pure_y_rule_blocked=true`

Familles de regles candidates :

1. `row_or_band_dependent_retarget`
   - La validite du retarget depend d'une interaction position + bande
     spatiale, pas de `x` ou `y` seuls.
   - Falsification : des retargets additionnels partageant la relation de
     bande proposee echouent, ou des retargets correspondant a la relation de
     l'echec reussissent sous le meme replay et les memes metriques grounded.

2. `local_patch_transformability`
   - Un retarget est valide si le patch local autour de la cible est
     transformable par `ACTION6` et si l'effet de positions d'objets est
     specifique au retarget.
   - Falsification : un retarget avec patch local similaire echoue, ou un
     retarget sans patch local transformable reussit sur
     `local_patch_before_after` / `object_positions_before_after`.

3. `specific_effect_over_global_pixels`
   - Le succes du retarget doit etre selectionne par les effets locaux / objets
     grounded, pas par la maximisation de `changed_pixels`.
   - Falsification : `changed_pixels` seul predit mieux les retargets reussis
     que les metriques locales / objets sur les follow-ups.

Hints M3.15 :

- Tester d'autres bandes `y` a `x=30`.
- Tester d'autres voisins `x` sur `y=0`.
- Comparer des cibles avec patch local similaire aux succes et a l'echec.

Lecture :

- M3.14 transforme le contre-exemple `{"x": 30, "y": 0}` en probleme
  explicite d'induction d'affordance.
- La regle n'est pas encore apprise comme verite : elle est formulee comme
  ensemble de candidats falsifiables.
- La prochaine etape naturelle est M3.15 : generer des follow-ups qui
  discriminent entre bande spatiale, patch local transformable et simple effet
  global.

## M3.15 - Selection rule follow-up planner

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_selection_rules.json`.
- Transformer les regles candidates M3.14 en requests discriminantes.
- Borner strictement le nombre de follow-ups.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire dans A33.

Ajouts :

- `theory/m3/dynamic_retarget_selection_followup_planner.py`
- `SelectionRuleFollowupRequest`
- `run_dynamic_retarget_selection_followup_planning`
- `diagnostics/m3/dynamic_retarget_selection_followup_requests.json`

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_selection_rules.json`
- sortie : `diagnostics/m3/dynamic_retarget_selection_followup_requests.json`
- `selection_rule_sets_consumed=1`
- `candidate_followup_requests=8`
- `followup_requests_generated=8`
- `truncated_followup_requests=0`
- `max_followup_requests=8`
- `request_budget_respected=true`
- `explicit_arg_requests=6`
- `dynamic_arg_resolution_requests=2`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Probes generees :

1. `same_x_different_band_probe`
   - Objectif : tester `row_or_band_dependent_retarget`.
   - Args explicites :
     - `{"x": 30, "y": 6}`
     - `{"x": 30, "y": 18}`
     - `{"x": 30, "y": 24}`
   - Points deja testes exclus :
     - `{"x": 30, "y": 0}` echec connu.
     - `{"x": 30, "y": 12}` succes connu.

2. `same_y_neighbor_x_probe`
   - Objectif : tester la frontiere / voisinage sur `y=0`.
   - Args explicites :
     - `{"x": 6, "y": 0}`
     - `{"x": 18, "y": 0}`
     - `{"x": 36, "y": 0}`
   - Points deja testes exclus :
     - `{"x": 12, "y": 0}` succes connu.
     - `{"x": 24, "y": 0}` succes connu.
     - `{"x": 30, "y": 0}` echec connu.

3. `local_patch_success_similarity_probe`
   - Objectif : resoudre dynamiquement des cibles dont le patch local ressemble
     aux retargets reussis.
   - `target_action_args=null`
   - `target_action_arg_policy=local_patch_similarity_after_repositioning`
   - Seeds succes :
     - `{"x": 12, "y": 0}`
     - `{"x": 24, "y": 0}`
     - `{"x": 30, "y": 12}`
     - `{"x": 36, "y": 12}`

4. `local_patch_failure_similarity_probe`
   - Objectif : resoudre dynamiquement des cibles dont le patch local ressemble
     au retarget echoue.
   - `target_action_args=null`
   - `target_action_arg_policy=local_patch_similarity_after_repositioning`
   - Seed echec :
     - `{"x": 30, "y": 0}`

Metriques :

- Metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- Metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`

Lecture :

- M3.15 ne choisit pas la bonne regle.
- Il fabrique la prochaine batterie de tests pour discriminer :
  - bande spatiale / relation positionnelle ;
  - patch local transformable ;
  - effet specifique local/objet plutot que `changed_pixels` global.
- Les requests restent candidate-only :
  `support=0`, `followup_request_counted_as_support=false`,
  `rule_counted_as_confirmation=false`.

## M3.16 - Selection rule follow-up executor

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_selection_followup_requests.json`.
- Rejouer exactement `["ACTION6", "ACTION3", "ACTION4"]`.
- Executer les follow-ups resolubles comme `ACTION6` target vs controles
  dynamiques `ACTION3` / `ACTION4`.
- Resoudre explicitement les probes `local_patch_similarity_after_repositioning`.
- Bloquer proprement les args explicites qui ne sont pas disponibles apres replay.
- Exclure les args deja testes, dont la cible consommee
  `{"x": 18, "y": 0}`.
- Mesurer les metriques de succes et les metriques diagnostiques separement.
- Ne pas confirmer, ne pas ecrire A32/A33, garder `support=0`.

Ajouts :

- `theory/m3/dynamic_retarget_selection_followup_executor.py`
- `run_dynamic_retarget_selection_followup_execution`
- `resolve_selection_followup_target_args`
- `resolve_local_patch_similarity_args`
- `diagnostics/m3/dynamic_retarget_selection_followup_results.json`

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_selection_followup_requests.json`
- sortie : `diagnostics/m3/dynamic_retarget_selection_followup_results.json`
- `selection_followup_requests_consumed=8`
- `selection_followup_requests_executed=2`
- `controlled_experiments_run=16`
- `explicit_requests=6`
- `explicit_requests_available=0`
- `explicit_requests_blocked_unavailable=5`
- `explicit_requests_blocked_excluded=1`
- `dynamic_arg_resolution_requests=2`
- `dynamic_arg_resolution_requests_resolved=2`
- `resolved_request_arg_pairs=2`
- `unique_resolved_target_arg_sets=1`
- `duplicate_resolved_target_arg_sets=1`
- `duplicate_resolved_target_arg_sets_counted_as_independent=false`
- `success_metric_support_events=8`
- `success_metric_contradiction_events=0`
- `diagnostic_support_events=0`
- `diagnostic_contradiction_events=4`
- `neutral_events=4`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Resolution des args :

- `{"x": 30, "y": 6}` : bloque,
  `explicit_args_not_available_after_replay`.
- `{"x": 30, "y": 18}` : bloque,
  `explicit_args_not_available_after_replay`.
- `{"x": 30, "y": 24}` : bloque,
  `explicit_args_not_available_after_replay`.
- `{"x": 6, "y": 0}` : bloque,
  `explicit_args_not_available_after_replay`.
- `{"x": 18, "y": 0}` : bloque,
  `explicit_args_excluded_known_arg` car deja present dans le replay causal.
- `{"x": 36, "y": 0}` : bloque,
  `explicit_args_not_available_after_replay`.
- `local_patch_success_similarity_probe` : resolu vers
  `{"x": 42, "y": 12}`.
- `local_patch_failure_similarity_probe` : resolu vers
  `{"x": 42, "y": 12}`.

Lecture :

- Les probes geometriques explicites de M3.15 etaient utiles comme questions,
  mais la plupart etaient hors de l'espace des `ACTION6` disponibles apres
  replay `ACTION6 -> ACTION3 -> ACTION4`.
- Le point `{"x": 18, "y": 0}` est explicitement exclu : M3.16 ne reteste pas
  l'ancienne cible consommee comme nouveau retarget.
- Les deux probes de similarite de patch convergent vers le meme arg live
  `{"x": 42, "y": 12}`.
- Cette convergence augmente l'interet experimental, mais elle n'est pas deux
  preuves independantes :
  `duplicate_resolved_target_arg_sets_counted_as_independent=false`.
- Les metriques de succes `local_patch_before_after` et
  `object_positions_before_after` donnent du support candidate-only pour
  l'arg resolu.
- Les metriques diagnostiques restent separees et ne deviennent pas un critere
  de succes retarget.
- M3.16 ne choisit toujours pas une regle vraie. Il prepare la consolidation
  M3.17 en distinguant :
  - questions explicites non executables dans l'espace live ;
  - retarget patch-similar resolu et testable ;
  - observation candidate-only sans verdict.

Schema guard ajoute :

- `candidate_arg_was_excluded=true` quand l'arg candidat est explicitement
  bloque par la liste d'exclusion.
- `excluded_known_args_guard_triggered=true` quand le guard d'exclusion s'est
  declenche.
- L'ancien champ `excluded_known_args_respected` reste present pour
  compatibilite, mais M3.17 lit le guard explicite pour eviter toute ambiguite.

## M3.17 - Selection rule evidence consolidation

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_selection_followup_results.json`.
- Consolider les observations M3.16 sans surinterpreter les probes bloquees.
- Identifier la meilleure famille de regle candidate actuelle.
- Garder `row_or_band_dependent_retarget` non contredite directement si ses
  probes n'ont pas ete executables.
- Preserver le role methodologique de `changed_pixels` comme radar d'effet,
  pas comme metrique de succes retarget.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/dynamic_retarget_selection_rule_consolidation.py`
- `RetargetSelectionRuleConsolidation`
- `run_dynamic_retarget_selection_rule_consolidation`
- `diagnostics/m3/dynamic_retarget_selection_rule_consolidation.json`

Run du 2026-06-20 :

- entree : `diagnostics/m3/dynamic_retarget_selection_followup_results.json`
- sortie : `diagnostics/m3/dynamic_retarget_selection_rule_consolidation.json`
- `selection_followup_results_consumed=1`
- `selection_rule_consolidations=1`
- `best_current_rule_families=["local_patch_transformability"]`
- `unique_new_successful_args=1`
- `row_or_band_directly_tested=false`
- `row_or_band_not_directly_contradicted=true`
- `blocked_probe_counted_as_contradiction=false`
- `duplicate_resolution_counted_as_independent=false`
- `source_success_metric_support_events=8`
- `source_diagnostic_contradiction_events=4`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Consolidation produite :

- `selection_rule_consolidation_id=m3_17::bp35::ACTION4_ACTION6::selection_rule_consolidation`
- `best_current_rule_family=local_patch_transformability`
- `confidence_basis=only_executed_followups_support_patch_similarity`
- `important_caveat=row_or_band probes were mostly not executable, so they are not direct contradictions`
- `unique_new_successful_args=[{"x": 42, "y": 12}]`
- `excluded_args_for_next_expansion` :
  - `{"x": 12, "y": 0}`
  - `{"x": 18, "y": 0}`
  - `{"x": 24, "y": 0}`
  - `{"x": 30, "y": 0}`
  - `{"x": 30, "y": 12}`
  - `{"x": 36, "y": 12}`
  - `{"x": 42, "y": 12}`
- metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`

Assessments :

1. `local_patch_transformability`
   - `assessment=supported_candidate_only_by_executed_followups`
   - `controlled_experiments_run=16`
   - `requests_with_success_metric_support=2`
   - `success_metric_support_events=8`
   - `support=0`

2. `row_or_band_dependent_retarget`
   - `assessment=not_directly_tested_blocked_or_excluded`
   - `controlled_experiments_run=0`
   - `blocked_requests=6`
   - `blocked_reason_counts` :
     - `explicit_args_not_available_after_replay=5`
     - `explicit_args_excluded_known_arg=1`
   - `directly_contradicted=false`
   - `blocked_probe_counted_as_contradiction=false`

3. `specific_effect_over_global_pixels`
   - `assessment=methodological_support_candidate_only`
   - `changed_pixels_role=effect_radar_not_retarget_success_metric`
   - `diagnostic_contradiction_events=4`

Lecture :

- M3.17 ne dit pas que la regle patch a gagne au sens fort.
- Il dit que, parmi les follow-ups reellement executes, seule la famille
  `local_patch_transformability` a trouve une affordance live avec metriques de
  succes positives.
- La famille `row_or_band_dependent_retarget` reste ouverte : ses probes
  explicites etaient majoritairement hors espace d'actions live, donc ce ne
  sont pas des contradictions directes.
- `changed_pixels` continue de jouer le role de radar global et non de critere
  de succes retarget.
- La prochaine etape naturelle est M3.18 : etendre la resolution
  patch-similarity en excluant `{"x": 42, "y": 12}` pour verifier si la regle
  trouve plusieurs affordances ou seulement cet attracteur.

## M3.18 - Patch-similarity expansion planner

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_selection_rule_consolidation.json`.
- Etendre la recherche patch-similarity en excluant tous les args deja testes,
  consommes ou resolus.
- Produire des requests `ACTION6` resolues ou resolubles.
- Generer trois familles :
  - `success_patch_similarity_expansion`
  - `failure_patch_negative_control_expansion`
  - `mixed_patch_boundary_probe`
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/dynamic_retarget_patch_similarity_expansion_planner.py`
- `PatchSimilarityExpansionRequest`
- `run_dynamic_retarget_patch_similarity_expansion_planning`
- `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_requests.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/dynamic_retarget_selection_rule_consolidation.json`
- sortie :
  `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_requests.json`
- `selection_rule_consolidations_consumed=1`
- `candidate_arg_groups=1`
- `live_available_args_seen=8`
- `excluded_args_count=7`
- `available_args_after_exclusion=1`
- `candidate_resolution_args=1`
- `expansion_requests_generated=3`
- `blocked_expansion_requests=0`
- `resolved_request_arg_pairs=3`
- `unique_resolved_target_arg_sets=1`
- `duplicate_resolved_target_arg_sets=2`
- `duplicate_resolved_target_arg_sets_counted_as_independent=false`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Args live apres replay `ACTION6 -> ACTION3 -> ACTION4` :

- `{"x": 18, "y": 0}`
- `{"x": 24, "y": 0}`
- `{"x": 30, "y": 0}`
- `{"x": 12, "y": 0}`
- `{"x": 30, "y": 12}`
- `{"x": 36, "y": 12}`
- `{"x": 42, "y": 12}`
- `{"x": 48, "y": 12}`

Args exclus :

- `{"x": 12, "y": 0}` succes connu.
- `{"x": 18, "y": 0}` cible consommee dans le replay.
- `{"x": 24, "y": 0}` succes connu.
- `{"x": 30, "y": 0}` echec connu.
- `{"x": 30, "y": 12}` succes connu.
- `{"x": 36, "y": 12}` succes connu.
- `{"x": 42, "y": 12}` nouvel attracteur resolu par M3.16.

Nouvel arg live disponible apres exclusion :

- `{"x": 48, "y": 12}`

Resolution patch :

- `success_patch_distance=0.0`
- `failure_patch_distance=7.0`
- `mixed_patch_boundary_score=7.0`
- `similarity_interpretation=success_like`

Requests generees :

1. `success_patch_similarity_expansion`
   - resolu vers `{"x": 48, "y": 12}`
   - basis :
     `nearest_patch_signature_to_success_seeds_excluding_known_args`

2. `failure_patch_negative_control_expansion`
   - resolu aussi vers `{"x": 48, "y": 12}`
   - basis :
     `nearest_patch_signature_to_failure_seed_excluding_known_args`
   - lecture prudente : il s'agit du plus proche disponible de l'echec, pas
     d'un vrai candidat failure-like, car `similarity_interpretation=success_like`.

3. `mixed_patch_boundary_probe`
   - resolu aussi vers `{"x": 48, "y": 12}`
   - basis :
     `nearest_ambiguous_patch_boundary_excluding_known_args`

Lecture :

- M3.18 teste si la regle patch-similarity generalise au-dela de
  `{"x": 42, "y": 12}`.
- Il trouve exactement un nouveau candidat live : `{"x": 48, "y": 12}`.
- Les trois probes convergent vers ce meme arg, donc cette convergence ne vaut
  pas trois preuves independantes.
- Le candidat est patch-similar aux succes, pas vraiment au contre-exemple.
- La prochaine etape naturelle est M3.19 : executer `ACTION6 {"x":48,"y":12}`
  une seule fois par signature experimentale, tout en conservant les trois
  rationales de provenance.

## M3.19 - Patch-similarity expansion executor

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_requests.json`.
- Executer les requests M3.18 une seule fois par signature experimentale unique.
- Preserver les trois rationales de provenance sans les compter comme
  observations independantes.
- Tester `ACTION6 {"x": 48, "y": 12}` contre `ACTION3` et `ACTION4`.
- Mesurer les metriques de succes et les metriques diagnostiques separement.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/dynamic_retarget_patch_similarity_expansion_executor.py`
- `run_dynamic_retarget_patch_similarity_expansion_execution`
- `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_results.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_requests.json`
- sortie :
  `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_results.json`
- `expansion_requests_consumed=3`
- `unique_execution_signatures=1`
- `unique_target_arg_sets_executed=1`
- `duplicate_request_rationales_preserved=3`
- `duplicate_request_rationales_counted_as_independent=false`
- `controlled_experiments_run=8`
- `success_metric_support_events=4`
- `success_metric_contradiction_events=0`
- `diagnostic_support_events=0`
- `diagnostic_contradiction_events=2`
- `neutral_events=2`
- `blocked_experiments=0`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Signature executee :

- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- target : `ACTION6 {"x": 48, "y": 12}`
- controles : `ACTION3`, `ACTION4`
- metriques :
  - `local_patch_before_after`
  - `object_positions_before_after`
  - `changed_pixels`
  - `contact_graph_before_after`

Rationales preservees :

- `success_patch_similarity_expansion`
- `failure_patch_negative_control_expansion`
- `mixed_patch_boundary_probe`

Lecture des metriques :

- `local_patch_before_after`
  - support contre `ACTION3`
  - support contre `ACTION4`
  - signal target `2.0` vs controles `0.0`
- `object_positions_before_after`
  - support contre `ACTION3`
  - support contre `ACTION4`
  - signal target `49.0` vs controles `5.0`
- `changed_pixels`
  - contradiction diagnostique contre `ACTION3`
  - contradiction diagnostique contre `ACTION4`
  - signal target `36.0` vs controles `47.0`
- `contact_graph_before_after`
  - neutre contre `ACTION3`
  - neutre contre `ACTION4`

Expansion summary :

- `tested_unique_arg_sets=1`
- `args_with_grounded_support=1`
- `args_with_success_metric_contradictions=0`
- `best_arg={"x": 48, "y": 12}`
- `mechanism_support_events=1`
- `signature_level_support_events=4`
- `signature_level_support_events_counted_as_mechanism_support=false`

Lecture :

- M3.19 montre que `{"x": 48, "y": 12}` se comporte comme les retargets
  utiles precedents : succes sur `local_patch_before_after` et
  `object_positions_before_after`, contradiction seulement sur le radar global
  `changed_pixels`.
- La regle patch-similarity devient donc une hypothese generative candidate :
  apres exclusion de `{"x": 42, "y": 12}`, elle retrouve une autre affordance
  success-like.
- Les trois rationales M3.18 ne sont pas trois preuves :
  elles sont conservees comme provenance et executees une seule fois.
- La prochaine etape naturelle est M3.20 : consolider la regle en distinguant
  `ligne/region d'affordances success-like` de `simple suite de points`.

## M3.20 - Patch-similarity generativity consolidation

Objectif :

- Consommer `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_results.json`.
- Consolider les succes initiaux M3.12, le succes d'expansion M3.16
  `{"x": 42, "y": 12}` et le nouveau succes M3.19
  `{"x": 48, "y": 12}`.
- Produire une lecture de generativite candidate-only pour
  `local_patch_transformability`.
- Marquer `ready_for_a32_revision_queue=true` seulement comme pret pour une
  revue, jamais comme verdict.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/dynamic_retarget_patch_similarity_generativity_consolidation.py`
- `PatchSimilarityGenerativityConsolidation`
- `run_dynamic_retarget_patch_similarity_generativity_consolidation`
- `diagnostics/m3/dynamic_retarget_patch_similarity_generativity_consolidation.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/dynamic_retarget_patch_similarity_expansion_results.json`
- sortie :
  `diagnostics/m3/dynamic_retarget_patch_similarity_generativity_consolidation.json`
- `patch_similarity_generativity_consolidations=1`
- `candidate_rule_families=["local_patch_transformability"]`
- `candidate_generativity=["SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"]`
- `successful_args_total_count=6`
- `failed_args_count=1`
- `new_expansion_successes_count=2`
- `ready_for_a32_revision_queue=true`
- `ready_for_a32_revision_queue_is_not_verdict=true`
- `source_success_metric_support_events=4`
- `source_success_metric_contradiction_events=0`
- `source_diagnostic_contradiction_events=2`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `generative_sequence_counted_as_confirmation=false`

Consolidation :

- candidat :
  `m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity`
- regle candidate :
  `local_patch_transformability`
- succes initiaux :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
  - `{"x": 30, "y": 12}`
  - `{"x": 36, "y": 12}`
- succes d'expansion :
  - `{"x": 42, "y": 12}`
  - `{"x": 48, "y": 12}`
- echec connu :
  - `{"x": 30, "y": 0}`
- pattern candidate :
  `success_like_patch_line_or_region_after_repositioning`
- metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`
- role de `changed_pixels` :
  `effect_radar_not_success_metric`

Lecture :

- M3.20 ne dit pas que la regle patch-similarity est confirmee.
- Il dit que la regle est generative candidate-only : elle n'a pas seulement
  explique des exemples deja vus, elle a propose une nouvelle cible non testee
  apres exclusion des anciennes, puis cette cible a produit les effets attendus.
- `ready_for_a32_revision_queue=true` indique que le dossier merite une revue
  scientifique A32/A15-A31, pas que M3 a le droit de reviser ou confirmer.
- La prochaine etape propre est M3.21 : construire un bridge A32/A15-A31
  dedie a ce dossier, sans declencher la revision.

## M3.21 - Patch-similarity A32 revision queue bridge

Objectif :

- Consommer
  `diagnostics/m3/dynamic_retarget_patch_similarity_generativity_consolidation.json`.
- Transformer le dossier M3.20 en candidat revisable par A32/A15-A31.
- Produire `diagnostics/m3/patch_similarity_a32_revision_queue.json`.
- Refuser un item si `ready_for_a32_revision_queue=false`.
- Refuser un item si `source_success_metric_contradiction_events>0`.
- Ne pas interpreter les contradictions diagnostiques comme refutation.
- Garder `changed_pixels` comme metrique diagnostique.
- Ne pas executer.
- Ne pas ecrire A32/A33.
- Ne pas confirmer.

Ajouts :

- `theory/m3/patch_similarity_a32_revision_queue.py`
- `PatchSimilarityA32RevisionQueueItem`
- `run_patch_similarity_a32_revision_queue_generation`
- `diagnostics/m3/patch_similarity_a32_revision_queue.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/dynamic_retarget_patch_similarity_generativity_consolidation.json`
- sortie :
  `diagnostics/m3/patch_similarity_a32_revision_queue.json`
- `generativity_consolidations_consumed=1`
- `queue_items=1`
- `candidate_records=1`
- `rejected_queue_items=0`
- `ready_for_a32_revision_candidates=1`
- `ready_for_a32_revision_is_not_verdict=true`
- `success_metric_support_events=4`
- `success_metric_contradiction_events=0`
- `diagnostic_contradiction_events=2`
- `diagnostic_contradictions_counted_as_refutation=false`
- `changed_pixels_kept_diagnostic=true`
- `a32_write_performed=false`
- `a33_write_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `generative_sequence_counted_as_confirmation=false`

Queue item :

- `queue_item_id=m3_21::bp35::ACTION4_ACTION6::local_patch_transformability`
- `source_generativity_consolidation_id=m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity`
- `candidate_rule_family=local_patch_transformability`
- `candidate_mechanic=repositioning_opens_patch_similar_action6_affordances`
- `candidate_generativity=SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY`
- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- target : `ACTION6`
- succes totaux :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
  - `{"x": 30, "y": 12}`
  - `{"x": 36, "y": 12}`
  - `{"x": 42, "y": 12}`
  - `{"x": 48, "y": 12}`
- echec connu :
  - `{"x": 30, "y": 0}`
- metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`
- role de `changed_pixels` :
  `effect_radar_not_success_metric`

Lecture :

- M3.21 ne revise pas et ne confirme rien.
- Le bridge transforme seulement le dossier M3.20 en queue item A32-ready.
- Les contradictions diagnostiques de `changed_pixels` sont conservees comme
  contexte scientifique, mais elles ne refutent pas le candidat.
- A32/A15-A31 restent les seuls lieux autorises pour une decision scientifique.

## M3.22 - A32 requested patch-similarity follow-up planner

Objectif :

- Lire `diagnostics/a32/patch_similarity_revision_decisions.json`.
- Consommer les `requested_followup_tests` produits par A32.3.
- Transformer les demandes A32 en requests M3 executables.
- Porter explicitement :
  - `source_a32_decision=SCOPE_LIMITED_CANDIDATE_ONLY`
  - `source_a32_recommended_next_step=REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS`
  - `a32_decision_counted_as_confirmation=false`
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/a32_requested_patch_similarity_followup_planner.py`
- `A32RequestedPatchSimilarityFollowupRequest`
- `run_a32_requested_patch_similarity_followup_planning`
- `diagnostics/m3/a32_requested_patch_similarity_followup_requests.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/a32/patch_similarity_revision_decisions.json`
- sortie :
  `diagnostics/m3/a32_requested_patch_similarity_followup_requests.json`
- `a32_revision_decisions_consumed=1`
- `a32_requested_followup_tests_seen=2`
- `candidate_arg_groups=2`
- `planned_followup_requests=2`
- `blocked_followup_requests=0`
- `outside_known_y12_region_requests=1`
- `alternate_repositioning_context_requests=1`
- `resolved_request_arg_pairs=4`
- `unique_resolved_target_arg_sets=4`
- `a32_decision_counted_as_confirmation=false`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Requests produites :

1. `outside_known_y12_region_probe`

- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}, {}]`
- target : `ACTION6`
- args resolus :
  - `{"x": 18, "y": 0}`
  - `{"x": 30, "y": 0}`
- interpretation :
  tester la limite hors de la region de succes connue, avec les succes
  precedents exclus par A32.

2. `alternate_repositioning_context_probe`

- contexte alternatif resolu : `["ACTION6", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}]`
- target : `ACTION6`
- args resolus :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
- interpretation :
  tester si des affordances deja success-like restent utiles sous un replay de
  repositionnement alternatif.

Metriques :

- succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`

Lecture :

- M3.22 ne traite pas la decision A32 comme une preuve.
- A32.3 discipline le scope ; M3.22 transforme cette discipline en agenda
  experimental executable.
- La prochaine etape naturelle est M3.23 : executer ces requests, puis renvoyer
  les observations a A32 si le scope s'elargit ou se contredit.

## M3.23 - A32 requested patch-similarity follow-up executor

Objectif :

- Consommer `diagnostics/m3/a32_requested_patch_similarity_followup_requests.json`.
- Executer les follow-ups demandes par A32.3 une seule fois par signature
  experimentale unique.
- Tester chaque arg `ACTION6` contre les controles dynamiques `ACTION3` et
  `ACTION4`.
- Separer les metriques de succes :
  `local_patch_before_after`, `object_positions_before_after`
  des metriques diagnostiques :
  `changed_pixels`, `contact_graph_before_after`.
- Ne pas interpreter les echecs outside-region comme une refutation globale.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/a32_requested_patch_similarity_followup_executor.py`
- `run_a32_requested_patch_similarity_followup_execution`
- `diagnostics/m3/a32_requested_patch_similarity_followup_results.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/a32_requested_patch_similarity_followup_requests.json`
- sortie :
  `diagnostics/m3/a32_requested_patch_similarity_followup_results.json`
- `a32_followup_requests_consumed=2`
- `unique_execution_signatures=4`
- `unique_target_arg_sets_executed=4`
- `controlled_experiments_run=32`
- `outside_known_y12_region_signatures=2`
- `alternate_repositioning_context_signatures=2`
- `success_metric_support_events=8`
- `success_metric_contradiction_events=4`
- `diagnostic_contradiction_events=8`
- `alternate_context_args_with_success_metric_support=2`
- `outside_boundary_failures_counted_as_rule_refutation=false`
- `diagnostic_contradictions_counted_as_refutation=false`
- `a32_decision_counted_as_confirmation=false`
- `a32_write_performed=false`
- `a33_write_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`

Famille `outside_known_y12_region_probe` :

- contexte : `["ACTION6", "ACTION3", "ACTION4"]`
- args testes :
  - `{"x": 18, "y": 0}`
  - `{"x": 30, "y": 0}`
- resultats :
  - `success_metric_support_events=0`
  - `success_metric_contradiction_events=4`
  - `diagnostic_contradiction_events=4`
- interpretation :
  `outside_known_region_boundary_reinforced_candidate_only`
- lecture :
  ces echecs bornent le scope local. Ils ne refutent pas la regle globale,
  car cette famille est une negative-control / boundary probe demandee par A32.

Famille `alternate_repositioning_context_probe` :

- contexte alternatif : `["ACTION6", "ACTION4"]`
- contexte args : `[{"x": 18, "y": 0}, {}]`
- args testes :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
- resultats :
  - `success_metric_support_events=8`
  - `success_metric_contradiction_events=0`
  - `diagnostic_contradiction_events=4`
  - metriques de succes grounded :
    `local_patch_before_after`, `object_positions_before_after`
- interpretation :
  `alternate_context_scope_expanded_candidate_only`

Lecture :

- M3.23 repond directement a la demande A32.3.
- La probe hors region connue montre que les cibles failure-like
  `{"x": 18, "y": 0}` et `{"x": 30, "y": 0}` ne doivent pas etre integrees
  abusivement au scope.
- La probe de contexte alternatif est plus forte scientifiquement :
  `{"x": 12, "y": 0}` et `{"x": 24, "y": 0}` restent utiles apres
  `ACTION6 -> ACTION4`.
- Le scope de la regle patch-similarity s'elargit donc candidate-only a un
  deuxieme contexte de repositionnement, sans devenir une confirmation.
- La prochaine etape naturelle est M3.24 : consolider les follow-ups demandes
  par A32 en distinguant `scope_expanded_candidate_only`,
  `scope_limited_reinforced_candidate_only` et
  `scope_contradicted_candidate_only`.

## M3.24 - A32 requested scope follow-up consolidation

Objectif :

- Consommer `diagnostics/m3/a32_requested_patch_similarity_followup_results.json`.
- Condenser les follow-ups A32.3 en une lecture de scope candidate-only.
- Produire un artefact leger pret pour un probe agentique experimental.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/a32_requested_patch_similarity_scope_consolidation.py`
- `run_a32_requested_patch_similarity_scope_consolidation`
- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/a32_requested_patch_similarity_followup_results.json`
- sortie :
  `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`
- `scope_consolidations=1`
- `scope_assessment=SCOPE_EXPANDED_CANDIDATE_ONLY`
- `initial_context_supported=true`
- `alternate_context_supported=true`
- `outside_region_boundary_reinforced=true`
- `ready_for_agent_policy_probe=true`
- `agent_policy_probe_status=EXPERIMENTAL_POLICY_CANDIDATE_ONLY`
- `a33_ready=false`
- `a32_write_performed=false`
- `a33_write_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `scope_expansion_counted_as_confirmation=false`

Consolidation :

- `scope_consolidation_id=m3_24::bp35-0a0ad940::patch_similarity_scope::ACTION6`
- regle candidate : `local_patch_transformability`
- target : `ACTION6`
- contexte initial soutenu par les succes herites M3.20 :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
  - `{"x": 30, "y": 12}`
  - `{"x": 36, "y": 12}`
  - `{"x": 42, "y": 12}`
  - `{"x": 48, "y": 12}`
- contexte alternatif soutenu :
  - replay : `["ACTION6", "ACTION4"]`
  - args : `{"x": 12, "y": 0}`, `{"x": 24, "y": 0}`
- bornes hors region :
  - replay : `["ACTION6", "ACTION3", "ACTION4"]`
  - args : `{"x": 18, "y": 0}`, `{"x": 30, "y": 0}`
- echec connu conserve :
  - `{"x": 30, "y": 0}`
- metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`

Lecture :

- M3.24 ne rend pas le candidat A33-ready.
- Le scope est elargi candidate-only : la regle fonctionne dans le replay
  initial et dans le replay alternatif `ACTION6 -> ACTION4`.
- Le scope reste borne : les args failure-like / outside-region ne doivent
  pas etre selectionnes par une policy naive.
- L'artefact autorise seulement un probe agentique experimental :
  utiliser `ACTION4` comme repositionnement, chercher des args `ACTION6`
  patch-similar success-like, eviter les failure-like connus, et ne jamais
  traiter cette regle comme une competence confirmee.

## M3.O1 - Objective stop/switch experiment planner

Objectif :

- Consommer `diagnostics/m2/objective_conditioned_hypotheses.json`.
- Transformer les 4 hypotheses objectif M2.O1 en protocoles M3 testables.
- Planifier seulement : aucune execution, aucun support, aucun verdict.
- Comparer continuer `ACTION6`, stop/hold, et switch vers
  `ACTION3`/`ACTION4`/`ACTION1`/`ACTION2`.

Ajouts :

- `theory/m3/objective_stop_switch_experiment_planner.py`
- `tests/test_m3_objective_stop_switch_experiment_planner.py`
- `diagnostics/m3/objective_stop_switch_experiment_requests.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m2/objective_conditioned_hypotheses.json`
- sortie :
  `diagnostics/m3/objective_stop_switch_experiment_requests.json`
- `objective_hypotheses_consumed=4`
- `objective_experiment_requests_generated=4`
- `skipped_objective_hypotheses=0`
- `prefix_lengths=[6, 12, 24, 48, 64]`
- `conditions_per_request=6`
- `planned_condition_cells=120`
- `primary_objective_metrics` :
  - `final_game_state`
  - `terminal_state_after_rollout`
  - `levels_completed_after_rollout`
  - `objective_progress_proxy`
- `local_diagnostic_metrics` :
  - `local_effect_metric`
  - `repeated_action6_count`
  - `useful_action6_steps`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `objective_request_counted_as_support=false`
- `policy_result_counted_as_scientific_verdict=false`
- `a32_remains_only_verdict_location=true`

Conditions planifiees par requete :

- `continue_action6`
- `stop_policy`
- `switch_ACTION3`
- `switch_ACTION4`
- `switch_ACTION1`
- `switch_ACTION2`

Lecture :

- M3.O1 n'interprete pas les hypotheses M2.O1 comme des preuves.
- La matrice prepare M3.O2 a tester si continuer `ACTION6` reste utile
  objectivement ou si un stop/switch reduit le risque terminal.
- Les metriques objectives pilotent le test; les metriques locales restent
  diagnostiques pour comprendre quand une affordance locale masque un echec
  global.

## M3.O2 - Objective stop/switch experiment executor

Objectif :

- Consommer `diagnostics/m3/objective_stop_switch_experiment_requests.json`.
- Deduplicater les cellules d'execution partagees par les 4 hypotheses.
- Executer chaque cellule unique une seule fois.
- Rattacher chaque resultat aux hypotheses via
  `hypothesis_observation_links`.
- Ne pas compter les liens dupliques comme observations independantes.
- Ne pas confirmer, ne pas refuter, ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/objective_stop_switch_experiment_executor.py`
- `tests/test_m3_objective_stop_switch_experiment_executor.py`
- `diagnostics/m3/objective_stop_switch_experiment_results.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/objective_stop_switch_experiment_requests.json`
- sortie :
  `diagnostics/m3/objective_stop_switch_experiment_results.json`
- `objective_requests_consumed=4`
- `planned_condition_cells=120`
- `unique_execution_cells=30`
- `duplicate_execution_cells=90`
- `duplicate_execution_cells_counted_as_independent=false`
- `hypothesis_observation_links=120`
- `objective_cells_executed=16`
- `blocked_cells=14`
- `blocked_control_unavailable_cells=8`
- `early_terminal_prefix_cells=6`
- `neutral_events=16`
- `support_events=0`
- `contradiction_events=0`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `blocked_cells_counted_as_contradictions=false`
- `policy_result_counted_as_scientific_verdict=false`

Resultats par prefixe :

- Prefixes `6`, `12`, `24`, `48` :
  - `continue_action6` execute, `final_game_state=NOT_FINISHED`.
  - `stop_policy` observe le point d'arret sans action supplementaire,
    `final_game_state=NOT_FINISHED`.
  - `switch_ACTION3` et `switch_ACTION4` executables,
    `final_game_state=NOT_FINISHED`.
  - `switch_ACTION1` et `switch_ACTION2` bloques comme controles
    indisponibles, pas comme contradictions.
- Prefixe `64` :
  - les 6 conditions sont `EARLY_TERMINAL_DURING_PREFIX`.
  - le prefixe atteint deja `GAME_OVER` avant que la condition
    post-prefixe puisse etre testee.

Lecture :

- M3.O2 confirme seulement le statut experimental de la matrice :
  les rollouts partages sont deduplicates correctement.
- Les controles indisponibles et les early-terminal prefix cells ne sont pas
  des refutations.
- Le signal utile pour M3.O3 est structurel : le seuil `64` atteint le
  terminal avant tout switch/stop, tandis que les prefixes plus courts restent
  testables et non terminaux.
- A32 reste le seul endroit autorise pour transformer ces observations en
  verdict scientifique.

## M3.O3 - Objective threshold consolidation

Objectif :

- Consommer `diagnostics/m3/objective_stop_switch_experiment_results.json`.
- Consolider la fenetre pre-terminale observee par M3.O2.
- Ne pas executer.
- Ne pas confirmer.
- Ne pas refuter.
- Produire une recommandation M3.O4 pour raffiner les prefixes.

Ajouts :

- `theory/m3/objective_threshold_consolidation.py`
- `tests/test_m3_objective_threshold_consolidation.py`
- `diagnostics/m3/objective_threshold_consolidation.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/objective_stop_switch_experiment_results.json`
- sortie :
  `diagnostics/m3/objective_threshold_consolidation.json`
- `objective_threshold_consolidations=1`
- `threshold_type=PRE_TERMINAL_PREFIX_WINDOW`
- `safe_tested_prefixes=[6, 12, 24, 48]`
- `safe_tested_prefix_max=48`
- `early_terminal_prefixes=[64]`
- `early_terminal_prefix_min=64`
- `critical_window=[49, 64]`
- `stop_switch_effectiveness_status=NOT_ESTABLISHED`
- `next_experiment_recommendation=REFINE_PREFIX_WINDOW`
- `recommended_refined_prefixes=[50, 54, 58, 60, 62, 63]`
- `recommended_conditions` :
  - `continue_action6`
  - `stop_policy`
  - `switch_ACTION3`
  - `switch_ACTION4`
- `excluded_conditions` :
  - `switch_ACTION1` :
    `blocked_control_unavailable_in_safe_prefixes`
  - `switch_ACTION2` :
    `blocked_control_unavailable_in_safe_prefixes`
- `execution_performed=false`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `threshold_consolidation_counted_as_confirmation=false`
- `stop_switch_effectiveness_counted_as_verdict=false`

Lecture :

- M3.O3 ne dit pas encore qu'il faut switcher a 48.
- M3.O3 dit que le stop/switch a `64` est trop tardif :
  le terminal est atteint pendant le prefixe, avant toute condition post-prefixe.
- Les prefixes `6`, `12`, `24`, `48` restent non terminaux et testables, mais
  ne produisent pas de level completion.
- La prochaine question experimentale est plus nette :
  existe-t-il un prefixe entre `49` et `63` ou stop/switch change l'issue
  terminale ?
- `ACTION1` et `ACTION2` restent documentes comme controles bloques, mais ils
  ne sont pas recommandes pour la sous-matrice M3.O4.

## M3.O4 - Refined objective window executor

Objectif :

- Consommer `diagnostics/m3/objective_threshold_consolidation.json`.
- Executer uniquement la sous-matrice recommandee par M3.O3.
- Tester les prefixes `50`, `54`, `58`, `60`, `62`, `63`.
- Tester seulement :
  - `continue_action6`
  - `stop_policy`
  - `switch_ACTION3`
  - `switch_ACTION4`
- Ne pas rejouer `ACTION1`/`ACTION2`, deja documentes comme controles
  indisponibles dans M3.O2/M3.O3.
- Ne pas confirmer, ne pas refuter, ne pas ecrire A32/A33.

Ajouts :

- `theory/m3/objective_refined_window_executor.py`
- `tests/test_m3_objective_refined_window_executor.py`
- `diagnostics/m3/objective_refined_window_results.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/objective_threshold_consolidation.json`
- sortie :
  `diagnostics/m3/objective_refined_window_results.json`
- `threshold_consolidations_consumed=1`
- `refined_prefixes=[50, 54, 58, 60, 62, 63]`
- `refined_conditions=[continue_action6, stop_policy, switch_ACTION3, switch_ACTION4]`
- `planned_refined_cells=24`
- `unique_execution_cells=24`
- `objective_cells_executed=24`
- `blocked_cells=0`
- `early_terminal_prefix_cells=0`
- `neutral_events=24`
- `support_events=0`
- `contradiction_events=0`
- `support=0`
- `revision_status=CANDIDATE_ONLY`
- `truth_status=NOT_EVALUATED_BY_M3`
- `wrong_confirmations=0`
- `stop_switch_effectiveness_status=CANDIDATE_TERMINAL_AVOIDANCE_OBSERVED`
- `prefixes_with_candidate_terminal_avoidance=[63]`
- `prefixes_without_terminal_separation=[50, 54, 58, 60, 62]`
- `refined_window_result_counted_as_confirmation=false`
- `stop_switch_effectiveness_counted_as_verdict=false`

Resultats clefs :

- Prefixes `50`, `54`, `58`, `60`, `62` :
  - `continue_action6` reste `NOT_FINISHED`.
  - `stop_policy`, `switch_ACTION3`, `switch_ACTION4` restent aussi
    `NOT_FINISHED`.
  - il n'y a donc pas encore de separation terminale.
- Prefixe `63` :
  - `continue_action6` : `GAME_OVER`.
  - `stop_policy` : `NOT_FINISHED`.
  - `switch_ACTION3` : `GAME_OVER`.
  - `switch_ACTION4` : `GAME_OVER`.

Lecture :

- M3.O4 ne confirme pas une policy.
- M3.O4 observe un signal candidate-only tres net :
  a `prefix=63`, continuer `ACTION6` est terminal tandis que stopper au point
  de prefixe reste non terminal.
- `ACTION3` et `ACTION4` ne sont pas de bons switchs a ce point precis dans
  ce protocole : ils n'evitent pas `GAME_OVER`.
- La prochaine etape agentique propre est P3.1 :
  tester une policy candidate qui stoppe ou change de mode autour du seuil
  `63`, sans traiter ce signal comme une mecanique confirmee.

## A32 - M3 revision intake

Objectif :

- Consommer `diagnostics/m3/a15_revision_queue.json` comme entree candidate
  cote A15-A31.
- Verifier les garde-fous avant acceptation.
- Construire des `HypothesisRecord` `UNRESOLVED`.
- Ne pas appeler la revision et ne pas produire de verdict.

Ajouts :

- `theory/a32_m3_revision_intake.py`
- `A32RevisionIntakeCandidate`
- `run_a32_m3_revision_intake`
- `diagnostics/a32/m3_revision_intake.json`

Conditions d'acceptation :

- `status=UNRESOLVED`
- `revision_status=CANDIDATE_ONLY`
- `support=0`
- `contradictions=0`
- `controlled_test_required=true`
- `support_events >= 3`
- `independent_support_events >= 2`
- `contradiction_events == 0`

Run du 2026-06-19 :

- entree : `diagnostics/m3/a15_revision_queue.json`
- sortie : `diagnostics/a32/m3_revision_intake.json`
- `accepted_candidates=1`
- `rejected_candidates=0`
- `candidate_records=1`
- `support_events=3`
- `independent_support_events=2`
- `reused_control_support_events=1`
- `contradiction_events=0`
- `hypotheses_confirmed=0`
- `hypotheses_refuted=0`
- `wrong_confirmations=0`
- `revision_performed=false`

Lecture :

- A32 prouve que la sortie M3.6 est consommable par le contrat A15-A31 sous
  forme candidate.
- A32 reste une intake, pas une revision : elle demande explicitement une
  decision A15-A31 ulterieure entre confirmation apres revision, refutation
  apres revision, ou test supplementaire.

## Commandes de verification

Tests M3 cibles :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_scientific_planner_state.py tests\test_m3_next_experiment_selector.py tests\test_m3_scientific_planning_loop.py tests\test_m3_a15_revision_queue.py tests\test_m3_m2_candidate_experiment_runner.py tests\test_m3_m2_observation_refinement.py tests\test_m3_refined_followup_planner.py tests\test_m3_refined_followup_executor.py tests\test_m3_dynamic_retarget_followup_planner.py tests\test_m3_dynamic_retarget_followup_executor.py tests\test_m3_dynamic_retarget_mechanism_consolidation.py tests\test_m3_dynamic_retarget_selection_rule_induction.py tests\test_m3_dynamic_retarget_selection_followup_planner.py tests\test_m3_dynamic_retarget_selection_followup_executor.py tests\test_m3_dynamic_retarget_selection_rule_consolidation.py tests\test_m3_dynamic_retarget_patch_similarity_expansion_planner.py tests\test_m3_dynamic_retarget_patch_similarity_expansion_executor.py tests\test_m3_dynamic_retarget_patch_similarity_generativity_consolidation.py tests\test_m3_patch_similarity_a32_revision_queue.py tests\test_m3_a32_requested_patch_similarity_followup_planner.py tests\test_m3_a32_requested_patch_similarity_followup_executor.py tests\test_m3_a32_requested_patch_similarity_scope_consolidation.py tests\test_a32_m3_revision_intake.py -q
```

Run bp35 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.scientific_planning_loop --game-id bp35-0a0ad940 --budget 3 --out diagnostics\m3\scientific_planning_bp35.json
```

Generation file A15 candidate :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.a15_revision_queue --planning-result diagnostics\m3\scientific_planning_bp35.json --out diagnostics\m3\a15_revision_queue.json
```

A32 intake :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a32_m3_revision_intake --queue diagnostics\m3\a15_revision_queue.json --out diagnostics\a32\m3_revision_intake.json
```

M3.7 batch M2 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.m2_candidate_experiment_runner --m2-requests diagnostics\m2\m3_candidate_experiment_requests.json --out diagnostics\m3\m2_candidate_experiment_results.json
```

M3.8 raffinement observations M2 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.m2_observation_refinement --m3-results diagnostics\m3\m2_candidate_experiment_results.json --out diagnostics\m3\refined_candidate_hypotheses_from_m2.json
```

M3.9 follow-up raffine :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.refined_followup_planner --refined diagnostics\m3\refined_candidate_hypotheses_from_m2.json --out diagnostics\m3\refined_followup_experiment_requests.json
```

M3.10 execution follow-up raffine :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.refined_followup_executor --requests diagnostics\m3\refined_followup_experiment_requests.json --out diagnostics\m3\refined_followup_experiment_results.json
```

M3.11 retarget dynamique :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_followup_planner --followup-results diagnostics\m3\refined_followup_experiment_results.json --out diagnostics\m3\dynamic_retarget_followup_requests.json --max-candidate-args 5
```

M3.12 execution retarget dynamique :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_followup_executor --requests diagnostics\m3\dynamic_retarget_followup_requests.json --out diagnostics\m3\dynamic_retarget_followup_results.json
```

M3.13 consolidation mecanistique retarget :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_mechanism_consolidation --retarget-results diagnostics\m3\dynamic_retarget_followup_results.json --out diagnostics\m3\dynamic_retarget_mechanism_candidates.json
```

M3.14 induction de regles de selection retarget :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_selection_rule_induction --mechanism-candidates diagnostics\m3\dynamic_retarget_mechanism_candidates.json --out diagnostics\m3\dynamic_retarget_selection_rules.json
```

M3.15 planning follow-up selection retarget :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_selection_followup_planner --selection-rules diagnostics\m3\dynamic_retarget_selection_rules.json --out diagnostics\m3\dynamic_retarget_selection_followup_requests.json --max-followup-requests 8
```

M3.16 execution follow-up selection retarget :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_selection_followup_executor --requests diagnostics\m3\dynamic_retarget_selection_followup_requests.json --out diagnostics\m3\dynamic_retarget_selection_followup_results.json --max-dynamic-args-per-request 1
```

M3.17 consolidation evidence selection retarget :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_selection_rule_consolidation --selection-followup-results diagnostics\m3\dynamic_retarget_selection_followup_results.json --out diagnostics\m3\dynamic_retarget_selection_rule_consolidation.json
```

M3.18 planning expansion patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_patch_similarity_expansion_planner --selection-rule-consolidation diagnostics\m3\dynamic_retarget_selection_rule_consolidation.json --out diagnostics\m3\dynamic_retarget_patch_similarity_expansion_requests.json --max-dynamic-args 3
```

M3.19 execution expansion patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_patch_similarity_expansion_executor --requests diagnostics\m3\dynamic_retarget_patch_similarity_expansion_requests.json --out diagnostics\m3\dynamic_retarget_patch_similarity_expansion_results.json
```

M3.20 consolidation generativite patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.dynamic_retarget_patch_similarity_generativity_consolidation --expansion-results diagnostics\m3\dynamic_retarget_patch_similarity_expansion_results.json --out diagnostics\m3\dynamic_retarget_patch_similarity_generativity_consolidation.json
```

M3.21 bridge queue A32 patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.patch_similarity_a32_revision_queue --generativity-consolidation diagnostics\m3\dynamic_retarget_patch_similarity_generativity_consolidation.json --out diagnostics\m3\patch_similarity_a32_revision_queue.json
```

M3.22 planning requests demandees par A32.3 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.a32_requested_patch_similarity_followup_planner --a32-decisions diagnostics\a32\patch_similarity_revision_decisions.json --out diagnostics\m3\a32_requested_patch_similarity_followup_requests.json --max-dynamic-args 2
```

M3.23 execution requests demandees par A32.3 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.a32_requested_patch_similarity_followup_executor --requests diagnostics\m3\a32_requested_patch_similarity_followup_requests.json --out diagnostics\m3\a32_requested_patch_similarity_followup_results.json
```

M3.24 consolidation scope demande A32.3 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.a32_requested_patch_similarity_scope_consolidation --followup-results diagnostics\m3\a32_requested_patch_similarity_followup_results.json --out diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json
```

M3.O1 planning objectif stop/switch :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_stop_switch_experiment_planner --objective-hypotheses diagnostics\m2\objective_conditioned_hypotheses.json --out diagnostics\m3\objective_stop_switch_experiment_requests.json
```

M3.O2 execution objectif stop/switch :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_stop_switch_experiment_executor --objective-requests diagnostics\m3\objective_stop_switch_experiment_requests.json --out diagnostics\m3\objective_stop_switch_experiment_results.json
```

M3.O3 consolidation seuil objectif :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_threshold_consolidation --objective-results diagnostics\m3\objective_stop_switch_experiment_results.json --out diagnostics\m3\objective_threshold_consolidation.json
```

M3.O4 execution fenetre objectif raffinee :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_refined_window_executor --threshold-consolidation diagnostics\m3\objective_threshold_consolidation.json --out diagnostics\m3\objective_refined_window_results.json
```

Guard anti-verdict :

- Rechercher les tokens de verdict automatique interdits dans `theory/m3`,
  `theory/a32_m3_revision_intake.py`, `diagnostics/m3`, `diagnostics/a32` et
  ce document.
- Resultat attendu : aucun match.

## M3.G0.1 - Generic mechanic experiment planner

Statut : fait.

But : consommer le ledger M1.G0 generaliste et choisir un petit lot de tests
M3 generiques, sans execution et sans verdict.

Entree :

- `diagnostics/m1/general_mechanic_candidates.json`

Sortie :

- `diagnostics/m3/generic_mechanic_experiment_requests.json`

Contrat respecte :

- lit uniquement M1.G0 ;
- ne modifie pas M1/M2/A32/A33 ;
- n'execute aucun test ;
- n'utilise que les actions observees dans la sweep M1.G0 ;
- transforme les scores M1 en priorite de planning, jamais en support ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, `execution_performed=false`.

Run bp35 :

- source_mechanic_hypotheses = 952 ;
- observed_actions = ACTION3, ACTION4, ACTION6 ;
- generic_experiment_requests_generated = 14 ;
- requests_by_source_family :
  - entity_role = 2 ;
  - action_effect = 4 ;
  - relation_change = 6 ;
  - dynamic_invariant = 2 ;
- test types :
  - actor_controllability_probe = 1 ;
  - entity_role_stability_probe = 1 ;
  - action_effect_causality_probe = 4 ;
  - relation_change_probe = 6 ;
  - dynamic_invariant_probe = 2.

Selection finale :

- role acteur candidat : `E182` comme `controllable_actor` ;
- role HUD/timer candidat : `E001` comme `timer_or_hud` ;
- effets d'action observes : `ACTION3/ACTION4` sur `move_entity` et
  `transform_entity` ;
- relations prioritaires : deltas `distance_decreases` impliquant
  `E182` et des cibles candidates ;
- invariants dynamiques : `E001::monotone_counter` avec
  `remaining_semantics_unknown=true`, et `E193::exogenous_motion`.

Interpretation :

M3.G0.1 ne dit pas quelle mecanique est vraie. Il transforme simplement la
masse M1.G0 en une matrice courte de tests controles prets pour M3.G0.2.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_experiment_planner --m1-candidates diagnostics\m1\general_mechanic_candidates.json --out diagnostics\m3\generic_mechanic_experiment_requests.json --max-requests 16
```

## M3.G0.2 - Generic mechanic experiment executor

Statut : fait.

But : executer les requetes M3.G0.1 en tests generiques deduplicates, puis
produire des observations candidate-only.

Entree :

- `diagnostics/m3/generic_mechanic_experiment_requests.json`
- `diagnostics/m1/general_mechanic_candidates.json` pour les profils
  d'entites candidates

Sortie :

- `diagnostics/m3/generic_mechanic_experiment_results.json`

Contrat respecte :

- consomme 14 requetes M3.G0.1 ;
- deduplique les cellules identiques par `game_id + replay_policy + action` ;
- execute seulement les actions observees `ACTION3`, `ACTION4`, `ACTION6` ;
- logge `entity_delta`, `relation_delta`, `invariant_delta` et `global_delta` ;
- garde les interpretations semantiques a `unknown` ;
- ne confirme pas `E001::monotone_counter` comme horizon terminal ;
- n'ecrit pas A32/A33 ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`.

Run bp35 :

- generic_requests_consumed = 14 ;
- planned_execution_cells = 42 ;
- unique_execution_cells = 3 ;
- duplicate_execution_cells = 39 ;
- duplicate_execution_cells_counted_as_independent = false ;
- controlled_experiments_run = 3 ;
- blocked_controls = 0 ;
- support_events = 10 ;
- contradiction_events = 0 ;
- neutral_events = 4 ;
- support = 0.

Lecture candidate-only :

- les signaux generiques sont suffisants pour nourrir une consolidation
  M3.G0.3 ;
- les support_events sont des observations de test, pas du support
  scientifique confirme ;
- la semantique du compteur HUD reste inconnue dans M3.G0.2.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_experiment_executor --requests diagnostics\m3\generic_mechanic_experiment_requests.json --out diagnostics\m3\generic_mechanic_experiment_results.json
```

## M3.G0.3 - Generic mechanic evidence consolidation

Statut : fait.

But : consolider les observations M3.G0.2 par hypothese et par famille, sans
transformer les evenements bruts en support scientifique.

Entree :

- `diagnostics/m3/generic_mechanic_experiment_results.json`

Sortie :

- `diagnostics/m3/generic_mechanic_evidence_consolidation.json`

Contrat respecte :

- ne lit que M3.G0.2 ;
- n'execute rien ;
- n'ecrit pas A32/A33 ;
- distingue `source_support_events`, `unique_execution_cells` et
  `independent_contexts` ;
- force `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3` ;
- garde les interpretations semantiques inconnues ;
- impose un follow-up contextuel avant induction de modele symbolique.

Run bp35 :

- generic_requests_consumed = 14 ;
- hypothesis_consolidations = 14 ;
- family_consolidations = 4 ;
- source_support_events = 10 ;
- source_neutral_events = 4 ;
- source_contradiction_events = 0 ;
- unique_execution_cells = 3 ;
- planned_execution_cells = 42 ;
- duplicate_execution_cells = 39 ;
- independent_contexts = 1 ;
- context_diversity_assessment = LOW_RESET_ONLY ;
- contextual_followup_required_before_symbolic_model = true ;
- support_events_counted_as_scientific_support = false.

Statuts produits :

- DUPLICATE_EVIDENCE_ONLY = 10 ;
- NEUTRAL_CANDIDATE_ONLY = 4 ;
- aucune contradiction ;
- aucun `SUPPORTED_CANDIDATE_ONLY`, car le contexte experimental est encore
  reset-only.

Lecture :

M3.G0.3 confirme la qualite du rattachement experimental, pas la verite des
mecaniques. Les signaux sont coherents, mais la diversite experimentale est
faible. La prochaine etape propre est M3.G0.4 : planner des follow-ups
contextuels generiques.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_evidence_consolidation --generic-results diagnostics\m3\generic_mechanic_experiment_results.json --out diagnostics\m3\generic_mechanic_evidence_consolidation.json
```

## M3.G0.4 - Generic contextual follow-up planner

Statut : fait.

But : transformer la consolidation reset-only M3.G0.3 en une petite matrice de
follow-ups contextuels, afin d'augmenter la diversite experimentale avant toute
induction de modele symbolique.

Entrees :

- `diagnostics/m3/generic_mechanic_evidence_consolidation.json`
- `diagnostics/m3/generic_mechanic_experiment_requests.json`

Sortie :

- `diagnostics/m3/generic_mechanic_contextual_followup_requests.json`

Contrat respecte :

- consomme la consolidation M3.G0.3 uniquement si
  `context_diversity_assessment=LOW_RESET_ONLY` ;
- ne promeut pas les signaux dupliques en support scientifique ;
- n'execute rien ;
- n'ecrit pas A32/A33 ;
- planifie des contextes non-reset et des sequences multi-actions ;
- conserve `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`.

Run bp35 :

- source_hypothesis_consolidations = 14 ;
- source_duplicate_evidence_hypotheses = 10 ;
- contextual_followup_requests_generated = 13 ;
- all_contextual_followup_requests_before_truncation = 13 ;
- truncated_contextual_followup_requests = 0 ;
- expected_independent_contexts_after_execution = 6 ;
- target_context_diversity = MULTI_PREFIX_CONTEXTS ;
- duplicate_evidence_promoted = false ;
- execution_performed = false ;
- support = 0.

Familles de follow-up :

- actor_persistence_outside_reset = 3 ;
- action_effect_stability = 4 ;
- relation_change_recurrence = 4 ;
- dynamic_invariant_temporal = 1 ;
- exogenous_motion_recurrence = 1.

Contextes planifies :

- prefixes non-reset : `[ACTION3]`, `[ACTION4]`, `[ACTION6]` ;
- sequences temporelles :
  `[ACTION3, ACTION4, ACTION6]`,
  `[ACTION4, ACTION4, ACTION3]`,
  `[ACTION6, ACTION3, ACTION4]`.

Lecture :

M3.G0.4 ne cherche pas a prouver les mecaniques M1.G0. Il cherche a sortir du
reset-only. Les hypotheses avec signaux compatibles sont converties en
questions contextuelles : l'acteur candidat reste-t-il tracable apres prefixe,
les effets d'action et les changements relationnels persistent-ils hors reset,
et les invariants dynamiques restent-ils reguliers sur plusieurs sequences ?

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_contextual_followup_planner --evidence-consolidation diagnostics\m3\generic_mechanic_evidence_consolidation.json --out diagnostics\m3\generic_mechanic_contextual_followup_requests.json --max-followup-requests 16
```

## M3.G0.5 - Generic contextual follow-up executor

Statut : fait.

But : executer les follow-ups M3.G0.4 dans des contextes non-reset et des
sequences temporelles courtes, afin de verifier si les signaux M1.G0/M3.G0
survivent hors du reset-only.

Entree :

- `diagnostics/m3/generic_mechanic_contextual_followup_requests.json`

Sortie :

- `diagnostics/m3/generic_mechanic_contextual_followup_results.json`

Contrat respecte :

- consomme 13 requetes M3.G0.4 ;
- deduplique les cellules par `game_id + replay_policy + contexte + action`
  ou par sequence temporelle ;
- execute les prefixes `[ACTION3]`, `[ACTION4]`, `[ACTION6]` et les trois
  sequences temporelles planifiees ;
- logge `entity_delta`, `relation_delta`, `invariant_delta`, `global_delta`
  et les transitions temporelles ;
- garde les interpretations semantiques a `unknown` ;
- ne confirme pas `E001::monotone_counter` comme horizon terminal ;
- n'ecrit pas A32/A33 ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`.

Run bp35 :

- contextual_followup_requests_consumed = 13 ;
- planned_contextual_cells = 28 ;
- unique_contextual_execution_cells = 9 ;
- duplicate_contextual_cells = 19 ;
- duplicate_contextual_cells_counted_as_independent = false ;
- independent_contexts = 6 ;
- context_diversity_assessment = MULTI_PREFIX_CONTEXTS ;
- controlled_experiments_run = 9 ;
- blocked_contexts = 0 ;
- support_events = 12 ;
- contradiction_events = 1 ;
- neutral_events = 0 ;
- support = 0.

Statuts candidate-only produits :

- CONTEXT_REPRODUCED_CANDIDATE_ONLY = 11 ;
- CONTEXT_FAILED_CANDIDATE_ONLY = 1 ;
- TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY = 1.

Lecture :

M3.G0.5 atteint le KPI central de M3.G0.4 : sortir du reset-only. Les signaux
acteur/action/relation se reproduisent dans des prefixes non-reset, mais ils
restent candidate-only. Le compteur `E001::monotone_counter` ne se reproduit
pas dans ce protocole generique et reste semantiquement inconnu ; le candidat
`E193::exogenous_motion` montre une regularite temporelle candidate-only. Ces
evenements ne sont pas du support scientifique : ils preparent une consolidation
contextuelle M3.G0.6.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_contextual_followup_executor --requests diagnostics\m3\generic_mechanic_contextual_followup_requests.json --out diagnostics\m3\generic_mechanic_contextual_followup_results.json
```

## M3.G0.6 - Generic contextual evidence consolidation

Statut : fait.

But : condenser les resultats M3.G0.5 par hypothese et par famille, en tenant
compte de la diversite contextuelle reelle, sans produire de verdict
scientifique.

Entree :

- `diagnostics/m3/generic_mechanic_contextual_followup_results.json`

Sortie :

- `diagnostics/m3/generic_mechanic_contextual_evidence_consolidation.json`

Contrat respecte :

- ne lit que M3.G0.5 ;
- n'execute rien ;
- n'ecrit pas A32/A33 ;
- distingue `contextual_events`, contradictions localisees et readiness
  candidate-only pour modele symbolique ;
- garde les interpretations semantiques a `unknown` ;
- ne transforme pas les evenements contextuels en support scientifique ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`.

Run bp35 :

- contextual_followup_requests_consumed = 13 ;
- hypothesis_consolidations = 13 ;
- family_consolidations = 4 ;
- source_support_events = 12 ;
- source_contradiction_events = 1 ;
- source_neutral_events = 0 ;
- unique_contextual_execution_cells = 9 ;
- planned_contextual_cells = 28 ;
- duplicate_contextual_cells = 19 ;
- independent_contexts = 6 ;
- context_diversity_assessment = MULTI_PREFIX_CONTEXTS ;
- contextual_events_counted_as_scientific_support = false ;
- contradiction_counted_as_refutation = false ;
- ready_for_symbolic_model_candidate_only = true ;
- symbolic_model_induction_performed = false ;
- support = 0.

Statuts produits :

- CONTEXT_STABLE_CANDIDATE_ONLY = 11 ;
- TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY = 1 ;
- TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY = 1.

Familles :

- action_effect : READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY ;
- entity_role : READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY ;
- relation_change : READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY ;
- dynamic_invariant : READY_FOR_SYMBOLIC_MODEL_CANDIDATE_ONLY, mais avec
  contradiction/follow-up requis.

Contradiction localisee :

- source_hypothesis_id = `m1g0::dynamic_invariant::E001::monotone_counter` ;
- candidate_status = TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY ;
- observation = dynamic_invariant_not_observed_across_contextual_temporal_sequences ;
- contradiction_counted_as_refutation = false ;
- followup_required_for_contradiction = true ;
- semantic_interpretation = unknown.

Lecture :

M3.G0.6 peut maintenant dire que plusieurs signaux acteur/action/relation sont
stables dans plusieurs contextes courts. Il ne dit toujours pas qu'ils sont
vrais. Le compteur `E001::monotone_counter` est affaibli dans ce protocole et
doit etre suivi explicitement. Le candidat `E193::exogenous_motion` reste une
regularite temporelle candidate-only. La prochaine etape peut etre M3.G0.7 :
induire un modele symbolique candidat, en incluant la contradiction E001 comme
caveat et non comme refutation globale.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_mechanic_contextual_evidence_consolidation --contextual-results diagnostics\m3\generic_mechanic_contextual_followup_results.json --out diagnostics\m3\generic_mechanic_contextual_evidence_consolidation.json
```

## M3.G0.7 - Candidate symbolic mechanism model induction

Statut : fait.

But : transformer les hypotheses contextuellement stables de M3.G0.6 en un
modele symbolique candidat, sans verdict scientifique et sans ecriture A32/A33.

Entrees :

- `diagnostics/m3/generic_mechanic_contextual_evidence_consolidation.json`
- `diagnostics/m3/generic_mechanic_contextual_followup_requests.json`

Sortie :

- `diagnostics/m3/generic_candidate_symbolic_mechanism_model.json`

Contrat respecte :

- consomme uniquement les consolidations ready candidate-only de M3.G0.6 ;
- utilise M3.G0.4 comme source structurale pour les entites, actions,
  relations et invariants ;
- inclut les contradictions comme caveats, pas comme refutations ;
- garde toutes les interpretations semantiques a `unknown` ;
- ne confirme pas E182 comme agent, ACTION3/ACTION4 comme mouvements vrais,
  E193 comme gravite, ni E001 comme compteur invalide ;
- n'ecrit pas A32/A33 ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`.

Run bp35 :

- source_hypothesis_consolidations = 13 ;
- source_context_diversity_assessment = MULTI_PREFIX_CONTEXTS ;
- source_independent_contexts = 6 ;
- actor_candidates = 1 ;
- action_models = 2 ;
- relation_effects = 4 ;
- dynamic_invariants = 1 ;
- caveats = 1 ;
- ready_for_policy_probe_candidate_only = true ;
- symbolic_model_induction_performed = true ;
- symbolic_model_induction_counted_as_verdict = false ;
- model_counted_as_confirmation = false ;
- support = 0.

Modele candidat :

- acteur candidat : `E182`, role `controllable_actor_candidate`,
  base `CONTEXT_STABLE_CANDIDATE_ONLY` ;
- `ACTION3` : effets candidats `move_entity`, `transform_entity`, relation
  candidate `distance_decreases` ;
- `ACTION4` : effets candidats `move_entity`, `transform_entity`, relation
  candidate `distance_decreases` ;
- relation model : effets acteur-candidat vers `E138`, `E139`, `E136`, `E137`,
  tous en `distance_decreases` candidate-only ;
- invariant dynamique candidat : `E193::exogenous_motion`,
  `TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY`, semantique inconnue ;
- caveat : `E001::monotone_counter`,
  `TEMPORAL_REGULARITY_FAILED_CANDIDATE_ONLY`,
  `followup_required=true`, semantique inconnue.

Lecture :

M3.G0.7 produit une representation symbolique utilisable par un probe de policy,
mais pas une mecanique confirmee. Le modele dit seulement : "voici les pieces
symboliques candidates qui ont survecu a plusieurs contextes courts". Le verdict
reste hors M3.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.generic_candidate_symbolic_model_induction --contextual-consolidation diagnostics\m3\generic_mechanic_contextual_evidence_consolidation.json --out diagnostics\m3\generic_candidate_symbolic_mechanism_model.json
```

## M3.G1 - Stop-state objective-conversion experiments (G1.1 -> G1.2 -> G1.3)

Objectif :

Compiler les 12 hypotheses M2.G1 (`diagnostics/m2/objective_conversion_hypotheses.json`)
en experiences controlees candidate-only, puis les executer depuis un etat
safe-stop rejoue, puis interpreter les mesures. Unite experimentale centrale :
`safe_stop_state -> candidate_action_or_sequence`, contre deux controles :
`hold_or_stop_state` (zero action) et `relation_progress_policy` (continuation
P3.G0 horizon-matchee a `candidate_len`).

Lineage : P3.G1 (terminal-safe-but-passive) -> P2.G1/G3 (frontier + handoff) ->
M2.G1 (12 hypotheses) -> M3.G1.

Garde-fous (tous les artefacts) : `support=0`, `revision_status=CANDIDATE_ONLY`,
`truth_status=NOT_EVALUATED_BY_M3`, `m2_hypothesis_counted_as_confirmation=false`,
`experiment_result_counted_as_scientific_verdict=false`, `a32_write_performed=false`,
`a33_write_performed=false`. A32 reste le seul lieu de verdict. Le mot "verdict"
n'apparait pas dans l'artefact M3.G1.3.

### M3.G1.1 - planner

- Entree : `diagnostics/m2/objective_conversion_hypotheses.json`.
- Sortie : `diagnostics/m3/objective_conversion_experiment_requests.json`.
- 12 hypotheses pretes -> 12 requests ; 22 conditions candidates au total ;
  4 familles couvertes.
- Metrique decisive : `delta_terminal_adjusted_progress_vs_hold` (en tete des
  `primary_metrics`).
- `relation_progress_policy` est horizon-matchee : `post_stop_horizon =
  candidate_len`.

### M3.G1.2 - executor (measurements seulement)

- Entree : requests M3.G1.1 + safe-stop P3.G1.
- Sortie : `diagnostics/m3/objective_conversion_experiment_results.json`.
- Safe-stop capture une fois, deterministe :
  - `safe_stop_capture_config.selection_rule =
    best_p3g1_condition_from_utility_consolidation`, fallback
    `objective_aware_abstract_policy_lambda_0` / budget 64 / seed 0,
    `selection_counted_as_support=false`.
  - condition selectionnee : `objective_aware_abstract_policy_lambda_0`,
    prefix safe-stop de 15 actions, `safe_stop_context_diversity =
    LOW_SINGLE_SAFE_STOP`.
- 46 cellules planifiees -> 10 cellules uniques (7 candidates, 1 hold,
  2 relation) ; 10 replay-exact ; 0 bloquee.
- Chaque cellule porte `captured_prefix_hash`, `replayed_prefix_hash`,
  `safe_stop_state_hash`, `safe_stop_replay_exact`.
- Statuts de blocage dedies (aucun resultat ambigu) :
  `SAFE_STOP_REPLAY_MISMATCH_BLOCKED`, `SAFE_STOP_NOT_REACHED_BLOCKED`,
  `CANDIDATE_ACTION_UNAVAILABLE_BLOCKED`,
  `TERMINAL_DURING_SAFE_STOP_PREFIX_BLOCKED`.
- L'executor ne produit AUCUN statut agrege
  (`produces_aggregated_outcome_enum=false`) : interpretation deferree a
  M3.G1.3.

Mesures cle (hold baseline `terminal_adjusted_progress` = 150, prefix 15) :

| Condition | taps | delta_vs_hold | terminal_reentry |
|---|---:|---:|---|
| hold_or_stop_state | 150 | 0 | non |
| candidate ACTION3 | 0 | -150 | oui |
| candidate ACTION4 | 0 | -150 | oui |
| candidate ACTION6 | 155 | +5 | non |
| candidate ACTION3,ACTION4 | 0 | -150 | oui |
| candidate ACTION4,ACTION3 | 0 | -150 | oui |
| candidate ACTION6,ACTION3 | 165 | +15 | non |
| candidate ACTION6,ACTION4 | 165 | +15 | non |
| relation_progress_policy (h1) | 0 | -150 | oui |
| relation_progress_policy (h2) | 0 | -150 | oui |

### M3.G1.3 - consolidation (interpretation candidate-only)

- Entree : results M3.G1.2.
- Sortie : `diagnostics/m3/objective_conversion_experiment_consolidation.json`.
- Regle de decision centrale : `candidate beats hold_or_stop_state on
  delta_terminal_adjusted_progress_vs_hold > 0 AND terminal_reentry == false` ;
  secondaire : `candidate beats relation_progress_policy`.
- Statuts candidate-only : `CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY`,
  `NO_CONVERSION_SIGNAL_CANDIDATE_ONLY`, `TERMINAL_REENTRY_CANDIDATE_ONLY`,
  `MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY`,
  `OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY`.

Resultat :

- `consolidation_outcome_status = MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY`
- `conversion_signal_candidates = 3` (`ACTION6`, `ACTION6,ACTION3`,
  `ACTION6,ACTION4`)
- `terminal_reentry_candidates = 4` (`ACTION3`, `ACTION4`, `ACTION3,ACTION4`,
  `ACTION4,ACTION3`)
- best candidate `ACTION6,ACTION3`, `delta_vs_hold = +15`, sans terminal
  re-entry
- `relation_progress_policy` re-entre en terminal sur les deux horizons
  (best_relation_taps = 0)
- `next_step = M3.G2 multi-safe-stop objective-conversion validation`

Lecture :

M3.G1 ferme un cycle SAGE complet : P3.G1 decouvre "safe mais passif" -> P2
formalise la frontier objective-conversion -> M2.G1 genere des hypotheses
falsifiables -> M3.G1 teste et observe un signal candidate-only. Depuis le
safe-stop, les sequences menees par `ACTION6` augmentent le progres
terminal-ajuste au-dessus de hold sans re-entrer en terminal, alors que
`ACTION3`/`ACTION4` et la continuation relation-progress re-entrent en terminal.
Ce n'est pas une mecanique confirmee : c'est un premier signal de conversion
objectif candidate-only, faible en diversite contextuelle
(`LOW_SINGLE_SAFE_STOP`), a valider en M3.G2 avec plusieurs safe-stops.

Commandes :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_experiment_planner --hypotheses diagnostics\m2\objective_conversion_hypotheses.json --out diagnostics\m3\objective_conversion_experiment_requests.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_experiment_executor --requests diagnostics\m3\objective_conversion_experiment_requests.json --out diagnostics\m3\objective_conversion_experiment_results.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_experiment_consolidation --results diagnostics\m3\objective_conversion_experiment_results.json --out diagnostics\m3\objective_conversion_experiment_consolidation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_objective_conversion_experiment_planner.py tests\test_m3_objective_conversion_experiment_executor.py tests\test_m3_objective_conversion_experiment_consolidation.py -q
```

## M3.G2 - Multi-safe-stop objective-conversion validation

Statut : fait.

But :

Tester si le signal M3.G1 `ACTION6`-led est robuste ou seulement local au
safe-stop unique capture par P3.G1. M3.G2 doit reposer la question exacte :
`safe-stop + ACTION6-led sequence -> progres marginal positif sans terminal`
sur plusieurs etats safe-stop, seeds et budgets.

Motivation :

- Avant M3.G1, la boucle objective-conversion etait surtout : safe mais passif.
- M3.G1 observe un premier signal candidate-only :
  - `ACTION6` : delta_vs_hold +5, sans terminal re-entry ;
  - `ACTION6,ACTION3` : delta_vs_hold +15, sans terminal re-entry ;
  - `ACTION6,ACTION4` : delta_vs_hold +15, sans terminal re-entry.
- Les controles ne disent pas "toute action apres stop marche" :
  - `ACTION3`, `ACTION4`, `ACTION3,ACTION4`, `ACTION4,ACTION3` re-entrent
    en terminal ;
  - `relation_progress_policy` re-entre aussi en terminal aux horizons testes.

Implementation :

- Ajout de `theory/m3/objective_conversion_multi_safe_stop_validation.py`.
- Ajout de `tests/test_m3_objective_conversion_multi_safe_stop_validation.py`.
- Sortie : `diagnostics/m3/objective_conversion_multi_safe_stop_validation.json`.

Contrat M3.G2 :

- Lire M3.G1.3 comme signal candidate-only, pas comme verdict.
- Generer ou selectionner plusieurs safe-stop states replay-exacts depuis la
  famille P3.G1 ou des variantes controlees de seed/budget.
- Retester au minimum `ACTION6,ACTION3` et `ACTION6,ACTION4`, avec `ACTION6`
  comme ablation courte.
- Garder les controles `hold_or_stop_state` et `relation_progress_policy`
  horizon-matchee.
- Mesurer `delta_terminal_adjusted_progress_vs_hold`, `terminal_reentry`,
  `objective_completion_signal`, `levels_completed_after_rollout` et les
  diagnostics relationnels deja presents.
- Ne pas ecrire A32/A33 et conserver `support=0`,
  `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_M3`.

Criteres de lecture attendus :

- `REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY` si les sequences
  ACTION6-led battent hold sans terminal re-entry sur plusieurs safe-stops.
- `LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY` si le signal ne survit que dans
  le safe-stop M3.G1.
- `TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY` si les sequences ACTION6-led
  re-entrent en terminal dans des contextes alternatifs.
- `OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY` seulement si un niveau est
  effectivement complete, toujours sans confirmation scientifique.

Run bp35 du 2026-06-21 :

- source : M3.G1.3 `MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY` ;
- cibles retestees : `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4` ;
- controles conserves : `hold_or_stop_state`,
  `relation_progress_policy_h1`, `relation_progress_policy_h2` ;
- safe-stop capture specs planifiees = 8
  (`lambda_0`/`lambda_1` x budgets 48/64 x seeds 0/1) ;
- unique_safe_stop_captures = 1 ;
- duplicate_safe_stop_captures = 7 ;
- duplicate_safe_stop_captures_counted_as_independent = false ;
- selected_cells_per_safe_stop = 6 ;
- cells_executed = 6 ;
- cells_blocked = 0 ;
- reproduced_signal_safe_stops = 1 ;
- terminal_risk_safe_stops = 0 ;
- objective_completion_safe_stops = 0 ;
- safe_stop_context_diversity = `LOW_DUPLICATE_SAFE_STOP_CONTEXTS` ;
- validation_outcome_status =
  `LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY` ;
- `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Mesures sur le safe-stop unique deduplique :

| Sequence | delta_vs_hold | terminal_reentry | objective_completion |
|---|---:|---|---|
| ACTION6 | +5 | non | non |
| ACTION6,ACTION3 | +15 | non | non |
| ACTION6,ACTION4 | +15 | non | non |

Lecture actuelle :

M3.G1 a transforme "eviter GAME_OVER" en une question plus fine : une sequence
ACTION6-led peut-elle convertir un safe-stop passif en progres objectif sans
rouvrir le risque terminal ? M3.G2 implemente cette validation, mais le run
actuel ne cree pas encore la diversite experimentale attendue : toutes les
variantes condition/budget/seed rejoignent le meme safe-stop replay-exact. Le
signal ACTION6-led reste positif localement, sans terminal re-entry, mais il ne
survit pas encore a un vrai test multi-safe-stop. La prochaine etape propre est
donc d'elargir la capture de safe-stops, pas de promouvoir le signal.

Commandes :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_multi_safe_stop_validation --requests diagnostics\m3\objective_conversion_experiment_requests.json --source-consolidation diagnostics\m3\objective_conversion_experiment_consolidation.json --out diagnostics\m3\objective_conversion_multi_safe_stop_validation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_objective_conversion_experiment_planner.py tests\test_m3_objective_conversion_experiment_executor.py tests\test_m3_objective_conversion_experiment_consolidation.py tests\test_m3_objective_conversion_multi_safe_stop_validation.py -q
```

## M3.G3 - Safe-stop diversity acquisition / anti-attractor sampler

Statut : fait.

But :

Lire M3.G2 comme diagnostic de diversite insuffisante, pas comme echec du
signal ACTION6-led, puis produire des safe-stops replay-exacts, non terminaux,
terminal-safe et structurellement distincts avant toute nouvelle validation
objective-conversion.

Implementation :

- Ajout de `theory/m3/objective_conversion_safe_stop_diversity_sampler.py`.
- Ajout de `tests/test_m3_objective_conversion_safe_stop_diversity_sampler.py`.
- Sortie :
  `diagnostics/m3/objective_conversion_safe_stop_diversity_sampler.json`.

Contrat respecte :

- lit uniquement M3.G2 comme diagnostic de diversite ;
- genere des endpoints anti-attracteur par troncatures du prefixe P3.G1,
  phase-shifts ACTION3/ACTION4, bursts simples et perturbations ACTION6 ;
- accepte seulement les candidats replay-exacts, non terminaux, terminal-safe,
  avec hold baseline mesurable ;
- impose la nouveaute par hash d'etat, hash de prefixe et signature
  relationnelle proxy ;
- ne teste aucune sequence objective-conversion ;
- ne compte pas la diversite comme support scientifique ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Run bp35 du 2026-06-21 :

- source G2 : `LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY`,
  8 captures planifiees -> 1 safe-stop unique ;
- safe_stop_candidates_planned = 26 ;
- safe_stop_candidates_executed = 26 ;
- unique_safe_stop_candidates = 26 ;
- accepted_diverse_safe_stops = 13 ;
- duplicate_or_near_duplicate_rejected = 13 ;
- terminal_or_unsafe_rejected = 0 ;
- replay_inexact_rejected = 0 ;
- blocked_candidates_rejected = 0 ;
- min_diverse_safe_stops_required = 3 ;
- diversity_status = `SUFFICIENT_FOR_M3_G4` ;
- ready_for_m3_g4 = true ;
- objective_conversion_sequences_tested = false ;
- `support=0`, aucun verdict.

Lecture :

M3.G3 confirme que le verrou de M3.G2 etait bien la capture deterministe du
safe-stop, pas l'absence d'etats safe-stop alternatifs. En sortant de la policy
safe-stop attractrice, on obtient assez d'endpoints replay-exacts et
terminal-safe pour relancer proprement la validation. Ce n'est toujours pas une
preuve du motif ACTION6-led : c'est seulement l'acquisition du support
experimental necessaire pour M3.G4.

Commandes :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_safe_stop_diversity_sampler --source-g2 diagnostics\m3\objective_conversion_multi_safe_stop_validation.json --out diagnostics\m3\objective_conversion_safe_stop_diversity_sampler.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_objective_conversion_multi_safe_stop_validation.py tests\test_m3_objective_conversion_safe_stop_diversity_sampler.py -q
```

## M3.G4 - Multi-diverse-safe-stop objective-conversion validation

Statut : fait.

But :

Relancer le protocole M3.G2 sur les safe-stops diversifies M3.G3, afin de
tester enfin la vraie question de generalisation : `ACTION6,ACTION3` /
`ACTION6,ACTION4` est-il seulement un operateur local opportuniste du safe-stop
P3.G1, ou un motif de conversion post-stop plus robuste ?

Implementation :

- Ajout de `theory/m3/objective_conversion_diverse_safe_stop_validation.py`.
- Ajout de `tests/test_m3_objective_conversion_diverse_safe_stop_validation.py`.
- Sortie :
  `diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json`.

Contrat respecte :

- lire M3.G3 seulement si `diversity_status=SUFFICIENT_FOR_M3_G4` ;
- consommer les `accepted_diverse_safe_stops` M3.G3 comme base states ;
- retester `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4` ;
- garder `hold_or_stop_state` et `relation_progress_policy` horizon-matchee ;
- agreger par safe-stop diversifie, puis par sequence ;
- separer l'analyse globale, par `sampling_family`, par horizon restant et par
  hold baseline ;
- produire uniquement des statuts candidate-only, par exemple
  `REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY`,
  `LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY`,
  `TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY`,
  `OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY` ;
- garder `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Run bp35 du 2026-06-21 :

- source G3 : `SUFFICIENT_FOR_M3_G4` ;
- accepted_diverse_safe_stops_consumed = 13 ;
- selected_cells_per_safe_stop = 6 ;
- execution_cells_total = 78 ;
- cells_executed = 78 ;
- cells_blocked = 0 ;
- candidate_cells_executed = 39 ;
- hold_cells_executed = 13 ;
- relation_progress_cells_executed = 26 ;
- safe_stops_with_weak_signal = 13 ;
- safe_stops_with_medium_signal = 1 ;
- safe_stops_with_terminal_risk = 1 ;
- safe_stops_with_objective_completion = 0 ;
- sampling_family_aggregates = 3 ;
- sequence_aggregates = 3 ;
- validation_outcome_status =
  `MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY` ;
- `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Agregats par sequence :

| Sequence | weak | medium | terminal | mean delta | min/max delta | Status |
|---|---:|---:|---:|---:|---|---|
| ACTION6 | 13/13 | 1/13 | 0/13 | +5.0 | +5 / +5 | `REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY` |
| ACTION6,ACTION3 | 12/13 | 1/13 | 1/13 | +3.076923 | -140 / +15 | `TERMINAL_REENTRY_CANDIDATE_ONLY` |
| ACTION6,ACTION4 | 12/13 | 1/13 | 1/13 | +3.076923 | -140 / +15 | `TERMINAL_REENTRY_CANDIDATE_ONLY` |

Agregats par famille :

| sampling_family | safe-stops | weak | medium | terminal | Status |
|---|---:|---:|---:|---:|---|
| action6_perturbation | 3 | 3 | 0 | 0 | `REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY` |
| base_prefix_truncation | 7 | 7 | 1 | 1 | `TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY` |
| single_action_burst | 3 | 3 | 0 | 0 | `REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY` |

Agregats horizon / baseline :

- `horizon_far_ge_55` : 8 safe-stops, weak 8, terminal 0,
  `REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY` ;
- `horizon_mid_45_54` : 4 safe-stops, weak 4, medium 1, terminal 1,
  `TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY` ;
- `horizon_near_lt_45` : 1 safe-stop, weak 1, terminal 0,
  `LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY` ;
- `hold_low_lt_50` et `hold_mid_50_119` : signal reproduit sans terminal ;
- `hold_high_ge_120` : 3 safe-stops, weak 3, medium 1, terminal 1,
  terminal risk localise.

Lecture :

M3.G4 transforme le signal ACTION6-led en carte de scope candidate-only. Le
signal faible `delta_terminal_adjusted_progress_vs_hold > 0` est present sur
tous les safe-stops diversifies. En revanche, le signal moyen
(`beats_relation_progress_policy`) reste rare, et les sequences longues
`ACTION6,ACTION3` / `ACTION6,ACTION4` reintroduisent un risque terminal dans un
sous-ensemble de safe-stops issus de `base_prefix_truncation`, surtout en
baseline haute / horizon moyen. Aucune completion objective n'est observee.

Conclusion candidate-only :

- `ACTION6` seul ressemble a un operateur post-stop faible mais robuste dans
  cette matrice ;
- `ACTION6,ACTION3` et `ACTION6,ACTION4` restent utiles dans beaucoup de
  contextes, mais leur scope est conditionnel et peut rouvrir le terminal ;
- le prochain verrou n'est plus la diversite des safe-stops, mais la selection
  de sequence post-stop conditionnee par famille/horizon/baseline.

Commandes :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.objective_conversion_diverse_safe_stop_validation --requests diagnostics\m3\objective_conversion_experiment_requests.json --source-g3 diagnostics\m3\objective_conversion_safe_stop_diversity_sampler.json --out diagnostics\m3\objective_conversion_diverse_safe_stop_validation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_objective_conversion_experiment_planner.py tests\test_m3_objective_conversion_experiment_executor.py tests\test_m3_objective_conversion_experiment_consolidation.py tests\test_m3_objective_conversion_multi_safe_stop_validation.py tests\test_m3_objective_conversion_safe_stop_diversity_sampler.py tests\test_m3_objective_conversion_diverse_safe_stop_validation.py -q
```

## M3.G5 - Objective-readiness / commit-action controlled experiment planner

Statut : fait.

But :

Compiler les 15 hypotheses M2.G2 en requests d'experiences controlees, sans
inventer une nouvelle policy P3 et sans executer l'environnement. M3.G5 teste
le nouveau verrou : savoir si le blocage vient d'un signal de readiness, d'une
action de commit, d'une mauvaise representation du but, d'un discriminateur
proxy-vs-completion, ou du gap du selector risk-aware.

Implementation :

- Ajout de `theory/m3/risk_aware_objective_completion_experiment_planner.py`.
- Ajout de `tests/test_m3_risk_aware_objective_completion_experiment_planner.py`.
- Sortie :
  `diagnostics/m3/risk_aware_objective_completion_experiment_requests.json`.

Contrat respecte :

- lit uniquement les hypotheses M2.G2 :
  `diagnostics/m2/risk_aware_objective_completion_hypotheses.json` ;
- consomme seulement les hypotheses `ready_for_m3_g5=true`,
  `support=0`, `truth_status=NOT_EVALUATED_BY_M2` ;
- compile une request par hypothese ;
- n'execute aucun rollout, aucune policy et aucun step environnement ;
- conserve les controles :
  `hold_or_stop_state`, `ACTION6`, `frozen_contextual_selector`,
  `static_ACTION6,ACTION3`, `static_ACTION6,ACTION4`,
  `relation_progress_policy` ;
- selectionne des substrats experimentaux explicites :
  `risk_aware_post_stop_safe_contexts`,
  `selector_action6_fallback_contexts`,
  `selector_extension_safe_contexts`,
  `static_extension_terminal_risk_contexts` ;
- met `objective_completion_signal`, `levels_completed_after_rollout` et
  `terminal_reentry_rate` en metriques primaires ;
- garde `terminal_adjusted_progress_after_stop` comme safety/proxy, pas comme
  succes objectif ;
- conserve `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4` comme substrats ou
  controles, pas comme hypotheses cibles ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Run bp35 du 2026-06-21 :

- risk_aware_objective_hypotheses_consumed = 15
- risk_aware_objective_experiment_requests_generated = 15
- skipped_risk_aware_objective_hypotheses = 0
- all_five_families_covered = true
- covered_hypothesis_families =
  `goal_state_representation_beyond_safe_progress`,
  `objective_readiness_detection`,
  `post_conversion_commit_action_search`,
  `proxy_progress_vs_completion_discriminator`,
  `risk_aware_selector_completion_gap`
- covered_experiment_styles =
  `horizon_conditioned_completion_trigger_search`,
  `post_conversion_commit_action_matrix`,
  `post_selector_objective_readiness_probe`,
  `risk_aware_policy_ablation_with_completion_metrics`,
  `terminal_safe_progress_vs_completion_discriminator`
- substrate_categories =
  `risk_aware_post_stop_safe_contexts`,
  `selector_action6_fallback_contexts`,
  `selector_extension_safe_contexts`,
  `static_extension_terminal_risk_contexts`
- controls_per_request =
  `hold_or_stop_state`,
  `ACTION6`,
  `frozen_contextual_selector`,
  `static_ACTION6,ACTION3`,
  `static_ACTION6,ACTION4`,
  `relation_progress_policy`
- action6_extension_retest_requests_generated = false
- planner_outcome_status =
  `OBJECTIVE_READINESS_EXPERIMENT_REQUESTS_COMPILED_CANDIDATE_ONLY`
- execution_performed = false
- policy_rollout_performed = false
- environment_step_performed = false
- support = 0
- a32_write_performed = false
- a33_write_performed = false

Regle de decision centrale :

`candidate protocol improves objective_completion_signal or
levels_completed_after_rollout over matched controls while keeping
terminal_reentry_rate at or below frozen_contextual_selector`.

Lecture :

M3.G5 ne teste encore rien ; il rend les hypotheses M2.G2 executables sous
forme de protocoles controles. Le point important est que le planner ne revient
pas a "tester ACTION6,A3/A4 encore une fois". Ces sequences restent des
controles ou des substrats. Les cibles M3.G5 sont les detecteurs de readiness,
les actions de commit, les representations du but, les discriminateurs
proxy/completion et les variantes selector-gap.

Suite naturelle :

- `M3.G6` - Objective-readiness / commit-action controlled executor.
- Puis une consolidation candidate-only qui pourra produire
  `COMMIT_ACTION_SIGNAL_CANDIDATE_ONLY`,
  `READINESS_DISCRIMINATOR_SIGNAL_CANDIDATE_ONLY`,
  `PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY` ou
  `NO_OBJECTIVE_COMPLETION_MECHANISM_FOUND_CANDIDATE_ONLY`.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.risk_aware_objective_completion_experiment_planner --hypotheses diagnostics\m2\risk_aware_objective_completion_hypotheses.json --out diagnostics\m3\risk_aware_objective_completion_experiment_requests.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m2_risk_aware_objective_completion_hypothesis_generator.py tests\test_m3_risk_aware_objective_completion_experiment_planner.py -q
```

## M3.G6 - Objective-completion experiment executor

Statut : fait.

But :

Executer les 15 requests M3.G5 contre le substrat post-stop risk-aware deja
mesure, sans modifier les hypotheses M2.G2, sans creer de nouvelle policy P3 et
sans ecrire A32/A33. M3.G6 teste la question : les protocoles readiness,
commit, representation du but, discriminateur proxy/completion ou selector-gap
font-ils apparaitre un signal objectif, ou observe-t-on seulement une divergence
proxy-vs-completion ?

Implementation :

- Ajout de `theory/m3/risk_aware_objective_completion_experiment_executor.py`.
- Ajout de `tests/test_m3_risk_aware_objective_completion_experiment_executor.py`.
- Sortie :
  `diagnostics/m3/risk_aware_objective_completion_experiment_results.json`.

Contrat respecte :

- entree principale :
  `diagnostics/m3/risk_aware_objective_completion_experiment_requests.json` ;
- lit P3.G4/M3.G4 seulement comme substrat mesure pour les controles et
  contextes post-stop ;
- consomme uniquement les requests M3.G5 `READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT` ;
- deduplique les cellules identiques avant mesure ;
- conserve les controles :
  `hold_or_stop_state`, `ACTION6`, `frozen_contextual_selector`,
  `static_ACTION6,ACTION3`, `static_ACTION6,ACTION4`,
  `relation_progress_policy` ;
- conserve `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4` comme controles ou
  substrats, pas comme targets candidates ;
- ne reoptimise pas P3.G2/P3.G4 ;
- ne transforme aucun signal candidat en verdict scientifique ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3`, aucun write A32/A33.

Run bp35 du 2026-06-21 :

- risk_aware_objective_experiment_requests_consumed = 15
- raw_execution_cells_planned = 336
- deduplicated_execution_cells = 72
- deduplicated_cells_removed = 264
- candidate_protocol_cells = 48
- control_cells = 24
- cells_executed = 66
- cells_blocked = 6
- commit_action_cells_blocked = 6
- candidate_protocol_cells_by_family =
  - `objective_readiness_detection` : 12
  - `post_conversion_commit_action_search` : 6
  - `goal_state_representation_beyond_safe_progress` : 12
  - `proxy_progress_vs_completion_discriminator` : 6
  - `risk_aware_selector_completion_gap` : 12
- objective_completion_signal = false
- objective_completion_candidate_cells = 0
- levels_completed_after_rollout_max = 0.0
- terminal_reentry_rate_max_executed_candidate = 0.0
- proxy_completion_divergence_candidate_cells = 42
- proxy_progress_without_completion_observed = true
- static_extension_terminal_options_from_p3g4 = 4
- unsafe_extension_options_avoided_from_p3g4 = 4
- objective_completion_experiment_outcome_status =
  `PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY`
- support = 0
- a32_write_performed = false
- a33_write_performed = false

Lecture :

M3.G6 ne trouve pas de mecanisme de completion. Les protocoles executes
readiness/goal/discriminator/selector-gap restent terminaux-safe mais ne
produisent ni `objective_completion_signal` ni `levels_completed_after_rollout`.
Le resultat principal est donc une divergence proxy/completion : le substrat
risk-aware conserve du progres terminal-adjusted et evite les extensions
terminales observees en P3.G4, mais ce progres ne se convertit toujours pas en
completion.

Les 6 cellules `post_conversion_commit_action_search` sont bloquees parce que
les actions de commit candidates `ACTION1`/`ACTION2`/`ACTION5` ne sont pas
presentes dans les mesures source M3.G6. Ce n'est pas une refutation de la
famille commit-action : c'est un verrou experimental concret. Il faudra un run
qui execute ces actions sur les substrats post-conversion, ou un nouveau
handoff qui explicite pourquoi ces actions sont indisponibles.

Suite naturelle :

- `M3.G7` - Concrete post-conversion commit-action executor, si les actions
  `ACTION1`/`ACTION2`/`ACTION5` sont executables sur les substrats replay-exacts.
- Sinon, ouvrir une frontier sur le manque d'observables/actions de commit
  concretes avant de revenir a M2/M3.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.risk_aware_objective_completion_experiment_executor --requests diagnostics\m3\risk_aware_objective_completion_experiment_requests.json --source-p3g4 diagnostics\p3\risk_targeted_contextual_post_stop_policy_validation.json --source-m3g4 diagnostics\m3\objective_conversion_diverse_safe_stop_validation.json --out diagnostics\m3\risk_aware_objective_completion_experiment_results.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_risk_aware_objective_completion_experiment_planner.py tests\test_m3_risk_aware_objective_completion_experiment_executor.py -q
```

## M3.G7 - State-conditioned LLM hypothesis execution

Statut : fait.

But : executer les requetes candidates M2.13a (LLM local state-conditioned) sur
le substrat post-stop deja mesure en M3.G6, et comparer honnetement le resultat
a M3.G6. La question n'est pas "le LLM a-t-il raison", mais "le LLM ajoute-t-il
un signal testable nouveau au-dela de M3.G6, ou retombe-t-il sur l'echec
proxy/completion deja connu". Cette brique remplace l'ancienne intention M3.G7
(commit-action concret), reportee car `ACTION1/ACTION2/ACTION5` restent
indisponibles sur le substrat.

Entrees :

- `diagnostics/m2/state_conditioned_llm_m3_candidate_requests.json` (M2.13a)
- `diagnostics/m3/risk_aware_objective_completion_experiment_results.json` (M3.G6)

Sortie :

- `diagnostics/m3/state_conditioned_llm_hypothesis_execution_results.json`

Contrat respecte :

- consomme uniquement `status=READY_FOR_M3` ;
- ignore `status=BLOCKED_NOT_TESTABLE` (la metrique unlock
  `available_actions_before_after`, non mesurable par l'executor courant) ;
- rejoue `context_replay=["ACTION6"]` ;
- teste `ACTION3`/`ACTION4` contre des controles dynamiques mesures
  (`hold_or_stop_state`, `ACTION6`, `contextual_post_stop_conversion_policy`,
  `relation_progress_policy`) ;
- mesure `objective_completion_signal` et `terminal_reentry_rate` en reutilisant
  le `source_measurement_context` de M3.G6 ;
- ne refait aucun pas d'environnement, aucun rollout de policy ;
- garde `support=0`, `revision_status=CANDIDATE_ONLY`,
  `truth_status=NOT_EVALUATED_BY_M3` ;
- n'ecrit pas A32/A33 ; aucune requete LLM comptee comme evidence.

Resultats possibles : `LLM_ADDS_NEW_TESTABLE_SIGNAL`,
`LLM_REDUCES_TO_EXISTING_G6_FAILURE`, `LLM_REQUESTS_TOO_WEAK_NEED_M2_14`,
`SEMANTIC_COMPILATION_GAP`.

Run bp35 :

- ready_requests_consumed = 5
- blocked_requests_ignored = 1 (unlock `available_actions_before_after`)
- cells_executed = 5 ; cells_unbound = 0
- metrics_executed = `objective_completion_signal`, `terminal_reentry_rate`
- objective_completion_signal = false ; objective_completion_candidate_cells = 0
- candidate_supported_cells = 0 ; cells_reduce_to_g6_failure = 5
- terminal_reentry_rate_max_executed_candidate = 0.285714
- g6_outcome_status = `PROXY_COMPLETION_DIVERGENCE_CANDIDATE_ONLY`
- reproduces_g6_proxy_completion_divergence = true
- adds_signal_beyond_g6 = false
- recommends_m2_14_world_model = true
- support = 0 ; a32_write_performed = false ; a33_write_performed = false
- outcome = `LLM_REDUCES_TO_EXISTING_G6_FAILURE_CANDIDATE_ONLY`

Run variants:
- mock_baseline: 5 READY_FOR_M3, 5 reduce to G6 failure.
- real_qwen_0_5b_smoke: 1 READY_FOR_M3, 1 reduce to G6 failure.
- real_qwen_3b_smoke: 1 READY_FOR_M3, 1 reduce to G6 failure.

M2.13d — Real LLM provenance/context anchoring guard:
- Override LLM fields (source_request_id, frontier_context_id, frontier_step) from SituationPacket
- Fill required_context_replay from packet if LLM provides empty list
- Add warnings: llm_provenance_fields_overridden, llm_context_replay_missing_filled_from_packet
- Add summary.llm_context_anchoring_failures counter
- Fix attention_mask warning when pad_token == eos_token
- Reduce max_new_tokens default from 10000 to 2048 (JSON output doesn't need more)

Lecture :

Les 5 requetes LLM se lient toutes au substrat M3.G6 (`ACTION6,ACTION3` et
`ACTION6,ACTION4`) et reproduisent exactement l'echec proxy/completion : aucune
completion d'objectif, aucun niveau complete, et les candidats
`terminal_reentry_rate` re-entrent en terminal (0.2857) plus que les controles
dynamiques (0.0). Le LLM ne fabrique pas de faux signal et ne refute rien : il
retombe sur la divergence proxy/completion deja etablie par M3.G6. Comme les
requetes generiques (action cible vs controles, metriques de surface) n'ont pas
de feature representationnelle discriminante, atteindre un signal de completion
genuinement nouveau exigerait un vrai world model. C'est exactement le signal
`recommends_m2_14_world_model=true`.

M2.13a réel fonctionne techniquement.
Qwen 0.5B ne suffit pas à produire une hypothèse plus discriminante que M3.G6.
Le verrou n’est plus l’intégration LLM ; c’est l’absence de représentation causale/world-model.

Suite naturelle :

- `M2.14` - Real world model activation, pour fournir des features
  representationnelles (au-dela du proxy) que M3 pourrait discriminer.
- Sinon ouvrir une frontier sur les observables/actions de commit concrets
  manquants (`ACTION1/ACTION2/ACTION5` indisponibles).

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.state_conditioned_llm_hypothesis_execution --requests diagnostics\m2\state_conditioned_llm_m3_candidate_requests.json --m3g6-results diagnostics\m3\risk_aware_objective_completion_experiment_results.json --out diagnostics\m3\state_conditioned_llm_hypothesis_execution_results.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_m3_state_conditioned_llm_hypothesis_execution.py -q
```
