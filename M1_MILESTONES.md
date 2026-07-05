# M1 milestones - journal de mise en place

Derniere mise a jour : 2026-06-19

Ce fichier sert de relais de communication avec l'agent ChatGPT de travail. Il
doit etre mis a jour a chaque milestone M1/M2 implementee.

## Principe fixe

M1/M2 ne confirment jamais une hypothese. Ils produisent uniquement de la
matiere candidate `UNRESOLVED`. Toute confirmation ou refutation reste reservee
au moteur scientifique A15-A31, apres experience reelle. Les champs
`trace_support_counted_as_proof=false` et `prior_counted_as_proof=false` doivent
rester explicites dans les artefacts M1.

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| M1.1 - Dataset d'observations brutes | Fait | `theory/m1/observation_dataset.py`, `theory/m1/__init__.py`, `tests/test_m1_observation_dataset.py`, `training/m1_observation_dataset.jsonl` | `3 passed`; artefact complet: 5711 observations, 18 traces, 6 jeux |
| M1.2 - Invariant mining multi-jeux | Fait | `theory/m1/invariant_miner.py`, `tests/test_m1_invariant_miner.py`, `diagnostics/m1/accepted_invariants.json`, `diagnostics/m1/rejected_invariants.json` | `9 passed` sur les tests M1; run complet: 23 acceptes, 12 rejetes |
| M1.3 - Injection A25 | Fait, opt-in | `theory/m1/predicate_generation.py`, patch DI minimal `theory/cross_game_correspondence_discovery.py` + `theory/non_ar25_multi_relation_agenda.py`, `tests/test_m1_predicate_generation.py`, `diagnostics/m1/predicate_coverage_pretest.json` | M1 desactive: sortie historique identique; pre-test bloquees: unique 4.67 -> 11.67, relation candidates 341.33 -> 701.0, paires 77 -> 77 |
| M1.3b - Source-target anchor expansion | Fait, opt-in M1 | `theory/m1/anchor_expansion.py`, propagation optionnelle `anchor_expander`, `tests/test_m1_anchor_expansion.py`, `diagnostics/m1/anchor_expansion_pretest.json` | Tests M1: 22 passes; pre-test bloquees: paires 77 -> 79.67, unique 4.67 -> 14.67, relation candidates 341.33 -> 776.67 |
| M1.3c - Live-grid compatible anchor ranking | Fait, opt-in M1 | `theory/m1/live_anchor_ranking.py`, `theory/m1/grounding_autopsy.py`, propagation optionnelle `candidate_ranker`, `tests/test_m1_live_anchor_ranking.py`, `tests/test_m1_grounding_autopsy.py`, `diagnostics/m1/live_anchor_compatibility_pretest.json`, `diagnostics/m1/grounding_autopsy.json`, `diagnostics/m1/grounding_autopsy.md` | Tests M1: 29 passes; autopsie: bp35 0 source actionnable, cd82 0 source actionnable, dc22 3 sources actionnables dont 1 entrant agenda |
| M1.3c+ - Grounding Metrics | Fait, diagnostic pur | `GroundingFunnel` dans `theory/m1/grounding_autopsy.py`, `diagnostics/m1/grounding_autopsy.json`, `diagnostics/m1/grounding_autopsy.md` | `blocked_by_unselectable_source` expose: bp35 20/20 bloquees, cd82 20/20 bloquees, dc22 15/20 bloquees mais 5 sources actionnables |
| M1.3d-a - Source reachability | Fait, diagnostic pur | `theory/m1/source_reachability.py`, `tests/test_m1_source_reachability.py`, `diagnostics/m1/source_reachability_problems.json` | Extraction typee des `SourceAlignmentProblem`: global bp35/cd82/dc22 = 20/20/15, nouvelles paires M1 = 5/4/3 |
| M1.3d-b - Actionable Source Alignment | Fait, pretest offline sans A25 | `theory/m1/actionable_source_alignment.py`, `tests/test_m1_actionable_source_alignment.py`, `diagnostics/m1/actionable_source_alignment_pretest.json`, `diagnostics/m1/actionable_source_alignment_pretest_wide.json` | Pipeline `SourceAlignmentProblem -> ExperimentalPrecondition` teste; run reel nouvelles paires M1: 12 problemes, 0 precondition courte trouvee, `CONFIRMED` absent |
| M1.3d-c - Game mechanic typing | Fait, diagnostic pur | `theory/m1/mechanic_typing.py`, `tests/test_m1_mechanic_typing.py`, `diagnostics/m1/mechanic_typing.json` | bp35/cd82: `color_source_schema_misaligned_with_observed_mechanics`; dc22: source-color partiellement actionnable |
| M1.3e - Mechanic-grounded experiment candidates | Fait, generation UNRESOLVED hors A25 | `theory/m1/mechanic_grounded_candidates.py`, `tests/test_m1_mechanic_grounded_candidates.py`, `diagnostics/m1/mechanic_grounded_candidates.json` | 64 candidats non-couleur: object_motion/contact/lifecycle/shape_zone/position_effect; bp35/cd82 sortent du format `source_color -> target_color` |
| M1.3f - Polymorphic A25 interface pretest | Fait, pretest d'affordance hors A25 | `theory/m1/polymorphic_a25_pretest.py`, `tests/test_m1_polymorphic_a25_pretest.py`, `diagnostics/m1/polymorphic_a25_pretest.json` | 44/64 candidats mecaniques testables; bp35 9/12, cd82 5/12; blocages types `action_not_available=13`, `missing_position_argument=7`; wrong_confirmations=0 |
| M1.3g - Minimal Polymorphic A25 Adapter | Fait, execution minimale opt-in hors revision | `theory/m1/polymorphic_a25_adapter.py`, `tests/test_m1_polymorphic_a25_adapter.py`, `diagnostics/m1/polymorphic_a25_adapter_bp35.json` | bp35 cible: 3/3 experiences mecaniques generees, env_actions=3, observable_deltas=3, wrong_confirmations=0 |
| M1.3h - Experiment Value Estimates | Fait, selection informative hors execution | `theory/m1/experiment_value_estimator.py`, `tests/test_m1_experiment_value_estimator.py`, `diagnostics/m1/experiment_value_estimates.json` | 44 candidats testables scores; 5 recommandations diversifiees, type_diversity=1.0, mean_information_score=0.5688, wrong_confirmations=0 |
| M1.3i - Recommended experiment to polymorphic A25 choice | Fait, pont opt-in hors revision | `theory/m1/recommended_experiment_choice.py`, `tests/test_m1_recommended_experiment_choice.py`, `diagnostics/m1/recommended_polymorphic_a25_choice.json` | Top recommandation bp35 ACTION6 position_effect convertie en choix A25 polymorphe; env_actions=1, observable_delta=true, wrong_confirmations=0 |
| M1.3j - Mechanic observation to revision candidate | Fait, proposition de revision sans verdict | `theory/m1/mechanic_revision_candidate.py`, `tests/test_m1_mechanic_revision_candidate.py`, `diagnostics/m1/mechanic_revision_candidates.json` | Observation M1.3i -> prediction `local_patch_changed` + proposition A15-A31 `UNRESOLVED`; controlled_tests_required=1, revision_performed=false, wrong_confirmations=0 |
| M1.3k - Controlled scientific integration pretest | Fait, entree ledger A15-A31 sans verdict | `theory/m1/scientific_integration_pretest.py`, `tests/test_m1_scientific_integration_pretest.py`, `diagnostics/m1/scientific_integration_pretest.json` | Proposition M1.3j admise dans le ledger epistemique: unresolved_records=1, support_total=0, experiments_spent_total=0, wrong_confirmations=0 |
| M1.3l - Controlled follow-up experiment | Fait, test controle opt-in sans verdict | `theory/m1/controlled_followup_experiment.py`, `tests/test_m1_controlled_followup_experiment.py`, `diagnostics/m1/controlled_experiment_results.json` | Ledger M1.3k -> test `ACTION6` vs controle `ACTION3`: controlled_experiments_run=1, support_events=1, contradiction_events=0, status UNRESOLVED, support=0, wrong_confirmations=0 |
| M1.4 - Stress-test A31bis | En place, run borne valide | `theory/m1/stress_test_a31bis.py`, `tests/test_m1_stress_test_a31bis.py`, `diagnostics/m1_stress_test_a31bis_bounded.json`, `diagnostics/m1_stress_test_a31bis_smoke.json`, `diagnostics/m1_stress_test_a31bis_anchor_bounded.json`, `diagnostics/m1_stress_test_a31bis_live_ranked_bounded.json` | M1.3c borne local off/on: blocage 8 -> 8, paires 65.5 -> 69.5, live-compatible nouvelles paires 0.0 sur les 2 traces bornees, wrong_confirmations=0 |
| M1.G0.1/G0.5 - General mechanic abstraction | Fait, candidate-only | `theory/m1/general_mechanic_abstraction.py`, `tests/test_m1_general_mechanic_abstraction.py`, `diagnostics/m1/general_mechanic_candidates.json` | Entity tracker + role/action-effect/relation/invariant hypotheses generiques; bp35 action sweep: 230 entites, 1 `controllable_actor`, 1 `timer_or_hud`, 512 hypotheses `relation_change`, 114 invariants dynamiques, support=0, wrong_confirmations=0 |
| M2/M1.5 - World model objet-centric | Differe | `theory/m1/object_world_model.py` | Demarrer seulement si M1.4 justifie l'accelerateur |
| M2/M1.6 - Proposeur LLM Qwen2.5-3B | Differe | `theory/m1/predicate_proposer_llm.py` | Fallback deterministe obligatoire |

## M1.1 - Implementation precise

- Ajout d'un sous-paquet `theory/m1/`.
- Ajout de `RawTransitionObservation`, une dataclass par transition brute de
  trace human_traces.
- Lecture JSONL de `*.steps.jsonl`, avec une observation par transition ayant
  `frame_before` et `frame_after`.
- Conservation de l'identite : `game_id`, `episode_id`, `step`, `action`,
  `action_args`, `available_actions`.
- Flags bruts de changement : `shape_changed`, `color_changed`,
  `position_changed`, `adjacency_changed`, `object_count_changed`,
  `num_cells_changed`, `player_moved`, `level_progressed`, `game_over`.
- Mesures brutes : formes de grille, couleur de fond inferee, comptes par
  couleur, composants flood-fill, tailles, bbox, centroides, signatures de forme
  stables, contacts de couleurs, paires de couleurs modifiees, vecteurs de
  mouvement, objets crees/supprimes.
- Contexte humain conserve uniquement comme supervision faible :
  `intent_text`, `hypothesis_text`.
- Anti-circularite : aucun champ du JSONL ne porte les noms des 5 predicats
  A12 historiques (`source_target_color_transform`, `paired_with`,
  `same_shape`, `aligned_with`, `adjacent_to`).
- CLI : `python -m theory.m1.observation_dataset` ecrit par defaut
  `training/m1_observation_dataset.jsonl` a partir de `human_traces/*.steps.jsonl`.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests/test_m1_observation_dataset.py -q` -> 3 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.observation_dataset` -> 5711 observations, 18 traces, 6 jeux (`ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`).
  - `rg "source_target_color_transform|paired_with|same_shape|aligned_with|adjacent_to" training\\m1_observation_dataset.jsonl` -> aucun match.

## M1.2 - Implementation precise

- Ajout de `LatentInvariant` et `InvariantMiningResult`.
- Le mineur lit le JSONL M1.1 et ne consomme que des champs bruts :
  flags de changement, comptes, listes before/after, paires de couleur changees,
  contacts couleur, mouvements d'objets, creations/suppressions, progression de
  niveau et etat terminal.
- Premiere couche : calcul de taux par cellule
  `(game_id, action, raw_attribute)` via `summarize_raw_outcome_rates`.
- Deuxieme couche : generation d'invariants par signature
  `(attribute, outcome)`, avec les actions conservees dans `contexts`.
- Filtre dur multi-jeux :
  - `MIN_GAMES = 2`.
  - `MIN_INTRA_GAME_SUPPORT = 0.6`.
  - un invariant mono-jeu, meme tres soutenu, est conserve dans les rejetes avec
    `rejected_reason="single_game"`.
- Score transfert :
  - `cross_game_score = min(per_game_support) * log(1 + n_games_supporting)`.
  - test de monotonie en nombre de jeux ajoute.
- Score nouveaute :
  - `novelty_score = 1 - max_similarity_to_existing_predicates`.
  - comparaison aux 5 predicats socles + invariants deja acceptes.
  - les quasi-doublons sont rejetes en `low_novelty`.
- Persistence :
  - acceptes -> `diagnostics/m1/accepted_invariants.json`.
  - rejetes -> `diagnostics/m1/rejected_invariants.json`.
  - raisons de rejet observees : `low_novelty`, `low_support`, `single_game`.
- Discipline epistemique :
  - chaque invariant garde `trace_support_counted_as_proof=false`.
  - chaque invariant garde `prior_counted_as_proof=false`.
  - aucun statut `CONFIRMED` n'est cree par M1.2.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_observation_dataset.py tests\\test_m1_invariant_miner.py -q` -> 9 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.invariant_miner` -> 23 invariants acceptes, 12 rejetes.
  - Rejets par raison : `low_novelty=4`, `low_support=2`, `single_game=6`.
- Lecture manuelle des 20 premiers acceptes :
  - Invariants tres generaux/stables : `grid_extent_preserved`,
    `terminal_state_absent`, `level_progress_absent`.
  - Cas proches du vocabulaire relationnel connu : `color_pair_change_appears`,
    `adjacency_modified`, `color_contact_modified`, `adjacency_preserved`,
    `color_contact_preserved`.
  - Dimensions nouvelles utiles pour M1.3/M1.4 : `object_creation_appears`,
    `object_removal_appears`, `object_motion_appears`, `object_motion_absent`,
    `player_position_modified`, `position_modified`, `object_count_preserved`.
  - Diagnostic : M1.2 ne fait pas que renommer les 5 predicats historiques, mais
    les premiers invariants restent larges. L'ancrage A25 de M1.3 devra verifier
    lesquels augmentent vraiment les couples relationnels instanciables.

## M1.3 - Implementation precise

- Ajout de `theory/m1/predicate_generation.py`.
- Patch DI minimal de `theory/cross_game_correspondence_discovery.py` :
  - nouveau parametre optionnel `predicate_generator`.
  - defaut `None` -> comportement historique strict via `_source_target_predicates`.
  - CLI opt-in `--m1-predicates`.
- Patch DI minimal de `theory/non_ar25_multi_relation_agenda.py` :
  - nouveau parametre optionnel `predicate_generator`.
  - defaut `None` -> A25 historique inchange.
- Mapping initial volontairement parcimonieux :
  - injectes si acceptes M1.2 et supportes par >= 2 jeux :
    `m1_object_creation_appears`,
    `m1_object_removal_appears`,
    `m1_object_motion_appears`,
    `m1_object_count_preserved`,
    `m1_color_pair_change_appears`,
    `m1_adjacency_preserved`,
    `m1_adjacency_modified`.
  - gardes hors injection initiale :
    `grid_extent_preserved`, `terminal_state_absent`,
    `level_progress_absent`.
- Les 5 predicats historiques restent toujours presents quand M1 est active :
  M1 ajoute des predicats, il ne remplace pas le socle.
- Les nouveaux predicats entrent uniquement dans
  `DiscoveredCorrespondenceCandidate.predicates`; aucun statut `CONFIRMED` n'est
  cree et aucun support trace n'est compte comme preuve.
- Pre-test offline ajoute :
  - `run_predicate_coverage_pretest`.
  - sortie `diagnostics/m1/predicate_coverage_pretest.json`.
  - aucune execution d'environnement, aucune revision scientifique.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_observation_dataset.py tests\\test_m1_invariant_miner.py tests\\test_m1_predicate_generation.py tests\\test_cross_game_correspondence_discovery.py tests\\test_non_ar25_multi_relation_agenda.py -q` -> 15 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.predicate_generation` -> pre-test sur traces bloquees `bp35`, `cd82`, `dc22`.
- Resultat pre-test M1.3 (`top_k=100`, `min_pixel_support=1`) :
  - `unique_predicates_per_trace` moyen : 4.6667 -> 11.6667.
  - `relation_candidates_generated` moyen : 341.3333 -> 701.0.
  - `candidate_pairs_per_trace` moyen : 77.0 -> 77.0.
  - Lecture : le vocabulaire et les predicats attaches aux paires ancrees
    augmentent nettement, mais M1.3 ne cree pas encore de nouvelles paires
    source/target. Si A31bis reste bloque, le goulot sera probablement dans la
    decouverte/ancrage des paires, pas seulement dans le vocabulaire.

## M1.3b - Implementation precise

- Ajout de `theory/m1/anchor_expansion.py`.
- Ajout de `ExpandedAnchor` et `build_m1_anchor_expander`.
- Objectif limite :
  - augmenter les couples source/target proposes a A12/A25.
  - ne pas toucher au moteur scientifique A15-A31.
  - ne jamais creer de statut `CONFIRMED`.
- Trois heuristiques object-centric deterministes :
  - `created_removed` : apparie des composants supprimes/crees par taille,
    signature de forme et proximite de bbox/centroide.
  - `motion` : apparie un composant avant/apres de meme couleur/forme, puis
    propose comme cible les couleurs dont le contact local change autour du
    mouvement.
  - `contact` : propose des paires couleur-couleur quand un contact/adjacency
    apparait ou disparait.
- Les ancres ajoutent des predicats de provenance M1 :
  `m1_anchor_created_removed`, `m1_anchor_motion`, `m1_anchor_contact`.
- Les ancres ajoutent aussi les predicats relationnels historiques observables
  quand ils tiennent (`paired_with`, `same_shape`, `aligned_with`,
  `adjacent_to`), car A25 consomme encore ces predicats pour construire son
  agenda relationnel.
- Important : pour les paires issues uniquement d'une ancre M1.3b, on n'ajoute
  pas automatiquement `source_target_color_transform`. Le support vient de
  l'ancrage objet/contact, pas d'une transformation couleur pixel-a-pixel.
- Patch DI minimal :
  - `theory/cross_game_correspondence_discovery.py` accepte maintenant
    `anchor_expander=None`.
  - `None` conserve le comportement historique.
  - `--m1-anchor-expansion` active l'expander en CLI.
  - les transitions sans `changed_color_pairs` peuvent maintenant compter si
    une ancre M1.3b existe.
- Propagation opt-in de `anchor_expander` dans :
  - `theory/non_ar25_multi_relation_agenda.py`.
  - `theory/multi_game_evaluation.py`.
  - `theory/multi_game_stress_test.py`.
  - `theory/m1/stress_test_a31bis.py`.
- Le wrapper A31bis active M1.3b par defaut avec M1, et fournit
  `--disable-m1-anchor-expansion` pour reproduire le run M1.3/M1.4 sans ancres.
- Pre-test offline M1.3b :
  - commande :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.predicate_generation --m1-anchor-expansion --out diagnostics\\m1\\anchor_expansion_pretest.json`.
  - moyenne traces bloquees `bp35`, `cd82`, `dc22` :
    - `candidate_pairs_per_trace` : 77.0 -> 79.6667.
    - `unique_predicates_per_trace` : 4.6667 -> 14.6667.
    - `relation_candidates_generated` : 341.3333 -> 776.6667.
  - detail :
    - `bp35` : paires 31 -> 39, candidates 122 -> 296, unique 4 -> 14.
    - `cd82` : paires 100 -> 100, candidates 449 -> 1002, unique 5 -> 15.
    - `dc22` : paires 100 -> 100, candidates 453 -> 1032, unique 5 -> 15.
  - lecture : M1.3b cree bien de nouvelles paires sur `bp35`; `cd82` et
    `dc22` sont au plafond `top_k=100` du pre-test, donc le KPI moyen sous-estime
    probablement l'expansion brute.
- A31bis borne avec M1.3b :
  - commande :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.stress_test_a31bis --budgets 5,10,25,50 --latest-per-game --exclude-ar25 --max-traces 2 --max-attempts-per-budget 1 --run-local-baseline --out diagnostics\\m1_stress_test_a31bis_anchor_bounded.json`.
  - resultat local M1 off/on :
    - `not_enough_relation_candidates` : 8 -> 8.
    - `wrong_confirmations` : 0.
    - `unique_predicates_per_trace` : 4.5 -> 14.5.
    - `relation_candidates_generated` : 285.5 -> 649.0.
    - `candidate_pairs_per_trace` : 65.5 -> 69.5.
    - `experiments_run` : 0 -> 0.
    - diagnostic : `anchor_expansion_pairs_up_blocker_not_reduced`.
- Interpretation scientifique :
  - M1.3b valide que l'ancrage source/target peut etre augmente sans casser la
    discipline epistemique.
  - Le run borne refute cependant l'hypothese plus forte
    "quelques paires supplementaires suffisent a debloquer A25".
  - Le prochain goulot semble etre la compatibilite des nouvelles paires avec
    la grille live et les preconditions relationnelles de l'agenda, pas M2.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_anchor_expansion.py tests\\test_m1_observation_dataset.py tests\\test_m1_invariant_miner.py tests\\test_m1_predicate_generation.py tests\\test_m1_stress_test_a31bis.py -q` -> 22 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_anchor_expansion.py tests\\test_m1_predicate_generation.py tests\\test_m1_stress_test_a31bis.py tests\\test_multi_game_evaluation.py tests\\test_multi_game_stress_test.py tests\\test_cross_game_correspondence_discovery.py tests\\test_non_ar25_multi_relation_agenda.py -q` -> 22 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 265 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\anchor_expansion_pretest.json diagnostics\\m1_stress_test_a31bis_anchor_bounded.json` -> aucun match.

## M1.3c - Implementation precise

- Ajout de `theory/m1/live_anchor_ranking.py`.
- Ajout de `LiveAnchorCompatibilityMetrics`,
  `LiveAnchorCompatibilityPretest` et `build_m1_live_candidate_ranker`.
- Objectif limite :
  - augmenter les paires reellement consommables par A25.
  - ne pas seulement augmenter les paires candidates brutes.
  - ne pas toucher au moteur scientifique A15-A31.
  - ne jamais creer de statut `CONFIRMED`.
- KPI ajoutes avant toute nouvelle heuristique :
  - `new_pairs_total`.
  - `new_pairs_live_color_compatible`.
  - `new_pairs_target_present` (diagnostic ajoute pour verifier que la cible est
    ancrable dans la grille live).
  - `new_pairs_with_2_preferred_predicates`.
  - `new_pairs_entering_agenda`.
- Predicats preferes M1.3c :
  `same_shape`, `aligned_with`, `adjacent_to`, `paired_with`.
- Patch DI minimal :
  - `theory/non_ar25_multi_relation_agenda.py` accepte maintenant
    `candidate_ranker=None`.
  - `None` conserve le comportement historique.
  - si `candidate_ranker` est fourni, A25 decouvre plus large
    (`max_candidates * pre_rank_candidate_multiplier`, defaut 5), puis ranke et
    reduit a `max_candidates` apres lecture de la grille live.
  - ajout non cassant de `raw_discovered_candidates` dans la sortie agenda.
- Propagation opt-in de `candidate_ranker` et `preferred_predicates` dans :
  - `theory/multi_game_evaluation.py`.
  - `theory/multi_game_stress_test.py`.
  - `theory/m1/stress_test_a31bis.py`.
- Le ranker M1.3c :
  - priorise les candidats dont la couleur source est selectionnable par une
    action live compatible.
  - exige une cible presente/ancrable dans la grille live.
  - augmente les predicats preferes si la relation est observable dans la grille
    live actuelle, sans compter cette observation comme preuve mecanique.
  - trie ensuite par nombre de predicats preferes, support transitionnel et
    support brut.
- Pre-test offline/live-grid M1.3c :
  - commande :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.live_anchor_ranking --out diagnostics\\m1\\live_anchor_compatibility_pretest.json`.
  - moyenne traces bloquees `bp35`, `cd82`, `dc22`, avant ranking :
    - `new_pairs_total` : 16.0.
    - `new_pairs_live_color_compatible` : 1.0.
    - `new_pairs_blocked_by_unselectable_source` : 15.0.
    - `new_pairs_target_present` : 10.6667.
    - `new_pairs_with_2_preferred_predicates` : 15.6667.
    - `new_pairs_entering_agenda` : 0.3333.
  - moyenne apres ranking top 20 :
    - `new_pairs_total` : 5.0.
    - `new_pairs_live_color_compatible` : 1.0.
    - `new_pairs_blocked_by_unselectable_source` : 4.0.
    - `new_pairs_target_present` : 4.3333.
    - `new_pairs_with_2_preferred_predicates` : 5.0.
    - `new_pairs_entering_agenda` : 0.3333.
  - detail apres ranking :
    - `bp35` : 5 nouvelles paires, 0 live-compatible, 0 entrant agenda.
    - `cd82` : 4 nouvelles paires, 0 live-compatible, 0 entrant agenda.
    - `dc22` : 6 nouvelles paires, 3 live-compatibles, 1 entrant agenda.
- A31bis borne avec M1.3c :
  - commande :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.stress_test_a31bis --budgets 5,10,25,50 --latest-per-game --exclude-ar25 --max-traces 2 --max-attempts-per-budget 1 --run-local-baseline --out diagnostics\\m1_stress_test_a31bis_live_ranked_bounded.json`.
  - resultat local M1 off/on :
    - `not_enough_relation_candidates` : 8 -> 8.
    - `wrong_confirmations` : 0.
    - `unique_predicates_per_trace` : 4.5 -> 14.5.
    - `relation_candidates_generated` : 285.5 -> 649.0.
    - `candidate_pairs_per_trace` : 65.5 -> 69.5.
    - `live_anchor_compatibility.new_pairs_entering_agenda` : 0.0.
    - diagnostic : `anchor_expansion_pairs_up_blocker_not_reduced`.
  - lecture : ce run borne ne couvre que `bp35` et `cd82`, les deux jeux ou le
    pre-test M1.3c trouve 0 nouvelle paire live-compatible.
- Checks directs A25 (hors surcouche A31bis, une seule execution agenda) :
  - `bp35` avec M1.3c :
    - `error=not_enough_relation_candidates_for_agenda`.
    - `raw_discovered_candidates=39`, `discovered_candidates=20`.
    - `candidate_prediction_count=0`, `relation_candidate_count=0`.
    - `env_actions=0`, `wrong_confirmations=0`.
  - `cd82` avec M1.3c :
    - `error=not_enough_relation_candidates_for_agenda`.
    - `raw_discovered_candidates=100`, `discovered_candidates=20`.
    - `candidate_prediction_count=0`, `relation_candidate_count=0`.
    - `env_actions=0`, `wrong_confirmations=0`.
  - `dc22` baseline historique direct :
    - `error=""`, `candidate_prediction_count=16`.
    - agenda `(8,0)` avec `same_shape`, `aligned_with`, `adjacent_to`.
    - `env_actions=1`, `wrong_confirmations=0`.
  - `dc22` avec M1.3c direct :
    - `error=""`, `raw_discovered_candidates=100`, `discovered_candidates=20`.
    - `candidate_prediction_count=38`, `relation_candidate_count=4`.
    - agenda `(8,12)` avec `same_shape`, `aligned_with`, `adjacent_to`,
      `paired_with`.
    - `env_actions=1`, `wrong_confirmations=0`.
- Autopsie comparative approfondie dc22 vs bp35/cd82 :
  - Ajout de `theory/m1/grounding_autopsy.py`.
  - Sorties :
    - `diagnostics/m1/grounding_autopsy.json`.
    - `diagnostics/m1/grounding_autopsy.md`.
  - commande :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.grounding_autopsy --json-out diagnostics\\m1\\grounding_autopsy.json --md-out diagnostics\\m1\\grounding_autopsy.md`.
  - tableau comparatif M1.3c :
    - `bp35` : nouvelles paires 5, live-compatible 0, entrant agenda 0,
      `relation_candidate_count=0`, `env_actions=0`,
      `error=not_enough_relation_candidates_for_agenda`.
    - `cd82` : nouvelles paires 4, live-compatible 0, entrant agenda 0,
      `relation_candidate_count=0`, `env_actions=0`,
      `error=not_enough_relation_candidates_for_agenda`.
    - `dc22` : nouvelles paires 6, live-compatible 3, entrant agenda 1,
      `relation_candidate_count=4`, `env_actions=1`, `error=""`.
  - raisons de blocage :
    - `bp35` : toutes les nouvelles paires sont bloquees par
      `source_not_selectable_for_action`; source live de `ACTION6` = `[5]`.
    - `cd82` : toutes les nouvelles paires sont bloquees par
      `source_not_selectable_for_action`; aucune couleur source selectionnable
      n'est exposee par les actions live au reset.
    - `dc22` : source live de `ACTION6` = `[8, 9]`; c'est le premier cas ou les
      ancres M1 atteignent une source actionnable.
  - les 3 nouvelles paires live-compatibles dc22 :
    - `ACTION6`, `8 -> 2`, support 291, transition_support 10,
      predicats preferes `aligned_with`, `adjacent_to`, `paired_with`,
      memes predicats vrais dans la grille live, `entering_agenda=true`.
    - `ACTION6`, `9 -> 12`, support 100, transition_support 36,
      predicats preferes `same_shape`, `aligned_with`, `adjacent_to`,
      `paired_with`, mais cible absente de la grille live,
      `entering_agenda=false`.
    - `ACTION6`, `8 -> 15`, support 713, transition_support 28,
      predicats preferes `same_shape`, `aligned_with`, `adjacent_to`,
      `paired_with`, mais cible absente de la grille live,
      `entering_agenda=false`.
  - conclusion de l'autopsie :
    - le verrou n'est plus seulement `Hypothesis generation`.
    - le verrou est `Hypothesis grounding` :
      `trace object -> anchor -> live object -> actionable source`.
    - `dc22` est le controle positif M1 -> A25 : une nouvelle paire ancree par
      M1 entre dans l'agenda et produit une experience sans fausse confirmation.
- M1.3c+ Grounding Metrics :
  - ajout de `GroundingFunnel` dans `theory/m1/grounding_autopsy.py`.
  - objectif : suivre le deplacement du blocage quand une etape du grounding est
    resolue.
  - funnel global = toutes les paires classees que A25 voit apres ranking.
  - funnel nouvelles paires = uniquement les paires ajoutees par M1.
  - champs :
    - `pairs_discovered`.
    - `pairs_target_present`.
    - `pairs_actionable_source`.
    - `pairs_blocked_by_unselectable_source`.
    - `pairs_live_compatible` (= source actionnable ET cible presente).
    - `pairs_with_2_preferred_predicates`.
    - `pairs_entering_agenda`.
    - `pairs_generating_env_action`.
  - funnel global apres ranking :
    - `bp35` : 20 -> 20 cibles presentes -> 0 sources actionnables /
      20 bloquees par source non selectionnable -> 0
      live-compatibles -> 0 entrant agenda -> 0 generant action.
    - `cd82` : 20 -> 20 cibles presentes -> 0 sources actionnables /
      20 bloquees par source non selectionnable -> 0
      live-compatibles -> 0 entrant agenda -> 0 generant action.
    - `dc22` : 20 -> 17 cibles presentes -> 5 sources actionnables /
      15 bloquees par source non selectionnable -> 2
      live-compatibles -> 2 entrant agenda -> 1 generant action.
  - funnel des nouvelles paires M1 apres ranking :
    - `bp35` : 5 -> 5 cibles presentes -> 0 sources actionnables /
      5 bloquees par source non selectionnable -> 0
      live-compatibles -> 0 entrant agenda -> 0 generant action.
    - `cd82` : 4 -> 4 cibles presentes -> 0 sources actionnables /
      4 bloquees par source non selectionnable -> 0
      live-compatibles -> 0 entrant agenda -> 0 generant action.
    - `dc22` : 6 -> 4 cibles presentes -> 3 sources actionnables /
      3 bloquees par source non selectionnable -> 1
      live-compatible -> 1 entrant agenda -> 0 generant action.
  - nuance importante :
    - la nouvelle paire M1 `ACTION6 8 -> 2` est eligible/entrant agenda.
    - l'action environnement directe observee dans le run A25 est portee par le
      funnel global (`pairs_generating_env_action=1`), pas encore par une
      nouvelle paire M1 (`new_pair pairs_generating_env_action=0`).
    - cela confirme que le blocage migre de la generation vers la transformation
      d'une idee testable en experience realisee.
- Runs cibles `dc22` via `stress_test_a31bis` :
  - `--budgets 5,10,25,50` sur une trace `dc22` -> timeout a 10 minutes, aucun
    JSON final.
  - `--budgets 5` sur une trace `dc22` -> timeout a 5 minutes, aucun JSON final.
  - Le check direct A25 ci-dessus remplace donc le stress wrapper pour le signal
    court terme sur `dc22`.
- Interpretation scientifique :
  - M1.3c confirme que la majorite des nouvelles paires ne sont pas
    selectionnables dans la grille live de depart.
  - `bp35` et `cd82` ne manquent plus de vocabulaire ni de paires brutes : ils
    manquent de paires dont la source est actionnable au reset.
  - `dc22` montre que le ranking live peut produire un agenda consommable et
    augmenter les predictions relationnelles sans fausse confirmation.
  - L'autopsie positive de `dc22` doit etre conservee comme controle avant toute
    nouvelle extension : elle explique pourquoi la chaine marche au moins une
    fois.
  - Le prochain goulot est donc l'alignement entre ancres de trace et source
    selectionnable/reachable dans la grille live, pas M2.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest (Get-ChildItem -Path tests -Filter test_m1_*.py).FullName -q` -> 29 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 272 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\live_anchor_compatibility_pretest.json diagnostics\\m1_stress_test_a31bis_live_ranked_bounded.json` -> aucun match.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\grounding_autopsy.json diagnostics\\m1\\grounding_autopsy.md diagnostics\\m1_stress_test_a31bis_smoke.json` -> aucun match.

## M1.3d - Cadrage Actionable Source Alignment

- Statut : M1.3d-a `source_reachability.py` implemente ; M1.3d-b
  `actionable_source_alignment.py` implemente en pretest offline, sans
  integration A25.
- Position dans l'architecture :
  - ce n'est pas M2.
  - ce n'est pas une nouvelle generation de paires.
  - c'est une premiere couche de planification experimentale locale entre
    `Correspondance` et `Experience`.
- Diagnostic prerequis maintenant observe :
  - `bp35` : 20 paires classees, 20 bloquees par source non selectionnable.
  - `cd82` : 20 paires classees, 20 bloquees par source non selectionnable.
  - `dc22` : 20 paires classees, 15 bloquees, mais 5 sources actionnables et
    2 paires entrant agenda dans le funnel global.
  - nouvelles paires M1 : `bp35` 5/5 bloquees, `cd82` 4/4 bloquees, `dc22`
    3/6 bloquees mais 1 paire entrant agenda.
- Objectif M1.3d :
  - partir d'une paire M1 interessante mais non actionnable au reset.
  - identifier que son blocage est `source_not_selectable_for_action`.
  - chercher une courte preparation qui rend la source selectionnable.
  - relancer A25 sur la paire, toujours sans confirmation automatique.
- Module 1 prevu : `theory/m1/source_reachability.py`.
  - statut : fait.
  - structure cible `SourceAlignmentProblem` :
    `desired_source_color`, `target_color`, `action`, `available_live_sources`,
    `candidate_pair`, `block_reason`.
  - role : formaliser les paires bloquees et leurs sources live disponibles,
    pas les resoudre.
- Module 2 prevu : `theory/m1/actionable_source_alignment.py`.
  - statut : fait en pretest offline, integration A25 non demarree.
  - recherche deterministe courte, profondeur 1 a 3 actions.
  - objectif de recherche :
    `desired_source_color in live_source_colors_by_action[action]`.
  - sortie cible : une sequence de preparation candidate, jamais une preuve.
- Integration cible :
  - ajouter une couche optionnelle de precondition avant A25, par exemple
    `ExperimentalPrecondition(prep_actions=[...],
    intended_effect="make_source_actionable", then_test_pair=(source, target),
    status="UNRESOLVED")`.
  - la preparation rend l'experience possible ; elle ne confirme pas
    l'hypothese.
  - A15-A31 restent le seul chemin de confirmation/refutation.
- KPI M1.3d :
  - `blocked_by_unselectable_source` baisse.
  - `pairs_actionable_source` augmente.
  - `pairs_entering_agenda` augmente.
  - `env_actions` augmente.
  - `wrong_confirmations == 0`.
- Garde-fous :
  - aucun statut `CONFIRMED` dans M1.3d.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
  - pas de world model / LLM tant que ce verrou local n'est pas compris.

## M1.3d-a - Implementation precise

- Ajout de `theory/m1/source_reachability.py`.
- Ajout de `SourceAlignmentProblem`, dataclass typant un blocage
  `source_not_selectable_for_action` :
  - `game_id`, `trace_path`.
  - `action`.
  - `desired_source_color`.
  - `target_color`.
  - `available_live_sources`.
  - `candidate_pair`.
  - `block_reason`.
  - `source_scope` (`ranked_pairs` ou `ranked_new_pairs`).
  - `support`, `transition_support`, `target_live_present`.
  - `predicates`, `preferred_predicates`, `live_preferred_predicates`.
  - `status="UNRESOLVED"`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Ajout de fonctions pures :
  - `load_grounding_autopsy`.
  - `extract_source_alignment_problems`.
  - `summarize_source_alignment_problems`.
  - `run_source_reachability_analysis`.
  - `write_source_reachability_analysis`.
- Entree :
  `diagnostics/m1/grounding_autopsy.json`.
- Sortie :
  `diagnostics/m1/source_reachability_problems.json`.
- Resultat sur scope global `ranked_pairs` :
  - `bp35-0a0ad940` : 20 problemes.
  - `cd82-fb555c5d` : 20 problemes.
  - `dc22-4c9bff3e` : 15 problemes.
- Resultat sur scope nouvelles paires M1 `ranked_new_pairs` :
  - `bp35-0a0ad940` : 5 problemes.
    - actions : `ACTION3=2`, `ACTION4=3`.
    - sources desirees : `[9, 11, 14]`.
    - sources live disponibles pour ces actions : aucune.
  - `cd82-fb555c5d` : 4 problemes.
    - actions : `ACTION1=1`, `ACTION2=1`, `ACTION5=2`.
    - sources desirees : `[0, 8, 12]`.
    - sources live disponibles pour ces actions : aucune.
  - `dc22-4c9bff3e` : 3 problemes.
    - actions : `ACTION1=2`, `ACTION2=1`.
    - sources desirees : `[2, 9]`.
    - sources live disponibles pour ces actions : aucune.
- Lecture :
  - le premier succes M1.3d-a est atteint :
    `source_not_selectable_for_action -> SourceAlignmentProblem` correctement
    type.
  - aucune preparation n'est encore cherchee.
  - aucune action environnement n'est executee.
  - aucune confirmation n'est creee.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_source_reachability.py tests\\test_m1_grounding_autopsy.py tests\\test_m1_live_anchor_ranking.py -q` -> 10 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest (Get-ChildItem -Path tests -Filter test_m1_*.py).FullName -q` -> 32 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 275 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\source_reachability_problems.json diagnostics\\m1\\grounding_autopsy.json diagnostics\\m1\\grounding_autopsy.md` -> aucun match.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.source_reachability --grounding-autopsy diagnostics\\m1\\grounding_autopsy.json --out diagnostics\\m1\\source_reachability_problems.json`.

## M1.3d-b - Implementation precise

- Ajout de `theory/m1/actionable_source_alignment.py`.
- Objectif limite :
  - consommer des `SourceAlignmentProblem`.
  - chercher une sequence de preparation courte, profondeur 1 a 3.
  - objectif : `desired_source_color in live_source_colors_by_action[action]`.
  - produire des `ExperimentalPrecondition` candidats, jamais des preuves.
  - ne pas integrer encore A25.
- Structures ajoutees :
  - `PreparationAction` : action concrete de preparation serialisable.
  - `ExperimentalPrecondition` :
    - `source_problem`.
    - `prep_actions`.
    - `intended_effect="make_source_actionable"`.
    - `then_test_pair=(action, source, target)`.
    - `source_becomes_actionable`.
    - `target_still_present`.
    - `final_available_sources`.
    - `status="UNRESOLVED"`.
    - `trace_support_counted_as_proof=false`.
    - `prior_counted_as_proof=false`.
  - `SearchState` et `SearchResult`.
- Recherche :
  - BFS bornee.
  - defaut : `max_depth=3`, `max_branching=4`,
    `max_nodes_per_problem=85`.
  - variante large testee : `max_branching=8`,
    `max_nodes_per_problem=585`.
  - chaque sequence est rejouee depuis `RESET` dans l'environnement offline.
  - cache par jeu et signature de sequence pour eviter de reevaluer les memes
    chemins.
- Pretest offline :
  - entree :
    `diagnostics/m1/source_reachability_problems.json`.
  - scope par defaut :
    `ranked_new_pairs`.
  - sortie conservative :
    `diagnostics/m1/actionable_source_alignment_pretest.json`.
  - sortie large :
    `diagnostics/m1/actionable_source_alignment_pretest_wide.json`.
- KPI exposes :
  - `problems_total`.
  - `preconditions_found`.
  - `mean_prep_length`.
  - `source_becomes_actionable`.
  - `target_still_present`.
  - `per_game`.
- Resultat conservative (`depth=3`, `branching=4`) :
  - `problems_total=12`.
  - `preconditions_found=0`.
  - `source_becomes_actionable=0`.
  - `target_still_present=0` car aucune precondition trouvee.
  - par jeu : `bp35=0/5`, `cd82=0/4`, `dc22=0/3`.
- Resultat large (`depth=3`, `branching=8`) :
  - `problems_total=12`.
  - `preconditions_found=0`.
  - par jeu : `bp35=0/5`, `cd82=0/4`, `dc22=0/3`.
- Lecture :
  - le pipeline logiciel M1.3d-b existe et sait produire un
    `ExperimentalPrecondition` sur test synthetique.
  - le run reel sur nouvelles paires M1 est negatif : aucune preparation courte
    simple ne rend les sources desirees selectionnables dans le beam actuel.
  - ce resultat indique que le verrou n'est pas seulement "chercher 1 a 3
    actions quelconques" ; il faut probablement mieux comprendre les effets
    locaux des actions de preparation avant toute integration A25.
  - aucune precondition n'est une confirmation ; la recherche reste
    `UNRESOLVED`.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_actionable_source_alignment.py tests\\test_m1_source_reachability.py -q` -> 7 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest (Get-ChildItem -Path tests -Filter test_m1_*.py).FullName -q` -> 36 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 279 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\actionable_source_alignment_pretest.json diagnostics\\m1\\actionable_source_alignment_pretest_wide.json diagnostics\\m1\\source_reachability_problems.json` -> aucun match.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.actionable_source_alignment --source-reachability diagnostics\\m1\\source_reachability_problems.json --source-scope ranked_new_pairs --max-depth 3 --max-branching 4 --max-nodes-per-problem 85 --out diagnostics\\m1\\actionable_source_alignment_pretest.json`.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.actionable_source_alignment --source-reachability diagnostics\\m1\\source_reachability_problems.json --source-scope ranked_new_pairs --max-depth 3 --max-branching 8 --max-nodes-per-problem 585 --out diagnostics\\m1\\actionable_source_alignment_pretest_wide.json`.

## M1.3d-c - Implementation precise

- Motivation :
  - M1.3d-b suppose implicitement que le bon verrou est
    "rendre une couleur source selectionnable".
  - Ce diagnostic teste l'hypothese inverse : bp35/cd82 changent peut-etre des
    couleurs, mais ne sont pas des jeux de type `source_color -> target_color`.
- Ajout de `theory/m1/mechanic_typing.py`.
- Entrees :
  - observations brutes M1.1 :
    `training/m1_observation_dataset.jsonl`.
  - grounding M1.3c+ :
    `diagnostics/m1/grounding_autopsy.json`.
- Sortie :
  - `diagnostics/m1/mechanic_typing.json`.
- Metriques brutes par jeu, resets exclus :
  - `color_change_rate`.
  - `position_change_rate`.
  - `object_motion_rate`.
  - `object_creation_rate`.
  - `object_removal_rate`.
  - `contact_change_rate`.
  - `shape_change_rate`.
  - `action_argument_rate`.
  - `action_argument_predictiveness`.
  - `mean_color_pairs_changed`.
  - `mean_motion_vectors`.
- Metriques de fit du schema couleur-source :
  - `source_color_predictiveness` =
    `pairs_actionable_source / pairs_discovered`.
  - `source_not_selectable_rate`.
  - `new_pair_source_color_predictiveness`.
  - `new_pair_source_not_selectable_rate`.
  - `pairs_entering_agenda_rate`.
  - `new_pairs_entering_agenda_rate`.
- Scores de mecanique :
  - `color_change`.
  - `color_source_grounding`.
  - `position_motion`.
  - `object_lifecycle`.
  - `topology_contact`.
  - `shape_object`.
  - `action_argument`.
- Resultats clefs :
  - `bp35-0a0ad940` :
    - `color_change_rate=1.0`.
    - `position_change_rate=0.9464`.
    - `object_motion_rate=0.9464`.
    - `object_creation_rate=1.0`.
    - `object_removal_rate=1.0`.
    - `contact_change_rate=0.7598`.
    - `source_color_predictiveness=0.0`.
    - `source_not_selectable_rate=1.0`.
    - warning :
      `color_source_schema_misaligned_with_observed_mechanics`.
  - `cd82-fb555c5d` :
    - `color_change_rate=0.8695`.
    - `position_change_rate=0.1345`.
    - `object_motion_rate=0.1345`.
    - `object_creation_rate=0.8682`.
    - `object_removal_rate=0.8695`.
    - `contact_change_rate=0.6485`.
    - `source_color_predictiveness=0.0`.
    - `source_not_selectable_rate=1.0`.
    - warning :
      `color_source_schema_misaligned_with_observed_mechanics`.
  - `dc22-4c9bff3e` :
    - `color_change_rate=0.8214`.
    - `position_change_rate=0.6964`.
    - `object_motion_rate=0.6964`.
    - `object_creation_rate=0.8005`.
    - `object_removal_rate=0.8005`.
    - `contact_change_rate=0.6922`.
    - `source_color_predictiveness=0.25`.
    - `new_pair_source_color_predictiveness=0.5`.
    - `new_pairs_entering_agenda_rate=0.1667`.
    - aucun warning de mismatch.
- Lecture :
  - bp35 et cd82 changent bien des couleurs, mais le schema
    "couleur source selectionnable" ne se grounde pas.
  - M1.3d-b echoue probablement parce qu'il essaie de preparer le mauvais type
    d'experience.
  - Le prochain verrou est mieux nomme `mechanic grounding` que
    `source reachability`.
  - Pour bp35/cd82, il faut chercher des hypotheses experimentales de type
    objet/mouvement/contact/topologie, pas seulement des paires couleur-source.
- Discipline epistemique :
  - chaque profil est `UNRESOLVED`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
  - aucune confirmation n'est produite.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_mechanic_typing.py -q` -> 3 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest (Get-ChildItem -Path tests -Filter test_m1_*.py).FullName -q` -> 39 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 282 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\mechanic_typing.json diagnostics\\m1\\actionable_source_alignment_pretest.json diagnostics\\m1\\source_reachability_problems.json` -> aucun match.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.mechanic_typing --observations training\\m1_observation_dataset.jsonl --grounding-autopsy diagnostics\\m1\\grounding_autopsy.json --out diagnostics\\m1\\mechanic_typing.json`.

## M1.3e - Implementation precise

- Ajout de `theory/m1/mechanic_grounded_candidates.py`.
- Objectif :
  - generer des candidats experimentaux non-couleur, a partir de M1.1 et
    M1.3d-c.
  - ne pas integrer A25.
  - ne pas executer d'environnement.
  - ne jamais confirmer.
- Entrees :
  - `training/m1_observation_dataset.jsonl`.
  - `diagnostics/m1/mechanic_typing.json`.
- Sortie :
  - `diagnostics/m1/mechanic_grounded_candidates.json`.
- Structure `MechanicGroundedExperimentCandidate` :
  - `game_id`.
  - `candidate_type`.
  - `action`.
  - `mechanism`.
  - `support`.
  - `support_rate`.
  - `test_goal`.
  - `evidence`.
  - `status="UNRESOLVED"`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Types generes :
  - `object_motion_candidate`.
  - `contact_change_candidate`.
  - `object_lifecycle_candidate`.
  - `shape_zone_candidate`.
  - `position_effect_candidate`.
- Selection :
  - support minimal par defaut : `min_support_count=3`.
  - taux minimal par action : `min_support_rate=0.35`.
  - maximum par jeu : `max_candidates_per_game=12`.
  - selection diversifiee : garder d'abord le meilleur candidat de chaque type
    mecanique, puis completer par score.
- Resultat global :
  - `candidate_count=64`.
  - par type :
    - `contact_change_candidate=5`.
    - `object_lifecycle_candidate=19`.
    - `object_motion_candidate=10`.
    - `position_effect_candidate=13`.
    - `shape_zone_candidate=17`.
- Resultat bp35 :
  - 12 candidats.
  - types :
    - `object_lifecycle_candidate=4`.
    - `shape_zone_candidate=4`.
    - `position_effect_candidate=2`.
    - `contact_change_candidate=1`.
    - `object_motion_candidate=1`.
  - meilleurs exemples :
    - `ACTION6 contact_change_candidate`, support_rate `0.9805`.
    - `ACTION6 object_lifecycle_candidate`, support_rate `1.0`.
    - `ACTION3 object_motion_candidate`, support_rate `0.9803`.
    - `ACTION6 position_effect_candidate`, support_rate `1.0`.
    - `ACTION6 shape_zone_candidate`, support_rate `1.0`.
- Resultat cd82 :
  - 12 candidats.
  - types :
    - `object_lifecycle_candidate=5`.
    - `shape_zone_candidate=4`.
    - `contact_change_candidate=1`.
    - `object_motion_candidate=1`.
    - `position_effect_candidate=1`.
  - meilleurs exemples :
    - `ACTION5 contact_change_candidate`, support_rate `0.7534`.
    - `ACTION5 object_lifecycle_candidate`, support_rate `0.9178`.
    - `ACTION6 object_motion_candidate`, support_rate `0.6818`.
    - `ACTION6 position_effect_candidate`, support_rate `1.0`.
    - `ACTION5 shape_zone_candidate`, support_rate `0.9178`.
- Resultat dc22 :
  - 12 candidats.
  - types :
    - `object_lifecycle_candidate=3`.
    - `position_effect_candidate=3`.
    - `shape_zone_candidate=3`.
    - `object_motion_candidate=2`.
    - `contact_change_candidate=1`.
  - dc22 reste le controle positif partiellement compatible couleur-source,
    mais dispose aussi de candidats mecanique-grounded.
- Lecture :
  - M1.3e valide le changement de representation : pour bp35/cd82, la
    prochaine experience ne doit pas seulement chercher une couleur source.
  - Le bon objet A25 futur devient polymorphe :
    "quel mecanisme testable ?" avant "quelle couleur source ?".
  - Ces candidats ne sont pas des hypotheses confirmees ; ils sont de la
    matiere experimentale `UNRESOLVED`.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_mechanic_grounded_candidates.py tests\\test_m1_mechanic_typing.py -q` -> 7 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest (Get-ChildItem -Path tests -Filter test_m1_*.py).FullName -q` -> 43 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\ -q` -> 286 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\mechanic_grounded_candidates.json diagnostics\\m1\\mechanic_typing.json diagnostics\\m1\\source_reachability_problems.json` -> aucun match.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.mechanic_grounded_candidates --observations training\\m1_observation_dataset.jsonl --mechanic-typing diagnostics\\m1\\mechanic_typing.json --min-support-count 3 --min-support-rate 0.35 --max-candidates-per-game 12 --out diagnostics\\m1\\mechanic_grounded_candidates.json`.

## M1.3f - Implementation precise

- Ajout de `theory/m1/polymorphic_a25_pretest.py`.
- Objectif :
  - tester si les candidats mecaniques M1.3e peuvent devenir une experience
    A25 polymorphe.
  - ne pas executer A25.
  - ne pas effectuer d'action environnement.
  - ne jamais confirmer.
- Entree :
  - `diagnostics/m1/mechanic_grounded_candidates.json`.
- Sortie :
  - `diagnostics/m1/polymorphic_a25_pretest.json`.
- Structure `PolymorphicA25PretestRow` :
  - `candidate_id`.
  - `game_id`.
  - `candidate_type`.
  - `action`.
  - `testability_status` (`testable` ou `blocked`).
  - `required_observation`.
  - `available_live_affordance`.
  - `blocking_reason`.
  - `status="UNRESOLVED"`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Observations requises par type :
  - `object_motion_candidate` -> `object_positions_before_after`.
  - `contact_change_candidate` -> `contact_graph_before_after`.
  - `object_lifecycle_candidate` -> `object_counts_before_after`.
  - `shape_zone_candidate` -> `object_shape_zone_before_after`.
  - `position_effect_candidate` -> `local_patch_before_after`.
- Raisons de blocage typees :
  - `action_not_available`.
  - `missing_position_argument`.
  - `no_live_object_anchor`.
  - `no_measurable_before_after_metric`.
  - `unsafe_or_terminal_action`.
- Critere de testabilite :
  - l'action candidate existe dans les actions live au reset.
  - l'action n'est pas terminale/unsafe.
  - la metrique before/after requise est mesurable sur la grille live.
  - les candidats positionnels exigent au moins une action avec arguments
    `x/y`.
  - les candidats objet/contact/shape exigent une ancre objet live quand cela
    est necessaire.
- KPI exposes :
  - `mechanic_candidates_total`.
  - `mechanic_candidates_testable`.
  - `testable_by_type`.
  - `testable_by_game`.
  - `blocking_reasons`.
  - `wrong_confirmations=0`.
- Resultat global :
  - `mechanic_candidates_total=64`.
  - `mechanic_candidates_testable=44`.
  - `wrong_confirmations=0`.
  - blocages :
    - `action_not_available=13`.
    - `missing_position_argument=7`.
  - testables par type :
    - `object_lifecycle_candidate=15/19`.
    - `shape_zone_candidate=13/17`.
    - `object_motion_candidate=9/10`.
    - `position_effect_candidate=4/13`.
    - `contact_change_candidate=3/5`.
- Resultat par jeu :
  - `bp35-0a0ad940` : 9/12 testables.
    - testables : `ACTION6 contact/lifecycle/position/shape`,
      `ACTION3 motion/lifecycle/shape`, `ACTION4 lifecycle/shape`.
    - bloques : `ACTION7` indisponible, `ACTION3 position` sans argument
      positionnel live.
  - `cd82-fb555c5d` : 5/12 testables.
    - testables : `ACTION1 lifecycle`, `ACTION2 lifecycle/shape`,
      `ACTION3 lifecycle/shape`.
    - bloques : les candidats forts de trace `ACTION5/ACTION6` ne sont pas
      disponibles au reset live dans ce pretest.
  - `dc22-4c9bff3e` : 10/12 testables.
  - `ar25-e3c63847` : 7/12 testables.
  - `cn04-65d47d14` : 9/12 testables.
  - `ft09-0d8bbf25` : 4/4 testables.
- Lecture :
  - le funnel polymorphe existe maintenant :
    `mechanic candidate -> measurable live affordance -> testable experiment`.
  - bp35 sort nettement du blocage couleur-source : la majorite des candidats
    mecaniques sont testables au reset.
  - cd82 est plus nuance : des candidats mecaniques sont testables, mais les
    meilleurs candidats de trace `ACTION5/ACTION6` sont indisponibles au reset ;
    cela indique un probleme de disponibilite d'action/precondition, pas de
    couleur source.
  - M1.3f ne prouve rien : il transforme seulement la matiere M1.3e en
    candidats experimentaux potentiellement consommables.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_polymorphic_a25_pretest.py -q` -> 6 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.polymorphic_a25_pretest --candidates diagnostics\\m1\\mechanic_grounded_candidates.json --out diagnostics\\m1\\polymorphic_a25_pretest.json`.

## M1.3g - Implementation precise

- Ajout de `theory/m1/polymorphic_a25_adapter.py`.
- Objectif :
  - consommer uniquement les lignes M1.3f `testability_status="testable"`.
  - executer une seule action concrete candidate depuis `RESET`.
  - mesurer le delta attendu par `required_observation`.
  - produire une experience mecanique `UNRESOLVED`.
  - ne pas reviser d'hypothese et ne jamais confirmer.
- Entree :
  - `diagnostics/m1/polymorphic_a25_pretest.json`.
- Sortie cible bp35 :
  - `diagnostics/m1/polymorphic_a25_adapter_bp35.json`.
- Structures ajoutees :
  - `ConcretePolymorphicAction`.
  - `PolymorphicMechanicExperiment`.
- Selection par defaut :
  - `game_id=bp35-0a0ad940`.
  - `ACTION6:object_lifecycle_candidate`.
  - `ACTION6:contact_change_candidate`.
  - `ACTION3:object_motion_candidate`.
  - `max_candidates=3`.
- Mesures before/after ajoutees :
  - `object_counts_before_after`.
  - `contact_graph_before_after`.
  - `object_positions_before_after`.
  - `object_shape_zone_before_after`.
  - `local_patch_before_after`.
- KPI exposes :
  - `mechanic_candidates_consumed`.
  - `mechanic_experiments_generated`.
  - `env_actions`.
  - `observable_deltas`.
  - `generated_by_type`.
  - `generated_by_game`.
  - `revision_performed`.
  - `wrong_confirmations`.
- Resultat bp35 du 2026-06-19 :
  - `mechanic_candidates_consumed=3`.
  - `mechanic_experiments_generated=3`.
  - `env_actions=3`.
  - `observable_deltas=3`.
  - `revision_performed=false`.
  - `wrong_confirmations=0`.
- Deltas observes :
  - `ACTION6 object_lifecycle_candidate` :
    - `changed_pixels=26`.
    - `object_count_before=190`.
    - `object_count_after=187`.
    - `object_count_delta=-3`.
    - delta couleurs : `3:-4`, `10:+1`, `14:-1`, `15:+1`.
  - `ACTION6 contact_change_candidate` :
    - `changed_pixels=26`.
    - `contact_pair_count_before=3`.
    - `contact_pair_count_after=4`.
    - contact ajoute : `[0, 15]`.
  - `ACTION3 object_motion_candidate` :
    - `changed_pixels=47`.
    - `matched_component_count=190`.
    - `moved_component_count=4`.
    - mouvements observes sur couleurs `10`, `9`, `11`, `0`.
- Lecture :
  - M1.3g atteint le premier critere operationnel polymorphe :
    `env_actions > 0` et `mechanic_experiments_generated > 0` sur bp35.
  - Le verrou couleur-source est contourne sans M2 et sans changer A15-A31.
  - Le module ne decide pas confirmation/refutation ; il produit seulement des
    observations experimentales mecaniques pour une future couche A25
    polymorphe.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_polymorphic_a25_adapter.py -q` -> 5 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.polymorphic_a25_adapter --pretest diagnostics\\m1\\polymorphic_a25_pretest.json --game-id bp35-0a0ad940 --candidate ACTION6:object_lifecycle_candidate --candidate ACTION6:contact_change_candidate --candidate ACTION3:object_motion_candidate --max-candidates 3 --out diagnostics\\m1\\polymorphic_a25_adapter_bp35.json`.

## M1.3h - Implementation precise

- Ajout de `theory/m1/experiment_value_estimator.py`.
- Objectif :
  - repondre a "quelle experience vaut le plus la peine d'etre executee ?".
  - scorer les candidats M1.3f testables.
  - utiliser les observations M1.3g quand elles existent.
  - estimer les autres candidats depuis support trace + affordances live.
  - recommander un petit set diversifie.
  - ne pas executer d'action, ne pas reviser, ne jamais confirmer.
- Entrees :
  - `diagnostics/m1/polymorphic_a25_pretest.json`.
  - `diagnostics/m1/mechanic_grounded_candidates.json`.
  - `diagnostics/m1/polymorphic_a25_adapter_bp35.json`.
- Sortie :
  - `diagnostics/m1/experiment_value_estimates.json`.
- Structure `ExperimentalValueEstimate` :
  - `candidate_id`.
  - `game_id`.
  - `candidate_type`.
  - `action`.
  - `required_observation`.
  - `score`.
  - `expected_information_gain`.
  - `delta_score`.
  - `novelty_score`.
  - `diversity_score`.
  - `expected_delta_magnitude`.
  - `expected_state_change`.
  - `expected_novelty`.
  - `expected_disambiguation_power`.
  - `score_basis`.
  - `recommended`.
  - `status="UNRESOLVED"`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Formule V1 :
  - `score = 0.4 * delta_score + 0.3 * novelty_score + 0.3 * diversity_score`.
  - `delta_score` vient du delta observe M1.3g si disponible, sinon d'une
    estimation par type mecanique.
  - `novelty_score` favorise les types moins redondants dans le pool.
  - `diversity_score` approxime le pouvoir de desambiguisation : une action
    liee a plusieurs types mecaniques est plus informative.
- Selection recommandee :
  - premier passage : meilleur candidat de chaque type mecanique, si le budget
    le permet.
  - second passage : remplissage par score.
  - but explicite : eviter `lifecycle/lifecycle/lifecycle` quand
    `motion/contact/shape/position` sont aussi testables.
- Resultat global du 2026-06-19 :
  - `candidates_total=44`.
  - `recommended_experiments=5`.
  - `candidate_types_total=5`.
  - `recommended_candidate_types=5`.
  - `type_diversity=1.0`.
  - `mean_information_score=0.5688`.
  - `wrong_confirmations=0`.
  - bases de score :
    - `observed_delta=3`.
    - `estimated_from_trace_lifecycle=14`.
    - `estimated_from_trace_shape=13`.
    - `estimated_from_trace_motion=8`.
    - `estimated_from_trace_position=4`.
    - `estimated_from_trace_contact=2`.
- Recommandations V1 :
  - `bp35-0a0ad940 ACTION6 position_effect_candidate`, score `0.8636`.
  - `bp35-0a0ad940 ACTION3 object_lifecycle_candidate`, score `0.8084`.
  - `ar25-e3c63847 ACTION1 object_motion_candidate`, score `0.7310`.
  - `cn04-65d47d14 ACTION6 contact_change_candidate`, score `0.7183`.
  - `bp35-0a0ad940 ACTION6 shape_zone_candidate`, score `0.5629`.
- Lecture :
  - M1.3h transforme M1.3g en selection d'experiences sous budget.
  - bp35 reste prioritaire : 3/5 recommandations globales sont bp35.
  - l'objectif n'est pas la verite de l'hypothese, mais la valeur attendue de
    l'observation.
  - cette milestone prefigure M3 : choisir d'abord ce qui reduit le plus
    l'incertitude.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_experiment_value_estimator.py -q` -> 3 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.experiment_value_estimator --pretest diagnostics\\m1\\polymorphic_a25_pretest.json --candidates diagnostics\\m1\\mechanic_grounded_candidates.json --observed diagnostics\\m1\\polymorphic_a25_adapter_bp35.json --max-recommended 5 --out diagnostics\\m1\\experiment_value_estimates.json`.

## M1.3i - Implementation precise

- Ajout de `theory/m1/recommended_experiment_choice.py`.
- Objectif :
  - prendre la meilleure recommandation M1.3h.
  - la convertir en `MechanicHypothesisCandidate`.
  - produire un `PolymorphicA25ExperimentalChoice` consommable par une future
    branche A25 opt-in.
  - executer l'action via M1.3g.
  - ecrire une `MechanicObservation` structuree.
  - ne pas reviser, ne pas confirmer.
- Entrees :
  - `diagnostics/m1/experiment_value_estimates.json`.
  - `diagnostics/m1/polymorphic_a25_pretest.json`.
- Sortie :
  - `diagnostics/m1/recommended_polymorphic_a25_choice.json`.
- Contrat fige :
  - `MechanicHypothesisCandidate` :
    - `mechanic_family`.
    - `action`.
    - `predicted_metric`.
    - `expected_outcome`.
    - `observed_delta`.
    - `status="UNRESOLVED"`.
    - `trace_support_counted_as_proof=false`.
    - `prior_counted_as_proof=false`.
  - `PolymorphicA25ExperimentalChoice` :
    - `a25_choice_type="polymorphic_mechanic_experiment"`.
    - `candidate_id`.
    - `action`.
    - `mechanic_family`.
    - `predicted_metric`.
    - `expected_outcome`.
    - `expected_information_gain`.
    - `selection_reason="m1_3h_recommended_highest_information_gain"`.
    - `competing_keys`, `prediction_families`, `predicted_outcomes` pour
      compatibilite conceptuelle avec un choix experimental A25.
  - `MechanicObservation` :
    - `observed_delta`.
    - `env_actions`.
    - `changed_pixels`.
    - `mechanic_experiment_generated`.
    - `revision_performed=false`.
    - `wrong_confirmations=0`.
- Run cible du 2026-06-19 :
  - selection automatique de la meilleure recommandation M1.3h :
    `m1e0015:bp35-0a0ad940:ACTION6:position_effect_candidate`.
  - `score=0.8636`.
  - `expected_information_gain=0.8636`.
  - `predicted_metric=local_patch_before_after`.
  - `action=ACTION6`.
- Observation reelle :
  - `env_actions=1`.
  - `changed_pixels=26`.
  - `local_changed_pixels=1`.
  - patch local avant :
    `[[5, 5, 5], [3, 5, 3]]`.
  - patch local apres :
    `[[5, 5, 5], [3, 5, 10]]`.
  - `observable_delta=true`.
  - `revision_performed=false`.
  - `wrong_confirmations=0`.
- Lecture :
  - M1.3i transforme enfin une recommandation M1.3h en choix experimental A25
    polymorphe, sans passer par le moule couleur-source.
  - Le choix est consommable conceptuellement par A25, mais reste opt-in et
    hors moteur A15-A31 tant que le contrat de revision mecanique n'est pas
    defini.
  - La chaine complete existe maintenant :
    `candidate -> testable -> executable -> prioritized -> A25 choice`.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_recommended_experiment_choice.py -q` -> 3 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.recommended_experiment_choice --estimates diagnostics\\m1\\experiment_value_estimates.json --pretest diagnostics\\m1\\polymorphic_a25_pretest.json --out diagnostics\\m1\\recommended_polymorphic_a25_choice.json`.

## M1.3j - Implementation precise

- Ajout de `theory/m1/mechanic_revision_candidate.py`.
- Objectif :
  - transformer une `MechanicObservation` M1.3i en prediction mecanique
    testable.
  - produire une proposition de revision pour A15-A31.
  - ne pas effectuer la revision.
  - ne pas compter l'observation M1 comme confirmation.
  - imposer un test controle ulterieur.
- Entree :
  - `diagnostics/m1/recommended_polymorphic_a25_choice.json`.
- Sortie :
  - `diagnostics/m1/mechanic_revision_candidates.json`.
- Structures ajoutees :
  - `MechanicPredictionCandidate`.
  - `MechanicRevisionCandidate`.
- Contrat `MechanicPredictionCandidate` :
  - `candidate_id`.
  - `game_id`.
  - `mechanic_family`.
  - `action`.
  - `predicted_metric`.
  - `expected_outcome`.
  - `observed_outcome`.
  - `observed_delta_summary`.
  - `key`.
  - `status="UNRESOLVED"`.
  - `controlled_test_required=true`.
  - `observation_counted_as_confirmation=false`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Contrat `MechanicRevisionCandidate` :
  - `revision_candidate_id`.
  - `prediction`.
  - `a15_a31_revision_proposal`.
  - `proposed_status="UNRESOLVED"`.
  - `support=0`.
  - `contradictions=0`.
  - `experiments_spent=0`.
  - `controlled_test_required=true`.
  - `revision_performed=false`.
  - `wrong_confirmations=0`.
- Run du 2026-06-19 :
  - entree issue de M1.3i :
    `m1e0015:bp35-0a0ad940:ACTION6:position_effect_candidate`.
  - prediction produite :
    - key :
      `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`.
    - `observed_outcome=local_patch_changed`.
    - `observed_delta_summary.local_changed_pixels=1`.
    - `observed_delta_summary.changed_pixels=26`.
  - proposition A15-A31 :
    - `proposed_status=UNRESOLVED`.
    - `support=0`.
    - `contradictions=0`.
    - `controlled_test_required=true`.
    - `observation_counted_as_confirmation=false`.
- KPI :
  - `mechanic_predictions=1`.
  - `revision_candidates=1`.
  - `a15_a31_revision_proposals=1`.
  - `controlled_tests_required=1`.
  - `revision_performed=false`.
  - `observation_counted_as_confirmation=false`.
  - `wrong_confirmations=0`.
- Lecture :
  - M1.3j ferme la chaine M1 sans casser la discipline epistemique :
    l'observation devient une prediction/revision candidate, pas un verdict.
  - La prochaine integration doit etre opt-in et faire passer cette proposition
    dans un vrai test controle A15-A31.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_mechanic_revision_candidate.py -q` -> 4 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.mechanic_revision_candidate --choice diagnostics\\m1\\recommended_polymorphic_a25_choice.json --out diagnostics\\m1\\mechanic_revision_candidates.json`.

## M1.3k - Implementation precise

- Ajout de `theory/m1/scientific_integration_pretest.py`.
- Objectif :
  - prendre `diagnostics/m1/mechanic_revision_candidates.json`.
  - convertir la proposition M1.3j en `HypothesisRecord` A15-A31.
  - faire passer ce record dans `score_beliefs`.
  - verifier que le moteur epistemique accepte l'entree sans la transformer en
    confirmation.
  - ne pas executer d'experience.
  - ne pas reviser.
- Entree :
  - `diagnostics/m1/mechanic_revision_candidates.json`.
- Sortie :
  - `diagnostics/m1/scientific_integration_pretest.json`.
- Structure `ScientificLedgerEntry` :
  - `revision_candidate_id`.
  - `game_id`.
  - `key`.
  - `description`.
  - `status=unresolved`.
  - `support=0`.
  - `contradictions=0`.
  - `experiments_spent=0`.
  - `controlled_test_required=true`.
  - `entered_scientific_ledger=true`.
  - `observation_counted_as_confirmation=false`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Conversion vers A15-A31 :
  - `ScientificLedgerEntry.to_record()` produit un `HypothesisRecord`.
  - ce record est score par `score_beliefs(records, MechanicsOracle(game_id))`.
  - l'oracle est volontairement vide pour cette mecanique nouvelle :
    `unverifiable=1`, ce qui prouve que l'entree est admise mais non jugee.
- Resultat du 2026-06-19 :
  - `revision_candidates_total=1`.
  - `entered_scientific_ledger=1`.
  - `unresolved_records=1`.
  - `confirmed_records=0`.
  - `refuted_records=0`.
  - `support_total=0`.
  - `contradictions_total=0`.
  - `experiments_spent_total=0`.
  - `controlled_tests_required=1`.
  - `observation_counted_as_confirmation=false`.
  - `revision_performed=false`.
  - `wrong_confirmations=0`.
- Score epistemique :
  - jeu : `bp35-0a0ad940`.
  - `hypotheses_confirmed=0`.
  - `hypotheses_refuted=0`.
  - `wrong_confirmations=0`.
  - `unverifiable=1`.
  - `experiment_actions=0`.
- Lecture :
  - M1.3k prouve que M1 peut entrer dans le ledger scientifique sans verdict
    premature.
  - Le prochain branchement doit ajouter un vrai test controle supplementaire,
    pas reutiliser l'observation M1 comme preuve.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_scientific_integration_pretest.py -q` -> 3 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.scientific_integration_pretest --revision-candidates diagnostics\\m1\\mechanic_revision_candidates.json --out diagnostics\\m1\\scientific_integration_pretest.json`.

## M1.3l - Implementation precise

- Ajout de `theory/m1/controlled_followup_experiment.py`.
- Objectif :
  - prendre une entree `ScientificLedgerEntry` M1.3k deja admise dans le
    ledger.
  - executer un test controle supplementaire depuis reset.
  - comparer une sequence baseline et une sequence perturbation sur la meme
    metrique.
  - ajouter des `support_events` ou `contradiction_events`.
  - ne jamais transformer cet evenement en confirmation.
- Entree :
  - `diagnostics/m1/scientific_integration_pretest.json`.
- Sortie :
  - `diagnostics/m1/controlled_experiment_results.json`.
- Structure `ControlledExperiment` :
  - `hypothesis_key`.
  - `baseline_sequence`.
  - `perturbation_sequence`.
  - `predicted_metric`.
  - `observed_baseline`.
  - `observed_perturbation`.
  - `delta`.
  - `support_events`.
  - `contradiction_events`.
  - `status="UNRESOLVED"`.
  - `revision_performed=false`.
  - `observation_counted_as_confirmation=false`.
  - `trace_support_counted_as_proof=false`.
  - `prior_counted_as_proof=false`.
- Principe du test controle V1 :
  - la perturbation rejoue l'action cible de l'hypothese ledger :
    `RESET -> ACTION6`.
  - la baseline execute une action controle prioritaire differente :
    `RESET -> ACTION3`.
  - pour `local_patch_before_after`, la baseline mesure le meme patch que
    l'action cible, meme si l'action controle n'a pas d'arguments `x/y`.
  - `support_event=1` si le signal perturbation depasse le signal baseline.
  - `contradiction_event=1` si le signal perturbation est inferieur au signal
    baseline.
  - `support` du ledger reste `0` : l'evenement est une evidence experimentale,
    pas un verdict scientifique.
- Run du 2026-06-19 :
  - hypothese testee :
    `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`.
  - baseline :
    - sequence : `RESET -> ACTION3`.
    - `changed_pixels=47`.
    - patch cible `[0, 17, 1, 19]`.
    - `local_changed_pixels=0`.
  - perturbation :
    - sequence : `RESET -> ACTION6`.
    - `changed_pixels=26`.
    - patch cible `[0, 17, 1, 19]`.
    - `local_changed_pixels=1`.
    - patch local avant : `[[5, 5, 5], [3, 5, 3]]`.
    - patch local apres : `[[5, 5, 5], [3, 5, 10]]`.
  - delta controle :
    - `baseline_signal=0.0`.
    - `perturbation_signal=1.0`.
    - `effect_size=1.0`.
    - `direction=support`.
- Resultat :
  - `hypotheses_tested=1`.
  - `controlled_experiments_run=1`.
  - `env_actions=2`.
  - `support_events=1`.
  - `contradiction_events=0`.
  - `unresolved_hypotheses=1`.
  - `wrong_confirmations=0`.
  - entree ledger mise a jour avec `support_events=1`, mais `support=0`,
    `contradictions=0`, `status=UNRESOLVED`.
- Lecture :
  - M1.3l prouve qu'une hypothese mecanique M1 peut survivre a un premier test
    controle et accumuler un evenement de support.
  - Ce n'est toujours pas M3 : le module ne choisit pas une sequence optimale,
    il verifie seulement qu'une hypothese issue du ledger peut recevoir une
    evidence controlee.
  - La chaine M1 complete devient :
    `hypothese -> experience -> observation -> ledger -> nouveau test -> support event`.
- Verification locale du 2026-06-19 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_controlled_followup_experiment.py -q` -> 4 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.controlled_followup_experiment --scientific-integration diagnostics\\m1\\scientific_integration_pretest.json --out diagnostics\\m1\\controlled_experiment_results.json`.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests -q -k m1` -> 71 tests M1 passes, 243 deselectionnes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests -q` -> 314 tests passes.
  - `rg "CONFIRMED|HypothesisStatus.CONFIRMED" theory\\m1 diagnostics\\m1\\controlled_experiment_results.json diagnostics\\m1\\scientific_integration_pretest.json diagnostics\\m1\\mechanic_revision_candidates.json` -> aucun match.

## M1.4 - Implementation precise

- Ajout de `theory/m1/stress_test_a31bis.py`.
- Propagation opt-in du `predicate_generator` M1 dans :
  - `theory/multi_game_evaluation.py`.
  - `theory/multi_game_stress_test.py`.
- Defaut toujours historique :
  - si `predicate_generator=None`, A30/A31 appellent A12/A25 comme avant.
  - M1 est active uniquement par `stress_test_a31bis.py`.
- Le wrapper M1.4 :
  - construit le generateur M1 depuis `diagnostics/m1/accepted_invariants.json`.
  - lance A31 avec M1 active.
  - charge le baseline archive `diagnostics/multi_game_stress_test.json`
    (UTF-8/UTF-16 supportes).
  - ajoute un mode `--run-local-baseline` pour comparer M1 off/on sur exactement
    la meme configuration reduite.
  - ajoute les KPI de couverture M1.3 dans la sortie :
    `unique_predicates_per_trace`, `relation_candidates_generated`,
    `candidate_pairs_per_trace`.
- Verification locale du 2026-06-18 :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_observation_dataset.py tests\\test_m1_invariant_miner.py tests\\test_m1_predicate_generation.py tests\\test_m1_stress_test_a31bis.py tests\\test_multi_game_evaluation.py tests\\test_multi_game_stress_test.py tests\\test_cross_game_correspondence_discovery.py tests\\test_non_ar25_multi_relation_agenda.py -q` -> 26 tests passes.
  - Smoke comparable :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.stress_test_a31bis --budgets 2 --latest-per-game --exclude-ar25 --max-traces 2 --max-attempts-per-budget 1 --run-local-baseline --out diagnostics\\m1_stress_test_a31bis_smoke.json`.
  - Resultat smoke local M1 off/on :
    - `not_enough_relation_candidates` : 2 -> 2.
    - `wrong_confirmations` : 0.
    - `unique_predicates_per_trace` : 4.5 -> 11.5.
    - `relation_candidates_generated` : 285.5 -> 583.5.
    - `candidate_pairs_per_trace` : 65.5 -> 65.5.
    - Diagnostic : `vocabulary_expanded_pairs_flat_blocker_not_reduced`.
  - Smoke regenere apres M1.3c+ avec ranking live/ancres :
    - `new_pairs_blocked_by_unselectable_source` : 4.5 en moyenne sur
      `bp35/cd82`.
    - `new_pairs_live_color_compatible` : 0.0.
    - `new_pairs_target_present` : 4.5.
    - `wrong_confirmations` : 0.
  - Run borne comparable sur 4 budgets :
    `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.stress_test_a31bis --budgets 5,10,25,50 --latest-per-game --exclude-ar25 --max-traces 2 --max-attempts-per-budget 1 --run-local-baseline --out diagnostics\\m1_stress_test_a31bis_bounded.json`.
  - Resultat run borne local M1 off/on :
    - `not_enough_relation_candidates` : 8 -> 8.
    - `wrong_confirmations` : 0.
    - `unique_predicates_per_trace` : 4.5 -> 11.5.
    - `relation_candidates_generated` : 285.5 -> 583.5.
    - `candidate_pairs_per_trace` : 65.5 -> 65.5.
    - Diagnostic : `vocabulary_expanded_pairs_flat_blocker_not_reduced`.
- Runs longs tentes mais non finalises dans la fenetre d'execution :
  - `--budgets 5,10,25,50 --latest-per-game` avec ar25 inclus -> timeout a 15 minutes, aucun JSON final.
  - `--budgets 5,10,25,50 --latest-per-game --exclude-ar25` -> timeout a 15 minutes, aucun JSON final.
  - `--budgets 5,10,25,50 --latest-per-game --exclude-ar25 --max-attempts-per-budget 2` -> timeout a 15 minutes, aucun JSON final.
- Lecture M1.4 actuelle :
  - Le chemin A31bis est implemente et testable.
  - Le smoke et le run borne confirment le diagnostic M1.3 : M1 augmente le
    vocabulaire et les predicats attaches, mais ne reduit pas le blocage quand
    les paires source/target restent plates.
  - Le run borne M1.3b montre maintenant que les paires peuvent augmenter
    (`65.5 -> 69.5`) sans reduire encore le blocage (`8 -> 8`). Le diagnostic
    passe donc de `vocabulary_expanded_pairs_flat_blocker_not_reduced` a
    `anchor_expansion_pairs_up_blocker_not_reduced`.
  - Le pre-test M1.3c affine encore : sur `bp35` et `cd82`, les nouvelles paires
    existent mais ne sont pas live-color-compatibles au reset ; sur `dc22`, une
    nouvelle paire entre dans l'agenda et A25 direct produit une experience.
  - Le run global exact doit etre relance avec un timeout plus large ou une
    ecriture incrementale des courbes pour eviter de perdre les resultats si le
    processus depasse la fenetre.

## M1.G0.1/G0.2/G0.3/G0.4/G0.5 - General mechanic abstraction layer

- Ajout de `theory/m1/general_mechanic_abstraction.py`.
- Objectif :
  - passer d'une logique patch/action locale a une couche generique
    `entites -> roles candidats -> hypotheses mecaniques candidate-only`.
  - garder l'agent detector comme role possible, pas comme branche bp35 codee.
  - ne pas injecter de notion specifique comme gravite, boite orange ou
    navigation.
- M1.G0.1 Entity tracker :
  - extraction de composants connectes par couleur, hors background infere.
  - suivi temporel par couleur, signature de forme, taille et proximite de
    centroide.
  - resume par entite :
    `bbox_sequence`, `centroid_sequence`, `size_sequence`,
    `persistence_score`, `shape_stability_score`, `motion_score`,
    `edge_contact_score`, `appears_after_action`, `disappears_after_action`.
- M1.G0.2 Role hypothesis generator :
  - roles candidats generiques :
    `controllable_actor`, `moving_object`, `passive_object`, `boundary`,
    `target_candidate`, `created_object`, `transformed_object`,
    `timer_or_hud`, `unknown`.
  - chaque role garde `support=0`,
    `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_M1`.
  - un accord de signaux augmente seulement un score de priorite, jamais un
    support scientifique.
- Mechanic hypothesis emitter V1 :
  - emission de hypotheses `mechanic_family=entity_role`.
  - emission de hypotheses `mechanic_family=action_effect`.
  - chaque hypothese a `status=UNRESOLVED`, `controlled_test_required=true`,
    `support=0`, `trace_support_counted_as_proof=false`,
    `prior_counted_as_proof=false`.
  - les scores de role/effet et les confidences restent des priorites
    candidates ; `confidence_counted_as_support=false`.
  - les tests suggeres restent generiques, par exemple comparer les actions sur
    le delta de centroide/relation d'une entite candidate.
- M1.G0.3 Action-effect abstraction :
  - pour chaque transition `frame_t -> frame_t+1`, l'action executee est liee a
    des deltas d'entites suivies.
  - familles d'effet candidates :
    `move_entity`, `transform_entity`, `create_entity`, `delete_entity`,
    `change_relation`, `tick_latent`, `global_transition`, `no_effect`,
    `unknown`.
  - separation explicite :
    - `entity_delta_summary` pour centroide/bbox/shape/creation/suppression.
    - `relation_delta_summary` pour un signal relationnel provisoire
      multi-entites, en attendant M1.G0.4.
    - `hud_or_latent_delta_summary` pour les changements de candidats HUD ou
      compteurs monotones.
    - `global_delta_summary` pour les pixels changes.
  - les actions demandees mais indisponibles dans la sweep ne produisent pas de
    pseudo-effets.
- M1.G0.4 Dynamic relation graph :
  - construction de graphes relationnels par frame :
    `relation_graphs_by_frame`.
  - relations V1 :
    `touches`, `adjacent_to`, `overlaps`, `above`, `below`, `left_of`,
    `right_of`, `contains`, `inside`, `near`, `near_edge`, `same_color`,
    `same_shape_signature`, `distance`.
  - extraction de deltas relationnels par transition :
    `relation_delta_rows`.
  - types de deltas V1 :
    `contact_created`, `contact_removed`, `distance_decreases`,
    `distance_increases`, `near_relation_created`,
    `near_relation_removed`, `containment_created`,
    `containment_removed`, `edge_relation_created`,
    `edge_relation_removed`, `relation_created`, `relation_removed`.
  - emission de hypotheses `mechanic_family=relation_change`.
  - garde-fou de volume :
    - graphe V1 borne a 16 entites prioritaires par frame.
    - hypotheses relationnelles filtrees par priorite role/delta.
    - cap `max_hypotheses=512`.
  - les deltas relationnels gardent `relation_delta_counted_as_confirmation=false`.
- M1.G0.5 Dynamic invariant detector :
  - detection de candidats dynamiques generiques :
    `monotone_counter`, `irreversible_change`, `exogenous_motion`,
    `phase_indicator`.
  - sources V1 :
    - `entity_size_sequence` pour HUD/compteurs lies a une entite.
    - `entity_centroid_sequence` pour derive/mouvement force candidat.
    - `entity_lifecycle_or_size_sequence` pour changements irreversibles.
    - `global_color_count_sequence` pour compteurs/couleurs ressources.
    - `global_grid_signature_sequence` pour cycles/phases.
  - le HUD devient un cas de `monotone_counter`, pas une branche speciale.
  - les sorties gardent `invariant_score_counted_as_support=false` et
    `invariant_counted_as_confirmation=false`.
  - la semantique exacte d'un compteur reste non confirmee :
    `remaining_semantics_unknown=true`.
- Runner bp35 :
  - capture une courte trajectoire visuelle `generic_action_sweep_observation`
    avec `ACTION3`, `ACTION4`, `ACTION1`, `ACTION2`, `ACTION6`.
  - cette source remplace la trajectoire P3 specialisee pour M1.G0, car une
    trajectoire qui exploite surtout `ACTION6` ne donne pas assez de mouvement
    pour proposer un acteur candidat.
  - l'action sweep est une observation perceptive M1, pas une policy et pas une
    experience scientifique.
- Run reel :
  - sortie : `diagnostics/m1/general_mechanic_candidates.json`.
  - `frames_consumed=21`.
  - `actions_consumed=20`.
  - `entities_tracked=230`.
  - `controllable_actor_candidates=1`.
  - `timer_or_hud_candidates=1`.
  - `actions_analyzed=3`.
  - `action_effect_rows=3`.
  - `entity_role_hypotheses_generated=308`.
  - `action_effect_hypotheses_generated=18`.
  - `relation_graph_frames=21`.
  - `relation_delta_rows=5634`.
  - `relation_change_hypotheses_generated=512`.
  - `dynamic_invariant_candidates=114`.
  - `dynamic_invariant_hypotheses_generated=114`.
  - familles invariantes :
    `monotone_counter=7`, `irreversible_change=104`,
    `exogenous_motion=3`.
  - deltas relationnels notables :
    `contact_created=86`, `contact_removed=100`,
    `distance_decreases=186`, `distance_increases=117`.
  - familles d'effet observees :
    `move_entity`, `transform_entity`, `create_entity`, `delete_entity`,
    `change_relation`, `tick_latent`.
  - `mechanic_hypotheses_generated=952`.
  - `ready_for_m3_g0=true`.
  - `support=0`, `wrong_confirmations=0`.
- Lecture :
  - M1.G0 detecte maintenant un acteur controlable candidat et un HUD candidat
    sans les confirmer.
  - M1.G0.3 lie les actions aux deltas d'entites/latents, mais le
    `change_relation` global reste un signal action-effect grossier.
  - M1.G0.4 remplace ce signal grossier par des deltas relationnels explicites,
    par exemple `distance_decreases` ou `contact_created` entre un acteur
    candidat et une autre entite.
  - M1.G0.5 expose le HUD bp35 comme `monotone_counter` generique
    `E001`, avec `action_correlation_score=1.0`, sans le confirmer comme
    horizon terminal.
  - La prochaine brique naturelle est M3.G0.1 `Generic mechanic experiment
    planner`, qui doit selectionner un petit nombre d'hypotheses
    `entity_role`, `action_effect`, `relation_change` et `dynamic_invariant`
    a tester.
- Verification locale :
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests\\test_m1_general_mechanic_abstraction.py -q` -> 10 tests passes.
  - `ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.general_mechanic_abstraction --out diagnostics\\m1\\general_mechanic_candidates.json`.

## Prochaines actions

1. Ne pas demarrer M2.
2. Considerer M1 comme conceptuellement complet apres M1.3l :
   generation, grounding, execution, priorisation, ledger et premier support
   controle existent sans confirmation prematuree.
3. Ne pas ajouter un nouveau sous-module M1.3x tant qu'un besoin experimental
   precis n'est pas identifie.
4. Prochaine bifurcation recommandee : cadrer M3 minimal
   `Scientific Planning`, centre sur le choix du prochain test controle a partir
   des entrees ledger et des `support_events`/`contradiction_events`.
5. Garder une etape de securite avant M3 : verifier que
   `controlled_experiment_results.json` peut etre relu comme evidence
   experimentale sans augmenter `support` ni creer de verdict automatique.
6. Ne pas integrer M1.3d-b couleur-source a A25 tant que
   `preconditions_found=0` sur les nouvelles paires M1.
7. Ne pas poursuivre aveuglement la preparation couleur-source sur bp35/cd82 :
   M1.3d-c indique un mismatch de representation.
8. Traiter cd82 separement comme probleme de disponibilite d'action/precondition :
   5/12 candidats sont testables, mais les candidats forts de trace
   `ACTION5/ACTION6` sont bloques par `action_not_available`.
9. Garder dc22 comme controle positif du schema couleur-source partiellement
   valide.
10. Garder M1.3i/M1.3j/M1.3k/M1.3l opt-in, sans branchement par defaut au
   stress-test A31 tant que M3 n'a pas defini une politique de sequence
   experimentale.
11. Ajouter une ecriture incrementale a `stress_test_a31bis.py` ou lancer le run
   complet avec une fenetre nettement plus large pour obtenir
   `diagnostics/m1_stress_test_a31bis.json`.
12. Garder `wrong_confirmations == 0` comme garde-fou non negociable.
