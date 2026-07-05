# Cognitive Taxonomy v1 vs v2 Experiment

Same dataset/features/model. Taxonomy v2 remaps labels only and drops `none`.

| run | rows | labels | group_by | accuracy | balanced_acc | macro_f1 | weighted_f1 | baseline_macro_f1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `v1_episode` | 5605 | 10 | episode | 0.2471 | 0.1808 | 0.1362 | 0.2296 | 0.0508 |
| `v2_episode` | 5541 | 17 | episode | 0.2269 | 0.1352 | 0.1001 | 0.2076 | 0.0301 |
| `v1_by_game` | 5605 | 10 | game | 0.3302 | 0.1061 | 0.0695 | 0.1928 | 0.0508 |
| `v2_by_game` | 5541 | 17 | game | 0.1855 | 0.0376 | 0.0332 | 0.1515 | 0.0301 |

## Criteria

- by_game_macro_f1_delta: -0.0363
- by_game_accuracy_delta: -0.1447
- episode_macro_f1_delta: -0.0361
- repeat_success_by_game_delta: -0.1242

## v2 Label Counts

- `explore_movement`: 254
- `explore_click`: 6
- `explore_boundary`: 42
- `explore_object`: 161
- `explore_global_rule`: 49
- `test_move`: 596
- `test_control_or_activation`: 238
- `test_object_interaction`: 586
- `probe_control_object`: 189
- `probe_color_object`: 113
- `probe_hazard_object`: 3
- `probe_bridge_or_path`: 126
- `probe_matching_object`: 221
- `reach_target`: 664
- `avoid_danger`: 275
- `repeat_success`: 1909
- `recover_after_failure`: 109

## v2 By-Game Nonzero F1

- `test_move`: f1=0.1193, recall=0.1124, support=596
- `test_object_interaction`: f1=0.0607, recall=0.0341, support=586
- `repeat_success`: f1=0.3839, recall=0.4929, support=1909

## v2 By-Game Main Confusions

- `explore_movement` -> [(185, 'recover_after_failure', 0.728), (64, 'repeat_success', 0.252), (3, 'avoid_danger', 0.012)]
- `explore_click` -> [(5, 'repeat_success', 0.833), (1, 'test_move', 0.167)]
- `explore_boundary` -> [(37, 'recover_after_failure', 0.881), (5, 'repeat_success', 0.119)]
- `explore_object` -> [(154, 'recover_after_failure', 0.957), (7, 'repeat_success', 0.043)]
- `explore_global_rule` -> [(35, 'test_move', 0.714), (14, 'repeat_success', 0.286)]
- `test_move` -> [(480, 'repeat_success', 0.805), (27, 'explore_click', 0.045), (20, 'test_object_interaction', 0.034)]
- `test_control_or_activation` -> [(182, 'repeat_success', 0.765), (30, 'test_move', 0.126), (21, 'explore_click', 0.088)]
- `test_object_interaction` -> [(331, 'repeat_success', 0.565), (96, 'recover_after_failure', 0.164), (81, 'test_move', 0.138)]
- `probe_control_object` -> [(109, 'repeat_success', 0.577), (67, 'explore_click', 0.354), (13, 'test_move', 0.069)]
- `probe_color_object` -> [(102, 'explore_click', 0.903), (11, 'repeat_success', 0.097)]
- `probe_hazard_object` -> [(3, 'repeat_success', 1.0)]
- `probe_bridge_or_path` -> [(102, 'explore_click', 0.81), (24, 'repeat_success', 0.19)]
- `probe_matching_object` -> [(126, 'test_move', 0.57), (95, 'repeat_success', 0.43)]
- `reach_target` -> [(339, 'repeat_success', 0.511), (150, 'explore_click', 0.226), (123, 'recover_after_failure', 0.185)]
- `avoid_danger` -> [(274, 'repeat_success', 0.996), (1, 'reach_target', 0.004)]
- `repeat_success` -> [(438, 'explore_click', 0.229), (370, 'recover_after_failure', 0.194), (143, 'test_move', 0.075)]
- `recover_after_failure` -> [(109, 'repeat_success', 1.0)]