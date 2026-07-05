# Cognitive Phase Taxonomy Audit

Dataset rows: 5605.

| phase | n | games | top game share | F1 state | F1 by-game | hyp n | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `explore_unknown` | 512 | 6 | 0.734 | 0.048 | 0.000 | 10 | meta-class probable: split into more specific exploration phases |
| `test_move` | 596 | 5 | 0.547 | 0.192 | 0.115 | 22 | relatively atomic, but confused with exploration/recovery |
| `test_click` | 238 | 5 | 0.630 | 0.075 | 0.000 | 19 | plausible concept, but undercovered and mixed with interaction |
| `test_interaction` | 586 | 6 | 0.292 | 0.033 | 0.069 | 19 | plausible concept, but too broad/mixed |
| `probe_object` | 652 | 5 | 0.416 | 0.183 | 0.003 | 16 | meta-class probable: split by object/probe type |
| `reach_target` | 664 | 6 | 0.411 | 0.131 | 0.000 | 20 | global intent, probably not readable from state-only features |
| `avoid_danger` | 275 | 3 | 0.651 | 0.000 | 0.000 | 6 | undercovered and game-dependent semantics |
| `repeat_success` | 1909 | 6 | 0.423 | 0.460 | 0.508 | 17 | fairly coherent, but dominant |
| `recover_after_failure` | 109 | 2 | 0.725 | 0.239 | 0.000 | 2 | possible coherent concept, but too little coverage |
| `none` | 64 | 1 | 1.000 | 0.000 | 0.000 | 1 | technical/noise label: exclude or handle separately |

## Per Phase Details

### explore_unknown

- support: 512 rows, 6 games, 12 episodes
- F1: state=0.048, by_game=0.000
- game counts: {'dc22-4c9bff3e': 376, 'ft09-0d8bbf25': 54, 'cd82-fb555c5d': 37, 'bp35-0a0ad940': 23, 'ar25-e3c63847': 21, 'cn04-65d47d14': 1}
- main state confusions: [[440, 'repeat_success', 0.859], [26, 'test_click', 0.051], [20, 'test_move', 0.039]]
- main by-game confusions: [[485, 'repeat_success', 0.947], [12, 'avoid_danger', 0.023], [8, 'test_move', 0.016]]
- top hypotheses: [['__none__', 293], ['fancy_colors_squares_are_teleporters', 125], ['colors_must_differ_is_shape_to_reproduce_says_so', 47], ['doted_line_act_as_miror', 18], ['black_and_green_squares_expand_into_4_new_pieces', 8]]
- top actions: [['ACTION6', 222], ['ACTION4', 86], ['ACTION2', 86], ['ACTION1', 63], ['ACTION3', 51]]
- flags: game-concentrated, hypothesis-heterogeneous, no-cross-game-generalization, weak-even-in-episode-cv
- verdict: meta-class probable: split into more specific exploration phases

### test_move

- support: 596 rows, 5 games, 13 episodes
- F1: state=0.192, by_game=0.115
- game counts: {'ar25-e3c63847': 326, 'cd82-fb555c5d': 151, 'cn04-65d47d14': 73, 'dc22-4c9bff3e': 27, 'bp35-0a0ad940': 19}
- main state confusions: [[151, 'recover_after_failure', 0.253], [78, 'explore_unknown', 0.131], [76, 'none', 0.128]]
- main by-game confusions: [[518, 'repeat_success', 0.869], [12, 'test_interaction', 0.02], [7, 'explore_unknown', 0.012]]
- top hypotheses: [['__none__', 176], ['you_need_to_match_at_least_one_shape', 157], ['which_action_correspond_to_which_move', 77], ['action5_rotates_cursor', 43], ['action1_moves_red_linned_black_sqaure_left', 32]]
- top actions: [['ACTION2', 168], ['ACTION1', 122], ['ACTION3', 112], ['ACTION4', 109], ['ACTION5', 53]]
- flags: hypothesis-heterogeneous
- verdict: relatively atomic, but confused with exploration/recovery

### test_click

- support: 238 rows, 5 games, 12 episodes
- F1: state=0.075, by_game=0.000
- game counts: {'cd82-fb555c5d': 150, 'cn04-65d47d14': 27, 'bp35-0a0ad940': 25, 'dc22-4c9bff3e': 21, 'ft09-0d8bbf25': 15}
- main state confusions: [[105, 'test_interaction', 0.441], [56, 'test_move', 0.235], [25, 'explore_unknown', 0.105]]
- main by-game confusions: [[199, 'repeat_success', 0.836], [37, 'test_move', 0.155], [2, 'test_interaction', 0.008]]
- top hypotheses: [['actions_6_click_on_colored_box_to_select_color', 46], ['click_on_the_top_cursor_to_enable_it', 44], ['action_5_to_apply_color_on_central_shape_from_point_of_view_of_the_cursor', 32], ['click_on_yellow_black square_moves_black_line_below', 24], ['action6_click_on_shapes_in_grey_area_to_open_barriers', 14]]
- top actions: [['ACTION6', 83], ['ACTION1', 50], ['ACTION3', 34], ['ACTION2', 30], ['ACTION4', 29]]
- flags: hypothesis-heterogeneous, no-cross-game-generalization, weak-even-in-episode-cv
- verdict: plausible concept, but undercovered and mixed with interaction

### test_interaction

- support: 586 rows, 6 games, 15 episodes
- F1: state=0.033, by_game=0.069
- game counts: {'cd82-fb555c5d': 171, 'dc22-4c9bff3e': 144, 'bp35-0a0ad940': 127, 'cn04-65d47d14': 111, 'ar25-e3c63847': 19, 'ft09-0d8bbf25': 14}
- main state confusions: [[147, 'repeat_success', 0.251], [138, 'test_move', 0.235], [101, 'reach_target', 0.172]]
- main by-game confusions: [[479, 'repeat_success', 0.817], [85, 'test_move', 0.145]]
- top hypotheses: [['find_way_via_removing_bleu_cells', 98], ['shapes_can_overlap', 96], ['use_clamp_to_move_colored_bridges', 92], ['how_to_use_the_upper_cursor', 80], ['action_5_to_apply_color_on_central_shape_from_point_of_view_of_the_cursor', 74]]
- top actions: [['ACTION6', 171], ['ACTION4', 110], ['ACTION3', 96], ['ACTION2', 90], ['ACTION1', 70]]
- flags: hypothesis-heterogeneous, weak-even-in-episode-cv
- verdict: plausible concept, but too broad/mixed

### probe_object

- support: 652 rows, 5 games, 11 episodes
- F1: state=0.183, by_game=0.003
- game counts: {'dc22-4c9bff3e': 271, 'ft09-0d8bbf25': 148, 'bp35-0a0ad940': 88, 'cn04-65d47d14': 80, 'ar25-e3c63847': 65}
- main state confusions: [[363, 'repeat_success', 0.557], [85, 'test_move', 0.13], [35, 'reach_target', 0.054]]
- main by-game confusions: [[550, 'repeat_success', 0.844], [81, 'test_move', 0.124], [20, 'reach_target', 0.031]]
- top hypotheses: [['__none__', 152], ['change_color_of_shapes_to_move_in_them', 102], ['match_centered_shapes_with_colors', 75], ['colors_must_differ_is_shape_to_reproduce_says_so', 73], ['click_ongrey_side_colored_shapes_to_move_objects_in_yellow_pane', 67]]
- top actions: [['ACTION6', 269], ['ACTION3', 105], ['ACTION4', 99], ['ACTION2', 91], ['ACTION1', 77]]
- flags: hypothesis-heterogeneous, no-cross-game-generalization
- verdict: meta-class probable: split by object/probe type

### reach_target

- support: 664 rows, 6 games, 15 episodes
- F1: state=0.131, by_game=0.000
- game counts: {'dc22-4c9bff3e': 273, 'bp35-0a0ad940': 159, 'ar25-e3c63847': 140, 'cd82-fb555c5d': 55, 'cn04-65d47d14': 33, 'ft09-0d8bbf25': 4}
- main state confusions: [[301, 'repeat_success', 0.453], [95, 'test_interaction', 0.143], [65, 'test_move', 0.098]]
- main by-game confusions: [[616, 'repeat_success', 0.928], [33, 'test_move', 0.05], [11, 'test_interaction', 0.017]]
- top hypotheses: [['__none__', 141], ['fancy_colors_squares_are_teleporters', 121], ['action3_left_action4_right', 98], ['use_brdiges_to_reach_target', 91], ['doted_line_act_as_miror', 53]]
- top actions: [['ACTION6', 193], ['ACTION4', 124], ['ACTION2', 119], ['ACTION3', 116], ['ACTION1', 92]]
- flags: hypothesis-heterogeneous, no-cross-game-generalization
- verdict: global intent, probably not readable from state-only features

### avoid_danger

- support: 275 rows, 3 games, 5 episodes
- F1: state=0.000, by_game=0.000
- game counts: {'cd82-fb555c5d': 179, 'bp35-0a0ad940': 89, 'ft09-0d8bbf25': 7}
- main state confusions: [[152, 'test_interaction', 0.553], [39, 'test_move', 0.142], [25, 'reach_target', 0.091]]
- main by-game confusions: [[275, 'repeat_success', 1.0]]
- top hypotheses: [['yelow_bar_bellow_represents_the_moves_and_clicks_left_before_game_over', 179], ['action3_left_action4_right', 48], ['use_clicks_to_find_way_using_gravity', 22], ['find_way_via_removing_bleu_cells', 17], ['match_centered_shapes_with_colors', 7]]
- top actions: [['ACTION6', 67], ['ACTION4', 67], ['ACTION3', 59], ['ACTION2', 34], ['ACTION1', 33]]
- flags: no-cross-game-generalization, weak-even-in-episode-cv
- verdict: undercovered and game-dependent semantics

### repeat_success

- support: 1909 rows, 6 games, 14 episodes
- F1: state=0.460, by_game=0.508
- game counts: {'dc22-4c9bff3e': 808, 'ar25-e3c63847': 455, 'bp35-0a0ad940': 341, 'ft09-0d8bbf25': 221, 'cn04-65d47d14': 76, 'cd82-fb555c5d': 8}
- main state confusions: [[241, 'recover_after_failure', 0.126], [173, 'probe_object', 0.091], [150, 'reach_target', 0.079]]
- main by-game confusions: [[81, 'test_move', 0.042], [37, 'reach_target', 0.019], [13, 'avoid_danger', 0.007]]
- top hypotheses: [['__none__', 691], ['shapes_mirrored_by_doted_lines', 269], ['use_brdiges_to_reach_target', 193], ['use_combination_of_clicks_and_moves_to_reach_orange_cross', 165], ['click_based_game', 134]]
- top actions: [['ACTION6', 651], ['ACTION2', 331], ['ACTION4', 318], ['ACTION1', 279], ['ACTION3', 261]]
- flags: hypothesis-heterogeneous, dominant/stable-signal
- verdict: fairly coherent, but dominant

### recover_after_failure

- support: 109 rows, 2 games, 3 episodes
- F1: state=0.239, by_game=0.000
- game counts: {'bp35-0a0ad940': 79, 'ar25-e3c63847': 30}
- main state confusions: [[31, 'probe_object', 0.284], [1, 'repeat_success', 0.009]]
- main by-game confusions: [[108, 'repeat_success', 0.991], [1, 'reach_target', 0.009]]
- top hypotheses: [['black_and_green_squares_expand_into_4_new_pieces', 79], ['doted_line_act_as_miror', 30]]
- top actions: [['ACTION6', 49], ['ACTION3', 19], ['ACTION4', 18], ['ACTION5', 12], ['ACTION2', 10]]
- flags: rare, undercovered, game-concentrated
- verdict: possible coherent concept, but too little coverage

### none

- support: 64 rows, 1 games, 1 episodes
- F1: state=0.000, by_game=0.000
- game counts: {'ar25-e3c63847': 64}
- main state confusions: [[58, 'probe_object', 0.906], [6, 'repeat_success', 0.094]]
- main by-game confusions: [[64, 'repeat_success', 1.0]]
- top hypotheses: [['doted_line_act_as_miror', 64]]
- top actions: [['ACTION3', 24], ['ACTION5', 18], ['ACTION1', 12], ['ACTION2', 8], ['ACTION4', 2]]
- flags: rare, undercovered, game-concentrated
- verdict: technical/noise label: exclude or handle separately
