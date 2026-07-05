from theory.m1.grounding_autopsy import (
    GroundingFunnel,
    grounding_block_reason,
    render_grounding_autopsy_markdown,
)


def test_grounding_block_reason_prioritizes_live_source_before_predicates():
    assert (
        grounding_block_reason(
            live_color_compatible=False,
            target_live_present=True,
            preferred_predicate_count=3,
        )
        == "source_not_selectable_for_action"
    )
    assert (
        grounding_block_reason(
            live_color_compatible=True,
            target_live_present=False,
            preferred_predicate_count=3,
        )
        == "target_not_present_in_live_grid"
    )
    assert (
        grounding_block_reason(
            live_color_compatible=True,
            target_live_present=True,
            preferred_predicate_count=1,
        )
        == "not_enough_preferred_predicates"
    )
    assert (
        grounding_block_reason(
            live_color_compatible=True,
            target_live_present=True,
            preferred_predicate_count=2,
        )
        == "agenda_eligible"
    )


def test_grounding_autopsy_markdown_includes_dc22_positive_pair():
    markdown = render_grounding_autopsy_markdown(
        {
            "summary_table": [
                {
                    "game_id": "dc22-4c9bff3e",
                    "new_pairs": 6,
                    "live_compatible": 3,
                    "entering_agenda": 1,
                    "relation_candidate_count": 4,
                    "env_actions": 1,
                    "error": "",
                }
            ],
            "games": [
                {
                    "game_id": "dc22-4c9bff3e",
                    "grounding_funnel": GroundingFunnel(
                        pairs_discovered=20,
                        pairs_target_present=8,
                        pairs_actionable_source=4,
                        pairs_blocked_by_unselectable_source=7,
                        pairs_live_compatible=3,
                        pairs_with_2_preferred_predicates=10,
                        pairs_entering_agenda=2,
                        pairs_generating_env_action=1,
                    ).to_dict(),
                    "new_pair_grounding_funnel": GroundingFunnel(
                        pairs_discovered=6,
                        pairs_target_present=4,
                        pairs_actionable_source=3,
                        pairs_blocked_by_unselectable_source=2,
                        pairs_live_compatible=3,
                        pairs_with_2_preferred_predicates=6,
                        pairs_entering_agenda=1,
                        pairs_generating_env_action=0,
                    ).to_dict(),
                    "ranked_live_compatible_new_pairs": [
                        {
                            "action": "ACTION1",
                            "source_color": 8,
                            "target_color": 12,
                            "support": 42,
                            "preferred_predicates": [
                                "same_shape",
                                "aligned_with",
                            ],
                            "live_preferred_predicates": [
                                "same_shape",
                                "aligned_with",
                            ],
                            "entering_agenda": True,
                        }
                    ],
                }
            ],
            "interpretation": ["dc22 is the positive control."],
        }
    )

    assert "| dc22-4c9bff3e | 6 | 3 | 1 | 4 | 1 |  |" in markdown
    assert "| dc22-4c9bff3e | 20 | 8 | 4 | 7 | 3 | 2 | 1 |" in markdown
    assert "| dc22-4c9bff3e | 6 | 4 | 3 | 2 | 3 | 1 | 0 |" in markdown
    assert "| ACTION1 | 8->12 | 42 | same_shape, aligned_with |" in markdown
    assert "dc22 is the positive control." in markdown
