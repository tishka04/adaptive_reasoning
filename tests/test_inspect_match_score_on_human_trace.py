from inspect_match_score_on_human_trace import _parse_pair_colors, _summarize_reports


def test_parse_pair_colors_accepts_space_or_comma_forms():
    assert _parse_pair_colors(["10", "11"]) == (10, 11)
    assert _parse_pair_colors(["10,11"]) == (10, 11)


def test_summarize_reports_counts_matched_pairs():
    summary = _summarize_reports(
        [
            {
                "match_score": {
                    "matched_pairs": 1,
                    "unmatched_second": 2,
                    "dotted_constraint_violations": 1,
                }
            },
            {
                "match_score": {
                    "matched_pairs": 3,
                    "unmatched_second": 0,
                    "dotted_constraint_violations": 0,
                }
            },
        ]
    )

    assert summary["max_matched_pairs"] == 3
    assert summary["min_unmatched_second"] == 0
    assert summary["matched_pairs_hist"] == {"1": 1, "3": 1}
