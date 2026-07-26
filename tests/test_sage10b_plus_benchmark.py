"""Paired SAGE.10b+ procedural benchmark tests."""

from theory.sage10b_plus_benchmark import run_sage10b_plus_benchmark


def test_sage10b_plus_procedural_pairs_pass_all_gates(tmp_path):
    output = tmp_path / "sage10b-plus.json"
    payload = run_sage10b_plus_benchmark(write_path=output)

    assert payload["schema_version"] == (
        "sage.sage10b_plus_procedural.v1"
    )
    assert payload["all_procedural_gates_passed"] is True
    assert all(payload["gates"].values())
    assert payload["relay"]["active"]["terminal_credits"] == 1
    assert payload["relay"]["ablated"]["terminal_credits"] == 0
    assert (
        payload["generalized_stall"]["active"]["interventions"]
        == 1
    )
    assert (
        payload["generalized_stall"]["ablated"]["interventions"]
        == 0
    )
    assert output.exists()
