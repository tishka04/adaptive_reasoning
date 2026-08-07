from __future__ import annotations

from theory.sage_t import bounded_active_v9_3b as r2
from theory.sage_t.contracts import ActionCandidate


def test_live_spatial_macros_are_west_first_and_bounded() -> None:
    east = ActionCandidate("ACTION6", {"x": 56, "y": 29})
    west = ActionCandidate("ACTION6", {"x": 4, "y": 29})

    macros = r2.live_spatial_macros((east, west))

    assert macros[0] == (west, west, west)
    assert macros[1] == (west, west)
    assert len(macros) <= 8
    assert all(len(macro) <= 8 for macro in macros)


def test_retry_manifest_registers_only_the_live_grounding_changes() -> None:
    manifest = r2.load_manifest()

    assert manifest["registered_changes"] == [
        "fallback repeat macros from materialized spatial legal actions",
        "westmost tie-break only among equal-utility admissible actions",
        "false-high metric matched to the action actually executed",
    ]
    assert manifest["firewall"]["holdout_opened"] is False
