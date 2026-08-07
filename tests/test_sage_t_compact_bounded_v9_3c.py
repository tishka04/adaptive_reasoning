from __future__ import annotations

from theory.sage_t import compact_bounded_v9_3c as compact
from theory.sage_t.contracts import ActionCandidate


def test_compact_manifest_binds_abort_and_winning_prefix_audit() -> None:
    manifest = compact.load_manifest()

    assert manifest["controller_caps"]["maximum_sequences"] == 16
    assert manifest["controller_caps"]["maximum_executor_cache_entries"] == 512
    assert manifest["compact_prefix_audit"]["passed"] is True
    assert manifest["compact_prefix_audit"]["correct_first_action"] == 9
    assert manifest["firewall"]["active_authority"] is False


def test_compact_controller_clears_pure_executor_cache_on_branch() -> None:
    controller = compact.build_controller(compact.load_manifest())
    controller.executor._step_cache[("p", "s", "a")] = object()  # type: ignore[assignment]

    controller.start_branch()

    assert controller.executor.summary()["cache_entries"] == 0
    macros = controller._latest_proposal.plan_sequences
    assert macros == ()
    assert ActionCandidate("ACTION6", {"x": 4, "y": 29}).key
