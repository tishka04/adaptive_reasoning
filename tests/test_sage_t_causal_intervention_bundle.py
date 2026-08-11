from dataclasses import replace

from tests.test_sage_t_causal_contract_executor import causal_program
from tests.test_sage_t_causal_posterior_memory import matching_evidence
from theory.sage_t.causal.contracts import (
    ActionProgram,
    GroundedAction,
    InterventionBranch,
    InterventionBundle,
)
from theory.sage_t.causal.replay import InterventionBundleRunner
from theory.sage_t.causal.runtime import CausalRuntime


class FakeReplayEnvironment:
    def __init__(self, *, diverge_at=-1):
        self.calls = 0
        self.diverge_at = diverge_at

    def reset_and_replay(self, prefix):
        frame = {"branch": self.calls}
        self.calls += 1
        return frame

    def state_hash(self, frame):
        return "diverged" if frame["branch"] == self.diverge_at else "exact-prefix-hash"

    def legal_action_names(self, frame):
        return ("CLICK",)

    def execute(self, frame, action):
        return {"branch": frame["branch"], "executed": action.key}


def bundle():
    prefix = ActionProgram((GroundedAction("CLICK"),), source="exact_route")
    return InterventionBundle(
        prefix=prefix,
        prefix_hash="exact-prefix-hash",
        branches=(
            InterventionBranch(
                GroundedAction("CLICK", {"probe": 1}),
                {"program-a": ("blue",), "program-b": ("red",)},
            ),
            InterventionBranch(
                GroundedAction("CLICK", {"probe": 2}),
                {"program-a": ("red",), "program-b": ("blue",)},
            ),
        ),
    )


def test_bundle_requires_same_prefix_and_updates_after_each_branch():
    runtime = CausalRuntime()
    runtime.seed(
        (
            causal_program(program_id="cycle", color_operator="cycle_attribute"),
            causal_program(program_id="identity", color_operator="identity"),
        )
    )
    runner = InterventionBundleRunner(runtime=runtime)

    def evidence_builder(before, action, after, index):
        return replace(
            matching_evidence(evidence_id=f"branch-{index}"),
            action=action,
            context_id=f"context-{index}",
        )

    result = runner.run(
        bundle(),
        environment=FakeReplayEnvironment(),
        evidence_builder=evidence_builder,
    )
    assert result.status == "BUNDLE_COMPLETE"
    assert result.predictions_registered_before_execution is True
    assert len(result.branches) == 2
    assert len(runtime.posterior.evidence) == 2


def test_bundle_fails_closed_before_divergent_branch_action():
    runtime = CausalRuntime()
    runtime.seed((causal_program(),))
    result = InterventionBundleRunner(runtime=runtime).run(
        bundle(),
        environment=FakeReplayEnvironment(diverge_at=1),
        evidence_builder=lambda before, action, after, index: replace(
            matching_evidence(evidence_id=f"branch-{index}"), action=action
        ),
    )
    assert result.status == "PREFIX_HASH_MISMATCH"
    assert len(result.branches) == 1
    assert len(runtime.posterior.evidence) == 1
