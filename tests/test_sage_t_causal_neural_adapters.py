from dataclasses import replace

import pytest
import torch

from tests.test_sage_t_causal_contract_executor import causal_program, initial_state
from tests.test_sage_t_causal_posterior_memory import matching_evidence
from theory.sage_t.causal.adapters import (
    CausalFamilyProposalAdapter,
    CausalProgramProposal,
    CausalProposalCoordinator,
    RouteReplayProposalAdapter,
    Sage9pProgramAdapter,
    Sage9pRelationProposal,
    causal_state_from_abstract,
)
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.executor import CausalExecutor
from theory.sage_t.causal.mechanisms import MechanismRegistry
from theory.sage_t.causal.neural import (
    GraphMaskedMechanismHead,
    SharedMechanismBank,
    TorchCategoricalMechanism,
    causal_mechanism_loss,
    module_content_hash,
)
from theory.sage_t.causal.posterior import CausalPosterior
from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact


def test_sage9p_and_route_adapters_are_proposal_only():
    proposal = Sage9pProgramAdapter().propose(
        Sage9pRelationProposal(
            proposal_id="sage9p_alignment",
            action_name="CLICK",
            source_variable="object.color",
            target_variable="target.color",
            relation_variable="pair.aligned",
        )
    )
    assert proposal.support == 0
    assert proposal.program is not None
    compiled = CausalExecutor().compile(proposal.program, action_catalog=("CLICK",))
    assert compiled.program.goal.success_predicate == "level.complete == true"

    route = RouteReplayProposalAdapter().action_program(
        (GroundedAction("CLICK"),), exact=True
    )
    assert route.source == "exact_route"
    assert CausalFamilyProposalAdapter().propose("stable_repeat", {}).support == 0


def test_new_proposals_are_rescored_on_existing_evidence():
    executor = CausalExecutor()
    posterior = CausalPosterior(executor=executor, mdl_beta=0.0)
    posterior.seed((causal_program(program_id="cycle"),))
    posterior.update(matching_evidence())
    rival = causal_program(program_id="identity", color_operator="identity")
    admitted = CausalProposalCoordinator().propose_into(
        posterior=posterior,
        proposals=(CausalProgramProposal(source="m2_or_llm", program=rival),),
        action_catalog=("CLICK",),
    )
    assert admitted == 1
    rival_particle = next(
        particle
        for particle in posterior.particles
        if particle.program.program_id == "identity"
    )
    assert rival_particle.evidence_ids == ("evidence-1",)
    assert rival_particle.latest_log_likelihood < 0.0


def test_graph_mask_and_shared_neural_module_are_enforced():
    torch.manual_seed(7)
    bank = SharedMechanismBank()
    head = GraphMaskedMechanismHead(
        parent_dim=2,
        action_dim=1,
        output_dim=2,
        hidden_dim=8,
    )
    bank.register_head("color_shared_v1", head)
    module = TorchCategoricalMechanism(
        bank=bank,
        module_id="color_shared_v1",
        parent_vocabulary=("red", "blue"),
        action_vocabulary=("CLICK",),
        output_vocabulary=("red", "blue"),
    )
    registry = MechanismRegistry()
    registry.register_neural("color_shared_v1", module)

    base = causal_program()
    neural_mechanism = replace(
        base.mechanisms[0],
        operator_type="neural_local_transition",
        neural_module_id="color_shared_v1",
        symbolic_fallback="cycle_attribute",
    )
    first = replace(
        base,
        program_id="neural_first",
        mechanisms=(neural_mechanism,) + base.mechanisms[1:],
    )
    second = replace(first, program_id="neural_second")
    executor = CausalExecutor(mechanism_registry=registry)
    left = executor.predict_step(first, initial_state(), GroundedAction("CLICK"))
    right = executor.predict_step(second, initial_state(), GroundedAction("CLICK"))
    assert left.state_after.value("object.color").probabilities == right.state_after.value("object.color").probabilities
    assert set(left.state_after.value("object.color").probabilities) == {'"red"', '"blue"'}
    assert module_content_hash(bank)

    with pytest.raises(ValueError, match="causal mask"):
        head(torch.zeros(1, 3), torch.zeros(1, 1))


def test_causal_neural_loss_contains_branch_invariance_sparsity_and_calibration():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    branch_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    parent_gate = torch.tensor([1.0, 0.2], requires_grad=True)
    losses = causal_mechanism_loss(
        logits,
        targets,
        branch_logits=branch_logits,
        branch_targets=torch.tensor([0]),
        invariant_pairs=(logits[:1], logits[1:]),
        parent_gate=parent_gate,
    )
    assert torch.isfinite(losses.total)
    assert losses.branch.item() > 0.0
    assert losses.calibration.item() > 0.0
    losses.total.backward()
    assert logits.grad is not None
    assert parent_gate.grad is not None


def test_causal_state_is_bounded_and_exposes_role_centers():
    player = AbstractEntity(
        "player_local",
        ("object", "player"),
        center=(38.0, 22.0),
    )
    facts = frozenset(
        GroundFact("exists", (f"entity_{index}",)) for index in range(2000)
    )
    state = causal_state_from_abstract(
        AbstractState(entities=(player,), true_facts=facts)
    )
    fact_variables = [key for key in state.variables if key.startswith("fact.")]
    assert len(fact_variables) <= 512
    assert state.value("summary.role.player.center").mode == [38.0, 22.0]
    assert state.value("summary.role.player.count").mode == 1
