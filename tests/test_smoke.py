"""
Smoke test — verifies all modules import correctly and the full
reasoning loop runs end-to-end without GPU or LLM.
"""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def test_imports():
    """All package modules import without error."""
    from v4_1_reasoning_system.parser.task_schema import TaskObject, Constraint, Objective
    from v4_1_reasoning_system.parser.llm_parser import LLMParser
    from v4_1_reasoning_system.world_model.encoder import StateEncoder
    from v4_1_reasoning_system.world_model.predictor import TransitionPredictor, ActionEncoder
    from v4_1_reasoning_system.world_model.aux_heads import AuxiliaryHeads, AuxPredictions
    from v4_1_reasoning_system.router.candidate_generator import CandidateGenerator, ReasoningCandidate
    from v4_1_reasoning_system.router.ebm_router import EBMRouter, RuleBasedRouter, RoutingDecision
    from v4_1_reasoning_system.router.routing_train import RouterTrainer
    from v4_1_reasoning_system.solvers.base import BaseSolver, SolverResult
    from v4_1_reasoning_system.solvers.hierarchical_solver import HierarchicalSolver
    from v4_1_reasoning_system.solvers.global_opt_solver import GlobalOptSolver
    from v4_1_reasoning_system.solvers.repair_solver import RepairSolver
    from v4_1_reasoning_system.solvers.llm_codegen_solver import LLMCodegenSolver
    from v4_1_reasoning_system.verifier.base import BaseVerifier, VerificationResult
    from v4_1_reasoning_system.verifier.planning_verifier import PlanningVerifier
    from v4_1_reasoning_system.verifier.scheduling_verifier import SchedulingVerifier
    from v4_1_reasoning_system.verifier.dispatch_verifier import DispatchVerifier
    from v4_1_reasoning_system.verifier.code_verifier import CodeVerifier
    from v4_1_reasoning_system.memory.replay import ReplayBuffer, TrajectoryRecord, StepRecord
    from v4_1_reasoning_system.memory.retrieval import EpisodicRetriever, StructuralMemory
    from v4_1_reasoning_system.orchestration.controller import ReasoningController
    from v4_1_reasoning_system.orchestration.state_update import StateManager, ReasoningState
    from v4_1_reasoning_system.training.collect_rollouts import RolloutCollector
    from v4_1_reasoning_system.training.train_world_model import WorldModelTrainer
    from v4_1_reasoning_system.training.train_router import train_router_from_buffer


def test_task_schema():
    """TaskObject creation and properties."""
    from v4_1_reasoning_system.parser.task_schema import (
        TaskObject, Constraint, ConstraintType, Objective, ObjectiveSense, DomainGuess
    )

    task = TaskObject(
        raw_input="test problem",
        domain=DomainGuess.PLANNING,
        description="A test planning problem",
        entities=["a", "b", "c"],
        constraints=[
            Constraint(name="c1", description="hard constraint", ctype=ConstraintType.HARD),
            Constraint(name="c2", description="soft constraint", ctype=ConstraintType.SOFT),
        ],
        objective=Objective(description="minimize cost", sense=ObjectiveSense.MINIMIZE),
    )

    assert task.domain == DomainGuess.PLANNING
    assert len(task.hard_constraints) == 1
    assert len(task.soft_constraints) == 1
    assert not task.has_ambiguity
    assert "planning" in task.summary()


def test_state_encoder():
    """StateEncoder forward pass."""
    from v4_1_reasoning_system.world_model.encoder import StateEncoder

    enc = StateEncoder(latent_dim=64)
    z = enc(
        torch.randn(2, 32),  # task
        torch.randn(2, 32),  # solution
        torch.randn(2, 16),  # feedback
        torch.randn(2, 64),  # history
    )
    assert z.shape == (2, 64)


def test_transition_predictor():
    """TransitionPredictor forward and batch prediction."""
    from v4_1_reasoning_system.world_model.predictor import TransitionPredictor

    pred = TransitionPredictor(latent_dim=64, action_dim=16)
    z_t = torch.randn(2, 64)
    action = torch.randn(2, 16)
    z_hat = pred(z_t, action)
    assert z_hat.shape == (2, 64)

    # Batch prediction
    actions_batch = torch.randn(2, 4, 16)  # 4 candidates
    z_hats = pred.predict_batch(z_t, actions_batch)
    assert z_hats.shape == (2, 4, 64)


def test_action_encoder():
    """ActionEncoder encodes candidate dicts."""
    from v4_1_reasoning_system.world_model.predictor import ActionEncoder

    ae = ActionEncoder(action_dim=16)
    candidate = {"mode": "hierarchical", "budget": "medium", "strictness": "verified", "tool_hint": "test"}
    emb = ae.encode_candidate(candidate)
    assert emb.shape == (1, 16)

    candidates = [
        {"mode": "hierarchical", "budget": "medium", "strictness": "verified"},
        {"mode": "global_opt", "budget": "high", "strictness": "strict"},
        {"mode": "repair", "budget": "low", "strictness": "fast"},
    ]
    embs = ae.encode_candidates(candidates)
    assert embs.shape == (1, 3, 16)


def test_aux_heads():
    """AuxiliaryHeads predictions."""
    from v4_1_reasoning_system.world_model.aux_heads import AuxiliaryHeads

    heads = AuxiliaryHeads(latent_dim=64)
    z_hat = torch.randn(2, 64)
    preds = heads(z_hat)
    assert preds.validity_prob.shape == (2,)
    assert preds.score_improvement.shape == (2,)
    assert preds.compute_cost.shape == (2,)
    assert preds.repair_prob.shape == (2,)
    # Probabilities in [0, 1]
    assert (preds.validity_prob >= 0).all() and (preds.validity_prob <= 1).all()


def test_candidate_generator():
    """CandidateGenerator produces valid candidates."""
    from v4_1_reasoning_system.router.candidate_generator import CandidateGenerator

    gen = CandidateGenerator()

    for domain in ["planning", "scheduling", "optimization", "coding", "unknown"]:
        candidates = gen.generate(domain=domain)
        assert len(candidates) >= 1
        assert all(c.mode in ["hierarchical", "global_opt", "repair", "llm_codegen"] for c in candidates)

    # Infeasible triggers repair priority (only after iteration 0)
    candidates = gen.generate(domain="scheduling", feasible=False, iteration=1)
    assert candidates[0].mode == "repair"


def test_ebm_router():
    """EBMRouter scoring and selection."""
    from v4_1_reasoning_system.router.ebm_router import EBMRouter

    router = EBMRouter(latent_dim=64, action_dim=16)
    z_t = torch.randn(1, 64)
    action_embs = torch.randn(1, 4, 16)
    z_hats = torch.randn(1, 4, 64)

    decision = router.score_candidates(z_t, action_embs, z_hats, top_k=2)
    assert 0 <= decision.selected_idx < 4
    assert len(decision.all_energies) == 4
    assert len(decision.top_k_indices) == 2


def test_rule_based_router():
    """RuleBasedRouter selects correct modes per domain."""
    from v4_1_reasoning_system.router.ebm_router import RuleBasedRouter
    from v4_1_reasoning_system.router.candidate_generator import ReasoningCandidate

    router = RuleBasedRouter()
    candidates = [
        ReasoningCandidate("hierarchical", "medium", "verified"),
        ReasoningCandidate("global_opt", "medium", "verified"),
        ReasoningCandidate("repair", "low", "verified"),
        ReasoningCandidate("llm_codegen", "medium", "verified"),
    ]

    # Planning prefers hierarchical
    idx = router.select(candidates, domain="planning")
    assert candidates[idx].mode == "hierarchical"

    # Scheduling prefers global_opt
    idx = router.select(candidates, domain="scheduling")
    assert candidates[idx].mode == "global_opt"

    # Infeasible prefers repair (after iteration 0)
    idx = router.select(candidates, domain="scheduling", feasible=False, iteration=1)
    assert candidates[idx].mode == "repair"


def test_hierarchical_solver():
    """HierarchicalSolver basic solve."""
    from v4_1_reasoning_system.solvers.hierarchical_solver import HierarchicalSolver

    solver = HierarchicalSolver()
    task = {
        "description": "Process a, b, c",
        "entities": ["a", "b", "c"],
        "constraints": [],
        "structured_data": {
            "subgoals": [
                {"name": "handle_a", "description": "Do a", "dependencies": []},
                {"name": "handle_b", "description": "Do b", "dependencies": ["handle_a"]},
                {"name": "handle_c", "description": "Do c", "dependencies": ["handle_b"]},
            ]
        },
    }
    result = solver.solve(task)
    assert result.success
    assert result.solution["complete"]


def test_global_opt_solver_knapsack():
    """GlobalOptSolver knapsack problem."""
    from v4_1_reasoning_system.solvers.global_opt_solver import GlobalOptSolver

    solver = GlobalOptSolver()
    task = {
        "domain": "optimization",
        "description": "Knapsack",
        "entities": [],
        "constraints": [],
        "objective": {"sense": "maximize"},
        "structured_data": {
            "type": "knapsack",
            "items": [
                {"name": "A", "weight": 5, "value": 10},
                {"name": "B", "weight": 4, "value": 8},
                {"name": "C", "weight": 6, "value": 12},
            ],
            "capacity": 10,
        },
    }
    result = solver.solve(task)
    assert result.success
    assert result.feasible
    assert "selected_items" in result.solution


def test_global_opt_solver_scheduling():
    """GlobalOptSolver scheduling problem."""
    from v4_1_reasoning_system.solvers.global_opt_solver import GlobalOptSolver

    solver = GlobalOptSolver()
    task = {
        "domain": "scheduling",
        "description": "Schedule jobs",
        "entities": ["job_0", "job_1"],
        "constraints": [],
        "objective": {"sense": "minimize"},
        "structured_data": {
            "type": "scheduling",
            "jobs": [
                {"name": "job_0", "duration": 3, "dependencies": []},
                {"name": "job_1", "duration": 5, "dependencies": ["job_0"]},
            ],
            "horizon": 20,
            "resources": {},
        },
    }
    result = solver.solve(task)
    assert result.success
    assert "job_0" in result.solution
    assert "job_1" in result.solution
    assert result.solution["job_1"]["start"] >= result.solution["job_0"]["end"]


def test_repair_solver():
    """RepairSolver with previous solution and violations."""
    from v4_1_reasoning_system.solvers.repair_solver import RepairSolver

    solver = RepairSolver()
    task = {
        "description": "Fix solution",
        "constraints": [
            {"name": "c1", "description": "x <= 10", "ctype": "hard",
             "params": {"type": "sum_leq", "bound": 10}},
        ],
    }
    context = {
        "previous_solution": {"x": 15, "y": 5},
        "verifier_feedback": {
            "violations": [
                {"name": "c1", "description": "sum > 10", "severity": 1.0,
                 "variable": "x", "suggested_value": 5}
            ],
        },
    }
    result = solver.solve(task, context=context)
    assert result.success


def test_planning_verifier():
    """PlanningVerifier checks a valid plan."""
    from v4_1_reasoning_system.verifier.planning_verifier import PlanningVerifier

    verifier = PlanningVerifier()
    task = {
        "entities": ["a", "b"],
        "constraints": [],
        "structured_data": {
            "subgoals": [
                {"name": "handle_a", "dependencies": []},
                {"name": "handle_b", "dependencies": ["handle_a"]},
            ]
        },
    }
    solution = {
        "steps": [
            {"subgoal": "handle_a", "solved": True, "solution": {"subgoal": "handle_a"}},
            {"subgoal": "handle_b", "solved": True, "solution": {"subgoal": "handle_b"}},
        ],
        "complete": True,
    }
    result = verifier.verify(task, solution)
    assert result.valid


def test_scheduling_verifier():
    """SchedulingVerifier checks a valid schedule."""
    from v4_1_reasoning_system.verifier.scheduling_verifier import SchedulingVerifier

    verifier = SchedulingVerifier()
    task = {
        "constraints": [],
        "structured_data": {
            "jobs": [
                {"name": "j0", "duration": 3, "dependencies": []},
                {"name": "j1", "duration": 5, "dependencies": ["j0"]},
            ],
            "horizon": 20,
            "resources": {},
        },
    }
    solution = {
        "j0": {"start": 0, "end": 3},
        "j1": {"start": 3, "end": 8},
    }
    result = verifier.verify(task, solution)
    assert result.valid
    assert result.feasible


def test_replay_buffer():
    """ReplayBuffer add, sample, and statistics."""
    from v4_1_reasoning_system.memory.replay import ReplayBuffer, TrajectoryRecord, StepRecord

    buf = ReplayBuffer(capacity=100)
    traj = TrajectoryRecord(
        trajectory_id="test_001",
        task_summary="test task",
        domain="planning",
        steps=[
            StepRecord(step_idx=0, mode="hierarchical", budget="medium",
                       strictness="verified", tool_hint="", success=True,
                       score_before=0.0, score_after=0.8),
        ],
        final_success=True,
        final_score=0.8,
    )
    buf.add(traj)
    assert len(buf) == 1

    samples = buf.sample(1)
    assert len(samples) == 1

    stats = buf.statistics()
    assert stats["count"] == 1
    assert stats["success_rate"] == 1.0


def test_structural_memory():
    """StructuralMemory templates and priors."""
    from v4_1_reasoning_system.memory.retrieval import StructuralMemory

    mem = StructuralMemory()
    mem.add_template("sched_basic", "scheduling", {"solver": "cp_sat"}, tags=["cp"])
    assert mem.get_template("sched_basic") is not None
    assert len(mem.find_templates(domain="scheduling")) == 1

    mem.update_domain_prior("scheduling", "global_opt", True)
    mem.update_domain_prior("scheduling", "global_opt", True)
    mem.update_domain_prior("scheduling", "global_opt", False)
    assert abs(mem.get_mode_prior("scheduling", "global_opt") - 2 / 3) < 0.01


def test_episodic_retriever():
    """EpisodicRetriever add and query."""
    from v4_1_reasoning_system.memory.retrieval import EpisodicRetriever

    ret = EpisodicRetriever(dim=32)
    for i in range(5):
        emb = torch.randn(32)
        ret.add(emb, {"id": i})

    query = torch.randn(32)
    results = ret.query(query, k=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_full_control_loop_planning():
    """Full end-to-end reasoning loop on a planning task (CPU, no LLM)."""
    from v4_1_reasoning_system.orchestration.controller import ReasoningController

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=3,
        max_time_seconds=30.0,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )

    result = controller.solve_from_dict(
        problem="Process a, b, c in order",
        parsed_json={
            "domain": "planning",
            "description": "Sequential processing",
            "entities": ["a", "b", "c"],
            "constraints": [
                {"name": "order", "description": "Process in dependency order", "ctype": "hard", "params": {}},
            ],
            "objective": {"description": "Complete all", "sense": "satisfy"},
            "ambiguities": [],
            "structured_data": {
                "subgoals": [
                    {"name": "handle_a", "description": "Process a", "dependencies": []},
                    {"name": "handle_b", "description": "Process b", "dependencies": ["handle_a"]},
                    {"name": "handle_c", "description": "Process c", "dependencies": ["handle_b"]},
                ]
            },
        },
    )

    assert result["iterations"] >= 1
    assert result["solution"] is not None
    assert "trajectory_id" in result
    assert len(controller.replay_buffer) >= 1


def test_full_control_loop_scheduling():
    """Full end-to-end reasoning loop on a scheduling task."""
    from v4_1_reasoning_system.orchestration.controller import ReasoningController

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=3,
        max_time_seconds=30.0,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )

    result = controller.solve_from_dict(
        problem="Schedule 3 jobs",
        parsed_json={
            "domain": "scheduling",
            "description": "Schedule 3 jobs",
            "entities": ["job_0", "job_1", "job_2"],
            "constraints": [],
            "objective": {"description": "Minimize makespan", "sense": "minimize"},
            "ambiguities": [],
            "structured_data": {
                "type": "scheduling",
                "jobs": [
                    {"name": "job_0", "duration": 3, "dependencies": []},
                    {"name": "job_1", "duration": 5, "dependencies": ["job_0"]},
                    {"name": "job_2", "duration": 2, "dependencies": []},
                ],
                "horizon": 20,
                "resources": {},
            },
        },
    )

    assert result["iterations"] >= 1
    assert result["solution"] is not None


def test_full_control_loop_optimization():
    """Full end-to-end reasoning loop on an optimization task."""
    from v4_1_reasoning_system.orchestration.controller import ReasoningController

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=3,
        max_time_seconds=30.0,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )

    result = controller.solve_from_dict(
        problem="Select items to maximize value",
        parsed_json={
            "domain": "optimization",
            "description": "Knapsack",
            "entities": ["item_0", "item_1", "item_2"],
            "constraints": [
                {"name": "capacity", "description": "weight <= 10", "ctype": "hard",
                 "params": {"type": "sum_leq", "bound": 10}},
            ],
            "objective": {"description": "Maximize value", "sense": "maximize"},
            "ambiguities": [],
            "structured_data": {
                "type": "knapsack",
                "items": [
                    {"name": "item_0", "weight": 5, "value": 10},
                    {"name": "item_1", "weight": 4, "value": 8},
                    {"name": "item_2", "weight": 6, "value": 12},
                ],
                "capacity": 10,
            },
        },
    )

    assert result["iterations"] >= 1
    assert result["solution"] is not None


def test_checkpoint_save_load(tmp_path):
    """Save and load checkpoint preserves state."""
    from v4_1_reasoning_system.orchestration.controller import ReasoningController

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=2,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )

    # Run one problem to populate buffer
    controller.solve_from_dict(
        problem="test",
        parsed_json={
            "domain": "planning",
            "description": "test",
            "entities": ["a"],
            "constraints": [],
            "objective": None,
            "ambiguities": [],
            "structured_data": {
                "subgoals": [{"name": "handle_a", "description": "do a", "dependencies": []}]
            },
        },
    )

    ckpt_dir = str(tmp_path / "ckpt")
    controller.save_checkpoint(ckpt_dir)

    # Load into fresh controller
    controller2 = ReasoningController(
        use_learned_router=False,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )
    controller2.load_checkpoint(ckpt_dir)

    assert len(controller2.replay_buffer) >= 1


def test_rollout_collector_generates_problems():
    """RolloutCollector generates valid synthetic problems."""
    from v4_1_reasoning_system.training.collect_rollouts import RolloutCollector

    planning = RolloutCollector.generate_planning_problems(3)
    assert len(planning) == 3
    assert all("parsed" in p for p in planning)
    assert all(p["parsed"]["domain"] == "planning" for p in planning)

    scheduling = RolloutCollector.generate_scheduling_problems(3)
    assert len(scheduling) == 3
    assert all(p["parsed"]["domain"] == "scheduling" for p in scheduling)

    optimization = RolloutCollector.generate_optimization_problems(3)
    assert len(optimization) == 3
    assert all(p["parsed"]["domain"] == "optimization" for p in optimization)

    suite = RolloutCollector.generate_benchmark_suite(
        n_planning=2, n_scheduling=2, n_optimization=2, n_coding=1
    )
    assert len(suite) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
