"""
Reasoning controller — the main control loop.

Implements the full adaptive reasoning cycle:
  1. Parse the problem into structured form
  2. Encode current reasoning state into latent z_t
  3. Generate candidate reasoning actions
  4. Predict latent consequences with the world model
  5. Score candidates with the router EBM
  6. Execute the best one (or top-k)
  7. Verify outputs externally
  8. Update state and memory
  9. Repeat until verified success or budget exhaustion
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import torch

from ..memory.replay import ReplayBuffer, StepRecord, TrajectoryRecord
from ..memory.retrieval import EpisodicRetriever, StructuralMemory
from ..parser.llm_parser import LLMParser
from ..parser.task_schema import TaskObject
from ..router.candidate_generator import CandidateGenerator
from ..router.ebm_router import EBMRouter, RuleBasedRouter
from ..solvers.base import BaseSolver, SolverResult
from ..solvers import SOLVER_REGISTRY
from ..verifier.base import BaseVerifier, VerificationResult
from ..verifier import VERIFIER_REGISTRY
from ..world_model.aux_heads import AuxiliaryHeads
from ..world_model.encoder import StateEncoder
from ..world_model.predictor import ActionEncoder, TransitionPredictor
from .state_update import ReasoningState, StateManager


class ReasoningController:
    """
    Main orchestrator for the adaptive reasoning system.

    Ties together all modules into the end-to-end control loop.
    Supports both rule-based (Phase 2) and learned (Phase 4+) routing.
    """

    def __init__(
        self,
        # Core modules
        parser: Optional[LLMParser] = None,
        state_encoder: Optional[StateEncoder] = None,
        transition_predictor: Optional[TransitionPredictor] = None,
        action_encoder: Optional[ActionEncoder] = None,
        aux_heads: Optional[AuxiliaryHeads] = None,
        ebm_router: Optional[EBMRouter] = None,
        # Memory
        replay_buffer: Optional[ReplayBuffer] = None,
        episodic_retriever: Optional[EpisodicRetriever] = None,
        structural_memory: Optional[StructuralMemory] = None,
        # Config
        use_learned_router: bool = False,
        max_iterations: int = 10,
        max_time_seconds: float = 300.0,
        top_k: int = 1,
        device: str = "cpu",
        latent_dim: int = 128,
        action_dim: int = 32,
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_iterations = max_iterations
        self.max_time_seconds = max_time_seconds
        self.top_k = top_k
        self.use_learned_router = use_learned_router

        # Initialize modules (create defaults if not provided)
        self.parser = parser or LLMParser()
        self.state_encoder = state_encoder or StateEncoder(latent_dim=latent_dim).to(device)
        self.transition_predictor = transition_predictor or TransitionPredictor(
            latent_dim=latent_dim, action_dim=action_dim
        ).to(device)
        self.action_encoder = action_encoder or ActionEncoder(action_dim=action_dim).to(device)
        self.aux_heads = aux_heads or AuxiliaryHeads(latent_dim=latent_dim).to(device)
        self.ebm_router = ebm_router or EBMRouter(
            latent_dim=latent_dim, action_dim=action_dim
        ).to(device)

        # State manager
        self.state_manager = StateManager(self.state_encoder, device=device)

        # Candidate generator
        self.candidate_generator = CandidateGenerator()

        # Rule-based fallback router
        self.rule_router = RuleBasedRouter()

        # Memory
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.episodic_retriever = episodic_retriever or EpisodicRetriever(dim=latent_dim)
        self.structural_memory = structural_memory or StructuralMemory()

        # Solver and verifier registries
        self._solvers: Dict[str, BaseSolver] = {}
        self._verifiers: Dict[str, BaseVerifier] = {}
        self._init_solvers_and_verifiers()

    def _init_solvers_and_verifiers(self) -> None:
        """Instantiate default solvers and verifiers."""
        for name, cls in SOLVER_REGISTRY.items():
            self._solvers[name] = cls()
        for name, cls in VERIFIER_REGISTRY.items():
            self._verifiers[name] = cls()

    def register_solver(self, name: str, solver: BaseSolver) -> None:
        self._solvers[name] = solver

    def register_verifier(self, name: str, verifier: BaseVerifier) -> None:
        self._verifiers[name] = verifier

    # ==================================================================
    # Main control loop
    # ==================================================================
    def solve(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        parsed_task: Optional[TaskObject] = None,
    ) -> Dict[str, Any]:
        """
        Full reasoning loop on a problem.

        Args:
            problem: natural-language problem statement
            context: optional structured context
            parsed_task: optional pre-parsed TaskObject (skips LLM parsing)

        Returns:
            Dict with solution, score, trajectory, etc.
        """
        t0 = time.time()
        trajectory_id = str(uuid.uuid4())[:8]
        logs: List[str] = []

        # ----------------------------------------------------------
        # Step 1: Parse the problem
        # ----------------------------------------------------------
        if parsed_task is not None:
            task = parsed_task
        else:
            try:
                task = self.parser.parse(problem, context)
            except Exception as e:
                logs.append(f"Parser failed: {e}, using minimal task")
                task = TaskObject(raw_input=problem)

        task_dict = task.model_dump()
        domain = task.domain.value
        logs.append(f"Parsed task: {task.summary()}")

        # Ensure structured_data has solver-ready content
        from ..parser.structured_builder import ensure_structured_data
        task_dict = ensure_structured_data(task_dict)
        if task_dict.get("structured_data"):
            sd_type = task_dict["structured_data"].get("type", "")
            sd_keys = list(task_dict["structured_data"].keys())
            logs.append(f"Structured data: type={sd_type}, keys={sd_keys}")

        # ----------------------------------------------------------
        # Step 2: Initialize state
        # ----------------------------------------------------------
        state = self.state_manager.initialize(task_dict)
        logs.append(f"Initial state encoded (latent dim={self.latent_dim})")

        # ----------------------------------------------------------
        # Step 3-9: Main reasoning loop
        # ----------------------------------------------------------
        step_records: List[StepRecord] = []
        best_solution = None
        best_score = -float("inf")
        best_verification = None

        for iteration in range(self.max_iterations):
            elapsed = time.time() - t0
            if elapsed > self.max_time_seconds:
                logs.append(f"Budget exhausted at iteration {iteration}")
                break

            logs.append(f"\n--- Iteration {iteration} ---")

            # Step 3: Generate candidates
            candidates = self.candidate_generator.generate(
                domain=domain,
                iteration=iteration,
                last_mode=state.last_mode,
                last_success=state.last_success,
                feasible=state.feasible,
                score=state.current_score,
            )
            logs.append(f"Generated {len(candidates)} candidates")

            if not candidates:
                logs.append("No candidates generated, stopping")
                break

            # Step 4: Predict latent consequences
            candidate_dicts = [c.to_dict() for c in candidates]
            action_embs = self.action_encoder.encode_candidates(
                candidate_dicts, device=torch.device(self.device)
            )  # (1, K, action_dim)

            self.transition_predictor.eval()
            self.aux_heads.eval()
            with torch.no_grad():
                z_hats = self.transition_predictor.predict_batch(
                    state.z_t, action_embs
                )  # (1, K, latent_dim)

                # Auxiliary predictions for each candidate
                aux_preds = []
                for k in range(z_hats.shape[1]):
                    aux = self.aux_heads(z_hats[:, k, :])
                    aux_preds.append(aux)

            # Step 5: Route — select the best candidate
            if self.use_learned_router:
                self.ebm_router.eval()
                with torch.no_grad():
                    decision = self.ebm_router.score_candidates(
                        state.z_t, action_embs, z_hats,
                        aux_list=aux_preds, top_k=self.top_k,
                    )
                selected_idx = decision.selected_idx
                logs.append(f"EBM router selected: idx={selected_idx} (E={decision.selected_energy:.3f})")
                logs.append(f"  All energies: {[f'{e:.3f}' for e in decision.all_energies]}")
            else:
                selected_idx = self.rule_router.select(
                    candidates, domain=domain,
                    feasible=state.feasible, iteration=iteration,
                )
                logs.append(f"Rule router selected: idx={selected_idx}")

            selected = candidates[selected_idx]
            logs.append(f"  Mode: {selected.mode}, Budget: {selected.budget}, Hint: {selected.tool_hint}")

            # Step 6: Execute solver
            solver = self._solvers.get(selected.mode)
            if solver is None:
                logs.append(f"No solver for mode {selected.mode}, skipping")
                continue

            solver_context = {
                "previous_solution": state.current_solution,
                "verifier_feedback": state.current_feedback or {},
                "structured_data": task_dict.get("structured_data", {}),
                "iteration": iteration,
            }

            step_t0 = time.time()
            solver_result = solver.solve(
                task_dict,
                budget=selected.budget,
                strictness=selected.strictness,
                tool_hint=selected.tool_hint,
                context=solver_context,
            )
            step_elapsed = time.time() - step_t0
            logs.append(f"  Solver: success={solver_result.success}, score={solver_result.score:.3f}, "
                        f"feasible={solver_result.feasible}, elapsed={step_elapsed:.2f}s")

            # Step 7: Verify
            verifier = self._verifiers.get(domain) or self._verifiers.get("planning")
            if verifier and solver_result.success:
                verification = verifier.verify(
                    task_dict, solver_result.solution, solver_context
                )
                feedback_dict = verification.to_feedback_dict()
                logs.append(f"  Verifier: valid={verification.valid}, score={verification.score:.3f}, "
                            f"passed={verification.tests_passed}/{verification.tests_total}")
            else:
                verification = VerificationResult(valid=False, feasible=False)
                feedback_dict = verification.to_feedback_dict()
                if not solver_result.success:
                    logs.append("  Solver failed, skipping verification")

            # Track best solution
            current_score = verification.score if verification.valid else solver_result.score * 0.5
            if current_score > best_score:
                best_score = current_score
                best_solution = solver_result.solution
                best_verification = verification

            # Step 8: Record step for memory
            step_record = StepRecord(
                step_idx=iteration,
                mode=selected.mode,
                budget=selected.budget,
                strictness=selected.strictness,
                tool_hint=selected.tool_hint,
                z_t=state.z_t.detach().cpu() if state.z_t is not None else None,
                action_emb=action_embs[:, selected_idx, :].detach().cpu(),
                z_hat=z_hats[:, selected_idx, :].detach().cpu(),
                solver_result=solver_result.to_dict(),
                verifier_result=feedback_dict,
                success=solver_result.success and verification.valid,
                score_before=state.current_score,
                score_after=current_score,
                elapsed_seconds=step_elapsed,
                candidates_considered=candidate_dicts,
            )
            step_records.append(step_record)

            # Update state
            prev_z_t = state.z_t.detach().cpu() if state.z_t is not None else None
            state = self.state_manager.update(
                state=state,
                task_dict=task_dict,
                solver_result=solver_result.to_dict(),
                verifier_feedback=feedback_dict,
                mode=selected.mode,
                budget=selected.budget,
                success=solver_result.success and verification.valid,
                elapsed=step_elapsed,
            )

            # Store the actual next latent state for world model training
            step_record.z_actual = state.z_t.detach().cpu() if state.z_t is not None else None

            # Update structural memory
            self.structural_memory.update_domain_prior(
                domain, selected.mode,
                solver_result.success and verification.valid,
            )

            # Check termination: verified success
            if verification.valid and verification.feasible and verification.score >= 0.99:
                logs.append(f"Verified success at iteration {iteration}!")
                break

        # ----------------------------------------------------------
        # Step 9: Store trajectory in memory
        # ----------------------------------------------------------
        total_elapsed = time.time() - t0
        trajectory = TrajectoryRecord(
            trajectory_id=trajectory_id,
            task_summary=task.description or problem[:100],
            domain=domain,
            steps=step_records,
            final_success=best_verification.valid if best_verification else False,
            final_score=best_score,
            total_elapsed=total_elapsed,
            metadata={"max_iterations": self.max_iterations},
        )
        self.replay_buffer.add(trajectory)

        # Add to episodic retriever
        if state.z_t is not None:
            self.episodic_retriever.add(
                state.z_t.squeeze(0),
                {"trajectory_id": trajectory_id, "domain": domain, "score": best_score},
            )

        logs.append(f"\n=== Done: {len(step_records)} steps, score={best_score:.3f}, "
                     f"elapsed={total_elapsed:.2f}s ===")

        return {
            "solution": best_solution,
            "score": best_score,
            "valid": best_verification.valid if best_verification else False,
            "feasible": best_verification.feasible if best_verification else False,
            "iterations": len(step_records),
            "trajectory_id": trajectory_id,
            "domain": domain,
            "task_summary": task.summary(),
            "elapsed_seconds": total_elapsed,
            "logs": logs,
            "trajectory": trajectory.to_dict(),
        }

    # ==================================================================
    # Convenience: solve from pre-parsed JSON (for Colab / testing)
    # ==================================================================
    def solve_from_dict(
        self,
        problem: str,
        parsed_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Solve using a pre-computed parse (no LLM needed)."""
        task = self.parser.parse_offline(problem, parsed_json)
        return self.solve(problem, parsed_task=task)

    # ==================================================================
    # Save / load all learned components
    # ==================================================================
    def save_checkpoint(self, directory: str) -> None:
        """Save all learned model weights and memory to disk."""
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_encoder.state_dict(), path / "state_encoder.pt")
        torch.save(self.transition_predictor.state_dict(), path / "transition_predictor.pt")
        torch.save(self.action_encoder.state_dict(), path / "action_encoder.pt")
        torch.save(self.aux_heads.state_dict(), path / "aux_heads.pt")
        torch.save(self.ebm_router.state_dict(), path / "ebm_router.pt")

        self.replay_buffer.save(str(path / "replay"))
        self.episodic_retriever.save(str(path / "episodic"))
        self.structural_memory.save(str(path / "structural"))

    def load_checkpoint(self, directory: str) -> None:
        """Load all learned model weights and memory from disk."""
        from pathlib import Path
        path = Path(directory)

        def _load(module, name):
            fp = path / name
            if fp.exists():
                module.load_state_dict(
                    torch.load(fp, map_location=self.device, weights_only=True)
                )

        _load(self.state_encoder, "state_encoder.pt")
        _load(self.transition_predictor, "transition_predictor.pt")
        _load(self.action_encoder, "action_encoder.pt")
        _load(self.aux_heads, "aux_heads.pt")
        _load(self.ebm_router, "ebm_router.pt")

        replay_dir = str(path / "replay")
        if (path / "replay").exists():
            self.replay_buffer.load(replay_dir)
        episodic_dir = str(path / "episodic")
        if (path / "episodic").exists():
            self.episodic_retriever.load(episodic_dir)
        structural_dir = str(path / "structural")
        if (path / "structural").exists():
            self.structural_memory.load(structural_dir)
