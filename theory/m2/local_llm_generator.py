"""M2.13a - State-conditioned local LLM hypothesis generator.

The local LLM is a strictly-encapsulated *candidate hypothesis source*. It never
produces a policy, support, evidence or a verdict. Every raw proposal it emits is
filtered by the existing M2 normalizer / validator / merger / testability
compiler. The real ``transformers`` backend is optional, offline and disabled by
default; a deterministic state-conditioned mock is the tested fallback.

Inputs (read-only summaries, no raw grid):
- P2.G5 risk-aware objective frontier handoff request
- M2.G2 hypotheses (negative-constraint / dedup base)
- M1.G0 general mechanic candidates (entities by role candidate, HUD/horizon,
  dynamic invariants, action-effect priors)
- M3.G6 failure summary (proxy/completion divergence, blocked commit cells)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .hypothesis_merger import assign_stable_hypothesis_ids, merge_hypotheses
from .metric_registry import (
    UNLOCK_HYPOTHESIS_FAMILY,
    is_metric_executor_supported,
)
from .normalizer import allowed_actions_for_frontier, normalize_raw_proposal
from .schema import (
    M2_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    FrontierConditionedHypothesis,
    RawHypothesisProposal,
    RejectedProposal,
)
from .testability_compiler import build_m3_requests_payload
from .validators import validate_hypothesis


# --------------------------------------------------------------------------- #
# Defaults and constants
# --------------------------------------------------------------------------- #

DEFAULT_FRONTIERS_PATH = (
    Path("diagnostics") / "p2" / "risk_aware_objective_frontier_handoff_requests.json"
)
DEFAULT_M2G2_HYPOTHESES_PATH = (
    Path("diagnostics") / "m2" / "risk_aware_objective_completion_hypotheses.json"
)
DEFAULT_M1_CANDIDATES_PATH = (
    Path("diagnostics") / "m1" / "general_mechanic_candidates.json"
)
DEFAULT_M3G6_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "risk_aware_objective_completion_experiment_results.json"
)
DEFAULT_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "state_conditioned_llm_hypotheses.json"
)
DEFAULT_M3_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "state_conditioned_llm_m3_candidate_requests.json"
)

STATE_CONDITIONED_LLM_SCHEMA_VERSION = "m2.state_conditioned_llm_hypotheses.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"

ACTION_UNIVERSE: Tuple[str, ...] = (
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)

ALLOWED_HYPOTHESIS_FAMILIES: Tuple[str, ...] = (
    "objective_readiness_detection",
    "post_conversion_commit_action_search",
    "goal_state_representation_beyond_safe_progress",
    "proxy_progress_vs_completion_discriminator",
    "risk_aware_selector_completion_gap",
    UNLOCK_HYPOTHESIS_FAMILY,
)

SUBSTRATE_ACTIONS_NOT_TARGETS: Tuple[str, ...] = (
    "ACTION6",
    "ACTION6,ACTION3",
    "ACTION6,ACTION4",
)

KNOWN_NEGATIVE_CONSTRAINTS: Tuple[str, ...] = (
    "do_not_retest_action6_extension_as_target",
    "proxy_progress_without_completion_observed",
    "completion_signal_absent",
)

FORBIDDEN_INTERPRETATIONS: Tuple[str, ...] = (
    "Do not treat proxy progress as objective completion.",
    "Do not propose unavailable ACTION1/ACTION2/ACTION5 as directly executable.",
    "Do not count mechanistic-context candidates or invariants as confirmed.",
    "Do not emit support, status CONFIRMED/REFUTED, or A32/A33 readiness.",
)

EXPECTED_SIGNAL_BY_FAMILY: Dict[str, str] = {
    "objective_readiness_detection": (
        "readiness_feature_predicts_completion_more_than_controls"
    ),
    "post_conversion_commit_action_search": (
        "commit_action_triggers_objective_completion_vs_controls"
    ),
    "goal_state_representation_beyond_safe_progress": (
        "goal_representation_predicts_completion_beyond_proxy_progress"
    ),
    "proxy_progress_vs_completion_discriminator": (
        "discriminator_separates_proxy_progress_from_completion"
    ),
    "risk_aware_selector_completion_gap": (
        "completion_targeted_selector_outperforms_progress_selector"
    ),
    UNLOCK_HYPOTHESIS_FAMILY: (
        "available_actions_set_changes_after_candidate_context"
    ),
}

SYSTEM_PROMPT = (
    "You are an abductive hypothesis generator inside a falsification pipeline. "
    "You only propose candidate, testable hypotheses. You MUST output JSON only: "
    "a list of objects with these exact fields:\n"
    "- proposal_id: string (e.g., 'llm_item_000')\n"
    "- source: 'local_llm'\n"
    "- source_request_id: string (use the frontier_context_id from input)\n"
    "- game_id: string (use the game_id from input)\n"
    "- frontier_context_id: string (use from input)\n"
    "- frontier_reason: string (use from input)\n"
    "- frontier_step: int or null (use from input)\n"
    "- hypothesis_family: string (must be one of allowed_hypothesis_families from input)\n"
    "- candidate_action: string (MUST be in available_actions from input)\n"
    "- predicted_metric: string (e.g., 'objective_completion_signal')\n"
    "- predicted_effect: string (describe the expected effect)\n"
    "- rationale: string (explain why this hypothesis is plausible)\n"
    "- suggested_control_actions: list of strings (optional, default empty)\n"
    "- required_context_replay: list of strings (optional, default empty)\n"
    "Never claim support, never confirm or refute, never set ready_for_a32 or ready_for_a33. "
    "candidate_action MUST be in available_actions; unavailable actions may appear only in "
    "unlock_target_actions.\n\n"
    "Example output:\n"
    "[\n"
    "  {\n"
    "    \"proposal_id\": \"llm_item_000\",\n"
    "    \"source\": \"local_llm\",\n"
    "    \"source_request_id\": \"frontier_ctx_123\",\n"
    "    \"game_id\": \"bp35-0a0ad940\",\n"
    "    \"frontier_context_id\": \"frontier_ctx_123\",\n"
    "    \"frontier_reason\": \"proxy_progress_without_completion\",\n"
    "    \"frontier_step\": 150,\n"
    "    \"hypothesis_family\": \"objective_readiness_detection\",\n"
    "    \"candidate_action\": \"ACTION3\",\n"
    "    \"predicted_metric\": \"objective_completion_signal\",\n"
    "    \"predicted_effect\": \"ACTION3 triggers objective completion when proxy progress > 0.5\",\n"
    "    \"rationale\": \"The agent shows proxy progress but never completes; ACTION3 may be the missing commit action.\"\n"
    "  }\n"
    "]"
)


# --------------------------------------------------------------------------- #
# Situation packet
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SituationPacket:
    frontier: Dict[str, Any]
    game_id: str
    available_actions: Tuple[str, ...]
    forbidden_target_actions: Tuple[str, ...]
    context_replay: Tuple[str, ...]
    known_negative_constraints: Tuple[str, ...]
    allowed_hypothesis_families: Tuple[str, ...]
    current_state_summary: Dict[str, Any]
    mechanistic_context_candidates: Dict[str, Any]
    m3g6_failure_summary: Dict[str, Any]
    m2g2_context: Dict[str, Any]
    forbidden_interpretations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier": dict(self.frontier),
            "game_id": self.game_id,
            "available_actions": list(self.available_actions),
            "forbidden_target_actions": list(self.forbidden_target_actions),
            "context_replay": list(self.context_replay),
            "known_negative_constraints": list(self.known_negative_constraints),
            "allowed_hypothesis_families": list(self.allowed_hypothesis_families),
            "current_state_summary": dict(self.current_state_summary),
            "mechanistic_context_candidates": dict(self.mechanistic_context_candidates),
            "m3g6_failure_summary": dict(self.m3g6_failure_summary),
            "m2g2_context": dict(self.m2g2_context),
            "required_output_schema": "RawM2HypothesisProposal[]",
            "forbidden_interpretations": list(self.forbidden_interpretations),
        }


def build_situation_packet(
    frontier_request: Mapping[str, Any],
    *,
    m1_payload: Mapping[str, Any] | None = None,
    m3g6_payload: Mapping[str, Any] | None = None,
    m2g2_payload: Mapping[str, Any] | None = None,
    available_actions: Sequence[str] | None = None,
) -> SituationPacket:
    game_id = str(frontier_request.get("game_id", ""))
    available = tuple(
        str(item) for item in (available_actions or ())
    ) or allowed_actions_for_frontier({"game_id": game_id}, game_id=game_id)
    forbidden = tuple(a for a in ACTION_UNIVERSE if a not in set(available))
    context_replay = _frontier_context_replay(frontier_request)

    entity_role_candidates = _entity_role_candidates(m1_payload or {})
    current_state_summary = {
        "available_actions": list(available),
        "entity_role_candidates": entity_role_candidates,
        "hud_or_horizon": _hud_summary(entity_role_candidates),
        "relation_summary": _relation_summary(m1_payload or {}),
    }
    mechanistic_context_candidates = {
        "dynamic_invariants": _dynamic_invariants_summary(m1_payload or {}),
        "action_effect_priors": _action_effect_priors(m1_payload or {}),
        "entity_role_candidates": entity_role_candidates,
        "prediction_confidence_is_not_evidence": True,
    }
    return SituationPacket(
        frontier={
            "type": str(frontier_request.get("frontier_type", "")),
            "blocked_capability": str(frontier_request.get("blocked_capability", "")),
            "frontier_reason": str(frontier_request.get("frontier_reason", "")),
            "known_failure": "proxy_progress_without_completion",
            "requested_hypothesis_families": list(
                frontier_request.get("requested_hypothesis_families", []) or []
            ),
            "requested_experiment_styles": list(
                frontier_request.get("requested_experiment_styles", []) or []
            ),
        },
        game_id=game_id,
        available_actions=available,
        forbidden_target_actions=forbidden,
        context_replay=context_replay,
        known_negative_constraints=KNOWN_NEGATIVE_CONSTRAINTS,
        allowed_hypothesis_families=ALLOWED_HYPOTHESIS_FAMILIES,
        current_state_summary=current_state_summary,
        mechanistic_context_candidates=mechanistic_context_candidates,
        m3g6_failure_summary=_m3g6_failure_summary(m3g6_payload or {}),
        m2g2_context=_m2g2_context(m2g2_payload or {}),
        forbidden_interpretations=FORBIDDEN_INTERPRETATIONS,
    )


def normalizer_frontier_request(
    frontier_request: Mapping[str, Any],
    packet: SituationPacket,
) -> Dict[str, Any]:
    """Synthesize the frontier mapping the shared normalizer expects."""
    context_id = str(
        frontier_request.get("source_frontier_id")
        or frontier_request.get("frontier_id")
        or frontier_request.get("request_id", "")
    )
    return {
        "request_id": str(frontier_request.get("request_id", "")),
        "source_step": None,
        "game_id": packet.game_id,
        "frontier_context_id": context_id,
        "context_signature": list(packet.context_replay),
        "reason": str(frontier_request.get("frontier_reason", "")),
        "live_state_signature": "state:risk_aware_post_stop_safe_conversion",
        "available_actions": list(packet.available_actions),
        "ready_for_m1_or_m3": True,
        "status": "OPEN",
    }


# --------------------------------------------------------------------------- #
# Local LLM config and backends
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalLLMConfig:
    enable_local_llm: bool = False
    model_path: str = ""
    device: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.2
    fallback_to_mock: bool = True


class LocalLLMUnavailable(RuntimeError):
    """Raised when the real local LLM backend cannot produce output."""


class StateConditionedMockLLM:
    """Deterministic, state-conditioned stand-in for an offline local LLM."""

    backend_name = "mock"

    def generate_json(self, packet: SituationPacket) -> str:
        return json.dumps(self.generate_items(packet))

    def generate_items(self, packet: SituationPacket) -> List[Dict[str, Any]]:
        available = list(packet.available_actions)
        move_action = "ACTION3" if "ACTION3" in available else (
            available[0] if available else "ACTION3"
        )
        commit_probe = "ACTION4" if "ACTION4" in available else move_action
        replay = list(packet.context_replay)
        forbidden = list(packet.forbidden_target_actions)
        unlock_target = next(
            (a for a in ("ACTION2", "ACTION1", "ACTION5") if a in forbidden),
            forbidden[0] if forbidden else "ACTION2",
        )
        specs = [
            (
                "objective_readiness_detection",
                move_action,
                "objective_completion_signal",
                "A relation-saturation readiness feature, not more relation "
                "progress, marks a post-stop state as completion-ready.",
                (),
            ),
            (
                "post_conversion_commit_action_search",
                commit_probe,
                "objective_completion_signal",
                "After a safe conversion, a distinct commit action (not an "
                "ACTION6 extension) may trigger objective completion.",
                (),
            ),
            (
                "goal_state_representation_beyond_safe_progress",
                move_action,
                "objective_completion_signal",
                "Completion depends on a global actor-target configuration, not "
                "on the count of relation states.",
                (),
            ),
            (
                "proxy_progress_vs_completion_discriminator",
                commit_probe,
                "terminal_reentry_rate",
                "High-hold proxy-progress states may score well yet stay unsafe "
                "or non-ready for completion.",
                (),
            ),
            (
                "risk_aware_selector_completion_gap",
                move_action,
                "terminal_reentry_rate",
                "The selector optimizes terminal-adjusted progress instead of a "
                "completion probability and lacks a commit branch.",
                (),
            ),
            (
                UNLOCK_HYPOTHESIS_FAMILY,
                move_action,
                "available_actions_before_after",
                "A currently unavailable commit action may become available "
                "after a specific spatial/geometric transition.",
                (unlock_target,),
            ),
        ]
        items: List[Dict[str, Any]] = []
        for index, (family, action, metric, text, unlock) in enumerate(specs, start=1):
            items.append(
                {
                    "source": "llm",
                    "hypothesis_family": family,
                    "hypothesis_id": f"llm_{family}_{index:03d}",
                    "hypothesis_text": text,
                    "candidate_action": action,
                    "context_replay": replay,
                    "unlock_target_actions": list(unlock),
                    "required_observables": _required_observables(family, metric),
                    "primary_metric": metric,
                    "secondary_metrics": _secondary_metrics(metric),
                    "expected_signal_type": EXPECTED_SIGNAL_BY_FAMILY[family],
                    "falsification_signal": _falsification_signal(family, metric),
                    "forbidden_interpretations": list(packet.forbidden_interpretations),
                    "requested_experiment_style": _experiment_style(family),
                    "support": 0,
                    "status": "UNRESOLVED",
                    "truth_status": M2_TRUTH_STATUS,
                }
            )
        return items


class RealLocalLLMGenerator:
    """Best-effort offline ``transformers`` backend (lazy import)."""

    backend_name = "transformers"

    def __init__(self, config: LocalLLMConfig):
        self.config = config

    def generate_json(self, packet: SituationPacket) -> str:
        model_path = self.config.model_path
        if not model_path or not Path(model_path).exists():
            raise LocalLLMUnavailable("model_path_not_found")
        try:  # pragma: no cover - exercised only with a real local model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise LocalLLMUnavailable(
                f"transformers_import_failed:{type(exc).__name__}"
            ) from exc
        try:  # pragma: no cover - exercised only with a real local model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device_map = "auto" if self.config.device == "auto" else self.config.device
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map=device_map
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_llm_prompt(packet)},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            # M2.13d: Fix attention_mask warning when pad_token == eos_token
            attention_mask = (inputs != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )
            raw_text = tokenizer.decode(
                output[0][inputs.shape[-1]:], skip_special_tokens=True
            )
            logger.info(f"Raw LLM output:\n{raw_text}")
            return raw_text
        except LocalLLMUnavailable:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            raise LocalLLMUnavailable(
                f"generation_failed:{type(exc).__name__}"
            ) from exc


def build_llm_prompt(packet: SituationPacket) -> str:
    return (
        "Situation packet (typed, candidate-only):\n"
        + json.dumps(packet.to_dict(), indent=2, sort_keys=True)
        + "\n\nReturn JSON only: RawM2HypothesisProposal[]."
    )


# --------------------------------------------------------------------------- #
# Parsing and boundary guards
# --------------------------------------------------------------------------- #


def parse_llm_json(text: str) -> Tuple[Dict[str, Any], ...]:
    """Strictly parse LLM output; any non-JSON yields no proposals."""
    # Extract JSON from markdown code blocks if present
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    
    try:
        payload = json.loads(text)
        logger.info(f"Parsed JSON payload type: {type(payload)}")
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(f"Failed to parse JSON: {exc}")
        logger.warning(f"Text to parse (first 500 chars): {text[:500]}")
        return ()
    
    # If single object, wrap in list
    if isinstance(payload, Mapping) and "hypotheses" not in payload and "proposals" not in payload:
        payload = [payload]
        logger.info("Wrapped single object in list")
    
    if isinstance(payload, list):
        result = tuple(item for item in payload if isinstance(item, Mapping))
        logger.info(f"Extracted {len(result)} items from list")
        return result
    if isinstance(payload, Mapping):
        rows = payload.get("hypotheses") or payload.get("proposals") or []
        if isinstance(rows, list):
            result = tuple(item for item in rows if isinstance(item, Mapping))
            logger.info(f"Extracted {len(result)} items from mapping")
            return result
    logger.warning(f"Payload structure not recognized: {list(payload.keys()) if isinstance(payload, Mapping) else type(payload)}")
    return ()


def trusted_assertion_violation(item: Mapping[str, Any]) -> str | None:
    if int(item.get("support", 0) or 0) > 0:
        return "asserts_support"
    if str(item.get("status", "")).upper() in {"CONFIRMED", "REFUTED"}:
        return "asserts_status"
    if str(item.get("truth_status", "")).upper() == "CONFIRMED":
        return "asserts_truth_confirmed"
    if bool(item.get("ready_for_a32")):
        return "asserts_ready_for_a32"
    if bool(item.get("ready_for_a33")):
        return "asserts_ready_for_a33"
    if bool(item.get("revision_performed")):
        return "asserts_revision_performed"
    return None


def _target_sequence(item: Mapping[str, Any]) -> str:
    explicit = item.get("target_sequence")
    if explicit:
        return ",".join(str(a) for a in explicit) if isinstance(
            explicit, (list, tuple)
        ) else str(explicit)
    return str(item.get("candidate_action", ""))


def guard_llm_item(
    item: Mapping[str, Any],
    *,
    available_actions: Sequence[str],
) -> str | None:
    """Return a rejection reason for an untrusted item, else ``None``."""
    violation = trusted_assertion_violation(item)
    if violation is not None:
        return violation
    candidate = str(item.get("candidate_action", ""))
    if not candidate:
        return "missing_candidate_action"
    if candidate not in set(str(a) for a in available_actions):
        return "direct_unavailable_action"
    if _target_sequence(item) in set(SUBSTRATE_ACTIONS_NOT_TARGETS):
        return "substrate_retest_target"
    return None


def apply_boundary_guards(
    items: Sequence[Mapping[str, Any]],
    *,
    available_actions: Sequence[str],
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        reason = guard_llm_item(item, available_actions=available_actions)
        if reason is None:
            accepted.append(dict(item))
        else:
            rejected.append(
                {
                    "proposal_id": str(
                        item.get("hypothesis_id", f"llm_item_{index:03d}")
                    ),
                    "hypothesis_family": str(item.get("hypothesis_family", "")),
                    "reason": reason,
                    "candidate_action": str(item.get("candidate_action", "")),
                }
            )
    return tuple(accepted), tuple(rejected)


def llm_item_to_raw_proposal(
    item: Mapping[str, Any],
    *,
    frontier_request: Mapping[str, Any],
    packet: SituationPacket | None = None,
    index: int,
) -> Tuple[RawHypothesisProposal, int] | RawHypothesisProposal:
    legacy_return = packet is None
    if packet is None:
        packet = build_situation_packet(frontier_request)
    context = str(
        frontier_request.get("source_frontier_id")
        or frontier_request.get("frontier_id")
        or frontier_request.get("request_id", "unknown")
    )
    family = str(item.get("hypothesis_family", ""))
    action = str(item.get("candidate_action", ""))
    metric = str(item.get("primary_metric", item.get("predicted_metric", "")))
    
    # M2.13d: Anchoring guard - override LLM provenance fields from packet
    llm_source_request_id = item.get("source_request_id")
    llm_frontier_context_id = item.get("frontier_context_id")
    llm_frontier_step = item.get("frontier_step")
    
    anchoring_failures = 0
    if llm_source_request_id and llm_source_request_id != str(frontier_request.get("request_id", "")):
        logger.warning(f"llm_provenance_fields_overridden: source_request_id {llm_source_request_id} -> {frontier_request.get('request_id', '')}")
        anchoring_failures += 1
    if llm_frontier_context_id and llm_frontier_context_id != context:
        logger.warning(f"llm_provenance_fields_overridden: frontier_context_id {llm_frontier_context_id} -> {context}")
        anchoring_failures += 1
    if llm_frontier_step is not None:
        logger.warning(f"llm_provenance_fields_overridden: frontier_step {llm_frontier_step} -> None")
        anchoring_failures += 1
    
    # M2.13d: Context replay anchoring - use packet if LLM provides empty list
    llm_replay = tuple(str(a) for a in item.get("context_replay", []) or ())
    if not llm_replay and packet.context_replay:
        logger.warning(f"llm_context_replay_missing_filled_from_packet: using {len(packet.context_replay)} actions from packet")
        anchoring_failures += 1
        replay = packet.context_replay
    else:
        replay = llm_replay
    
    unlock = tuple(str(a) for a in item.get("unlock_target_actions", []) or ())
    expected = str(
        item.get("expected_signal_type")
        or EXPECTED_SIGNAL_BY_FAMILY.get(family, "")
    )
    proposal = RawHypothesisProposal(
        proposal_id=f"raw::{context}::llm::{family}::{index:03d}",
        source="llm",
        source_request_id=str(frontier_request.get("request_id", "")),
        game_id=packet.game_id,
        frontier_context_id=context,
        frontier_reason=str(frontier_request.get("frontier_reason", "")),
        frontier_step=None,
        hypothesis_family=family,
        candidate_action=action,
        predicted_metric=metric,
        predicted_effect=str(item.get("hypothesis_text", item.get("predicted_effect", ""))),
        rationale=(
            "Candidate-only LLM abductive proposal; the LLM is not trusted and "
            "this is never evidence or support."
        ),
        required_context_replay=replay,
        unlock_target_actions=unlock,
        expected_signal_type=expected,
        raw_status=str(item.get("status", "")),
        raw_support=item.get("support"),
        raw_truth_status=str(item.get("truth_status", "")),
        raw_payload={
            "llm_required_observables": list(item.get("required_observables", []) or []),
            "llm_secondary_metrics": list(item.get("secondary_metrics", []) or []),
            "llm_falsification_signal": str(item.get("falsification_signal", "")),
            "llm_forbidden_interpretations": list(
                item.get("forbidden_interpretations", []) or []
            ),
            "llm_requested_experiment_style": str(
                item.get("requested_experiment_style", "")
            ),
        },
    )
    if legacy_return:
        return proposal
    return proposal, anchoring_failures


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_state_conditioned_llm_generation(
    *,
    frontiers_path: str | Path = DEFAULT_FRONTIERS_PATH,
    m2g2_path: str | Path = DEFAULT_M2G2_HYPOTHESES_PATH,
    m1_candidates_path: str | Path = DEFAULT_M1_CANDIDATES_PATH,
    m3g6_results_path: str | Path = DEFAULT_M3G6_RESULTS_PATH,
    config: LocalLLMConfig | None = None,
) -> Dict[str, Any]:
    config = config or LocalLLMConfig()
    frontier_payload = _load_json(frontiers_path)
    frontier_request = _first_frontier_request(frontier_payload)
    m1_payload = _load_json(m1_candidates_path)
    m3g6_payload = _load_json(m3g6_results_path)
    m2g2_payload = _load_json(m2g2_path)

    packet = build_situation_packet(
        frontier_request,
        m1_payload=m1_payload,
        m3g6_payload=m3g6_payload,
        m2g2_payload=m2g2_payload,
    )
    return run_generation_from_packet(
        packet,
        frontier_request=frontier_request,
        config=config,
        sources={
            "frontiers_path": str(frontiers_path),
            "m2g2_path": str(m2g2_path),
            "m1_candidates_path": str(m1_candidates_path),
            "m3g6_results_path": str(m3g6_results_path),
        },
    )


def run_generation_from_packet(
    packet: SituationPacket,
    *,
    frontier_request: Mapping[str, Any],
    config: LocalLLMConfig | None = None,
    sources: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    config = config or LocalLLMConfig()
    backend, backend_name, llm_enabled, fallback_used, llm_error = _select_backend(config)
    text = backend.generate_json(packet)
    raw_items = parse_llm_json(text)
    accepted_items, guard_rejected = apply_boundary_guards(
        raw_items, available_actions=packet.available_actions
    )

    normalizer_frontier = normalizer_frontier_request(frontier_request, packet)
    frontiers_by_request_id = {
        str(normalizer_frontier.get("request_id", "")): normalizer_frontier
    }
    raw_proposals = []
    anchoring_failures_total = 0
    for index, item in enumerate(accepted_items, start=1):
        proposal, failures = llm_item_to_raw_proposal(item, frontier_request=frontier_request, packet=packet, index=index)
        raw_proposals.append(proposal)
        anchoring_failures_total += failures
    normalized: List[FrontierConditionedHypothesis] = []
    norm_rejected: List[RejectedProposal] = []
    for proposal in raw_proposals:
        result = normalize_raw_proposal(proposal, frontier_request=normalizer_frontier)
        if isinstance(result, RejectedProposal):
            norm_rejected.append(result)
        else:
            normalized.append(result)
    merged = merge_hypotheses(
        normalized, frontiers_by_request_id=frontiers_by_request_id
    )
    hypotheses = assign_stable_hypothesis_ids(merged)

    hypothesis_payload = _build_hypothesis_payload(
        packet=packet,
        hypotheses=hypotheses,
        guard_rejected=guard_rejected,
        norm_rejected=norm_rejected,
        raw_items_count=len(raw_items),
        anchoring_failures=anchoring_failures_total,
        backend_name=backend_name,
        llm_enabled=llm_enabled,
        fallback_used=fallback_used,
        llm_error=llm_error,
        sources=dict(sources or {}),
        config=config,
    )
    m3_payload = build_m3_requests_payload(
        hypotheses, source_hypothesis_path=str(DEFAULT_HYPOTHESES_OUTPUT_PATH)
    )
    return {
        "situation_packet": packet.to_dict(),
        "hypothesis_payload": hypothesis_payload,
        "m3_payload": m3_payload,
    }


def _select_backend(
    config: LocalLLMConfig,
) -> Tuple[Any, str, bool, bool, str | None]:
    if not config.enable_local_llm:
        return StateConditionedMockLLM(), "mock", False, True, None
    real = RealLocalLLMGenerator(config)
    if not config.model_path or not Path(config.model_path).exists():
        if not config.fallback_to_mock:
            raise LocalLLMUnavailable("model_path_not_found")
        return StateConditionedMockLLM(), "mock", False, True, "model_path_not_found"
    return real, real.backend_name, True, False, None


def _build_hypothesis_payload(
    *,
    packet: SituationPacket,
    hypotheses: Sequence[FrontierConditionedHypothesis],
    guard_rejected: Sequence[Mapping[str, Any]],
    norm_rejected: Sequence[RejectedProposal],
    raw_items_count: int,
    anchoring_failures: int = 0,
    backend_name: str,
    llm_enabled: bool,
    fallback_used: bool,
    llm_error: str | None,
    sources: Mapping[str, Any],
    config: LocalLLMConfig,
) -> Dict[str, Any]:
    testable = [h for h in hypotheses if h.testability.testable]
    blocked = [h for h in hypotheses if not h.testability.testable]
    valid = [h for h in hypotheses if validate_hypothesis(h).valid]
    direct_unavailable = [
        h
        for h in hypotheses
        if h.candidate_action not in set(packet.available_actions)
    ]
    substrate_rejected = [
        r for r in guard_rejected if r.get("reason") == "substrate_retest_target"
    ]
    unavailable_rejected = [
        r for r in guard_rejected if r.get("reason") == "direct_unavailable_action"
    ]
    families = sorted({h.hypothesis_family for h in hypotheses})
    return {
        "config": {
            "schema_version": STATE_CONDITIONED_LLM_SCHEMA_VERSION,
            "m2_schema_version": M2_SCHEMA_VERSION,
            "inputs_read": ["P2.G5", "M2.G2", "M1.G0", "M3.G6"],
            "artifacts_not_modified": ["M3", "A32", "A33", "A40", "P2", "P3"],
            "local_llm_enabled": llm_enabled,
            "local_llm_backend": backend_name,
            "model_path": config.model_path,
            "device": config.device,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "sources": dict(sources),
        },
        "candidate_hypotheses": [h.to_dict() for h in hypotheses],
        "rejected_llm_proposals": [dict(r) for r in guard_rejected],
        "rejected_normalized_proposals": [r.to_dict() for r in norm_rejected],
        "summary": {
            "llm_enabled": llm_enabled,
            "local_llm_backend": backend_name,
            "fallback_used": fallback_used,
            "local_llm_error": llm_error,
            "raw_llm_items": raw_items_count,
            "hypotheses_generated": len(hypotheses),
            "valid_hypotheses": len(valid),
            "invalid_hypotheses_rejected": len(guard_rejected) + len(norm_rejected),
            "direct_unavailable_action_hypotheses": len(direct_unavailable),
            "direct_unavailable_action_rejected": len(unavailable_rejected),
            "substrate_retest_rejected": len(substrate_rejected),
            "action6_extension_retest_hypotheses_generated": False,
            "blocked_not_testable_hypotheses": len(blocked),
            "ready_for_m3_candidate_experiment_request": len(testable),
            "hypothesis_families_covered": families,
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "truth_status": M2_TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "llm_context_anchoring_failures": anchoring_failures,
        },
    }


# --------------------------------------------------------------------------- #
# Defensive artifact extraction helpers
# --------------------------------------------------------------------------- #


def _first_frontier_request(payload: Mapping[str, Any]) -> Dict[str, Any]:
    requests = payload.get("risk_aware_objective_handoff_requests", []) or []
    for row in requests:
        if isinstance(row, Mapping):
            return dict(row)
    # Permit being handed a bare request mapping directly.
    if payload.get("request_id"):
        return dict(payload)
    return {
        "request_id": "p2g5::bp35-0a0ad940::risk_aware_objective_completion::missing",
        "game_id": "bp35-0a0ad940",
        "frontier_type": "RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER",
        "blocked_capability": "objective_completion_after_risk_aware_safe_conversion",
        "frontier_reason": "RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION",
    }


def _frontier_context_replay(frontier_request: Mapping[str, Any]) -> Tuple[str, ...]:
    matrix = frontier_request.get("suggested_initial_experiment_matrix", {}) or {}
    options = matrix.get("source_policy_options", []) or []
    for option in options:
        if str(option).startswith("ACTION6"):
            return ("ACTION6",)
    return ("ACTION6",)


def _entity_role_candidates(
    m1_payload: Mapping[str, Any],
    *,
    cap_per_role: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    ledger = m1_payload.get("role_hypothesis_ledger", []) or []
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    for entry in ledger:
        if not isinstance(entry, Mapping):
            continue
        entity_id = str(entry.get("entity_id", ""))
        best_role = ""
        best_score = float("-inf")
        for role_hyp in entry.get("role_hypotheses", []) or []:
            if not isinstance(role_hyp, Mapping):
                continue
            score = float(role_hyp.get("score", 0) or 0)
            if score > best_score:
                best_score = score
                best_role = str(role_hyp.get("role", ""))
        if not entity_id or not best_role:
            continue
        bucket = by_role.setdefault(best_role, [])
        if len(bucket) < cap_per_role:
            bucket.append(
                {
                    "entity_id": entity_id,
                    "role_candidate": best_role,
                    "score": round(best_score, 4),
                }
            )
    return by_role


def _hud_summary(
    entity_role_candidates: Mapping[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    hud = entity_role_candidates.get("timer_or_hud", []) or []
    return {
        "detected": bool(hud),
        "hud_or_timer_entities": [item["entity_id"] for item in hud],
        "horizon_source": "hud_candidate" if hud else "unknown",
    }


def _relation_summary(m1_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = m1_payload.get("relation_delta_rows", []) or []
    return {"relation_delta_rows": len(rows)}


def _dynamic_invariants_summary(
    m1_payload: Mapping[str, Any],
    *,
    cap: int = 8,
) -> Dict[str, Any]:
    invariants = m1_payload.get("dynamic_invariant_candidates", []) or []
    sample: List[Dict[str, Any]] = []
    for entry in invariants[:cap]:
        if not isinstance(entry, Mapping):
            continue
        fields = {
            key: entry.get(key)
            for key in ("invariant_type", "family", "kind", "description", "entity_id")
            if key in entry
        }
        if fields:
            sample.append(fields)
    return {"count": len(invariants), "sample": sample}


def _action_effect_priors(m1_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    priors: List[Dict[str, Any]] = []
    for entry in m1_payload.get("action_effect_abstractions", []) or []:
        if not isinstance(entry, Mapping):
            continue
        priors.append(
            {
                "action": str(entry.get("action", "")),
                "effect_families": [
                    str(f) for f in entry.get("effect_families", []) or []
                ],
            }
        )
    return priors


def _m3g6_failure_summary(m3g6_payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = m3g6_payload.get("summary", {}) or {}
    return {
        "outcome_status": str(
            m3g6_payload.get("objective_completion_experiment_outcome_status")
            or summary.get("objective_completion_experiment_outcome_status", "")
        ),
        "objective_completion_signal": bool(
            summary.get("objective_completion_signal", False)
        ),
        "proxy_progress_without_completion_observed": bool(
            summary.get("proxy_progress_without_completion_observed", False)
        ),
        "commit_action_cells_blocked": int(
            summary.get("commit_action_cells_blocked", 0) or 0
        ),
        "levels_completed_after_rollout_max": float(
            summary.get("levels_completed_after_rollout_max", 0.0) or 0.0
        ),
        "cells_executed": int(summary.get("cells_executed", 0) or 0),
    }


def _m2g2_context(m2g2_payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = m2g2_payload.get("summary", {}) or {}
    return {
        "prior_hypotheses_generated": int(summary.get("hypotheses_generated", 0) or 0),
        "substrate_actions_not_target": list(SUBSTRATE_ACTIONS_NOT_TARGETS),
    }


def _required_observables(family: str, metric: str) -> List[str]:
    base = [metric]
    if family == UNLOCK_HYPOTHESIS_FAMILY:
        return [
            "available_actions_before_after",
            "object_positions_before_after",
            "objective_completion_signal",
        ]
    base.extend(["terminal_reentry_rate", "objective_completion_signal"])
    return list(dict.fromkeys(base))


def _secondary_metrics(metric: str) -> List[str]:
    options = ["objective_completion_signal", "terminal_reentry_rate"]
    return [m for m in options if m != metric]


def _falsification_signal(family: str, metric: str) -> str:
    if family == UNLOCK_HYPOTHESIS_FAMILY:
        return (
            "No new action becomes available after the proposed transition, or "
            "completion remains absent under all unlocked-action probes."
        )
    if metric == "objective_completion_signal":
        return "No completed-level delta versus matched controls."
    return "Candidate signal does not beat matched controls under the metric."


def _experiment_style(family: str) -> str:
    return {
        "objective_readiness_detection": "post_selector_objective_readiness_probe",
        "post_conversion_commit_action_search": "post_conversion_commit_action_matrix",
        "goal_state_representation_beyond_safe_progress": (
            "terminal_safe_progress_vs_completion_discriminator"
        ),
        "proxy_progress_vs_completion_discriminator": (
            "terminal_safe_progress_vs_completion_discriminator"
        ),
        "risk_aware_selector_completion_gap": (
            "risk_aware_policy_ablation_with_completion_metrics"
        ),
        UNLOCK_HYPOTHESIS_FAMILY: "post_conversion_action_availability_probe",
    }.get(family, "post_selector_objective_readiness_probe")


def _load_json(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_payload(payload: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M2.13a state-conditioned local LLM hypotheses.",
    )
    parser.add_argument("--frontiers", type=Path, default=DEFAULT_FRONTIERS_PATH)
    parser.add_argument("--m2-hypotheses", type=Path, default=DEFAULT_M2G2_HYPOTHESES_PATH)
    parser.add_argument("--m1-candidates", type=Path, default=DEFAULT_M1_CANDIDATES_PATH)
    parser.add_argument("--m3g6-results", type=Path, default=DEFAULT_M3G6_RESULTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_HYPOTHESES_OUTPUT_PATH)
    parser.add_argument("--m3-out", type=Path, default=DEFAULT_M3_REQUESTS_OUTPUT_PATH)
    parser.add_argument("--enable-local-llm", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--no-fallback-to-mock",
        action="store_true",
        help="Fail instead of falling back to the deterministic mock.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = LocalLLMConfig(
        enable_local_llm=bool(args.enable_local_llm),
        model_path=str(args.model_path),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        fallback_to_mock=not bool(args.no_fallback_to_mock),
    )
    outputs = run_state_conditioned_llm_generation(
        frontiers_path=args.frontiers,
        m2g2_path=args.m2_hypotheses,
        m1_candidates_path=args.m1_candidates,
        m3g6_results_path=args.m3g6_results,
        config=config,
    )
    write_payload(outputs["hypothesis_payload"], args.out)
    write_payload(outputs["m3_payload"], args.m3_out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "m3_output_path": str(args.m3_out),
                "summary": outputs["hypothesis_payload"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
