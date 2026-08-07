from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage_t import t10_2_protocol as protocol

REPO_ROOT = Path(__file__).resolve().parents[1]


def _roomy(_root=None) -> protocol.ResourceSnapshot:
    return protocol.ResourceSnapshot(
        repository_bytes=1,
        scratch_bytes=0,
        cache_bytes=0,
        resident_bytes=1,
        free_bytes=200 * protocol.GIB,
    )


@pytest.fixture()
def frozen_manifest(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "manifest.json"
    manifest = protocol.freeze_manifest(
        output_path=path,
        repo_root=REPO_ROOT,
        verify_repository=False,
    )
    return path, manifest


def _projections() -> dict:
    projections, _frames = _compact_frames_and_projections()
    return projections


def _quotient(*, step_count: int) -> dict:
    return {
        "format_version": protocol.COMPACT_QUOTIENT_FORMAT_VERSION,
        "summary_kind": "non_injective_multiset_counts",
        "entity_count": 0,
        "fact_count": 0,
        "role_rows": [],
        "fact_rows": [],
        "counter_rows": [{"amount": step_count, "name": "step_count"}],
        "register_rows": [],
        "topology_rows": [],
        "regime_index": 0,
    }


def _compact_frames_and_projections() -> tuple[dict, dict]:
    projections: dict[str, dict] = {}
    frames: dict[str, dict] = {}
    for frame_id in protocol.REGISTERED_FRAME_ORDER:
        before = _quotient(step_count=0)
        after = _quotient(step_count=1)
        observation = {
            "object_deltas": {},
            "relation_deltas": {"touching": 1.0},
            "topology_deltas": {},
            "known_channels": ["relations"],
            "residual": [],
        }
        frame = {
            "before": before,
            "after": after,
            "before_hash": protocol.canonical_sha256(before),
            "after_hash": protocol.canonical_sha256(after),
            "observation": observation,
            "observation_hash": protocol.canonical_sha256(observation),
            "complete": True,
            "missing": [],
            "covered_channels": ["objects", "relations", "topology"],
            "provenance": [f"deterministic-{frame_id}-encoder"],
        }
        safe_projection = {
            "format_version": protocol.COMPACT_PROJECTION_FORMAT_VERSION,
            "frame_id": frame_id,
            **{
                key: value
                for key, value in frame.items()
                if key not in {"before", "after"}
            },
        }
        projections[frame_id] = {
            **safe_projection,
            "canonical_hash": protocol.canonical_sha256(safe_projection),
        }
        frames[frame_id] = frame
    return projections, frames


def _event(
    manifest: dict,
    index: int,
    *,
    positive: bool,
    game_id: str | None = None,
    partial_transport: bool = False,
) -> dict:
    game = game_id or protocol.SOURCE_GAMES[index % 2]
    transport = {
        "mapping_kind": "exact",
        "comparable": True,
        "round_trip_exact": True,
        "entity_permutation_invariant": True,
        "commutative_exact": True,
        "live_graph_exact_attested": True,
        "summary_commutative_exact": True,
        "certificate_count": 1,
        "exact_certificate_count": 1,
        "partial_certificate_count": 0,
        "identity_root_certificate_exact": False,
    }
    if partial_transport:
        transport = {
            "mapping_kind": "partial",
            "comparable": False,
            "round_trip_exact": False,
            "entity_permutation_invariant": False,
            "commutative_exact": False,
            "live_graph_exact_attested": False,
            "summary_commutative_exact": False,
            "certificate_count": 1,
            "exact_certificate_count": 0,
            "partial_certificate_count": 1,
            "identity_root_certificate_exact": False,
        }
    transport_certificate = {
        "source_frame": "root_only",
        "target_frame": "allocentric_object_relative",
        "transport_hash": "a" * 64,
        "certificate_hash": "b" * 64,
        "coverage": 1.0 if not partial_transport else 0.5,
        "exact": not partial_transport,
        "comparable": not partial_transport,
        "mapping_kind": "exact" if not partial_transport else "partial",
        "round_trip_exact": not partial_transport,
        "certifies_gauge_equivalence": not partial_transport,
        "projection_complete": True,
        "live_graph_exact_attested": not partial_transport,
        "summary_commutative_exact": not partial_transport,
        "commutativity": {
            "before": "c" * 64,
            "after": "d" * 64,
            "dynamics": "e" * 64,
            "exact": not partial_transport,
        },
    }
    projections, model_frames = _compact_frames_and_projections()
    return protocol.seal_event(
        {
            "event_id": f"event-{index:04d}",
            "game_id": game,
            "seed": 0,
            "split": "discovery",
            "reset_index": 0,
            "step_index": index % protocol.SOURCE_ACTIONS_PER_RESET,
            "action": {
                "schema": "apply-relation-v1",
                "name": "ACTION1",
                "data": {"role": "target"},
                "executed": True,
            },
            "outcome": {
                "progression": int(positive),
                "terminal": False,
                "goal": False,
            },
            "projections": projections,
            "model_view": {"frames": model_frames},
            "learned_predicates": ["opens_relation"],
            "labels": {
                "opens_relation": positive,
                "diagnostic_always_false": False,
            },
            "correspondence": {
                "confident_matches": 95,
                "fully_ambiguous_matches": 0,
                "fraction_denominator": 100,
                "confident_fraction": 0.95,
                "fully_ambiguous_fraction": 0.0,
            },
            "transport": transport,
            "transport_certificates": [transport_certificate],
            "transport_orbits": [],
            "prefix": {
                "nonterminal": True,
                "evaluable": True,
                "coherent_frames": 4,
            },
            "provenance": {
                "kind": "fresh_source_trajectory",
                "game_id": game,
                "seed": 0,
                "split": "discovery",
                "manifest_checksum": manifest["manifest_checksum"],
                "environment_sha256": manifest["environment_sha256"],
                "collector": protocol.COMPACT_PROJECTION_FORMAT_VERSION,
                "projector_bank": list(protocol.REGISTERED_FRAME_ORDER),
                "summary_hashes": [
                    model_frames[frame_id][stage]
                    for frame_id in sorted(model_frames)
                    for stage in ("before_hash", "after_hash")
                ],
                "observation_hashes": [
                    model_frames[frame_id]["observation_hash"]
                    for frame_id in sorted(model_frames)
                ],
                "transport_orbit_hashes": [],
                "transport_certificate_hashes": ["b" * 64],
                "physical_outcome_known_channels": [
                    "goal",
                    "progress",
                    "terminal",
                ],
                "raw_runtime_state_retained": False,
            },
        }
    )


def _raw_event(event_id: str) -> dict:
    projections, model_frames = _compact_frames_and_projections()
    return {
        "event_id": event_id,
        "reset_index": 0,
        "step_index": 0,
        "action": {
            "schema": "apply-relation-v1",
            "name": "ACTION1",
            "data": {},
            "executed": True,
        },
        "outcome": {"progression": 0, "terminal": False, "goal": False},
        "projections": projections,
        "model_view": {"frames": model_frames},
        "transport": {
            "mapping_kind": "partial",
            "comparable": False,
            "round_trip_exact": False,
            "entity_permutation_invariant": True,
            "commutative_exact": False,
            "live_graph_exact_attested": False,
            "summary_commutative_exact": False,
            "certificate_count": 0,
            "exact_certificate_count": 0,
            "partial_certificate_count": 0,
            "identity_root_certificate_exact": False,
        },
        "transport_certificates": [],
        "transport_orbits": [],
        "correspondence": {
            "confident_matches": 1,
            "fully_ambiguous_matches": 0,
            "fraction_denominator": 1,
            "confident_fraction": 1.0,
            "fully_ambiguous_fraction": 0.0,
        },
        "prefix": {
            "nonterminal": True,
            "evaluable": True,
            "coherent_frames": 4,
        },
        "provenance": {
            "collector": protocol.COMPACT_PROJECTION_FORMAT_VERSION,
            "projector_bank": list(protocol.REGISTERED_FRAME_ORDER),
            "summary_hashes": [
                model_frames[frame_id][stage]
                for frame_id in sorted(model_frames)
                for stage in ("before_hash", "after_hash")
            ],
            "observation_hashes": [
                model_frames[frame_id]["observation_hash"]
                for frame_id in sorted(model_frames)
            ],
            "transport_orbit_hashes": [],
            "transport_certificate_hashes": [],
            "physical_outcome_known_channels": [
                "goal",
                "progress",
                "terminal",
            ],
            "raw_runtime_state_retained": False,
        },
    }


def _as_replay(event: dict, manifest: dict, *, source_line: int = 1) -> dict:
    unsigned = dict(event)
    unsigned.pop("event_checksum", None)
    game = str(unsigned["game_id"])
    conversion_code_sha256 = {
        role: manifest["code_sha256"][path]
        for role, path in protocol.REPLAY_CONVERSION_CODE_PATHS.items()
        if path in manifest["code_sha256"]
    }
    unsigned["split"] = protocol.REPLAY_SPLIT
    replay_provenance = dict(unsigned.get("provenance", {}))
    replay_provenance.update(
        {
            "kind": "frozen_source_replay",
            "game_id": game,
            "seed": int(unsigned["seed"]),
            "split": protocol.REPLAY_SPLIT,
            "manifest_checksum": manifest["manifest_checksum"],
            "source_format": "sage12-bound-trajectory-v4.3",
            "source_shard_sha256": manifest["frozen_source_shards"][game]["sha256"],
            "source_row_sha256": "1" * 64,
            "source_line": source_line,
            "pair_digest": "2" * 64,
            "arm": "left",
            "trace_digest": "3" * 64,
            "expected_pre_state_sha256": "4" * 64,
            "replay_pre_state_sha256": "5" * 64,
            "post_state_sha256": "6" * 64,
            "frame_before_sha256": "7" * 64,
            "frame_after_sha256": "8" * 64,
            "conversion_code_sha256": conversion_code_sha256,
            **{
                f"{role}_sha256": digest
                for role, digest in conversion_code_sha256.items()
            },
            "raw_frames_retained": False,
            "graphs_retained": False,
        }
    )
    unsigned["provenance"] = replay_provenance
    return protocol.seal_event(unsigned)


def _source_metrics(*, all_controls: bool = True) -> dict:
    completed = {name: all_controls for name in protocol.REGISTERED_SOURCE_CONTROLS}
    control_results = {
        name: {
            "attempted": all_controls,
            "execution_ok": all_controls,
            "scientific_pass": all_controls,
            "completed": all_controls,
            "passed": all_controls,
        }
        for name in protocol.REGISTERED_SOURCE_CONTROLS
    }
    control_results["no_transport"]["degradation"] = 0.1
    control_results["binding_swap"]["degradation"] = 0.1
    control_results["transport_oracle"].update(
        {
            "nontrivial_exact_commutative_certificate_count": 1,
            "certified_orbit_witness_candidate_count": 1,
            "posterior_merged_gauge_class_count": 1,
        }
    )
    return {
        "attempted_controls": dict(completed),
        "execution_ok_controls": dict(completed),
        "completed_controls": completed,
        "grammar_oracle": {
            "progress_games": 2,
            "levels": 2,
            "errors": 0,
            "illegal_actions": 0,
            "game_overs": 0,
            "positive_folds": list(protocol.SOURCE_GAMES),
        },
        "learned": {
            "positive_fold_ranks": {
                game: index + 1 for index, game in enumerate(protocol.SOURCE_GAMES)
            },
            "oracle_level_recovery": 0.5,
            "nonnegative_games": 2,
            "paired_rate_ci_lower": 0.01,
            "game_seed_probe_accuracy_increment": 0.1,
            "illegal_actions": 0,
            "errors": 0,
            "common_posterior_passed": True,
            "option_synthesis_passed": True,
        },
        "controls": {
            "no_transport_degradation": 0.1,
            "binding_swap_degradation": 0.1,
            "capacity_matched_independent_posterior_passed": all_controls,
            "transport_oracle_passed": all_controls,
            "dynamics_oracle_passed": all_controls,
            "goal_oracle_passed": all_controls,
            "best_executed_sequence_oracle_passed": all_controls,
            "option_oracle_passed": all_controls,
            "complete_program_oracle_passed": all_controls,
        },
        "control_results": control_results,
        "safety_gate_passed": True,
        "resource_gate_passed": True,
        "challenger_recipe": {
            "bound": True,
            "path": protocol.CHALLENGER_RECIPE_FILENAME,
            "artifact": {"bytes": 1, "sha256": "a" * 64},
            "recipe_checksum": "b" * 64,
        },
    }


def _validation_metrics() -> dict:
    return {
        "all_pairs_executed": True,
        "counterbalanced_and_reset": True,
        "total_level_advantage": 1,
        "nonnegative_games": 2,
        "paired_rate_ci_lower": 0.01,
        "illegal_actions": 0,
        "errors": 0,
        "unregistered_stops": 0,
        "game_over_rate_delta": 0.0,
        "budget_configuration_exact": True,
        "within_action_caps": True,
        "maximum_actions_per_controller": (
            protocol.VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
        ),
        "completed_budget_fraction": 0.95,
        "decision_latency_sample_count": 1,
        "observation_latency_sample_count": 1,
        "latency_samples_complete": True,
        "decision_p95_ms": 700.0,
        "decision_p99_ms": 2_000.0,
        "observation_p95_ms": 450.0,
        "observation_p99_ms": 2_500.0,
        "wall_seconds": 1_000.0,
    }


def _validation_arm(*, levels: int = 0, game_overs: int = 0) -> dict:
    reasons = ["option_exhausted"] * protocol.VALIDATION_RESETS_PER_GAME_SEED
    if game_overs:
        reasons[0] = "game_over"
    elif levels:
        reasons[0] = "progression"
    resets = [
        {
            "reset_index": reset_index,
            "planned_actions": 1,
            "completed_actions": 1,
            "stop_reason": reason,
        }
        for reset_index, reason in enumerate(reasons)
    ]
    actions = len(resets)
    return {
        "levels": levels,
        "legal_actions": actions,
        "game_overs": game_overs,
        "illegal_actions": 0,
        "errors": 0,
        "planned_actions": actions,
        "completed_actions": actions,
        "unregistered_stops": 0,
        "reset_summaries": resets,
        "decision_latency_ms": [10.0] * actions,
        "observation_latency_ms": [5.0] * actions,
    }


def _validation_runs() -> list[dict]:
    runs = []
    for pair_index, (game, seed) in enumerate(
        (game, seed)
        for game in protocol.VALIDATION_GAMES
        for seed in protocol.VALIDATION_SEEDS
    ):
        baseline = _validation_arm()
        candidate = _validation_arm(
            levels=1,
            game_overs=int(pair_index == 0),
        )
        runs.append(
            {
                "game_id": game,
                "seed": seed,
                "baseline": baseline,
                "t10_2": candidate,
                "registered_resets_per_controller": (
                    protocol.VALIDATION_RESETS_PER_GAME_SEED
                ),
                "registered_max_actions_per_reset": (
                    protocol.VALIDATION_ACTIONS_PER_RESET
                ),
                "controller_order": (
                    ["t10_1", "t10_2"] if pair_index % 2 == 0 else ["t10_2", "t10_1"]
                ),
                "counterbalanced": True,
                "posterior_reset": True,
                "learning_between_controllers": False,
                "wall_seconds": 1.0,
            }
        )
    return runs


def test_manifest_binds_baseline_code_inputs_runtime_and_source_shards(
    frozen_manifest: tuple[Path, dict],
) -> None:
    path, manifest = frozen_manifest
    loaded = protocol.load_manifest(
        path,
        repo_root=REPO_ROOT,
        verify_repository=False,
    )
    assert loaded == manifest
    assert loaded["baseline_commit"] == protocol.BASELINE_COMMIT
    assert loaded["registered_phases"] == list(protocol.PHASES)
    assert loaded["artifact_contract"] == {
        "physical_event_format": protocol.EVENT_FORMAT_VERSION,
        "projection_format": protocol.COMPACT_PROJECTION_FORMAT_VERSION,
        "structural_quotient_format": protocol.COMPACT_QUOTIENT_FORMAT_VERSION,
        "observer_frames": list(protocol.REGISTERED_FRAME_ORDER),
        "maximum_model_view_bytes": protocol.MAXIMUM_MODEL_VIEW_BYTES,
        "maximum_compact_event_bytes": protocol.MAXIMUM_COMPACT_EVENT_BYTES,
        "raw_frames_persisted": False,
        "full_graphs_persisted": False,
    }
    assert set(loaded["frozen_source_shards"]) == set(protocol.SOURCE_GAMES)
    assert set(loaded["source_environment_metadata"]) == set(protocol.SOURCE_GAMES)
    assert set(loaded["environment"]["runtime_versions"]) == {
        "arc-agi",
        "arcengine",
    }
    assert "theory/sage_t/frame_transport_v10_2.py" in loaded["code_sha256"]
    assert "theory/sage_t/frame_adapters_v10_2.py" in loaded["code_sha256"]
    assert "theory/sage12/mt/graph.py" in loaded["code_sha256"]
    assert "theory/sage12/topological_invariants_v4_19.py" in loaded["code_sha256"]
    assert {
        "theory/sage_t/compiler.py",
        "theory/sage_t/progress_witness_v10.py",
        "theory/live_transition_loop.py",
        "theory/sage12/scene_graph.py",
        "theory/sage12/mt/transition.py",
        "theory/m1/polymorphic_a25_adapter.py",
        "theory/m2/m3_execution_smoke.py",
        "theory/non_ar25_active_micro_run.py",
        "theory/real_env_option_adapter.py",
        "theory/unified_cognition_ab_benchmark.py",
    } <= set(loaded["code_sha256"])
    assert "tests/test_sage_t_t10_2_protocol.py" in loaded["code_sha256"]


def test_manifest_rejects_semantic_and_environment_drift(
    frozen_manifest: tuple[Path, dict],
) -> None:
    path, manifest = frozen_manifest
    drifted = dict(manifest)
    drifted["baseline_commit"] = "0" * 40
    protocol.write_compact_json(
        path,
        protocol.signed_payload(drifted, checksum_key="manifest_checksum"),
    )
    with pytest.raises(protocol.ManifestDriftError, match="baseline"):
        protocol.load_manifest(
            path,
            repo_root=REPO_ROOT,
            verify_repository=False,
        )

    protocol.write_compact_json(path, manifest)
    with pytest.raises(protocol.ManifestDriftError, match="environment"):
        protocol.load_manifest(
            path,
            repo_root=REPO_ROOT,
            environment={"runtime": "drifted"},
            verify_repository=False,
        )


def test_manifest_rejects_recursive_gate_budget_and_resource_drift(
    frozen_manifest: tuple[Path, dict],
) -> None:
    path, manifest = frozen_manifest
    mutations = (
        (
            "artifact_contract",
            lambda value: value["artifact_contract"].__setitem__(
                "maximum_compact_event_bytes",
                protocol.MAXIMUM_COMPACT_EVENT_BYTES + 1,
            ),
        ),
        (
            "qa",
            lambda value: value["qa_gate"].__setitem__("minimum_predicate_support", 31),
        ),
        (
            "source_plan",
            lambda value: value["source_plan"]["splits"]["discovery"].__setitem__(
                "resets_per_game_seed", 3
            ),
        ),
        (
            "source_gate",
            lambda value: value["source_gate"].__setitem__(
                "maximum_positive_fold_rank", 9
            ),
        ),
        (
            "validation_gate",
            lambda value: value["validation_gate"].__setitem__(
                "maximum_decision_p95_ms", 751.0
            ),
        ),
        (
            "resource_limits",
            lambda value: value["resource_limits"].__setitem__(
                "maximum_resident_bytes",
                value["resource_limits"]["maximum_resident_bytes"] + 1,
            ),
        ),
        (
            "firewall",
            lambda value: value["firewall"].__setitem__("ar25_opened", True),
        ),
    )
    for label, mutate in mutations:
        drifted = json.loads(json.dumps(manifest))
        mutate(drifted)
        protocol.write_compact_json(
            path,
            protocol.signed_payload(drifted, checksum_key="manifest_checksum"),
        )
        with pytest.raises(protocol.ManifestDriftError, match=label):
            protocol.load_manifest(
                path,
                repo_root=REPO_ROOT,
                verify_code=False,
                verify_inputs=False,
                verify_environment=False,
                verify_repository=False,
            )


def test_environment_metadata_hash_is_json_canonical_not_eol_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{"game_id":"bp35","tags":["a"]}\r\n')
    second.write_bytes(b'{\n  "tags": ["a"],\n  "game_id": "bp35"\n}\n')
    assert protocol.canonical_json_file_sha256(
        first
    ) == protocol.canonical_json_file_sha256(second)


def test_cli_registers_exactly_seven_phases_and_emits_compact_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = protocol.build_parser()
    phase_action = next(action for action in parser._actions if action.dest == "phase")
    assert tuple(phase_action.choices) == protocol.PHASES
    assert "all" not in phase_action.choices
    with pytest.raises(SystemExit):
        parser.parse_args(["all"])

    manifest_path = tmp_path / "cli-manifest.json"
    assert (
        protocol.main(
            [
                "freeze",
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(REPO_ROOT),
                "--skip-repository-check",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert json.loads(output)["status"] == "FROZEN_BEFORE_T10_2_COLLECTION"


def test_cli_rejects_unsigned_inputs_and_binds_source_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = protocol.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["source-train", "--input", str(tmp_path / "metrics.json")])

    captured: dict[str, object] = {}

    def source_train(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "TEST_CODE_BOUND_SOURCE_TRAIN", "passed": True}

    monkeypatch.setattr(protocol, "source_train_phase", source_train)
    assert (
        protocol.main(
            [
                "source-train",
                "--manifest",
                str(tmp_path / "unused-manifest.json"),
                "--output-dir",
                str(tmp_path / "output"),
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        == 0
    )
    assert callable(captured["trainer"])
    assert captured["trainer"].__name__ == "run_source_trainer"  # type: ignore[union-attr]
    assert "metrics" not in captured
    assert json.loads(capsys.readouterr().out)["status"] == (
        "TEST_CODE_BOUND_SOURCE_TRAIN"
    )


def test_cli_validate_injects_both_code_bound_frozen_policy_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from theory.sage_t import t10_2_runtime as runtime

    captured: dict[str, object] = {}

    class BaselineFactory:
        def __init__(self, **kwargs: object) -> None:
            captured["baseline_kwargs"] = kwargs

    class ChallengerFactory:
        def __init__(self, **kwargs: object) -> None:
            captured["challenger_kwargs"] = kwargs

    class ValidationFactory:
        def __init__(self, **kwargs: object) -> None:
            captured["validation_kwargs"] = kwargs

    manifest = {"manifest_checksum": "a" * 64}
    authorized_source = {
        "status": "PASS_SOURCE_GATE",
        "manifest_checksum": manifest["manifest_checksum"],
    }

    def validate(**kwargs: object) -> dict[str, object]:
        captured["validate_kwargs"] = kwargs
        return {"status": "TEST_CODE_BOUND_VALIDATION", "passed": True}

    monkeypatch.setattr(
        runtime, "T10_1BehaviorFrozenPolicyFactory", BaselineFactory, raising=False
    )
    monkeypatch.setattr(
        runtime, "T10_2GaugePolicyFactory", ChallengerFactory, raising=False
    )
    monkeypatch.setattr(runtime, "T10_2ValidationFactory", ValidationFactory)
    monkeypatch.setattr(protocol, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        protocol,
        "require_source_gate",
        lambda **_kwargs: authorized_source,
    )
    monkeypatch.setattr(protocol, "validate_phase", validate)

    output = tmp_path / "output"
    source_path = output / "source_report.json"
    assert (
        protocol.main(
            [
                "validate",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--output-dir",
                str(output),
                "--source-report",
                str(source_path),
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        == 0
    )
    validation_kwargs = captured["validation_kwargs"]
    assert isinstance(validation_kwargs, dict)
    assert validation_kwargs["t10_1_policy_factory"].__class__ is BaselineFactory
    assert validation_kwargs["t10_2_policy_factory"].__class__ is ChallengerFactory
    assert captured["baseline_kwargs"] == {"repo_root": str(REPO_ROOT)}
    assert captured["challenger_kwargs"] == {
        "source_report": authorized_source,
        "manifest": manifest,
        "output_dir": output,
    }
    assert json.loads(capsys.readouterr().out)["status"] == (
        "TEST_CODE_BOUND_VALIDATION"
    )


def test_forbidden_game_is_rejected_before_environment_factory(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    path, _manifest = frozen_manifest
    calls: list[str] = []

    def factory(game_id: str):
        calls.append(game_id)
        return []

    with pytest.raises(protocol.FirewallError, match="non-source"):
        protocol.collect_phase(
            manifest_path=path,
            output_dir=tmp_path / "output",
            repo_root=REPO_ROOT,
            env_factory=factory,
            games=(protocol.VALIDATION_GAMES[0],),
            resource_probe=_roomy,
        )
    assert calls == []


def test_collection_finishes_discovery_before_confirmation_and_marks_holdout_fold(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    path, _manifest = frozen_manifest
    calls: list[tuple[str, int, str, str | None, tuple[str, ...]]] = []

    def factory(
        game_id: str,
        seed: int,
        split: str,
        held_out_game: str | None,
        training_games: tuple[str, ...],
    ):
        calls.append((game_id, seed, split, held_out_game, training_games))
        return [_raw_event(f"{split}:{game_id}:{seed}")]

    report = protocol.collect_phase(
        manifest_path=path,
        output_dir=tmp_path / "output",
        repo_root=REPO_ROOT,
        env_factory=factory,
        resource_probe=_roomy,
        _test_only_allow_factory=True,
    )
    assert report["event_count"] == 18
    assert all(call[2] == "discovery" for call in calls[:9])
    assert all(call[3] is None for call in calls[:9])
    assert all(call[2] == "leave_one_game_out_confirmation" for call in calls[9:])
    assert all(call[3] == call[0] for call in calls[9:])
    assert all(call[0] not in call[4] for call in calls[9:])


def test_collection_wall_limit_refuses_before_next_factory_call(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    path, _manifest = frozen_manifest
    calls: list[str] = []
    ticks = iter((0.0, 5_401.0))

    def factory(game_id: str):
        calls.append(game_id)
        return []

    with pytest.raises(protocol.ResourceGateError, match="5,400"):
        protocol.collect_phase(
            manifest_path=path,
            output_dir=tmp_path / "output",
            repo_root=REPO_ROOT,
            env_factory=factory,
            resource_probe=_roomy,
            clock=lambda: next(ticks),
            _test_only_allow_factory=True,
            _test_only_allow_clock=True,
        )
    assert calls == []


def test_collection_wall_limit_includes_persistence_and_clock_is_monotone(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, _manifest = frozen_manifest

    def factory(game_id: str, seed: int, split: str, **_context: object):
        return [_raw_event(f"{split}:{game_id}:{seed}")]

    # Start + two samples for each of 18 lanes + ledger preflight + the two
    # persistence samples remain in budget; only the post-persistence recheck
    # crosses the registered cap.
    ticks = iter([0.0] * 40 + [5_401.0])
    output = tmp_path / "post-persistence"
    with pytest.raises(protocol.ResourceGateError, match="after persistence"):
        protocol.collect_phase(
            manifest_path=manifest_path,
            output_dir=output,
            repo_root=REPO_ROOT,
            env_factory=factory,
            resource_probe=_roomy,
            clock=lambda: next(ticks),
            _test_only_allow_factory=True,
            _test_only_allow_clock=True,
        )
    assert (output / "collection_report.json").is_file()

    calls: list[str] = []

    def unopened_factory(game_id: str):
        calls.append(game_id)
        return []

    regressing_ticks = iter((1.0, 0.0))
    with pytest.raises(protocol.DataGateError, match="regressed"):
        protocol.collect_phase(
            manifest_path=manifest_path,
            output_dir=tmp_path / "regressing",
            repo_root=REPO_ROOT,
            env_factory=unopened_factory,
            resource_probe=_roomy,
            clock=lambda: next(regressing_ticks),
            _test_only_allow_factory=True,
            _test_only_allow_clock=True,
        )
    assert calls == []


def test_duplicate_events_and_identity_bearing_model_views_fail_closed(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    row = _event(manifest, 0, positive=True)
    with pytest.raises(protocol.DataGateError, match="duplicate physical event_id"):
        protocol.validate_source_events([row, row], manifest=manifest)

    contaminated = dict(_event(manifest, 1, positive=True))
    contaminated["model_view"] = {"game_id": protocol.SOURCE_GAMES[0]}
    contaminated = protocol.seal_event(contaminated)
    with pytest.raises(protocol.FirewallError, match="identity-bearing"):
        protocol.validate_source_events([contaminated], manifest=manifest)

    complete_graph = dict(_event(manifest, 2, positive=True))
    complete_graph.pop("event_checksum")
    complete_graph["model_view"] = {
        "frames": {"root_only": {"entities": [], "true_facts": []}}
    }
    with pytest.raises(protocol.FirewallError, match="identity-bearing"):
        protocol.validate_source_events(
            [protocol.seal_event(complete_graph)], manifest=manifest
        )


def test_strict_event_schema_and_recursive_transfer_firewall(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    missing_action = dict(_event(manifest, 0, positive=True))
    missing_action.pop("event_checksum")
    missing_action.pop("action")
    with pytest.raises(protocol.DataGateError, match="strict action"):
        protocol.validate_source_events(
            [protocol.seal_event(missing_action)], manifest=manifest
        )

    raw_graph = dict(_event(manifest, 1, positive=True))
    raw_graph.pop("event_checksum")
    raw_graph["transport_certificates"] = [
        {
            **raw_graph["transport_certificates"][0],
            "audit": {"nested": {"raw_graph": {"nodes": [1]}}},
        }
    ]
    with pytest.raises(protocol.FirewallError, match="raw or identity-bearing"):
        protocol.validate_source_events(
            [protocol.seal_event(raw_graph)], manifest=manifest
        )

    incoherent = dict(_event(manifest, 2, positive=True))
    incoherent.pop("event_checksum")
    incoherent["prefix"] = {**incoherent["prefix"], "coherent_frames": 1}
    with pytest.raises(protocol.DataGateError, match="coherence count"):
        protocol.validate_source_events(
            [protocol.seal_event(incoherent)], manifest=manifest
        )

    missing_frame = dict(_event(manifest, 3, positive=True))
    missing_frame.pop("event_checksum")
    missing_frame["projections"] = dict(missing_frame["projections"])
    missing_frame["projections"].pop("action_rooted_topological")
    missing_frame["prefix"] = {**missing_frame["prefix"], "coherent_frames": 3}
    with pytest.raises(protocol.DataGateError, match="exactly four"):
        protocol.validate_source_events(
            [protocol.seal_event(missing_frame)], manifest=manifest
        )

    invalid_outcome = dict(_event(manifest, 4, positive=True))
    invalid_outcome.pop("event_checksum")
    invalid_outcome["outcome"] = {
        "progression": "1",
        "terminal": False,
        "goal": False,
    }
    with pytest.raises(protocol.DataGateError, match="finite progression"):
        protocol.validate_source_events(
            [protocol.seal_event(invalid_outcome)], manifest=manifest
        )

    unknown_terminal = dict(_event(manifest, 5, positive=True))
    unknown_terminal.pop("event_checksum")
    unknown_terminal["outcome"] = {
        "progression": 0.0,
        "terminal": None,
        "goal": False,
    }
    with pytest.raises(protocol.DataGateError, match="boolean terminal"):
        protocol.validate_source_events(
            [protocol.seal_event(unknown_terminal)], manifest=manifest
        )


def test_compact_projection_and_model_view_are_strictly_hash_bound(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    runtime_shape = dict(_event(manifest, 19, positive=True))
    protocol.validate_source_events([runtime_shape], manifest=manifest)

    tampered = dict(_event(manifest, 20, positive=True))
    tampered.pop("event_checksum")
    tampered["projections"] = dict(tampered["projections"])
    tampered_root = dict(tampered["projections"]["root_only"])
    tampered_root["complete"] = False
    tampered["projections"]["root_only"] = tampered_root
    with pytest.raises(protocol.DataGateError, match="canonical hash mismatch"):
        protocol.validate_source_events(
            [protocol.seal_event(tampered)], manifest=manifest
        )

    unknown = dict(_event(manifest, 21, positive=True))
    unknown.pop("event_checksum")
    unknown["projections"] = dict(unknown["projections"])
    unknown_root = dict(unknown["projections"]["root_only"])
    unknown_root["canonical_payload"] = {"forbidden": "alternate encoding"}
    unknown["projections"]["root_only"] = unknown_root
    with pytest.raises(protocol.DataGateError, match="schema drifted"):
        protocol.validate_source_events(
            [protocol.seal_event(unknown)], manifest=manifest
        )

    invalid_quotient = dict(_event(manifest, 22, positive=True))
    invalid_quotient.pop("event_checksum")
    invalid_quotient["model_view"] = json.loads(
        json.dumps(invalid_quotient["model_view"])
    )
    invalid_quotient["model_view"]["frames"]["root_only"]["before"]["counter_rows"][0][
        "name"
    ] = "unregistered_counter"
    with pytest.raises(protocol.DataGateError, match="invalid model frame"):
        protocol.validate_source_events(
            [protocol.seal_event(invalid_quotient)], manifest=manifest
        )


def test_compact_event_and_model_view_byte_budgets_fail_closed(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    _path, manifest = frozen_manifest
    event = dict(_event(manifest, 23, positive=False))
    event.pop("event_checksum")
    event["audit_padding"] = "z" * protocol.MAXIMUM_COMPACT_EVENT_BYTES
    with pytest.raises(protocol.DataGateError, match="sealed physical event exceeds"):
        protocol.seal_event(event)

    _projections_by_frame, frames = _compact_frames_and_projections()
    frames["root_only"] = dict(frames["root_only"])
    frames["root_only"]["provenance"] = ["p" * protocol.MAXIMUM_MODEL_VIEW_BYTES]
    with pytest.raises(protocol.DataGateError, match="model_view exceeds"):
        protocol._validate_compact_model_view(
            {"frames": frames}, event_id="oversized-model"
        )

    with pytest.raises(protocol.DataGateError, match="finite canonical JSON"):
        protocol.seal_event({"event_id": "nan-event", "value": float("nan")})

    oversized_path = tmp_path / "oversized.jsonl"
    oversized_path.write_text(
        '{"padding":"' + "z" * protocol.MAXIMUM_COMPACT_EVENT_BYTES + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises(protocol.DataGateError, match="oversized JSONL row"):
        protocol.read_event_ledger(oversized_path)


def test_transport_exactness_and_certificate_provenance_are_jointly_bound(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    inconsistent = dict(_event(manifest, 24, positive=False))
    inconsistent.pop("event_checksum")
    inconsistent["transport_certificates"] = [
        {
            **inconsistent["transport_certificates"][0],
            "summary_commutative_exact": False,
        }
    ]
    with pytest.raises(protocol.DataGateError, match="inconsistent transport"):
        protocol.validate_source_events(
            [protocol.seal_event(inconsistent)], manifest=manifest
        )

    unbound = dict(_event(manifest, 25, positive=False))
    unbound.pop("event_checksum")
    unbound["provenance"] = {
        **unbound["provenance"],
        "transport_certificate_hashes": ["f" * 64],
    }
    with pytest.raises(protocol.DataGateError, match="certificate provenance"):
        protocol.validate_source_events(
            [protocol.seal_event(unbound)], manifest=manifest
        )


def test_qa_accepts_registered_partial_transport_but_rejects_universal_label(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    events = [
        _event(
            manifest,
            index,
            positive=index < 32,
            partial_transport=index == 0,
        )
        for index in range(64)
    ]
    report = protocol.build_qa_report(manifest=manifest, events=events)
    assert report["passed"] is True
    assert report["metrics"]["partial_noncomparable_transports"] == 1

    universal = [_event(manifest, index + 100, positive=True) for index in range(64)]
    failed = protocol.build_qa_report(manifest=manifest, events=universal)
    assert failed["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert failed["checks"]["learned_predicate_prevalence_and_support"] is False


def test_qa_requires_declared_targets_and_at_least_one_exact_certificate(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    events = [_event(manifest, index, positive=index < 32) for index in range(64)]

    undeclared = []
    for event in events:
        row = dict(event)
        row.pop("event_checksum")
        row.pop("learned_predicates")
        undeclared.append(protocol.seal_event(row))
    undeclared_report = protocol.build_qa_report(manifest=manifest, events=undeclared)
    assert undeclared_report["checks"]["learned_predicates_present"] is False

    all_partial = [
        _event(manifest, index + 100, positive=index < 32, partial_transport=True)
        for index in range(64)
    ]
    partial_report = protocol.build_qa_report(manifest=manifest, events=all_partial)
    assert partial_report["checks"]["transport_round_trip_exact"] is True
    assert partial_report["checks"]["exact_transport_evidence_present"] is False
    assert partial_report["passed"] is False

    rare_only = []
    for event in events:
        row = dict(event)
        row.pop("event_checksum")
        row["learned_predicates"] = ["game_over"]
        rare_only.append(protocol.seal_event(row))
    rare_report = protocol.build_qa_report(manifest=manifest, events=rare_only)
    assert rare_report["checks"]["learned_predicates_present"] is False


def test_correspondence_fractions_require_auditable_bounded_ratios(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    event = _event(manifest, 690, positive=True)
    unsigned = dict(event)
    unsigned.pop("event_checksum")
    unsigned["correspondence"] = {
        **unsigned["correspondence"],
        "confident_matches": 101,
    }
    with pytest.raises(protocol.DataGateError, match="exceeds denominator"):
        protocol.validate_source_events(
            [protocol.seal_event(unsigned)],
            manifest=manifest,
            replay=False,
        )

    unsigned["correspondence"] = {
        **unsigned["correspondence"],
        "confident_matches": 95,
        "confident_fraction": 0.94,
    }
    with pytest.raises(protocol.DataGateError, match="registered ratio"):
        protocol.validate_source_events(
            [protocol.seal_event(unsigned)],
            manifest=manifest,
            replay=False,
        )


def test_qa_pools_correspondence_trials_instead_of_averaging_events(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    events = [_event(manifest, index, positive=index < 32) for index in range(64)]
    pooled = []
    for index, event in enumerate(events):
        unsigned = dict(event)
        unsigned.pop("event_checksum")
        if index == 0:
            correspondence = {
                "confident_matches": 0,
                "fully_ambiguous_matches": 0,
                "fraction_denominator": 1_000,
                "confident_fraction": 0.0,
                "fully_ambiguous_fraction": 0.0,
            }
        else:
            correspondence = {
                "confident_matches": 1,
                "fully_ambiguous_matches": 0,
                "fraction_denominator": 1,
                "confident_fraction": 1.0,
                "fully_ambiguous_fraction": 0.0,
            }
        unsigned["correspondence"] = correspondence
        pooled.append(protocol.seal_event(unsigned))

    report = protocol.build_qa_report(manifest=manifest, events=pooled)
    assert report["metrics"]["correspondence_trials"] == 1_063
    assert report["metrics"]["confident_correspondence_matches"] == 63
    assert report["checks"]["persistent_correspondence"] is False


def test_source_gate_requires_every_registered_control_and_causal_utility(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    passed = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    assert passed["status"] == "PASS_T10_2_SOURCE_GATE"
    assert all(passed["registered_controls"].values())

    incomplete = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(all_controls=False),
    )
    assert incomplete["status"] == "FAIL_T10_2_SOURCE_GATE"
    assert incomplete["checks"]["all_registered_controls_complete"] is False

    missing_recipe = _source_metrics()
    missing_recipe.pop("challenger_recipe")
    unexecutable = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=missing_recipe,
    )
    assert unexecutable["status"] == "FAIL_T10_2_SOURCE_GATE"
    assert unexecutable["verdict"] == "SOURCE_GROUNDING_MISS"
    assert unexecutable["checks"]["frozen_challenger_recipe_bound"] is False


def test_source_controls_are_derived_and_require_nontrivial_transport_evidence(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    inconsistent = _source_metrics()
    inconsistent["controls"]["goal_oracle_passed"] = False
    with pytest.raises(protocol.DataGateError, match="disagree"):
        protocol.build_source_gate_report(manifest=manifest, metrics=inconsistent)

    missing = _source_metrics()
    missing["control_results"].pop("goal_oracle")
    with pytest.raises(protocol.DataGateError, match="exhaustive"):
        protocol.build_source_gate_report(manifest=manifest, metrics=missing)

    identity_only = _source_metrics()
    transport = identity_only["control_results"]["transport_oracle"]
    transport["nontrivial_exact_commutative_certificate_count"] = 0
    transport["certified_orbit_witness_candidate_count"] = 0
    transport["posterior_merged_gauge_class_count"] = 0
    identity_only["controls"]["transport_oracle_passed"] = False
    report = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=identity_only,
    )
    assert report["verdict"] == "FRAME_TRANSPORT_MISS"

    invalid_comparator = _source_metrics()
    invalid_comparator["control_results"]["capacity_matched_independent_posterior"][
        "passed"
    ] = False
    invalid_comparator["control_results"]["capacity_matched_independent_posterior"][
        "scientific_pass"
    ] = False
    invalid_comparator["controls"]["capacity_matched_independent_posterior_passed"] = (
        False
    )
    report = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=invalid_comparator,
    )
    assert report["verdict"] == "SOURCE_GROUNDING_MISS"


def test_source_gate_cannot_hide_control_execution_failure_behind_scientific_pass(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _manifest_path, manifest = frozen_manifest
    metrics = _source_metrics()
    control = "capacity_matched_independent_posterior"
    metrics["control_results"][control]["execution_ok"] = False
    metrics["control_results"][control]["completed"] = False
    metrics["execution_ok_controls"][control] = False
    metrics["completed_controls"][control] = False
    metrics["controls"]["capacity_matched_independent_posterior_passed"] = False

    report = protocol.build_source_gate_report(manifest=manifest, metrics=metrics)
    assert metrics["control_results"][control]["scientific_pass"] is True
    assert report["checks"]["all_registered_controls_attempted"] is True
    assert report["checks"]["all_registered_controls_execution_ok"] is False
    assert report["checks"]["all_registered_controls_complete"] is False
    assert report["passed"] is False


def test_cross_fit_audit_fails_closed_on_empty_missing_mismatched_and_tampered_units(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    _manifest_path, manifest = frozen_manifest
    source_path = tmp_path / "source_events.jsonl"
    protocol.write_event_ledger(source_path, [])
    factory_binding = {
        "module": "theory.sage_t.t10_2_runtime",
        "class": "T10_2SourceFactory",
        "source_sha256": manifest["code_sha256"]["theory/sage_t/t10_2_runtime.py"],
        "manifest_checksum": manifest["manifest_checksum"],
        "code_bound": True,
    }
    units = protocol._fallback_cross_fit_units([])
    for unit in units:
        for reset in unit["resets"]:
            reset["initial_particle_count"] = 4
            reset["initial_class_count"] = 2
            reset["final_particle_count"] = 4
            reset["final_class_count"] = 2

    empty_audit = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=[],
        units=units,
        factory_binding=factory_binding,
    )
    assert empty_audit["checks"]["exact_nine_units"] is True
    assert empty_audit["checks"]["exact_two_resets_per_arm"] is True
    assert empty_audit["checks"]["effective_capacity_matched_by_fold"] is True
    assert empty_audit["checks"]["every_unit_has_both_arms_observed"] is False
    assert empty_audit["passed"] is False

    missing_units = json.loads(json.dumps(units))
    missing_units.pop()
    missing = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=[],
        units=missing_units,
        factory_binding=factory_binding,
    )
    assert missing["checks"]["exact_nine_units"] is False

    wrong_reset_count = json.loads(json.dumps(units))
    wrong_reset_count[0]["resets"].pop()
    wrong_reset = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=[],
        units=wrong_reset_count,
        factory_binding=factory_binding,
    )
    assert wrong_reset["checks"]["exact_two_resets_per_arm"] is False

    mismatched_capacity = json.loads(json.dumps(units))
    mismatched_capacity[0]["resets"][1]["initial_particle_count"] = 5
    capacity = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=[],
        units=mismatched_capacity,
        factory_binding=factory_binding,
    )
    assert capacity["checks"]["effective_capacity_matched_by_fold"] is False

    online_leak = json.loads(json.dumps(units))
    online_leak[0]["resets"][0]["online_observations"] = 1
    online = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=[],
        units=online_leak,
        factory_binding=factory_binding,
    )
    assert online["checks"]["online_observations_only_from_executed_actions"] is False

    audit_path = tmp_path / protocol.CROSS_FIT_AUDIT_FILENAME
    protocol.write_compact_json(audit_path, empty_audit)
    tampered = dict(empty_audit)
    tampered["registered_unit_count"] = 8
    protocol.write_compact_json(audit_path, tampered)
    with pytest.raises(protocol.ManifestDriftError, match="checksum"):
        protocol.read_cross_fit_audit(
            audit_path,
            manifest=manifest,
            source_event_path=source_path,
            source_events=[],
        )


def test_source_train_production_rejects_metric_and_callable_injection(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, _manifest = frozen_manifest
    with pytest.raises(
        protocol.RuntimeUnavailableError, match="forbids injected metrics"
    ):
        protocol.source_train_phase(
            metrics=_source_metrics(),
            manifest_path=manifest_path,
            output_dir=tmp_path,
            repo_root=REPO_ROOT,
            resource_probe=_roomy,
        )
    with pytest.raises(protocol.RuntimeUnavailableError, match="manifest-bound"):
        protocol.source_train_phase(
            trainer=lambda **_kwargs: _source_metrics(),
            manifest_path=manifest_path,
            output_dir=tmp_path,
            repo_root=REPO_ROOT,
            resource_probe=_roomy,
        )


def test_source_recipe_binding_is_recomputed_and_tamper_fails_closed(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    _manifest_path, manifest = frozen_manifest
    recipe_path = tmp_path / protocol.CHALLENGER_RECIPE_FILENAME
    recipe = protocol.signed_payload(
        {
            "format_version": "sage-t10.2-frozen-challenger-recipe-v1",
            "kind": "immutable_source_posterior_recipe",
            "manifest_checksum": manifest["manifest_checksum"],
        },
        checksum_key="recipe_checksum",
    )
    protocol.write_compact_json(recipe_path, recipe)
    metrics = {
        "challenger_recipe": {
            "bound": True,
            "path": protocol.CHALLENGER_RECIPE_FILENAME,
            "artifact": protocol.artifact_descriptor(recipe_path),
            "recipe_checksum": recipe["recipe_checksum"],
        }
    }
    binding = protocol._verified_challenger_recipe_binding(
        metrics,
        output_dir=tmp_path,
        manifest=manifest,
        limits=protocol.DEFAULT_RESOURCE_LIMITS,
    )
    assert binding["bound"] is True

    recipe_path.write_text("{}\n", encoding="utf-8")
    refused = protocol._verified_challenger_recipe_binding(
        metrics,
        output_dir=tmp_path,
        manifest=manifest,
        limits=protocol.DEFAULT_RESOURCE_LIMITS,
    )
    assert refused["bound"] is False
    assert refused["binding_error"] == "ManifestDriftError"


def test_source_verdict_ladder_separates_goal_posterior_option_and_grounding(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    goal_miss = _source_metrics()
    goal_miss["controls"]["goal_oracle_passed"] = False
    goal_miss["control_results"]["goal_oracle"]["passed"] = False
    goal_miss["control_results"]["goal_oracle"]["scientific_pass"] = False
    assert (
        protocol.build_source_gate_report(manifest=manifest, metrics=goal_miss)[
            "verdict"
        ]
        == "GOAL_OR_DYNAMICS_MISS"
    )

    option_miss = _source_metrics()
    option_miss["learned"]["option_synthesis_passed"] = False
    assert (
        protocol.build_source_gate_report(manifest=manifest, metrics=option_miss)[
            "verdict"
        ]
        == "OPTION_SYNTHESIS_MISS"
    )

    grounding_miss = _source_metrics()
    grounding_miss["learned"]["oracle_level_recovery"] = 0.0
    assert (
        protocol.build_source_gate_report(manifest=manifest, metrics=grounding_miss)[
            "verdict"
        ]
        == "SOURCE_GROUNDING_MISS"
    )

    safety_miss = _source_metrics()
    safety_miss["learned"]["illegal_actions"] = 1
    assert (
        protocol.build_source_gate_report(manifest=manifest, metrics=safety_miss)[
            "verdict"
        ]
        == "SAFETY_OR_RESOURCE_MISS"
    )


def test_source_gate_requires_per_fold_ranks_incremental_probe_and_oracles(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest

    rank_list = _source_metrics()
    rank_list["learned"]["positive_fold_ranks"] = [1, 2, 3]
    rank_report = protocol.build_source_gate_report(
        manifest=manifest, metrics=rank_list
    )
    assert rank_report["checks"]["positive_fold_rank_coverage"] is False
    assert rank_report["verdict"] == "COMMON_POSTERIOR_MISS"

    missing_increment = _source_metrics()
    missing_increment["learned"].pop("game_seed_probe_accuracy_increment")
    missing_increment["learned"]["game_seed_probe_accuracy"] = 0.0
    probe_report = protocol.build_source_gate_report(
        manifest=manifest, metrics=missing_increment
    )
    assert probe_report["checks"]["identity_probe_increment_closed"] is False
    assert probe_report["verdict"] == "SOURCE_GROUNDING_MISS"

    for oracle, expected in (
        ("best_executed_sequence_oracle_passed", "SOURCE_GROUNDING_MISS"),
        ("option_oracle_passed", "OPTION_SYNTHESIS_MISS"),
        ("complete_program_oracle_passed", "GOAL_OR_DYNAMICS_MISS"),
    ):
        oracle_miss = _source_metrics()
        oracle_miss["controls"][oracle] = False
        control_name = oracle.removesuffix("_passed")
        oracle_miss["control_results"][control_name]["passed"] = False
        oracle_miss["control_results"][control_name]["scientific_pass"] = False
        report = protocol.build_source_gate_report(
            manifest=manifest, metrics=oracle_miss
        )
        assert report["verdict"] == expected


def test_replay_is_revalidated_jointly_and_cannot_duplicate_fresh_event(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    _path, manifest = frozen_manifest
    fresh = _event(manifest, 0, positive=True)
    replay = _as_replay(fresh, manifest)
    fresh_path = tmp_path / "fresh.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    protocol.write_event_ledger(fresh_path, [fresh])
    protocol.write_event_ledger(replay_path, [replay])
    with pytest.raises(protocol.DataGateError, match="duplicate physical event_id"):
        protocol.validate_prefit_evidence(
            manifest=manifest,
            fresh_event_path=fresh_path,
            replay_event_path=replay_path,
        )


def test_replay_requires_complete_manifest_bound_provenance(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    replay = _as_replay(_event(manifest, 80, positive=False), manifest)
    incomplete = dict(replay)
    incomplete.pop("event_checksum")
    incomplete["provenance"] = dict(incomplete["provenance"])
    incomplete["provenance"].pop("trace_digest")
    with pytest.raises(protocol.DataGateError, match="trace_digest"):
        protocol.validate_source_events(
            [protocol.seal_event(incomplete)], manifest=manifest, replay=True
        )

    drifted = dict(replay)
    drifted.pop("event_checksum")
    drifted["provenance"] = {
        **drifted["provenance"],
        "source_shard_sha256": "f" * 64,
    }
    with pytest.raises(protocol.ManifestDriftError, match="shard binding"):
        protocol.validate_source_events(
            [protocol.seal_event(drifted)], manifest=manifest, replay=True
        )

    missing_converter = dict(replay)
    missing_converter.pop("event_checksum")
    missing_converter["provenance"] = dict(missing_converter["provenance"])
    missing_converter["provenance"].pop("converter_sha256")
    with pytest.raises(protocol.DataGateError, match="flat replay converter"):
        protocol.validate_source_events(
            [protocol.seal_event(missing_converter)],
            manifest=manifest,
            replay=True,
        )

    missing_compiler = dict(replay)
    missing_compiler.pop("event_checksum")
    missing_compiler["provenance"] = dict(missing_compiler["provenance"])
    missing_compiler["provenance"].pop("compiler_sha256")
    with pytest.raises(protocol.DataGateError, match="flat replay compiler"):
        protocol.validate_source_events(
            [protocol.seal_event(missing_compiler)],
            manifest=manifest,
            replay=True,
        )

    conversion_drift = dict(replay)
    conversion_drift.pop("event_checksum")
    conversion_drift["provenance"] = dict(conversion_drift["provenance"])
    conversion_drift["provenance"]["conversion_code_sha256"] = dict(
        conversion_drift["provenance"]["conversion_code_sha256"]
    )
    conversion_drift["provenance"]["conversion_code_sha256"]["projector"] = "f" * 64
    conversion_drift["provenance"]["projector_sha256"] = "f" * 64
    with pytest.raises(protocol.ManifestDriftError, match="projector"):
        protocol.validate_source_events(
            [protocol.seal_event(conversion_drift)],
            manifest=manifest,
            replay=True,
        )


def test_fresh_statistical_qa_does_not_block_replay_before_combined_gate(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    fresh_path = output / "source_events.jsonl"
    fresh = [_event(manifest, 0, positive=True)]
    protocol.write_event_ledger(fresh_path, fresh)
    cross_fit_path = output / protocol.CROSS_FIT_AUDIT_FILENAME
    cross_fit_audit = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=fresh_path,
        source_events=fresh,
        units=protocol._fallback_cross_fit_units(fresh),
        factory_binding={
            "module": "test",
            "class": "SyntheticFactory",
            "source_sha256": "",
            "manifest_checksum": manifest["manifest_checksum"],
            "code_bound": False,
        },
    )
    protocol.write_compact_json(cross_fit_path, cross_fit_audit)
    collection = protocol.signed_payload(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "collect",
            "status": "T10_2_SOURCE_COLLECTION_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "events": protocol.artifact_descriptor(fresh_path),
            "cross_fit_audit": protocol.artifact_descriptor(cross_fit_path),
        },
        checksum_key="report_checksum",
    )
    collection_path = output / "collection_report.json"
    protocol.write_compact_json(collection_path, collection)

    compile_report = protocol.compile_phase(
        manifest_path=manifest_path,
        output_dir=output,
        event_path=fresh_path,
        collection_report_path=collection_path,
        repo_root=REPO_ROOT,
        resource_probe=_roomy,
    )
    assert compile_report["status"] == "T10_2_FRESH_INTEGRITY_COMPLETE"
    assert compile_report["integrity_passed"] is True
    assert compile_report["fresh_scientific_qa"]["passed"] is False

    replay_input = output / "replay_input.jsonl"
    replay = _as_replay(_event(manifest, 1, positive=False), manifest)
    protocol.write_event_ledger(replay_input, [replay])
    replay_report = protocol.replay_phase(
        replay_input_path=replay_input,
        manifest_path=manifest_path,
        output_dir=output,
        compile_report_path=output / "compile_report.json",
        repo_root=REPO_ROOT,
        resource_probe=_roomy,
    )
    assert replay_report["status"] == "T10_2_SOURCE_REPLAY_COMPLETE"


def test_validate_refuses_failed_signed_source_report_before_factory(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    failed_metrics = _source_metrics()
    failed_metrics["grammar_oracle"] = {
        "progress_games": 0,
        "levels": 0,
        "errors": 0,
        "illegal_actions": 0,
        "game_overs": 0,
    }
    failed = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=failed_metrics,
    )
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, failed)
    calls: list[str] = []

    def factory(game_id: str):
        calls.append(game_id)
        return []

    # Authorization reconstructs the evidence before it considers a signed
    # top-level verdict; a report without bound ledgers/recipe is provenance
    # invalid even when its scientific status is already FAIL.
    with pytest.raises(protocol.ManifestDriftError, match="input binding drifted"):
        protocol.validate_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            repo_root=REPO_ROOT,
            env_factory=factory,
            resource_probe=_roomy,
        )
    assert calls == []


def test_validate_reconstructs_signed_pass_report_before_opening_factory(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    shallow_pass = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    assert shallow_pass["passed"] is True
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, shallow_pass)
    calls: list[str] = []

    def factory(game_id: str):
        calls.append(game_id)
        return []

    with pytest.raises(protocol.ManifestDriftError, match="input binding drifted"):
        protocol.validate_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            repo_root=REPO_ROOT,
            env_factory=factory,
            resource_probe=_roomy,
        )
    assert calls == []


def test_validate_executes_all_fifteen_counterbalanced_frozen_pairs(
    frozen_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    source = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, source)
    monkeypatch.setattr(protocol, "require_source_gate", lambda **_kwargs: source)
    calls: list[dict[str, object]] = []
    closed: list[tuple[str, int]] = []

    class PairedEnvironment:
        def __init__(self, game_id: str, seed: int) -> None:
            self.game_id = game_id
            self.seed = seed

        def run_validation(self, **context: object) -> dict[str, object]:
            calls.append({"game_id": self.game_id, "seed": self.seed, **context})
            arm = _validation_arm()
            return {
                "game_id": self.game_id,
                "seed": self.seed,
                "baseline": dict(arm),
                "t10_2": dict(arm),
                "controller_order": list(context["controller_order"]),
                "counterbalanced": True,
                "posterior_reset": context["posterior_reset"],
                "learning_between_controllers": False,
                "wall_seconds": 0.01,
            }

        def close(self) -> None:
            closed.append((self.game_id, self.seed))

    def factory(game_id: str, seed: int, **_context: object) -> PairedEnvironment:
        return PairedEnvironment(game_id, seed)

    report = protocol.validate_phase(
        manifest_path=manifest_path,
        output_dir=output,
        source_report_path=source_path,
        repo_root=REPO_ROOT,
        env_factory=factory,
        resource_probe=_roomy,
    )
    assert len(calls) == 15
    assert len(closed) == 15
    assert [tuple(call["controller_order"]) for call in calls] == [
        ("t10_1", "t10_2") if index % 2 == 0 else ("t10_2", "t10_1")
        for index in range(15)
    ]
    assert all(call["posterior_reset"] is True for call in calls)
    assert all(call["learning_enabled"] is False for call in calls)
    assert all(call["resets"] == 14 for call in calls)
    assert all(call["action_budget"] == 96 for call in calls)
    assert report["metrics"]["all_pairs_executed"] is True
    assert report["metrics"]["counterbalanced_and_reset"] is True


def test_validate_rejects_raw_nested_payload_before_persistence(
    frozen_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    source = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, source)
    monkeypatch.setattr(protocol, "require_source_gate", lambda **_kwargs: source)

    class RawLeakingEnvironment:
        def __init__(self, game_id: str, seed: int) -> None:
            self.game_id = game_id
            self.seed = seed

        def run_validation(self, **context: object) -> dict[str, object]:
            candidate = _validation_arm()
            candidate["grid"] = [[0, 1], [1, 0]]
            return {
                "game_id": self.game_id,
                "seed": self.seed,
                "baseline": _validation_arm(),
                "t10_2": candidate,
                "controller_order": list(context["controller_order"]),
                "counterbalanced": True,
                "posterior_reset": True,
                "learning_between_controllers": False,
                "wall_seconds": 0.01,
            }

    with pytest.raises(protocol.FirewallError, match="validation_summary.t10_2.grid"):
        protocol.validate_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            repo_root=REPO_ROOT,
            env_factory=lambda game_id, seed, **_context: RawLeakingEnvironment(
                game_id, seed
            ),
            resource_probe=_roomy,
        )
    assert not (output / "validation_runs.jsonl").exists()


def test_validate_uses_external_phase_wall_time_for_gate(
    frozen_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    source = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, source)
    monkeypatch.setattr(protocol, "require_source_gate", lambda **_kwargs: source)
    rows = {(row["game_id"], row["seed"]): row for row in _validation_runs()}

    class SummaryEnvironment:
        def __init__(self, game_id: str, seed: int) -> None:
            self.game_id = game_id
            self.seed = seed

        def run_validation(self, **_context: object) -> dict:
            return dict(rows[(self.game_id, self.seed)])

    ticks = iter((100.0, 21_701.0, 21_702.0, 21_703.0))
    with pytest.raises(protocol.ResourceGateError, match="after persistence"):
        protocol.validate_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            repo_root=REPO_ROOT,
            env_factory=lambda game_id, seed, **_context: SummaryEnvironment(
                game_id, seed
            ),
            resource_probe=_roomy,
            clock=lambda: next(ticks),
            _test_only_allow_clock=True,
        )
    report = protocol.read_checked_json(
        output / "validation_report.json", checksum_key="report_checksum"
    )
    assert report["metrics"]["reported_pair_wall_seconds"] == pytest.approx(15.0)
    assert report["metrics"]["wall_seconds"] == pytest.approx(21_602.0)
    assert report["checks"]["wall_time"] is False
    timing_path = output / protocol.VALIDATION_TIMING_PROOF_FILENAME
    timing = protocol.read_checked_json(
        timing_path,
        checksum_key="timing_proof_checksum",
    )
    assert report["inputs"]["validation_timing_proof"] == (
        protocol.artifact_descriptor(timing_path)
    )
    assert timing["monotonic_elapsed_seconds"] == pytest.approx(21_602.0)
    assert timing["reported_pair_wall_seconds"] == pytest.approx(15.0)
    assert timing["validation_report_linked"] is False
    assert timing["cycle_free"] is True
    assert set(timing["code_sha256"]) == set(protocol.VALIDATION_TIMING_CODE_PATHS)


def test_validate_rejects_non_code_bound_clock_in_production(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, _manifest = frozen_manifest
    with pytest.raises(protocol.RuntimeUnavailableError, match="monotonic clock"):
        protocol.validate_phase(
            manifest_path=manifest_path,
            output_dir=tmp_path,
            repo_root=REPO_ROOT,
            clock=lambda: 0.0,
        )


def test_compile_qa_failure_writes_terminal_source_report_without_trainer(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    compile_report = protocol.signed_payload(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "compile",
            "status": "DATA_OR_PROVENANCE_INVALID",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": False,
            "checks": {"learned_predicates": False},
        },
        checksum_key="report_checksum",
    )
    compile_path = output / "compile_report.json"
    protocol.write_compact_json(compile_path, compile_report)
    trainer_calls: list[str] = []

    def trainer(**_kwargs):
        trainer_calls.append("called")
        return _source_metrics()

    source = protocol.source_train_phase(
        trainer=trainer,
        _test_only_allow_injection=True,
        manifest_path=manifest_path,
        output_dir=output,
        compile_report_path=compile_path,
        repo_root=REPO_ROOT,
        resource_probe=_roomy,
    )
    assert source["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert source["checks"]["trainer_invoked"] is False
    assert source["checks"]["compile_integrity_passed"] is False
    assert trainer_calls == []
    final = protocol.report_phase(
        manifest_path=manifest_path,
        output_dir=output,
        source_report_path=output / "source_report.json",
        repo_root=REPO_ROOT,
    )
    assert final["verdict"] == "DATA_OR_PROVENANCE_INVALID"


def test_combined_prefit_qa_failure_is_signed_and_never_invokes_trainer(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    fresh = [_event(manifest, index, positive=index < 32) for index in range(64)]
    fresh_path = output / "source_events.jsonl"
    protocol.write_event_ledger(fresh_path, fresh)
    cross_fit_path = output / protocol.CROSS_FIT_AUDIT_FILENAME
    cross_fit_audit = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=fresh_path,
        source_events=fresh,
        units=protocol._fallback_cross_fit_units(fresh),
        factory_binding={
            "module": "test",
            "class": "SyntheticFactory",
            "source_sha256": "",
            "manifest_checksum": manifest["manifest_checksum"],
            "code_bound": False,
        },
    )
    protocol.write_compact_json(cross_fit_path, cross_fit_audit)
    compile_report = protocol.build_qa_report(manifest=manifest, events=fresh)
    compile_report["inputs"] = {
        "source_events": protocol.artifact_descriptor(fresh_path),
        "cross_fit_audit": protocol.artifact_descriptor(cross_fit_path),
    }
    compile_report = protocol.signed_payload(
        compile_report, checksum_key="report_checksum"
    )
    compile_path = output / "compile_report.json"
    protocol.write_compact_json(compile_path, compile_report)

    replay = _as_replay(_event(manifest, 1000, positive=True), manifest)
    unsigned_replay = dict(replay)
    unsigned_replay.pop("event_checksum")
    # Keep the v2 event structurally valid, but introduce a learned label
    # that has neither the registered support nor two-game coverage.  The
    # combined pre-fit QA must fail before the trainer is invoked.
    unsigned_replay["learned_predicates"] = ["replay_only_predicate"]
    unsigned_replay["labels"] = {
        **unsigned_replay["labels"],
        "replay_only_predicate": True,
    }
    replay = protocol.seal_event(unsigned_replay)
    replay_path = output / "replay_events.jsonl"
    protocol.write_event_ledger(replay_path, [replay])
    replay_report = protocol.signed_payload(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "replay",
            "status": "T10_2_SOURCE_REPLAY_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "events": protocol.artifact_descriptor(replay_path),
        },
        checksum_key="report_checksum",
    )
    replay_report_path = output / "replay_report.json"
    protocol.write_compact_json(replay_report_path, replay_report)
    trainer_calls: list[str] = []

    def trainer(**_kwargs):
        trainer_calls.append("called")
        return _source_metrics()

    source = protocol.source_train_phase(
        trainer=trainer,
        _test_only_allow_injection=True,
        manifest_path=manifest_path,
        output_dir=output,
        compile_report_path=compile_path,
        replay_report_path=replay_report_path,
        fresh_event_path=fresh_path,
        replay_event_path=replay_path,
        repo_root=REPO_ROOT,
        resource_probe=_roomy,
    )
    assert source["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert source["prefit_evidence"]["combined_qa_passed"] is False
    assert source["prefit_evidence"]["combined_qa_report_checksum"]
    assert trainer_calls == []


def test_resource_floor_fails_before_expensive_phase() -> None:
    low_disk = protocol.ResourceSnapshot(
        repository_bytes=1,
        scratch_bytes=0,
        cache_bytes=0,
        resident_bytes=1,
        free_bytes=protocol.GIB,
    )
    with pytest.raises(protocol.ResourceGateError, match="free-disk"):
        protocol.enforce_resource_limits(
            low_disk,
            limits=protocol.DEFAULT_RESOURCE_LIMITS,
            expensive=True,
        )


def test_validation_uses_exact_budget_episode_denominator_and_complete_latencies(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    metrics = protocol.aggregate_validation_runs(_validation_runs())
    assert metrics["all_pairs_executed"] is True
    assert metrics["budget_configuration_exact"] is True
    assert metrics["within_action_caps"] is True
    assert metrics["maximum_actions_per_controller"] == 20_160
    assert metrics["scheduled_episodes_per_controller"] == 210
    assert metrics["game_over_rate_delta"] == pytest.approx(1 / 210)
    assert metrics["latency_samples_complete"] is True

    source = protocol.build_source_gate_report(
        manifest=manifest, metrics=_source_metrics()
    )
    report = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=metrics,
    )
    assert report["verdict"] == "SAFETY_OR_RESOURCE_MISS"
    assert report["checks"]["game_over_not_worse"] is False


def test_validation_abstention_cannot_shrink_planned_budget(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    runs = _validation_runs()
    for row in runs:
        candidate = row["t10_2"]
        first_reset = candidate["reset_summaries"][0]
        first_reset["stop_reason"] = "policy_abstained"
        first_reset["planned_actions"] = protocol.VALIDATION_ACTIONS_PER_RESET
        candidate["planned_actions"] += protocol.VALIDATION_ACTIONS_PER_RESET - 1
        candidate["unregistered_stops"] = 1
        candidate["game_overs"] = 0
    metrics = protocol.aggregate_validation_runs(runs)
    assert metrics["candidate_completed_budget_fraction"] == pytest.approx(14 / 109)
    assert metrics["unregistered_stops"] == len(runs)

    source = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=_source_metrics(),
    )
    report = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=metrics,
    )
    assert report["checks"]["planned_budget_completion"] is False
    assert report["checks"]["zero_unregistered_stops"] is False
    assert report["verdict"] == "SAFETY_OR_RESOURCE_MISS"


def test_validation_closed_schema_rejects_unknown_summary_field() -> None:
    runs = _validation_runs()
    runs[0]["t10_2"]["debug_payload"] = {"compact": True}
    with pytest.raises(protocol.DataGateError, match="unknown=.*debug_payload"):
        protocol.aggregate_validation_runs(runs)


def test_validation_missing_latency_or_budget_fails_and_transfer_has_priority(
    frozen_manifest: tuple[Path, dict],
) -> None:
    _path, manifest = frozen_manifest
    source = protocol.build_source_gate_report(
        manifest=manifest, metrics=_source_metrics()
    )

    missing_latency = _validation_metrics()
    missing_latency["decision_latency_sample_count"] = 0
    missing_latency["observation_latency_sample_count"] = 0
    missing_latency["latency_samples_complete"] = False
    latency_report = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=missing_latency,
    )
    assert latency_report["checks"]["latency_samples_present"] is False
    assert latency_report["verdict"] == "SAFETY_OR_RESOURCE_MISS"

    wrong_budget = _validation_metrics()
    wrong_budget["maximum_actions_per_controller"] = 20_159
    budget_report = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=wrong_budget,
    )
    assert budget_report["checks"]["exact_validation_budget"] is False

    transfer_and_safety = _validation_metrics()
    transfer_and_safety["total_level_advantage"] = 0
    transfer_and_safety["illegal_actions"] = 1
    ordered_report = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=transfer_and_safety,
    )
    assert ordered_report["verdict"] == "SOURCE_VALIDATION_TRANSFER_MISS"


def test_final_report_rebuild_is_byte_idempotent(
    frozen_manifest: tuple[Path, dict], tmp_path: Path
) -> None:
    manifest_path, manifest = frozen_manifest
    output = tmp_path / "output"
    output.mkdir()
    fresh = [
        _event(manifest, 700 + index, positive=True, game_id=game)
        for index, game in enumerate(protocol.SOURCE_GAMES)
    ]
    event_index = 800
    for held_out_game in protocol.SOURCE_GAMES:
        for seed in protocol.CONFIRMATION_SEEDS:
            for reset_index, controller in enumerate(
                protocol._expected_cross_fit_resets(seed)
            ):
                event = _event(
                    manifest,
                    event_index,
                    positive=True,
                    game_id=held_out_game,
                )
                unsigned = dict(event)
                unsigned.pop("event_checksum")
                unsigned.update(
                    {
                        "seed": seed,
                        "split": "leave_one_game_out_confirmation",
                        "reset_index": reset_index,
                        "step_index": 0,
                        "selection": {
                            "controller": controller,
                            "reset_index": reset_index,
                        },
                        "provenance": {
                            "kind": "fresh_source_trajectory",
                            "game_id": held_out_game,
                            "seed": seed,
                            "split": "leave_one_game_out_confirmation",
                            "manifest_checksum": manifest["manifest_checksum"],
                            "environment_sha256": manifest["environment_sha256"],
                        },
                    }
                )
                fresh.append(protocol.seal_event(unsigned))
                event_index += 1
    replay = [_as_replay(_event(manifest, 9000, positive=True), manifest)]
    fresh_path = output / "source_events.jsonl"
    replay_path = output / "replay_events.jsonl"
    protocol.write_event_ledger(fresh_path, fresh)
    protocol.write_event_ledger(replay_path, replay)
    cross_fit_path = output / protocol.CROSS_FIT_AUDIT_FILENAME
    cross_fit_units = protocol._fallback_cross_fit_units(fresh)
    for unit in cross_fit_units:
        for reset in unit["resets"]:
            reset["online_observations"] = reset["action_count"]
            reset["initial_particle_count"] = 4
            reset["initial_class_count"] = 2
            reset["final_particle_count"] = 4
            reset["final_class_count"] = 2
            reset["stop_reason"] = "budget_exhausted"
    cross_fit_audit = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=fresh_path,
        source_events=fresh,
        units=cross_fit_units,
        factory_binding={
            "module": "theory.sage_t.t10_2_runtime",
            "class": "T10_2SourceFactory",
            "source_sha256": manifest["code_sha256"]["theory/sage_t/t10_2_runtime.py"],
            "manifest_checksum": manifest["manifest_checksum"],
            "code_bound": True,
        },
    )
    assert cross_fit_audit["passed"] is True
    protocol.write_compact_json(cross_fit_path, cross_fit_audit)

    recipe = protocol.signed_payload(
        {
            "format_version": "sage-t10.2-frozen-challenger-recipe-v1",
            "kind": "immutable_source_posterior_recipe",
            "manifest_checksum": manifest["manifest_checksum"],
        },
        checksum_key="recipe_checksum",
    )
    recipe_path = output / protocol.CHALLENGER_RECIPE_FILENAME
    protocol.write_compact_json(recipe_path, recipe)
    source_metrics = _source_metrics()
    source_metrics["challenger_recipe"] = {
        "bound": True,
        "path": protocol.CHALLENGER_RECIPE_FILENAME,
        "artifact": protocol.artifact_descriptor(recipe_path),
        "recipe_checksum": recipe["recipe_checksum"],
    }
    source_metrics["cross_fit_audit"] = {
        "artifact": protocol.artifact_descriptor(cross_fit_path),
        "audit_checksum": cross_fit_audit["audit_checksum"],
        "registered_unit_count": cross_fit_audit["registered_unit_count"],
        "checks": dict(cross_fit_audit["checks"]),
        "passed": cross_fit_audit["passed"],
    }
    source_metrics["control_results"]["capacity_matched_independent_posterior"][
        "cross_fit_schedule_checks"
    ] = dict(cross_fit_audit["checks"])
    source_metrics["source_evidence"] = protocol._source_evidence_binding(
        manifest=manifest,
        fresh_path=fresh_path,
        replay_path=replay_path,
        cross_fit_path=cross_fit_path,
        fresh=fresh,
        replay=replay,
        control_results=source_metrics["control_results"],
        cross_fit_audit=cross_fit_audit,
    )
    source = protocol.build_source_gate_report(
        manifest=manifest,
        metrics=source_metrics,
    )
    source["inputs"] = {
        "fresh_events": protocol.artifact_descriptor(fresh_path),
        "replay_events": protocol.artifact_descriptor(replay_path),
        "cross_fit_audit": protocol.artifact_descriptor(cross_fit_path),
    }
    source = protocol.signed_payload(source, checksum_key="report_checksum")
    source_path = output / "source_report.json"
    protocol.write_compact_json(source_path, source)

    runs = _validation_runs()
    runs[0]["t10_2"]["game_overs"] = 0
    runs[0]["t10_2"]["reset_summaries"][0]["stop_reason"] = "progression"
    runs_path = output / "validation_runs.jsonl"
    protocol._atomic_write_lines(
        runs_path,
        (protocol.canonical_json(row) + "\n" for row in runs),
    )
    validation_metrics = protocol.aggregate_validation_runs(runs)
    validation_metrics["reported_pair_wall_seconds"] = validation_metrics[
        "wall_seconds"
    ]
    timing_path = output / protocol.VALIDATION_TIMING_PROOF_FILENAME
    timing_proof = protocol._build_validation_timing_proof(
        manifest=manifest,
        source_report=source,
        source_path=source_path,
        runs_path=runs_path,
        runs=runs,
        monotonic_started=10.0,
        monotonic_finished=25.0,
        reported_pair_wall_seconds=validation_metrics["reported_pair_wall_seconds"],
    )
    protocol.write_compact_json(timing_path, timing_proof)
    validation_metrics["wall_seconds"] = timing_proof["monotonic_elapsed_seconds"]
    validation = protocol.build_validation_report(
        manifest=manifest,
        source_report=source,
        metrics=validation_metrics,
    )
    validation["inputs"] = {
        "source_report": protocol.artifact_descriptor(source_path),
        "validation_runs": protocol.artifact_descriptor(runs_path),
        "validation_timing_proof": protocol.artifact_descriptor(timing_path),
    }
    validation = protocol.signed_payload(validation, checksum_key="report_checksum")
    validation_path = output / "validation_report.json"
    protocol.write_compact_json(validation_path, validation)

    first = protocol.report_phase(
        manifest_path=manifest_path,
        output_dir=output,
        source_report_path=source_path,
        validation_report_path=validation_path,
        repo_root=REPO_ROOT,
    )
    first_bytes = (output / "report.json").read_bytes()
    second = protocol.report_phase(
        manifest_path=manifest_path,
        output_dir=output,
        source_report_path=source_path,
        validation_report_path=validation_path,
        repo_root=REPO_ROOT,
    )
    assert second == first
    assert (output / "report.json").read_bytes() == first_bytes
    assert first["verdict"] == "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED"
    assert first["firewall"]["holdout_opened"] is False
    inventory_path = output.parent / protocol.ARTIFACT_INVENTORY_FILENAME
    binding_path = output.parent / protocol.REPORT_INVENTORY_BINDING_FILENAME
    inventory = protocol.read_checked_json(
        inventory_path,
        checksum_key="inventory_checksum",
    )
    binding = protocol.read_checked_json(
        binding_path,
        checksum_key="binding_checksum",
    )
    assert binding["report"] == protocol.artifact_descriptor(output / "report.json")
    assert binding["inventory"] == protocol.artifact_descriptor(inventory_path)
    assert binding["inventory_checksum"] == inventory["inventory_checksum"]
    assert binding["cycle_free"] is True

    tampered_wall = dict(validation)
    tampered_wall["metrics"] = {
        **tampered_wall["metrics"],
        "wall_seconds": 0.0,
    }
    tampered_wall = protocol.signed_payload(
        tampered_wall,
        checksum_key="report_checksum",
    )
    protocol.write_compact_json(validation_path, tampered_wall)
    with pytest.raises(protocol.ManifestDriftError, match="validation metrics drifted"):
        protocol.report_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            validation_report_path=validation_path,
            repo_root=REPO_ROOT,
        )

    tampered_proof = {
        **timing_proof,
        "injected_elapsed_seconds": 0.0,
    }
    tampered_proof = protocol.signed_payload(
        tampered_proof,
        checksum_key="timing_proof_checksum",
    )
    protocol.write_compact_json(timing_path, tampered_proof)
    rebound_validation = dict(validation)
    rebound_validation["inputs"] = {
        **rebound_validation["inputs"],
        "validation_timing_proof": protocol.artifact_descriptor(timing_path),
    }
    rebound_validation = protocol.signed_payload(
        rebound_validation,
        checksum_key="report_checksum",
    )
    protocol.write_compact_json(validation_path, rebound_validation)
    with pytest.raises(protocol.ManifestDriftError, match="timing proof schema"):
        protocol.report_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            validation_report_path=validation_path,
            repo_root=REPO_ROOT,
        )

    protocol.write_compact_json(timing_path, timing_proof)
    protocol.write_compact_json(validation_path, validation)
    tampered_validation = dict(validation)
    tampered_validation["checks"] = {
        **tampered_validation["checks"],
        "total_level_advantage": False,
    }
    protocol.write_compact_json(
        validation_path,
        protocol.signed_payload(
            tampered_validation,
            checksum_key="report_checksum",
        ),
    )
    with pytest.raises(protocol.ManifestDriftError, match="reconstructed checks"):
        protocol.report_phase(
            manifest_path=manifest_path,
            output_dir=output,
            source_report_path=source_path,
            validation_report_path=validation_path,
            repo_root=REPO_ROOT,
        )
