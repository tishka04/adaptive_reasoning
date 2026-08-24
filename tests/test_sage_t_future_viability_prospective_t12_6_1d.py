from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from theory.sage_t.causal import (
    future_viability_prospective_experiment as experiment_module,
)
from theory.sage_t.causal import (
    future_viability_prospective_protocol as protocol_module,
)
from theory.sage_t.causal.archive import abstract_state_to_payload
from theory.sage_t.causal.future_viability_prospective_cli import build_parser
from theory.sage_t.causal.future_viability_prospective_confirmation import (
    ExactStateExtraction,
    _exact_productive_reach,
    adjudicate_prediction_commitment,
    commit_label_blind_predictions,
    extract_exact_state_candidates,
    verify_prediction_commitment,
)
from theory.sage_t.causal.future_viability_prospective_experiment import (
    adjudicate_future_viability_confirmation,
    classify_prospective_adjudication,
    collect_future_viability_batch,
    commit_future_viability_predictions,
    preflight_future_viability_confirmation,
)
from theory.sage_t.causal.future_viability_prospective_protocol import (
    FutureViabilityProspectiveProtocol,
    freeze_future_viability_prospective_confirmation,
    load_future_viability_prospective_manifest,
)
from theory.sage_t.contracts import AbstractState


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _reliability_parent() -> Path:
    return (
        _repo() / "training" / "sage_t" / "future_viability_reliability_t12_6_1c_bp35"
    )


def _hazard_parent() -> Path:
    return _repo() / "training" / "sage_t" / "hazard_diversity_t12_4a_4d_1_bp35"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edge(
    edge_id: str,
    *,
    source_cell: str,
    source_exact: str,
    action: str,
    target_cell: str,
    target_exact: str,
    terminal: bool = False,
    changed: bool = True,
    novel: bool = False,
) -> dict[str, object]:
    return {
        "action": {"action_data": {}, "action_name": action},
        "changed": changed,
        "edge_id": edge_id,
        "level_delta": 0,
        "novel": novel,
        "source_cell_id": source_cell,
        "source_exact_hash": source_exact,
        "success": False,
        "target_cell_id": target_cell,
        "target_exact_hash": target_exact,
        "terminal": terminal,
    }


def _archive(path: Path, edges: list[dict[str, object]]) -> None:
    cell_ids = sorted(
        {
            str(edge[side])
            for edge in edges
            for side in ("source_cell_id", "target_cell_id")
        }
    )
    payload = {
        "cells": [
            {
                "cell_id": cell_id,
                "state": abstract_state_to_payload(AbstractState()),
            }
            for cell_id in cell_ids
        ],
        "edges": edges,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metas(path: Path, *, seed: int = 9301) -> tuple[dict[str, object], ...]:
    checksum = _sha(path)
    return tuple(
        {
            "arm": arm,
            "lineage_seed": 8701,
            "path": str(path),
            "search_seed": seed,
            "sha256": checksum,
        }
        for arm in FutureViabilityProspectiveProtocol().search_arms
    )


def _extract(
    path: Path,
    *,
    seed: int = 9301,
    include_labels: bool = True,
) -> ExactStateExtraction:
    return extract_exact_state_candidates(
        archive_metas=_metas(path, seed=seed),
        root=path.parent,
        expected_search_seeds=(9301,),
        expected_lineages=(8701,),
        expected_arms=FutureViabilityProspectiveProtocol().search_arms,
        future_horizon=4,
        local_radius=7,
        include_labels=include_labels,
    )


def test_protocol_and_cli_freeze_the_full_staged_design() -> None:
    protocol = FutureViabilityProspectiveProtocol()
    assert protocol.prospective_search_seeds == (9301, 9302, 9303)
    assert protocol.pilot_search_seeds == (9301,)
    assert protocol.completion_search_seeds == (9302, 9303)
    assert protocol.expected_archive_count == 18
    assert protocol.minimum_unique_archive_count == 12
    assert protocol.sdk_calls_per_archive == 2048
    assert protocol.maximum_total_sdk_calls == 38_000
    assert protocol.maximum_cells_per_archive == 10_000
    assert protocol.maximum_artifact_bytes == 1024**3
    assert protocol.minimum_gain_over_exact_first == 0.02
    parser = build_parser()
    for phase in (
        "freeze",
        "preflight",
        "collect-batch",
        "seal-collection",
        "predict",
        "adjudicate",
        "status",
    ):
        assert phase in parser._subparsers._group_actions[0].choices
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityProspectiveProtocol(prospective_search_seeds=(9201, 9202, 9203))
    with pytest.raises(ValueError, match="preregistered value changed"):
        FutureViabilityProspectiveProtocol(minimum_gain_over_exact_first=0.019)


def test_exact_graph_reach_is_bounded_and_cycle_safe() -> None:
    outgoing = {
        "a": (
            _edge(
                "a-b",
                source_cell="a",
                source_exact="a",
                action="ACTION3",
                target_cell="b",
                target_exact="b",
            ),
        ),
        "b": (
            _edge(
                "b-c",
                source_cell="b",
                source_exact="b",
                action="ACTION3",
                target_cell="c",
                target_exact="c",
            ),
        ),
        "c": (
            _edge(
                "c-a",
                source_cell="c",
                source_exact="c",
                action="ACTION3",
                target_cell="a",
                target_exact="a",
            ),
        ),
    }
    assert (
        _exact_productive_reach(
            "a", outgoing=outgoing, remaining_horizon=4, visited=frozenset()
        )
        == 3
    )
    assert (
        _exact_productive_reach(
            "a", outgoing=outgoing, remaining_horizon=2, visited=frozenset()
        )
        == 2
    )
    outgoing["b"] = (
        {
            **outgoing["b"][0],
            "terminal": True,
        },
    )
    assert (
        _exact_productive_reach(
            "a", outgoing=outgoing, remaining_horizon=4, visited=frozenset()
        )
        == 1
    )


def test_exact_extraction_ignores_novelty_and_deduplicates_arm_archives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.json"
    _archive(
        path,
        [
            _edge(
                "source-a",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="a",
                target_exact="exact-a",
                novel=True,
            ),
            _edge(
                "source-a-repeat",
                source_cell="source-copy",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="a-copy",
                target_exact="exact-a",
                novel=False,
            ),
            _edge(
                "source-b",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION4",
                target_cell="b",
                target_exact="exact-b",
                terminal=True,
            ),
            _edge(
                "a-next",
                source_cell="a",
                source_exact="exact-a",
                action="ACTION3",
                target_cell="leaf",
                target_exact="exact-leaf",
            ),
        ],
    )
    extraction = _extract(path)
    assert extraction.metrics["raw_archive_count"] == 3
    assert extraction.metrics["scored_archive_count"] == 1
    assert extraction.metrics["unique_archive_count"] == 1
    assert extraction.metrics["novelty_only_repetitions"] == 1
    assert extraction.metrics["exact_transition_conflicts"] == 0
    assert extraction.metrics["source_arm_multiplicities"] == {"3": 1}
    assert len(extraction.candidates) == 2
    assert {candidate.source_arms for candidate in extraction.candidates} == {
        tuple(sorted(FutureViabilityProspectiveProtocol().search_arms))
    }
    assert sorted(extraction.labels.values()) == [0, 1]


def test_same_abstract_cell_does_not_merge_distinct_exact_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.json"
    _archive(
        path,
        [
            _edge(
                f"{exact}-{action}",
                source_cell="same-abstract-cell",
                source_exact=exact,
                action=action,
                target_cell=f"{exact}-{action}-target",
                target_exact=f"{exact}-{action}-target",
                terminal=action == "ACTION4",
            )
            for exact in ("exact-one", "exact-two")
            for action in ("ACTION3", "ACTION4")
        ],
    )
    extraction = _extract(path)
    assert extraction.metrics["multi_action_exact_groups"] == 2
    assert extraction.metrics["exact_transition_conflicts"] == 0
    assert len({item.group_id for item in extraction.candidates}) == 2


def test_exact_decision_group_is_consolidated_across_distinct_arm_archives(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"arm-{index}.json" for index in range(3)]
    _archive(
        paths[0],
        [
            _edge(
                "action-a",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="a",
                target_exact="exact-a",
            )
        ],
    )
    _archive(
        paths[1],
        [
            _edge(
                "action-b",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION4",
                target_cell="b",
                target_exact="exact-b",
                terminal=True,
            )
        ],
    )
    _archive(
        paths[2],
        [
            _edge(
                "future",
                source_cell="a",
                source_exact="exact-a",
                action="ACTION3",
                target_cell="leaf",
                target_exact="exact-leaf",
            )
        ],
    )
    metas = tuple(
        {
            "arm": arm,
            "lineage_seed": 8701,
            "path": str(path),
            "search_seed": 9301,
            "sha256": _sha(path),
        }
        for arm, path in zip(
            FutureViabilityProspectiveProtocol().search_arms,
            paths,
            strict=True,
        )
    )
    extraction = extract_exact_state_candidates(
        archive_metas=metas,
        root=tmp_path,
        expected_search_seeds=(9301,),
        expected_lineages=(8701,),
        expected_arms=FutureViabilityProspectiveProtocol().search_arms,
        future_horizon=4,
        local_radius=7,
        include_labels=True,
    )
    assert extraction.metrics["multi_action_exact_groups"] == 1
    assert extraction.metrics["scored_archive_count"] == 3
    assert extraction.metrics["exact_transition_conflicts"] == 0
    assert len(extraction.candidates) == 2
    assert sorted(extraction.labels.values()) == [0, 1]
    assert all(
        len(candidate.archive_sha256s) == 2 for candidate in extraction.candidates
    )


def test_cross_archive_exact_transition_conflict_is_detected(tmp_path: Path) -> None:
    paths = [tmp_path / f"arm-{index}.json" for index in range(3)]
    targets = ("exact-a", "exact-b", "exact-c")
    actions = ("ACTION3", "ACTION3", "ACTION4")
    for index, path in enumerate(paths):
        _archive(
            path,
            [
                _edge(
                    f"edge-{index}",
                    source_cell="source",
                    source_exact="exact-source",
                    action=actions[index],
                    target_cell=f"target-{index}",
                    target_exact=targets[index],
                )
            ],
        )
    metas = tuple(
        {
            "arm": arm,
            "lineage_seed": 8701,
            "path": str(path),
            "search_seed": 9301,
            "sha256": _sha(path),
        }
        for arm, path in zip(
            FutureViabilityProspectiveProtocol().search_arms,
            paths,
            strict=True,
        )
    )
    extraction = extract_exact_state_candidates(
        archive_metas=metas,
        root=tmp_path,
        expected_search_seeds=(9301,),
        expected_lineages=(8701,),
        expected_arms=FutureViabilityProspectiveProtocol().search_arms,
        future_horizon=4,
        local_radius=7,
        include_labels=True,
    )
    assert extraction.metrics["exact_transition_conflicts"] == 1


def test_different_transition_for_same_exact_action_is_integrity_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.json"
    _archive(
        path,
        [
            _edge(
                "first",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="a",
                target_exact="exact-a",
            ),
            _edge(
                "conflict",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="b",
                target_exact="exact-b",
            ),
            _edge(
                "other",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION4",
                target_cell="c",
                target_exact="exact-c",
            ),
        ],
    )
    extraction = _extract(path)
    assert extraction.metrics["exact_transition_conflicts"] == 1
    with pytest.raises(ValueError, match="transition integrity"):
        commit_label_blind_predictions(
            ExactStateExtraction(
                extraction.candidates,
                {},
                extraction.metrics,
            ),
            future_model=SimpleNamespace(),
            immediate_model=SimpleNamespace(),
            incumbent_model=SimpleNamespace(),
            binding_shift=1,
        )


class _FutureModel:
    def score_with_audit(self, item):
        value = 1.0 if item.base.action_name == "ACTION3" else 0.0
        return (
            value,
            "local_composition_signature",
            {
                "exact_candidate_present": True,
                "exact_candidate_reliable": False,
                "exact_rejection_reasons": ["insufficient_search_seed_span"],
            },
        )


class _ControlModel:
    def score(self, item):
        value = 1.0 if item.base.action_name == "ACTION4" else 0.0
        return value, "exact_local_signature"


def test_prediction_is_label_blind_and_adjudication_requires_open_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.json"
    _archive(
        path,
        [
            _edge(
                "best",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="a",
                target_exact="exact-a",
            ),
            _edge(
                "worse",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION4",
                target_cell="b",
                target_exact="exact-b",
                terminal=True,
            ),
            _edge(
                "reach",
                source_cell="a",
                source_exact="exact-a",
                action="ACTION3",
                target_cell="leaf",
                target_exact="exact-leaf",
            ),
        ],
    )
    closed = _extract(path, include_labels=False)
    commitment = commit_label_blind_predictions(
        closed,
        future_model=_FutureModel(),
        immediate_model=_ControlModel(),
        incumbent_model=_ControlModel(),
        binding_shift=1,
    )
    verify_prediction_commitment(commitment)
    encoded = json.dumps(commitment)
    assert "productive_reach" not in encoded
    assert '"hit"' not in encoded
    with pytest.raises(ValueError, match="candidate registry changed"):
        adjudicate_prediction_commitment(
            commitment,
            closed,
            bootstrap_repetitions=10,
            bootstrap_seed=1261,
            bootstrap_lower_quantile=0.05,
        )
    opened = _extract(path, include_labels=True)
    result = adjudicate_prediction_commitment(
        commitment,
        opened,
        bootstrap_repetitions=10,
        bootstrap_seed=1261,
        bootstrap_lower_quantile=0.05,
    )
    assert result["metrics"]["eligible_groups"] == 1
    assert result["metrics"]["future_binding_top1_accuracy"] == 1.0


def _passing_metrics() -> tuple[dict[str, object], dict[str, object]]:
    extraction = {
        "all_archive_conditions_present": True,
        "exact_transition_conflicts": 0,
        "raw_archive_count": 18,
        "unique_archive_count": 12,
    }
    seed = {
        "9301": {"future_gain_over_incumbent": 0.02},
        "9302": {"future_gain_over_incumbent": 0.03},
        "9303": {"future_gain_over_incumbent": 0.0},
    }
    ranked = {
        "binding_swap_top1_accuracy": 0.45,
        "bootstrap_gain_lower_bound_90": 0.0,
        "eligible_groups": 250,
        "exact_rejection_exercised_rate": 0.25,
        "future_binding_top1_accuracy": 0.70,
        "future_gain_over_binding_swap": 0.25,
        "future_gain_over_immediate": 0.10,
        "future_gain_over_incumbent": 0.02,
        "hierarchy_coverage": 0.70,
        "immediate_binding_top1_accuracy": 0.60,
        "per_lineage": {
            "8701": {"future_binding_top1_accuracy": 0.65},
            "8705": {"future_binding_top1_accuracy": 0.65},
        },
        "per_search_seed": seed,
        "recommendation_coverage": 0.60,
        "unique_top_rate": 0.85,
    }
    return extraction, ranked


@pytest.mark.parametrize(
    ("mutation", "expected_passed", "expected_classification"),
    (
        ("pass_at_two_points", True, "PROSPECTIVE_RELIABILITY_SUPERIORITY_CONFIRMED"),
        ("gain_at_1_9_points", False, "NO_PROSPECTIVE_RELIABILITY_SUPERIORITY"),
        ("negative_seed", False, "NO_PROSPECTIVE_RELIABILITY_SUPERIORITY"),
        ("insufficient_support", False, "INSUFFICIENT_PROSPECTIVE_SUPPORT"),
        ("exact_conflict", False, "PROSPECTIVE_ADJUDICATION_INTEGRITY_FAILURE"),
    ),
)
def test_frozen_verdict_boundaries(
    mutation: str,
    expected_passed: bool,
    expected_classification: str,
) -> None:
    extraction, ranked = _passing_metrics()
    if mutation == "gain_at_1_9_points":
        ranked["future_gain_over_incumbent"] = 0.019
    elif mutation == "negative_seed":
        ranked["per_search_seed"]["9303"]["future_gain_over_incumbent"] = -0.001
    elif mutation == "insufficient_support":
        ranked["eligible_groups"] = 249
    elif mutation == "exact_conflict":
        extraction["exact_transition_conflicts"] = 1
    verdict = classify_prospective_adjudication(
        protocol=FutureViabilityProspectiveProtocol(),
        extraction_metrics=extraction,
        ranked=ranked,
        elapsed_seconds=1.0,
    )
    assert verdict["passed"] is expected_passed
    assert verdict["classification"] == expected_classification


def _manifest() -> dict[str, object]:
    return {
        "game_id": "bp35",
        "manifest_checksum": "m" * 64,
        "parents": {
            "hazard_compile_receipt": {"receipt_checksum": "h" * 64},
            "reliability_compile_receipt": {"receipt_checksum": "r" * 64},
        },
        "protocol": asdict(FutureViabilityProspectiveProtocol()),
        "protocol_checksum": FutureViabilityProspectiveProtocol().checksum,
    }


def test_fake_environment_runs_full_three_by_two_by_three_matrix_within_budgets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    preflight_path = tmp_path / "preflight_receipt.json"
    preflight_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        experiment_module,
        "load_future_viability_prospective_manifest",
        lambda *args, **kwargs: manifest,
    )

    def fake_receipt(*args, **kwargs):
        phase = kwargs["expected_phase"]
        status = {
            "preflight": "PASS_T12_6_1D_PREFLIGHT",
            "collection_pilot": "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY",
        }[phase]
        return {"passed": True, "status": status}

    monkeypatch.setattr(experiment_module, "load_prospective_receipt", fake_receipt)
    witnesses = (SimpleNamespace(source_seed=8701), SimpleNamespace(source_seed=8705))
    monkeypatch.setattr(
        experiment_module,
        "_load_collector",
        lambda *args, **kwargs: (witnesses, object(), object(), object(), object()),
    )
    monkeypatch.setattr(experiment_module, "_clone_shield", lambda shield: object())
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        catalog = f"catalog-{kwargs['search_seed']}-{kwargs['witness'].source_seed}"
        return SimpleNamespace(
            archive=object(),
            candidate_catalog_checksum=catalog,
            entry_exact=True,
            metrics=lambda: {
                "cells": 10_000,
                "replay_exact_rate": 1.0,
                "sdk_calls": 2_048,
            },
        )

    def fake_write(path, archive, *, storage_budget):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"path": str(path.resolve()), "sha256": _sha(path)}

    monkeypatch.setattr(experiment_module, "run_hazard_diversity_arm", fake_run)
    monkeypatch.setattr(experiment_module, "_write_archive", fake_write)
    pilot = collect_future_viability_batch(
        manifest_path=tmp_path / "manifest.json",
        preflight_receipt_path=preflight_path,
        output_dir=tmp_path / "pilot",
        batch="pilot",
        root=tmp_path,
        env_factory=lambda game_id: object(),
    )
    pilot_receipt_path = tmp_path / "pilot" / "collection_receipt.json"
    completion = collect_future_viability_batch(
        manifest_path=tmp_path / "manifest.json",
        preflight_receipt_path=preflight_path,
        pilot_receipt_path=pilot_receipt_path,
        output_dir=tmp_path / "completion",
        batch="completion",
        root=tmp_path,
        env_factory=lambda game_id: object(),
    )
    assert pilot["status"] == "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY"
    assert completion["status"] == ("PASS_T12_6_1D_COMPLETION_COLLECTION_INTEGRITY")
    assert pilot["metrics"]["archive_count"] == 6
    assert completion["metrics"]["archive_count"] == 12
    assert pilot["metrics"]["sdk_calls_used"] == 6 * 2048
    assert completion["metrics"]["sdk_calls_used"] == 12 * 2048
    assert len(calls) == 18
    assert {call["search_seed"] for call in calls} == {9301, 9302, 9303}
    assert {call["witness"].source_seed for call in calls} == {8701, 8705}
    assert {call["arm"] for call in calls} == set(
        FutureViabilityProspectiveProtocol().search_arms
    )
    assert all(call["sdk_call_budget"] == 2048 for call in calls)
    assert all(call["maximum_cells"] == 10_000 for call in calls)
    assert sum(call["sdk_call_budget"] for call in calls) == 36_864
    assert sum(call["sdk_call_budget"] for call in calls) <= 38_000


def test_batch_firewalls_reject_missing_preflight_and_missing_pilot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        experiment_module,
        "load_future_viability_prospective_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment_module,
        "load_prospective_receipt",
        lambda *args, **kwargs: {
            "passed": True,
            "status": "NOT_A_PASSED_PREFLIGHT",
        },
    )
    with pytest.raises(ValueError, match="passed preflight"):
        collect_future_viability_batch(
            manifest_path=tmp_path / "manifest.json",
            preflight_receipt_path=tmp_path / "preflight.json",
            output_dir=tmp_path / "pilot",
            batch="pilot",
            root=tmp_path,
            env_factory=lambda game_id: object(),
        )
    monkeypatch.setattr(
        experiment_module,
        "load_prospective_receipt",
        lambda *args, **kwargs: {
            "passed": True,
            "status": "PASS_T12_6_1D_PREFLIGHT",
        },
    )
    with pytest.raises(ValueError, match="requires the pilot receipt"):
        collect_future_viability_batch(
            manifest_path=tmp_path / "manifest.json",
            preflight_receipt_path=tmp_path / "preflight.json",
            output_dir=tmp_path / "completion",
            batch="completion",
            root=tmp_path,
            env_factory=lambda game_id: object(),
        )


def test_prediction_and_adjudication_firewalls_require_prior_commitments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        experiment_module,
        "load_future_viability_prospective_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment_module,
        "load_prospective_receipt",
        lambda *args, **kwargs: {
            "passed": True,
            "status": "NOT_A_COLLECTION_SEAL",
        },
    )
    with pytest.raises(ValueError, match="requires the sealed collection"):
        commit_future_viability_predictions(
            manifest_path=tmp_path / "manifest.json",
            collection_seal_receipt_path=tmp_path / "seal.json",
            output_dir=tmp_path / "prediction",
            root=tmp_path,
        )

    def fake_receipt(*args, **kwargs):
        if kwargs["expected_phase"] == "collection_seal":
            return {
                "passed": True,
                "status": "PASS_T12_6_1D_COLLECTION_SEAL",
            }
        return {"passed": True, "status": "NOT_A_PREDICTION_COMMITMENT"}

    monkeypatch.setattr(experiment_module, "load_prospective_receipt", fake_receipt)
    with pytest.raises(ValueError, match="requires committed predictions"):
        adjudicate_future_viability_confirmation(
            manifest_path=tmp_path / "manifest.json",
            collection_seal_receipt_path=tmp_path / "seal.json",
            prediction_receipt_path=tmp_path / "prediction.json",
            output_dir=tmp_path / "adjudication",
            root=tmp_path,
        )


def test_freeze_binds_real_parents_without_old_archives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_future_viability_prospective_confirmation(
        output_path=manifest_path,
        reliability_manifest_path=_reliability_parent() / "manifest.json",
        reliability_compile_receipt_path=(
            _reliability_parent() / "compile" / "compile_receipt.json"
        ),
        hazard_manifest_path=_hazard_parent() / "manifest.json",
        hazard_compile_receipt_path=(
            _hazard_parent() / "compile" / "compile_receipt.json"
        ),
        root=_repo(),
    )
    loaded = load_future_viability_prospective_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert "inputs" not in loaded
    assert loaded["parents"]["reliability_model_bundle"]["bundle_checksum"]
    assert loaded["firewall"]["preflight_authorized"] is True
    assert loaded["firewall"]["pilot_collection_authorized"] is False
    assert loaded["firewall"]["t12_6_2_freeze_authorized"] is False
    preflight = preflight_future_viability_confirmation(
        manifest_path=manifest_path,
        output_dir=tmp_path / "preflight",
        root=_repo(),
    )
    assert preflight["status"] == "PASS_T12_6_1D_PREFLIGHT"
    assert preflight["metrics"]["sdk_calls_used"] == 0
    assert preflight["metrics"]["environment_collection_executed"] is False


def test_extractor_rejects_old_evaluation_seed(tmp_path: Path) -> None:
    path = tmp_path / "archive.json"
    _archive(
        path,
        [
            _edge(
                "one",
                source_cell="source",
                source_exact="exact-source",
                action="ACTION3",
                target_cell="target",
                target_exact="exact-target",
            )
        ],
    )
    with pytest.raises(ValueError, match="unregistered search seed"):
        _extract(path, seed=9201)
