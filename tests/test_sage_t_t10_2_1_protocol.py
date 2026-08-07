from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from theory.sage_t import t10_2_1_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARENT_PATHS = (
    protocol.PARENT_T10_1_MANIFEST_PATH,
    protocol.PARENT_T10_1_REPORT_PATH,
    protocol.PARENT_T10_2_MANIFEST_PATH,
    protocol.PARENT_T10_2_REPORT_PATH,
)


def _copy_parent_lineage(destination: Path) -> None:
    for relative_path in PARENT_PATHS:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative_path, target)


def _manifest_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, tuple[str, ...], tuple[str, ...], dict[str, str]]:
    _copy_parent_lineage(tmp_path)
    for registered in protocol.DEFAULT_PREREGISTRATION_PUBLICATION_FILES:
        publication = tmp_path / registered
        publication.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / registered, publication)

    code_path = tmp_path / "fixture_code.py"
    input_path = tmp_path / "fixture_input.md"
    code_path.write_bytes(b"VALUE = 1\n")
    input_path.write_bytes(b"# Frozen input\n")

    shard_paths: dict[str, str] = {}
    metadata_paths: dict[str, str] = {}
    for index, game_id in enumerate(protocol.SOURCE_GAMES):
        shard_relative = Path("fixtures") / f"source_{index}.jsonl"
        metadata_relative = Path("fixtures") / f"source_{index}.metadata.json"
        shard = tmp_path / shard_relative
        metadata = tmp_path / metadata_relative
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(
            protocol.canonical_json(
                {"event_id": f"fixture-{index}", "game_id": game_id}
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        metadata.write_text(
            protocol.canonical_json({"game_id": game_id, "verified": True})
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        shard_paths[game_id] = shard_relative.as_posix()
        metadata_paths[game_id] = metadata_relative.as_posix()

    monkeypatch.setattr(protocol, "DEFAULT_SOURCE_SHARD_FILES", shard_paths)
    monkeypatch.setattr(protocol, "DEFAULT_SOURCE_METADATA_FILES", metadata_paths)
    code_paths = (code_path.relative_to(tmp_path).as_posix(),)
    input_paths = (input_path.relative_to(tmp_path).as_posix(),)
    environment = {"fixture_runtime": "deterministic", "python": "test"}
    manifest = protocol.build_manifest(
        repo_root=tmp_path,
        code_paths=code_paths,
        input_paths=input_paths,
        environment=environment,
        verify_repository=False,
    )
    return manifest, code_path, code_paths, input_paths, environment


def _write_signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    signed = protocol.signed_payload(payload, checksum_key="report_checksum")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        protocol.canonical_json(signed) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return signed


def _source_fit_evidence(
    tmp_path: Path,
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    compile_report = _write_signed(
        destination / "compile_report.json",
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "compile",
            "status": "PASS_T10_2_1_QA",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": True,
            "checks": {"qa_passed": True},
        },
    )
    replay_report = _write_signed(
        destination / "replay_report.json",
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "replay",
            "status": "T10_2_1_SOURCE_REPLAY_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "compile_report_checksum": compile_report["report_checksum"],
        },
    )
    for name, material in (
        ("source_events.jsonl", b"{}\n"),
        ("replay_events.jsonl", b"{}\n"),
        (protocol.CROSS_FIT_AUDIT_FILENAME, b"{}\n"),
    ):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(material)
    return destination, compile_report, replay_report


def test_manifest_is_deterministic_and_self_authenticating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _, code_paths, input_paths, environment = _manifest_fixture(
        tmp_path, monkeypatch
    )
    second = protocol.build_manifest(
        repo_root=tmp_path,
        code_paths=code_paths,
        input_paths=input_paths,
        environment=environment,
        verify_repository=False,
    )

    assert first == second
    unsigned = dict(first)
    checksum = unsigned.pop("manifest_checksum")
    assert checksum == protocol.canonical_sha256(unsigned)
    assert first["registered_phases"] == list(protocol.PHASES)
    assert first["artifact_contract"]["artifact_root"] == (
        protocol.DEFAULT_OUTPUT_DIR.as_posix()
    )
    assert first["preregistration_publication"] == {
        "files": list(protocol.DEFAULT_PREREGISTRATION_PUBLICATION_FILES),
        "portable_initial_sha256": {
            path: protocol.canonical_file_sha256(tmp_path / path)
            for path in protocol.DEFAULT_PREREGISTRATION_PUBLICATION_FILES
        },
        "revalidated_as_runtime_input": False,
    }
    assert first["firewall"] == {
        "source_train_games": list(protocol.SOURCE_GAMES),
        "source_validation_games": list(protocol.VALIDATION_GAMES),
        "source_validation_opened": False,
        "ar25_opened": False,
        "holdout_opened": False,
        "production_authority": False,
    }


def test_portable_hash_normalizes_lf_crlf_and_manifest_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"alpha = 1\nbeta = 2\n")
    crlf.write_bytes(b"alpha = 1\r\nbeta = 2\r\n")

    assert protocol.canonical_file_sha256(lf) == protocol.canonical_file_sha256(crlf)
    assert protocol.raw_file_sha256(lf) != protocol.raw_file_sha256(crlf)

    manifest, code_path, code_paths, input_paths, environment = _manifest_fixture(
        tmp_path / "checkout", monkeypatch
    )
    manifest_path = tmp_path / "manifest.json"
    protocol.write_compact_json(manifest_path, manifest)
    code_path.write_bytes(code_path.read_bytes().replace(b"\n", b"\r\n"))

    loaded = protocol.load_manifest(
        manifest_path,
        repo_root=tmp_path / "checkout",
        code_paths=code_paths,
        input_paths=input_paths,
        environment=environment,
        verify_repository=False,
    )
    assert loaded["portable_code_sha256"] == manifest["portable_code_sha256"]


def test_jsonl_portable_digest_and_ordered_root_ignore_newlines_not_row_order(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    reversed_rows = tmp_path / "reversed.jsonl"
    lf.write_bytes(b'{"b":2,"a":1}\n{"row":2}\n')
    crlf.write_bytes(b'{ "a": 1, "b": 2 }\r\n{"row":2}\r\n')
    reversed_rows.write_bytes(b'{"row":2}\n{"a":1,"b":2}\n')

    assert protocol.canonical_file_sha256(lf) == protocol.canonical_file_sha256(crlf)
    assert protocol.ordered_jsonl_root_sha256(lf) == (
        protocol.ordered_jsonl_root_sha256(crlf)
    )
    assert protocol.raw_file_sha256(lf) != protocol.raw_file_sha256(crlf)
    assert protocol.ordered_jsonl_root_sha256(lf) != (
        protocol.ordered_jsonl_root_sha256(reversed_rows)
    )


def test_manifest_and_output_namespaces_are_fixed(tmp_path: Path) -> None:
    manifest = {
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        }
    }
    assert protocol.DEFAULT_MANIFEST_RELATIVE_PATH == Path(
        "theory/sage_t/sage_t10_2_1_protocol_manifest.json"
    )
    assert protocol._registered_output_dir(
        manifest=manifest,
        output_dir=protocol.DEFAULT_OUTPUT_DIR,
        repo_root=tmp_path,
    ) == (tmp_path / protocol.DEFAULT_OUTPUT_DIR).resolve()

    with pytest.raises(protocol.FirewallError, match="registered artifact namespace"):
        protocol._registered_output_dir(
            manifest=manifest,
            output_dir=tmp_path / "alternate-output",
            repo_root=tmp_path,
        )
    with pytest.raises(protocol.ManifestDriftError, match="namespace drifted"):
        protocol._registered_output_dir(
            manifest={"artifact_contract": {"artifact_root": "elsewhere"}},
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )


def test_parent_lineage_is_exact_and_tampering_is_refused(tmp_path: Path) -> None:
    before = {
        path.as_posix(): hashlib.sha256(
            (REPOSITORY_ROOT / path).read_bytes()
        ).hexdigest()
        for path in PARENT_PATHS
    }
    assert protocol._verify_parent_lineage(REPOSITORY_ROOT) == {
        "t10_1_manifest_checksum": protocol.PARENT_T10_1_MANIFEST_CHECKSUM,
        "t10_1_report_checksum": protocol.PARENT_T10_1_REPORT_CHECKSUM,
        "t10_2_manifest_checksum": protocol.PARENT_T10_2_MANIFEST_CHECKSUM,
        "t10_2_report_checksum": protocol.PARENT_T10_2_REPORT_CHECKSUM,
        "t10_2_verdict": protocol.PARENT_T10_2_VERDICT,
    }
    after = {
        path.as_posix(): hashlib.sha256(
            (REPOSITORY_ROOT / path).read_bytes()
        ).hexdigest()
        for path in PARENT_PATHS
    }
    assert after == before

    _copy_parent_lineage(tmp_path)
    report_path = tmp_path / protocol.PARENT_T10_2_REPORT_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["verdict"] = "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED"
    unsigned = dict(report)
    unsigned.pop("report_checksum", None)
    report["report_checksum"] = protocol.canonical_sha256(unsigned)
    report_path.write_text(protocol.canonical_json(report) + "\n", encoding="utf-8")

    with pytest.raises(protocol.ManifestDriftError, match="parent lineage drifted"):
        protocol._verify_parent_lineage(tmp_path)


def test_registered_seeds_budgets_and_firewalls_are_closed() -> None:
    source_seeds = set(protocol.DISCOVERY_SEEDS) | set(protocol.CONFIRMATION_SEEDS)
    auxiliary_seeds = {
        protocol.FIT_SEED,
        protocol.BOOTSTRAP_SEED,
        protocol.PERMUTATION_SEED,
    }

    assert protocol.DISCOVERY_SEEDS == (101, 102, 103)
    assert protocol.CONFIRMATION_SEEDS == (111, 112, 113)
    assert set(protocol.DISCOVERY_SEEDS).isdisjoint(protocol.CONFIRMATION_SEEDS)
    assert source_seeds.isdisjoint(auxiliary_seeds)
    assert source_seeds.isdisjoint(protocol.VALIDATION_SEEDS)
    assert auxiliary_seeds.isdisjoint(protocol.VALIDATION_SEEDS)
    assert protocol.SOURCE_LANE_COUNT == len(protocol.SOURCE_GAMES) * len(source_seeds)
    assert (
        protocol.SOURCE_RESET_REPORT_COUNT
        == protocol.SOURCE_LANE_COUNT * protocol.SOURCE_RESETS_PER_GAME_SEED
    )
    assert protocol.SOURCE_MAXIMUM_ACTIONS == (
        len(protocol.SOURCE_GAMES)
        * len(source_seeds)
        * protocol.SOURCE_RESETS_PER_GAME_SEED
        * protocol.SOURCE_ACTIONS_PER_RESET
    )
    assert protocol.SOURCE_MAXIMUM_NEW_ACTIONS == 4_608
    assert protocol.RESET_COOPERATIVE_STOP_SECONDS < protocol.RESET_HARD_TIMEOUT_SECONDS
    assert (
        protocol.SOURCE_STOP_NEW_ACTIONS_SECONDS
        < protocol.SOURCE_MAXIMUM_WALL_SECONDS
    )

    for game_id in protocol.SOURCE_GAMES:
        protocol.enforce_environment_firewall(phase="collect", game_id=game_id)
    for game_id in protocol.VALIDATION_GAMES:
        protocol.enforce_environment_firewall(
            phase="validate", game_id=game_id, source_gate_passed=True
        )

    with pytest.raises(protocol.FirewallError):
        protocol.enforce_environment_firewall(
            phase="collect", game_id=protocol.VALIDATION_GAMES[0]
        )
    with pytest.raises(protocol.GateRefusalError):
        protocol.enforce_environment_firewall(
            phase="validate",
            game_id=protocol.VALIDATION_GAMES[0],
            source_gate_passed=False,
        )
    with pytest.raises(protocol.FirewallError):
        protocol.enforce_environment_firewall(
            phase="validate",
            game_id=protocol.SOURCE_GAMES[0],
            source_gate_passed=True,
        )
    with pytest.raises(protocol.FirewallError, match="ar25 remains closed"):
        protocol.enforce_environment_firewall(
            phase="validate",
            game_id=protocol.AR25_GAME,
            source_gate_passed=True,
        )


def test_unattested_acquisition_miss_becomes_data_and_blocks_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "a" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    output_dir = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    _write_signed(
        output_dir / "collection_report.json",
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "collect",
            "status": "SOURCE_ACQUISITION_OR_RESOURCE_MISS",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": {"acquisition_gate_passed": False},
        },
    )

    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)

    def forbidden_legacy_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the frozen compiler or trainer must not be invoked")

    monkeypatch.setattr(protocol._t10_2, "compile_phase", forbidden_legacy_call)
    monkeypatch.setattr(protocol._t10_2, "source_train_phase", forbidden_legacy_call)

    compile_report = protocol.compile_phase(
        manifest_path=tmp_path / "manifest.json",
        output_dir=output_dir,
        repo_root=tmp_path,
    )
    assert compile_report["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert compile_report["verdict"] == "DATA_OR_PROVENANCE_INVALID"
    assert compile_report["passed"] is False
    assert compile_report["checks"] == {
        "acquisition_gate_passed": False,
        "compiler_invoked": False,
        "trainer_invoked": False,
        "validation_closed": True,
    }

    with pytest.raises(
        protocol.GateRefusalError,
        match="compile integrity did not pass",
    ):
        protocol.replay_phase(
            manifest_path=tmp_path / "manifest.json",
            output_dir=output_dir,
            repo_root=tmp_path,
        )

    source_report = protocol.source_train_phase(
        manifest_path=tmp_path / "manifest.json",
        output_dir=output_dir,
        repo_root=tmp_path,
    )
    assert source_report["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert source_report["terminal_stage"] == "acquisition"
    assert source_report["checks"]["trainer_invoked"] is False
    assert source_report["registered_controls"] == {
        name: False for name in protocol.REGISTERED_SOURCE_CONTROLS
    }
    assert source_report["firewall"]["source_validation_opened"] is False
    protocol._reconstruct_negative_source_report(
        manifest=manifest,
        source=source_report,
        destination=output_dir,
    )
    tampered_source = dict(source_report)
    tampered_source["checks"] = {
        **source_report["checks"],
        "trainer_invoked": True,
    }
    tampered_source = protocol.signed_payload(
        tampered_source,
        checksum_key="report_checksum",
    )
    with pytest.raises(protocol.ManifestDriftError, match="did not reconstruct"):
        protocol._reconstruct_negative_source_report(
            manifest=manifest,
            source=tampered_source,
            destination=output_dir,
        )

    def forbidden_bindings() -> None:
        raise AssertionError("validation factories must not be constructed")

    monkeypatch.setattr(protocol, "_legacy_bindings", forbidden_bindings)
    with pytest.raises(protocol.GateRefusalError, match="source gate did not pass"):
        protocol.validate_phase(
            manifest_path=tmp_path / "manifest.json",
            output_dir=output_dir,
            repo_root=tmp_path,
        )


def test_signed_but_incomplete_source_gate_cannot_open_validation(
    tmp_path: Path,
) -> None:
    manifest = {"manifest_checksum": "b" * 64}
    current_gate_path = tmp_path / "current_source_report.json"
    current_gate = _write_signed(
        current_gate_path,
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "PASS_T10_2_1_SOURCE_GATE",
            "verdict": "PASS_T10_2_1_SOURCE_GATE",
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": {"source_gate_passed": True},
            "registered_controls": {
                name: True for name in protocol.REGISTERED_SOURCE_CONTROLS
            },
            "registered_randomness": protocol._registered_randomness_spec(),
            "passed": True,
            "firewall": {
                "source_validation_opened": True,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
    )

    with pytest.raises(protocol.ManifestDriftError, match="reconstructable evidence"):
        protocol._require_source_gate(
            manifest=manifest,
            source_report_path=current_gate_path,
            output_dir=tmp_path,
        )

    legacy_gate_path = tmp_path / "legacy_source_report.json"
    _write_signed(
        legacy_gate_path,
        {
            **{
                key: value
                for key, value in current_gate.items()
                if key != "report_checksum"
            },
            "status": "PASS_T10_2_SOURCE_GATE",
            "verdict": "PASS_T10_2_SOURCE_GATE",
        },
    )
    with pytest.raises(protocol.GateRefusalError, match="source gate did not pass"):
        protocol._require_source_gate(
            manifest=manifest,
            source_report_path=legacy_gate_path,
            output_dir=tmp_path,
        )


def test_acquisition_taxonomy_requires_exact_resource_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {"manifest_checksum": "c" * 64}
    reconstruct_attestation = protocol._collection_failure_is_attested
    collection = {
        "status": "SOURCE_ACQUISITION_OR_RESOURCE_MISS",
        "manifest_checksum": manifest["manifest_checksum"],
        "report_checksum": "d" * 64,
    }
    monkeypatch.setattr(
        protocol,
        "_collection_failure_is_attested",
        lambda *args, **kwargs: True,
    )
    resource = protocol._build_acquisition_failure_report(
        manifest=manifest,
        collection=collection,
        output_dir=tmp_path,
    )
    assert resource["status"] == "SOURCE_ACQUISITION_OR_RESOURCE_MISS"

    monkeypatch.setattr(
        protocol,
        "_collection_failure_is_attested",
        lambda *args, **kwargs: False,
    )
    data = protocol._build_acquisition_failure_report(
        manifest=manifest,
        collection=collection,
        output_dir=tmp_path,
    )
    assert data["status"] == "DATA_OR_PROVENANCE_INVALID"

    boolean_counts = {
        "status": "SOURCE_ACQUISITION_OR_RESOURCE_MISS",
        "event_count": True,
        "action_accounting": {
            "authorized_intent_count": 0,
            "sealed_event_count": 0,
            "explicitly_unresolved_intent_count": 0,
            "unknown_intent_count": 0,
        },
        "durability": {
            "lane_report_count": 0,
            "reset_report_count": 0,
            "physical_steps_replayed_on_resume": 0,
        },
    }
    boolean_path = tmp_path / "boolean_collection_report.json"
    _write_signed(boolean_path, boolean_counts)
    resigned_boolean_counts = protocol._read_signed_json(
        boolean_path,
        checksum_key="report_checksum",
    )
    assert not reconstruct_attestation(resigned_boolean_counts)


def test_complete_lanes_finalization_timeout_is_resource_only_with_hard_receipt(
) -> None:
    full_timeout = {
        "event_count": 1,
        "action_accounting": {
            "authorized_intent_count": 1,
            "sealed_event_count": 1,
            "explicitly_unresolved_intent_count": 0,
            "unknown_intent_count": 0,
            "maximum_authorized_intents": protocol.SOURCE_MAXIMUM_ACTIONS,
            "equation_holds": True,
        },
        "durability": {
            "lane_report_count": protocol.SOURCE_LANE_COUNT,
            "reset_report_count": protocol.SOURCE_RESET_REPORT_COUNT,
            "journal_reconstructed": True,
            "checkpoint_reconstructed": True,
            "physical_steps_replayed_on_resume": 0,
        },
        "checks": {
            name: True
            for name in (
                "action_equation_holds",
                "no_unknown_intents",
                "authorized_action_cap",
                "sealed_events_bound",
                "journal_reconstructed",
                "checkpoint_reconstructed",
                "physical_steps_not_replayed",
                "absolute_wall_bound",
                "source_firewall_closed",
            )
        },
        "timing": {
            "stop_new_actions_seconds": protocol.SOURCE_STOP_NEW_ACTIONS_SECONDS,
            "absolute_seconds": protocol.SOURCE_MAXIMUM_WALL_SECONDS,
        },
        "terminal_reason": "registered_collection_deadline",
        "invocation": {"terminal_status": "HARD_TIMEOUT"},
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }

    assert protocol._collection_failure_is_attested(full_timeout)
    closed_timeout_claim = {
        **full_timeout,
        "invocation": {"terminal_status": "CLOSED"},
    }
    assert not protocol._collection_failure_is_attested(closed_timeout_claim)


def test_validation_opening_marker_is_one_shot_and_reconstructible(
    tmp_path: Path,
) -> None:
    manifest = {"manifest_checksum": "d" * 64}
    source_path = tmp_path / "source_report.json"
    source = _write_signed(
        source_path,
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "PASS_T10_2_1_SOURCE_GATE",
            "manifest_checksum": manifest["manifest_checksum"],
        },
    )
    marker_path = tmp_path / protocol.VALIDATION_OPENING_MARKER_FILENAME
    payload = protocol._validation_opening_payload(
        manifest=manifest,
        source_report=source,
        source_path=source_path,
    )
    protocol._create_one_shot_marker(
        marker_path,
        payload,
        label="validation",
    )

    assert protocol._verify_validation_opening_marker(
        path=marker_path,
        manifest=manifest,
        source_report=source,
        source_path=source_path,
    ) == payload
    with pytest.raises(protocol.GateRefusalError, match="already consumed"):
        protocol._create_one_shot_marker(
            marker_path,
            payload,
            label="validation",
        )


def test_source_fit_marker_binds_all_inputs_and_freezes_upstream_phases(
    tmp_path: Path,
) -> None:
    manifest = {
        "manifest_checksum": "3" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination, compile_report, replay_report = _source_fit_evidence(
        tmp_path,
        manifest,
    )
    marker_path = destination / protocol.SOURCE_FIT_OPENING_MARKER_FILENAME
    payload = protocol._source_fit_opening_payload(
        manifest=manifest,
        destination=destination,
        compile_report=compile_report,
        replay_report=replay_report,
    )
    protocol._create_one_shot_marker(
        marker_path,
        payload,
        label="source fit",
    )

    assert protocol._verify_source_fit_opening_marker(
        path=marker_path,
        manifest=manifest,
        destination=destination,
        compile_report=compile_report,
        replay_report=replay_report,
    ) == payload
    with pytest.raises(protocol.GateRefusalError, match="already consumed"):
        protocol._create_one_shot_marker(
            marker_path,
            payload,
            label="source fit",
        )
    for phase in ("collect", "compile", "replay"):
        with pytest.raises(protocol.GateRefusalError, match="source fit opening"):
            protocol._refuse_upstream_mutation_after_source_fit(
                destination=destination,
                phase=phase,
            )


def test_source_fit_partial_run_refuses_and_complete_run_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "4" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination, compile_report, replay_report = _source_fit_evidence(
        tmp_path,
        manifest,
    )
    protocol._create_one_shot_marker(
        destination / protocol.SOURCE_FIT_OPENING_MARKER_FILENAME,
        protocol._source_fit_opening_payload(
            manifest=manifest,
            destination=destination,
            compile_report=compile_report,
            replay_report=replay_report,
        ),
        label="source fit",
    )
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)

    with pytest.raises(protocol.GateRefusalError, match="lacks a reconstructible"):
        protocol.source_train_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )

    (destination / "source_report.json").write_text("{}\n", encoding="utf-8")
    reconstructed = {"status": "PASS_T10_2_1_SOURCE_GATE"}
    monkeypatch.setattr(
        protocol,
        "_reconstruct_existing_source_report",
        lambda **kwargs: reconstructed,
    )
    assert protocol.source_train_phase(
        manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        output_dir=protocol.DEFAULT_OUTPUT_DIR,
        repo_root=tmp_path,
    ) == reconstructed


def test_validation_marker_partial_run_refuses_but_terminal_run_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "e" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    source_path = destination / "source_report.json"
    source = _write_signed(
        source_path,
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "PASS_T10_2_1_SOURCE_GATE",
            "manifest_checksum": manifest["manifest_checksum"],
        },
    )
    marker_path = destination / protocol.VALIDATION_OPENING_MARKER_FILENAME
    protocol._create_one_shot_marker(
        marker_path,
        protocol._validation_opening_payload(
            manifest=manifest,
            source_report=source,
            source_path=source_path,
        ),
        label="validation",
    )
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        protocol,
        "_require_source_gate",
        lambda **kwargs: source,
    )

    with pytest.raises(protocol.GateRefusalError, match="did not reach"):
        protocol.validate_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )

    (destination / "validation_report.json").write_text("{}\n", encoding="utf-8")
    terminal = {"status": "PASS_T10_2_1_VALIDATION"}
    monkeypatch.setattr(
        protocol,
        "_reconstruct_existing_validation_report",
        lambda **kwargs: terminal,
    )
    assert protocol.validate_phase(
        manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        output_dir=protocol.DEFAULT_OUTPUT_DIR,
        repo_root=tmp_path,
    ) == terminal


def test_validation_requires_canonical_source_and_freezes_upstream_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "f" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True)
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)

    with pytest.raises(protocol.ManifestDriftError, match="canonical source report"):
        protocol.validate_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            source_report_path=tmp_path / "external-source-report.json",
            repo_root=tmp_path,
        )

    marker = destination / protocol.VALIDATION_OPENING_MARKER_FILENAME
    marker.write_text("claimed\n", encoding="utf-8")
    for phase in ("collect", "compile", "replay", "source-train"):
        with pytest.raises(protocol.GateRefusalError, match="frozen after"):
            protocol._refuse_upstream_mutation_after_validation(
                destination=destination,
                phase=phase,
            )


def test_validation_external_clock_starts_before_source_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "1" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    source_path = destination / "source_report.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}\n", encoding="utf-8")
    clock_calls: list[str] = []

    def external_clock() -> float:
        clock_calls.append("clock")
        return 10.0

    def refuse_source_gate(**kwargs: Any) -> dict[str, Any]:
        assert clock_calls == ["clock"]
        raise protocol.GateRefusalError("fixture source gate closed")

    monkeypatch.setattr(protocol.time, "perf_counter", external_clock)
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(protocol, "_require_source_gate", refuse_source_gate)

    with pytest.raises(protocol.GateRefusalError, match="fixture source gate"):
        protocol.validate_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )
    assert clock_calls == ["clock"]


def test_final_report_requires_validation_marker_and_negative_forbids_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "manifest_checksum": "2" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    source_path = destination / "source_report.json"
    positive = _write_signed(
        source_path,
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "PASS_T10_2_1_SOURCE_GATE",
            "verdict": "PASS_T10_2_1_SOURCE_GATE",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": True,
        },
    )
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        protocol,
        "_require_source_gate",
        lambda **kwargs: positive,
    )

    with pytest.raises(protocol.ManifestDriftError, match="signed JSON"):
        protocol.report_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )

    negative = _write_signed(
        source_path,
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "DATA_OR_PROVENANCE_INVALID",
            "verdict": "DATA_OR_PROVENANCE_INVALID",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": False,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
    )
    assert negative["passed"] is False
    (destination / protocol.VALIDATION_OPENING_MARKER_FILENAME).write_text(
        "claimed\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol.ManifestDriftError, match="closed source gate"):
        protocol.report_phase(
            manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
            output_dir=protocol.DEFAULT_OUTPUT_DIR,
            repo_root=tmp_path,
        )


def test_cli_has_separate_gated_phases_and_returns_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = protocol.build_parser()
    assert parser.parse_args(["freeze"]).phase == "freeze"
    assert parser.parse_args(["source-train"]).phase == "source-train"
    with pytest.raises(SystemExit):
        parser.parse_args(["all"])

    def refuse_compile(**kwargs: Any) -> dict[str, Any]:
        raise protocol.GateRefusalError("fixture gate closed")

    monkeypatch.setattr(protocol, "compile_phase", refuse_compile)
    exit_code = protocol.main(
        [
            "compile",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--repo-root",
            str(tmp_path),
        ]
    )
    emitted = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert emitted["phase"] == "compile"
    assert emitted["error"].startswith("GateRefusalError:")


def test_parent_code_faithful_digests_admits_lf_and_crlf(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_bytes(b"VALUE = 1\nOTHER = 2\n")

    lf_digest = hashlib.sha256(b"VALUE = 1\nOTHER = 2\n").hexdigest()
    crlf_digest = hashlib.sha256(b"VALUE = 1\r\nOTHER = 2\r\n").hexdigest()

    digests = protocol._parent_code_faithful_digests(source)

    assert lf_digest in digests
    assert crlf_digest in digests
    assert protocol.canonical_file_sha256(source) == lf_digest


def test_verify_parent_code_accepts_crlf_frozen_registry_entry() -> None:
    # The parent T10.2 manifest froze a handful of files from a CRLF working
    # tree.  This guard must pass against the real repository even though the
    # in-tree bytes are LF; regression for the T10.2.1 freeze CRLF drift.
    verified = protocol._verify_parent_code(REPOSITORY_ROOT)

    parent_manifest = json.loads(
        (REPOSITORY_ROOT / protocol.PARENT_T10_2_MANIFEST_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert verified == {
        str(path): str(digest)
        for path, digest in parent_manifest["code_sha256"].items()
    }
