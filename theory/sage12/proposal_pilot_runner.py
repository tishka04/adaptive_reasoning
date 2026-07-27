"""Run the frozen Qwen benchmark and SAGE12 grounded-proposal evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from theory.live_transition_loop import build_observation
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import (
    SAGE11_SPLITS,
    SOURCE_TRAIN,
    SOURCE_VALIDATION,
)
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _make_real_env,
)

from .compiler import HypothesisCompiler
from .controller import SemanticActionCandidate
from .hypotheses import SemanticHypothesis, hypotheses_from_json
from .llm import (
    LocalHypothesisGenerator,
    TemplateHypothesisGenerator,
    TransformersJSONModel,
    TransformersModelConfig,
)
from .proposal_pilot_data import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    ProposalPilotTrace,
    graph_from_mapping,
    load_frozen_manifest,
    read_trace_shard,
)
from .scene_graph import GroundedRelation, SceneGraph, build_scene_graph


BENCHMARK_FORMAT_VERSION = "sage12-qwen-device-benchmark-v1"
RESULT_FORMAT_VERSION = "sage12-grounded-proposal-result-v1"


def run_device_benchmark(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    graphs = _benchmark_graphs(
        frozen,
        environment_root=(
            Path(environments_dir)
            if environments_dir is not None
            else _env_dir()
        ),
    )
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda:0")
    except ImportError:
        pass
    results: dict[str, Any] = {}
    outputs: dict[str, list[str]] = {}
    for device in devices:
        measurement, raw = _measure_device(
            frozen,
            graphs=graphs,
            device=device,
        )
        results[device] = measurement
        outputs[device] = raw
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    equality = 1.0
    if "cuda:0" in outputs:
        equality = sum(
            left == right
            for left, right in zip(outputs["cpu"], outputs["cuda:0"])
        ) / len(outputs["cpu"])
    selected = "cpu"
    speedup = 1.0
    if "cuda:0" in results:
        cpu_median = float(results["cpu"]["median_inference_seconds"])
        gpu_median = float(results["cuda:0"]["median_inference_seconds"])
        speedup = cpu_median / max(1e-9, gpu_median)
        if speedup >= float(
            frozen["device_benchmark"]["minimum_gpu_speedup"]
        ):
            selected = "cuda:0"
    payload = {
        "format_version": BENCHMARK_FORMAT_VERSION,
        "status": "COMPLETE",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "model_sha256": frozen["model"]["weights_sha256"],
        "decoding": frozen["model"]["decoding"],
        "devices": results,
        "cpu_gpu_exact_output_equality": equality,
        "gpu_speedup": speedup,
        "selected_device": selected,
        "selection_uses_quality_outcomes": False,
    }
    payload["result_checksum"] = _payload_checksum(payload)
    _write_json_atomic(destination / "device_benchmark.json", payload)
    return payload


def run_evaluation(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    collection = json.loads(
        (destination / "collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        collection.get("frozen_manifest_checksum")
        != frozen["manifest_checksum"]
    ):
        raise ValueError("collection does not match frozen pilot")
    benchmark = json.loads(
        (destination / "device_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        benchmark.get("frozen_manifest_checksum")
        != frozen["manifest_checksum"]
    ):
        raise ValueError("device benchmark does not match frozen pilot")
    traces = _load_collection(frozen, destination)
    sample = _representative_sample(traces, frozen)
    output_path = destination / "model_outputs.jsonl"
    output_records = _run_or_resume_model_outputs(
        frozen,
        sample=sample,
        output_path=output_path,
        device=str(benchmark["selected_device"]),
    )
    result = _score(
        frozen,
        traces=traces,
        sample=sample,
        model_outputs=output_records,
    )
    result.update(
        {
            "format_version": RESULT_FORMAT_VERSION,
            "frozen_manifest_checksum": frozen["manifest_checksum"],
            "collection_manifest_checksum": _payload_checksum(collection),
            "model_outputs_sha256": _file_sha256(output_path),
            "device_benchmark_checksum": benchmark["result_checksum"],
            "selected_device": benchmark["selected_device"],
            "source_train_rows": collection["source_train_rows"],
            "source_validation_rows": collection[
                "source_validation_rows"
            ],
            "representative_rows": len(sample),
            "world_model_fit_started": False,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
        }
    )
    result["result_checksum"] = _payload_checksum(result)
    _write_json_atomic(destination / "pilot_result.json", result)
    return result


def _benchmark_graphs(
    frozen: Mapping[str, Any],
    *,
    environment_root: Path,
) -> tuple[tuple[str, SceneGraph, tuple[str, ...]], ...]:
    graphs = []
    for game in frozen["device_benchmark"]["games"]:
        SAGE11_SPLITS.assert_authorized(
            (game,),
            purpose=(
                "train"
                if game in SOURCE_TRAIN
                else "validate_source"
            ),
        )
        env = _make_real_env(str(game), environment_root)
        try:
            frame = _reset_env(env)
        except ModuleNotFoundError as exc:
            if exc.name != "arcengine":
                raise
            frame = env.step(0)
        snapshot = snapshot_frame(frame)
        legal = tuple(
            item
            for item in _valid_actions(env)
            if item.name not in {"", "RESET"}
        )
        names = tuple(sorted(set(_available_action_names(legal))))
        observation = build_observation(
            snapshot.grid,
            available_actions=names,
            game_state=snapshot.game_state,
            levels_completed=snapshot.levels_completed,
            infer_players=True,
        )
        graphs.append((str(game), build_scene_graph(observation), names))
    return tuple(graphs)


def _measure_device(
    frozen: Mapping[str, Any],
    *,
    graphs: Sequence[tuple[str, SceneGraph, tuple[str, ...]]],
    device: str,
) -> tuple[dict[str, Any], list[str]]:
    model_config = TransformersModelConfig(
        model_path=str(frozen["model"]["path"]),
        device=device,
        temperature=float(
            frozen["model"]["decoding"]["temperature"]
        ),
        local_files_only=True,
    )
    started = time.perf_counter()
    backend = TransformersJSONModel(model_config)
    backend._load()
    load_seconds = time.perf_counter() - started
    generator = LocalHypothesisGenerator(
        backend,
        maximum_hypotheses=int(frozen["model"]["maximum_hypotheses"]),
        maximum_tokens=int(
            frozen["model"]["decoding"]["maximum_new_tokens"]
        ),
    )
    timings = []
    outputs = []
    valid = 0
    for _, graph, actions in graphs:
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        started = time.perf_counter()
        result = generator.generate(
            graph=graph,
            available_actions=actions,
            subgoal=str(frozen["evaluation"]["subgoal"]),
        )
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        outputs.append(result.raw_response)
        valid += int(not result.parse_error)
    return (
        {
            "device": device,
            "model_load_seconds": load_seconds,
            "inference_seconds": timings,
            "median_inference_seconds": statistics.median(timings),
            "mean_inference_seconds": statistics.fmean(timings),
            "strict_json_valid_rate": valid / len(graphs),
            "prompt_count": len(graphs),
        },
        outputs,
    )


def _load_collection(
    frozen: Mapping[str, Any],
    destination: Path,
) -> tuple[ProposalPilotTrace, ...]:
    records = []
    for game, quota in frozen["game_quotas"].items():
        shard = read_trace_shard(destination / "shards" / f"{game}.jsonl")
        if len(shard) != int(quota):
            raise ValueError(f"{game} proposal shard row mismatch")
        records.extend(shard)
    return tuple(records)


def _representative_sample(
    traces: Sequence[ProposalPilotTrace],
    frozen: Mapping[str, Any],
) -> tuple[ProposalPilotTrace, ...]:
    per_game = int(frozen["evaluation"]["representative_rows_per_game"])
    selected = []
    by_game: dict[str, list[ProposalPilotTrace]] = defaultdict(list)
    for trace in traces:
        by_game[trace.game_id].append(trace)
    for game in frozen["game_quotas"]:
        ranked = sorted(
            by_game[game],
            key=lambda trace: hashlib.sha256(
                (
                    frozen["evaluation"]["sample_salt"]
                    + trace.digest
                ).encode("ascii")
            ).hexdigest(),
        )
        selected.extend(ranked[:per_game])
    return tuple(selected)


def _run_or_resume_model_outputs(
    frozen: Mapping[str, Any],
    *,
    sample: Sequence[ProposalPilotTrace],
    output_path: Path,
    device: str,
) -> tuple[dict[str, Any], ...]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    existing[(item["trace_digest"], item["variant"])] = item
    expected = {
        (trace.digest, variant)
        for trace in sample
        for variant in ("original", "relation_shuffle")
    }
    if set(existing) - expected:
        raise ValueError("model-output checkpoint contains unexpected rows")
    backend = TransformersJSONModel(
        TransformersModelConfig(
            model_path=str(frozen["model"]["path"]),
            device=device,
            temperature=float(
                frozen["model"]["decoding"]["temperature"]
            ),
            local_files_only=True,
        )
    )
    generator = LocalHypothesisGenerator(
        backend,
        maximum_hypotheses=int(frozen["model"]["maximum_hypotheses"]),
        maximum_tokens=int(
            frozen["model"]["decoding"]["maximum_new_tokens"]
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for trace in sample:
            graph = graph_from_mapping(trace.scene_graph)
            for variant in ("original", "relation_shuffle"):
                key = (trace.digest, variant)
                if key in existing:
                    continue
                candidate_graph = (
                    graph
                    if variant == "original"
                    else _shuffle_relations(graph, trace.digest)
                )
                started = time.perf_counter()
                generated = generator.generate(
                    graph=candidate_graph,
                    available_actions=trace.available_action_names,
                    subgoal=str(frozen["evaluation"]["subgoal"]),
                )
                record = {
                    "trace_digest": trace.digest,
                    "game_id": trace.game_id,
                    "source_split": trace.source_split,
                    "variant": variant,
                    "raw_response": generated.raw_response,
                    "parse_error": generated.parse_error,
                    "inference_seconds": time.perf_counter() - started,
                }
                handle.write(
                    json.dumps(record, sort_keys=True) + "\n"
                )
                handle.flush()
                existing[key] = record
    return tuple(existing[key] for key in sorted(existing))


def _shuffle_relations(graph: SceneGraph, salt: str) -> SceneGraph:
    entity_ids = sorted(entity.entity_id for entity in graph.entities)
    if len(entity_ids) < 2:
        return graph
    shift = int(salt[:8], 16) % (len(entity_ids) - 1) + 1
    mapped = {
        entity_id: entity_ids[(index + shift) % len(entity_ids)]
        for index, entity_id in enumerate(entity_ids)
    }
    relations = tuple(
        GroundedRelation(
            kind=relation.kind,
            subject_id=mapped[relation.subject_id],
            object_id=mapped[relation.object_id],
        )
        for relation in graph.relations
    )
    state = {
        predicate
        for predicate in graph.state_predicates
        if predicate.split("|", 1)[0]
        not in {relation.kind for relation in graph.relations}
    }
    state.update(relation.key for relation in relations)
    signature = hashlib.sha256(
        json.dumps(
            {
                "entities": [
                    {
                        "id": entity.entity_id,
                        "roles": entity.roles,
                        "area": entity.area_bucket,
                        "aspect": entity.aspect_bucket,
                    }
                    for entity in graph.entities
                ],
                "relations": sorted(state),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return SceneGraph(
        entities=graph.entities,
        relations=relations,
        state_predicates=frozenset(state),
        signature=signature,
    )


def _score(
    frozen: Mapping[str, Any],
    *,
    traces: Sequence[ProposalPilotTrace],
    sample: Sequence[ProposalPilotTrace],
    model_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_by_key = {
        (str(item["trace_digest"]), str(item["variant"])): item
        for item in model_outputs
    }
    compiler = HypothesisCompiler()
    template = TemplateHypothesisGenerator()
    action_baseline = _fit_action_baseline(traces)
    rows = []
    for trace in sample:
        graph = graph_from_mapping(trace.scene_graph)
        observed = _mechanism_names(trace.observed_effects)
        candidates = tuple(
            SemanticActionCandidate(name)
            for name in trace.available_action_names
        )
        variants = {}
        for variant in ("original", "relation_shuffle"):
            output = output_by_key[(trace.digest, variant)]
            hypotheses: tuple[SemanticHypothesis, ...] = ()
            if not output["parse_error"]:
                try:
                    hypotheses = hypotheses_from_json(
                        str(output["raw_response"]),
                        maximum=int(
                            frozen["model"]["maximum_hypotheses"]
                        ),
                    )
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    hypotheses = ()
            candidate_graph = (
                graph
                if variant == "original"
                else _shuffle_relations(graph, trace.digest)
            )
            compilation = compiler.compile(
                hypotheses,
                graph=candidate_graph,
                legal_candidates=candidates,
            )
            hit = _compiled_hit(
                compilation.options,
                action_name=trace.selected_action_name,
                observed_mechanisms=observed,
            )
            variants[variant] = {
                "json_valid": not bool(output["parse_error"]),
                "parsed": len(hypotheses),
                "support_zero": sum(
                    hypothesis.support == 0 for hypothesis in hypotheses
                ),
                "grounded_hypotheses": len(
                    {
                        option.hypothesis_id
                        for option in compilation.options
                    }
                ),
                "rejected_hypotheses": len(compilation.rejected),
                "hit": hit,
            }
        template_hypotheses = template.generate(
            graph=graph,
            available_actions=trace.available_action_names,
            subgoal=str(frozen["evaluation"]["subgoal"]),
        ).hypotheses
        template_options = compiler.compile(
            template_hypotheses,
            graph=graph,
            legal_candidates=candidates,
        ).options
        baseline_effect = action_baseline.get(trace.selected_action_name, "")
        rows.append(
            {
                "game_id": trace.game_id,
                "source_split": trace.source_split,
                "eligible": bool(trace.productive and observed),
                "llm_hit": variants["original"]["hit"],
                "shuffle_hit": variants["relation_shuffle"]["hit"],
                "template_hit": _compiled_hit(
                    template_options,
                    action_name=trace.selected_action_name,
                    observed_mechanisms=observed,
                ),
                "action_hit": baseline_effect in observed,
                **{
                    f"original_{key}": value
                    for key, value in variants["original"].items()
                    if key != "hit"
                },
            }
        )
    validation = [
        row for row in rows if row["source_split"] == "source_validation"
    ]
    eligible = [row for row in validation if row["eligible"]]
    llm_recall = _mean(row["llm_hit"] for row in eligible)
    shuffle_recall = _mean(row["shuffle_hit"] for row in eligible)
    template_recall = _mean(row["template_hit"] for row in eligible)
    action_recall = _mean(row["action_hit"] for row in eligible)
    stronger = max(template_recall, action_recall)
    parsed = sum(row["original_parsed"] for row in validation)
    support_zero = sum(row["original_support_zero"] for row in validation)
    grounded = sum(
        row["original_grounded_hypotheses"] for row in validation
    )
    strict_json = _mean(row["original_json_valid"] for row in validation)
    per_game = {}
    for game in SOURCE_VALIDATION:
        game_rows = [
            row
            for row in eligible
            if row["game_id"] == game
        ]
        game_llm = _mean(row["llm_hit"] for row in game_rows)
        game_template = _mean(row["template_hit"] for row in game_rows)
        game_action = _mean(row["action_hit"] for row in game_rows)
        per_game[game] = {
            "eligible_rows": len(game_rows),
            "llm_recall_at_8": game_llm,
            "template_recall": game_template,
            "action_only_recall": game_action,
            "gain_vs_stronger_baseline": (
                game_llm - max(game_template, game_action)
            ),
        }
    leakage = _game_identity_probe(
        [trace for trace in traces if trace.source_split == "source_train"]
    )
    metrics = {
        "validation_rows": len(validation),
        "eligible_validation_rows": len(eligible),
        "strict_json_validity": strict_json,
        "support_zero_rate": support_zero / parsed if parsed else 0.0,
        "grounded_hypothesis_rate": grounded / parsed if parsed else 0.0,
        "llm_productive_mechanism_recall_at_8": llm_recall,
        "template_recall": template_recall,
        "action_only_recall": action_recall,
        "stronger_baseline_recall": stronger,
        "gain_vs_stronger_baseline": llm_recall - stronger,
        "relation_shuffle_recall": shuffle_recall,
        "relation_shuffle_degradation": llm_recall - shuffle_recall,
        "per_game": per_game,
        "game_identity_probe": leakage,
    }
    thresholds = frozen["gates"]
    gates = {
        "strict_json_validity": strict_json
        >= float(thresholds["minimum_strict_json_validity"]),
        "support_zero": (
            support_zero / parsed if parsed else 0.0
        )
        >= float(thresholds["minimum_support_zero_rate"]),
        "grounded_hypotheses": (
            grounded / parsed if parsed else 0.0
        )
        >= float(thresholds["minimum_grounded_hypothesis_rate"]),
        "recall_gain": llm_recall - stronger
        >= float(thresholds["minimum_recall_gain"]),
        "relation_shuffle": llm_recall - shuffle_recall
        >= float(thresholds["minimum_relation_shuffle_degradation"]),
        "per_game_nonnegative": all(
            item["eligible_rows"] > 0
            and item["gain_vs_stronger_baseline"] >= 0.0
            for item in per_game.values()
        ),
        "game_identity": leakage["accuracy"]
        <= leakage["majority_accuracy"]
        + float(thresholds["maximum_identity_gain_over_majority"]),
    }
    passed = all(gates.values())
    return {
        "status": "PASS" if passed else "FAIL_CLOSED",
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": passed,
        "authorized_next_stage": (
            "semantic_world_model_pilot" if passed else "none"
        ),
    }


def _fit_action_baseline(
    traces: Sequence[ProposalPilotTrace],
) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        if trace.source_split != "source_train":
            continue
        for mechanism in _mechanism_names(trace.observed_effects):
            counts[trace.selected_action_name][mechanism] += 1
    return {
        action: counter.most_common(1)[0][0]
        for action, counter in counts.items()
        if counter
    }


def _compiled_hit(
    options: Sequence[Any],
    *,
    action_name: str,
    observed_mechanisms: set[str],
) -> bool:
    return any(
        option.action_name == action_name
        and any(
            effect.split("|", 1)[0] in observed_mechanisms
            for effect in option.asserted_effects
        )
        for option in options
    )


def _mechanism_names(effects: Iterable[str]) -> set[str]:
    return {
        str(effect).split("|", 1)[0]
        for effect in effects
        if str(effect).split("|", 1)[0] not in {"changed", "progress"}
    }


def _game_identity_probe(
    traces: Sequence[ProposalPilotTrace],
) -> dict[str, Any]:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    feature_rows = [_identity_features(trace) for trace in traces]
    labels = np.asarray([trace.game_id for trace in traces])
    matrix = DictVectorizer(sparse=True).fit_transform(feature_rows)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=12)
    scores = cross_val_score(
        LogisticRegression(max_iter=1_000, random_state=12),
        matrix,
        labels,
        cv=folds,
        scoring="accuracy",
    )
    majority = max(Counter(labels).values()) / len(labels)
    return {
        "model": "logistic_regression_3fold",
        "rows": len(labels),
        "games": len(set(labels)),
        "accuracy": float(np.mean(scores)),
        "fold_accuracies": [float(value) for value in scores],
        "majority_accuracy": float(majority),
        "gain_over_majority": float(np.mean(scores) - majority),
    }


def _identity_features(trace: ProposalPilotTrace) -> dict[str, float]:
    graph = trace.scene_graph
    features: Counter[str] = Counter()
    entities = list(graph.get("entities", ()))
    features["entity_count"] = len(entities)
    for entity in entities:
        features[f"area:{entity['area_bucket']}"] += 1
        features[f"aspect:{entity['aspect_bucket']}"] += 1
        for role in entity["roles"]:
            features[f"role:{role}"] += 1
    for relation in graph.get("relations", ()):
        features[f"relation:{relation['kind']}"] += 1
    for action in trace.available_action_names:
        features[f"available:{action}"] = 1
    return {key: float(value) for key, value in features.items()}


def _mean(values: Iterable[Any]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_checksum", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen SAGE12 proposal pilot."
    )
    parser.add_argument(
        "command",
        choices=("benchmark", "evaluate"),
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST_PATH,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        result = run_device_benchmark(
            frozen_manifest_path=args.frozen_manifest,
            output_dir=args.out_dir,
            environments_dir=args.environments_dir,
        )
    else:
        result = run_evaluation(
            frozen_manifest_path=args.frozen_manifest,
            output_dir=args.out_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
