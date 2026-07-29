"""Teacher transition encoder and deployable causal predictor for SAGE-MT."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .graph import MorphoTopologicalGraph
from .transition import MTTransitionRecord

MODEL_FORMAT_VERSION = "sage12-mt-model-v4.16"
DELTA_BUCKETS = 128


def _token_id(token: str, buckets: int) -> int:
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % max(1, int(buckets) - 1)


def _feature_tokens(prefix: str, values: Mapping[str, Any]) -> list[str]:
    output = []
    for key, raw in sorted(values.items()):
        items = raw if isinstance(raw, (list, tuple)) else (raw,)
        for item in items:
            output.append(f"{prefix}:{key}={item}")
    return output


def _event_targets(record: MTTransitionRecord, buckets: int = DELTA_BUCKETS) -> np.ndarray:
    values = np.zeros(buckets, dtype=np.float32)
    for event in record.events:
        values[_token_id(f"event:{event}", buckets) - 1] = 1.0
    return values


def _invariant_keys(records: Sequence[MTTransitionRecord]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(key)
                for record in records
                for key in record.invariant_deltas
            }
        )
    )


@dataclass(frozen=True)
class MTModelConfig:
    hash_buckets: int = 2048
    embedding_width: int = 32
    hidden_width: int = 128
    latent_width: int = 64
    message_passing_layers: int = 3
    maximum_nodes: int = 64
    maximum_edges: int = 256
    maximum_node_tokens: int = 16
    maximum_global_tokens: int = 24
    history_width: int = 8
    delta_buckets: int = DELTA_BUCKETS
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.10
    delta_loss_weight: float = 1.0
    contrastive_loss_weight: float = 1.0
    query_loss_weight: float = 0.5
    invariant_loss_weight: float = 0.25
    variance_loss_weight: float = 0.10
    identity_loss_weight: float = 0.10
    seed: int = 5_160


@dataclass(frozen=True)
class TensorizedGraphBatch:
    node_ids: np.ndarray
    node_mask: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_types: np.ndarray
    edge_mask: np.ndarray
    global_ids: np.ndarray


@dataclass(frozen=True)
class TensorizedTransitions:
    before: TensorizedGraphBatch
    after: TensorizedGraphBatch
    delta_targets: np.ndarray
    invariant_targets: np.ndarray
    signature_ids: np.ndarray
    state_ids: np.ndarray
    game_ids: np.ndarray
    invariant_keys: tuple[str, ...]


@dataclass(frozen=True)
class TransformationEmbedding:
    transition_id: str
    vector: tuple[float, ...]
    predicted_vector: tuple[float, ...]
    uncertainty: float
    delta_signature: str
    source_game_id: str


@dataclass(frozen=True)
class MTGraphPrediction:
    vector: tuple[float, ...]
    uncertainty: float
    delta_probabilities: tuple[float, ...]


def _tensorize_graphs(
    graphs: Sequence[MorphoTopologicalGraph],
    config: MTModelConfig,
    *,
    histories: Sequence[Sequence[str]] | None = None,
) -> TensorizedGraphBatch:
    if histories is not None and len(histories) != len(graphs):
        raise ValueError("SAGE-MT histories must align with graphs")
    count = len(graphs)
    node_ids = np.zeros(
        (
            count,
            config.maximum_nodes,
            config.maximum_node_tokens,
        ),
        dtype=np.int64,
    )
    node_mask = np.zeros((count, config.maximum_nodes), dtype=np.float32)
    edge_sources = np.zeros(
        (count, config.maximum_edges),
        dtype=np.int64,
    )
    edge_targets = np.zeros_like(edge_sources)
    edge_types = np.zeros_like(edge_sources)
    edge_mask = np.zeros((count, config.maximum_edges), dtype=np.float32)
    global_ids = np.zeros(
        (count, config.maximum_global_tokens),
        dtype=np.int64,
    )
    for graph_index, graph in enumerate(graphs):
        by_id = {
            node.node_id: index
            for index, node in enumerate(graph.nodes[: config.maximum_nodes])
        }
        for node_index, node in enumerate(graph.nodes[: config.maximum_nodes]):
            tokens = _feature_tokens("node", node.model_view())
            for token_index, token in enumerate(tokens[: config.maximum_node_tokens]):
                node_ids[graph_index, node_index, token_index] = _token_id(
                    token,
                    config.hash_buckets,
                )
            node_mask[graph_index, node_index] = 1.0
        relations = [
            relation
            for relation in graph.relations
            if relation.subject_id in by_id and relation.object_id in by_id
        ]
        for edge_index, relation in enumerate(relations[: config.maximum_edges]):
            edge_sources[graph_index, edge_index] = by_id[relation.subject_id]
            edge_targets[graph_index, edge_index] = by_id[relation.object_id]
            edge_types[graph_index, edge_index] = _token_id(
                f"edge:{relation.kind}",
                config.hash_buckets,
            )
            edge_mask[graph_index, edge_index] = 1.0
        globals_view = {
            "action_name": graph.action_name,
            "action_family": graph.action_family,
            **{
                f"invariant_{key}": min(max(int(value), -16), 16)
                for key, value in graph.invariants.items()
            },
        }
        tokens = _feature_tokens("global", globals_view)
        if histories is not None:
            tokens.extend(
                f"history:{offset}={value}"
                for offset, value in enumerate(
                    tuple(histories[graph_index])[-config.history_width :]
                )
            )
        for token_index, token in enumerate(tokens[: config.maximum_global_tokens]):
            global_ids[graph_index, token_index] = _token_id(
                token,
                config.hash_buckets,
            )
    return TensorizedGraphBatch(
        node_ids=node_ids,
        node_mask=node_mask,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_types=edge_types,
        edge_mask=edge_mask,
        global_ids=global_ids,
    )


def tensorize_transitions(
    records: Sequence[MTTransitionRecord],
    config: MTModelConfig,
    *,
    game_vocabulary: Sequence[str] | None = None,
) -> TensorizedTransitions:
    if not records:
        raise ValueError("cannot tensorize an empty SAGE-MT corpus")
    invariant_keys = _invariant_keys(records)
    signatures = {
        value: index
        for index, value in enumerate(
            sorted({record.delta_signature for record in records})
        )
    }
    states = {
        value: index
        for index, value in enumerate(
            sorted({record.graph_before.signature for record in records})
        )
    }
    games = tuple(
        sorted(
            set(game_vocabulary or ())
            | {record.source_game_id for record in records}
        )
    )
    game_index = {value: index for index, value in enumerate(games)}
    histories: list[tuple[str, ...]] = []
    sequence_history: dict[tuple[str, str], list[str]] = {}
    for record in records:
        sequence_key = (
            record.source_game_id,
            str(
                record.audit.get(
                    "episode_id",
                    record.audit.get("panel_id", record.transition_id),
                )
            ),
        )
        previous = sequence_history.setdefault(sequence_key, [])
        histories.append(tuple(previous[-config.history_width :]))
        previous.append(record.delta_signature)
    return TensorizedTransitions(
        before=_tensorize_graphs(
            [record.graph_before for record in records],
            config,
            histories=histories,
        ),
        after=_tensorize_graphs(
            [record.graph_after for record in records],
            config,
            histories=histories,
        ),
        delta_targets=np.stack(
            [_event_targets(record, config.delta_buckets) for record in records]
        ),
        invariant_targets=np.asarray(
            [
                [
                    float(np.clip(record.invariant_deltas.get(key, 0), -8, 8)) / 8.0
                    for key in invariant_keys
                ]
                for record in records
            ],
            dtype=np.float32,
        ),
        signature_ids=np.asarray(
            [signatures[record.delta_signature] for record in records],
            dtype=np.int64,
        ),
        state_ids=np.asarray(
            [states[record.graph_before.signature] for record in records],
            dtype=np.int64,
        ),
        game_ids=np.asarray(
            [game_index[record.source_game_id] for record in records],
            dtype=np.int64,
        ),
        invariant_keys=invariant_keys,
    )


def _torch_model(
    config: MTModelConfig,
    *,
    invariant_width: int,
    game_count: int,
) -> Any:
    import torch

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, values: Any) -> Any:
            return values.view_as(values)

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> Any:
            return -gradient

    class GraphEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(
                config.hash_buckets,
                config.embedding_width,
                padding_idx=0,
            )
            self.node_projection = torch.nn.Sequential(
                torch.nn.Linear(config.embedding_width, config.hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.hidden_width),
            )
            self.edge_projection = torch.nn.Linear(
                config.embedding_width + config.hidden_width,
                config.hidden_width,
            )
            self.updates = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_width * 2, config.hidden_width),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(config.hidden_width),
                )
                for _ in range(config.message_passing_layers)
            )
            self.global_projection = torch.nn.Sequential(
                torch.nn.Linear(config.embedding_width, config.hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.hidden_width),
            )
            self.output = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_width * 2, config.hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.hidden_width),
            )

        @staticmethod
        def _masked_mean(values: Any, mask: Any, dim: int) -> Any:
            weights = mask.to(values.dtype)
            while weights.ndim < values.ndim:
                weights = weights.unsqueeze(-1)
            return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)

        def forward(
            self,
            node_ids: Any,
            node_mask: Any,
            edge_sources: Any,
            edge_targets: Any,
            edge_types: Any,
            edge_mask: Any,
            global_ids: Any,
        ) -> Any:
            embedded_nodes = self.embedding(node_ids)
            token_mask = node_ids != 0
            nodes = self.node_projection(
                self._masked_mean(embedded_nodes, token_mask, dim=2)
            )
            batch_size, node_count, _ = nodes.shape
            for update in self.updates:
                source = torch.gather(
                    nodes,
                    1,
                    edge_sources.unsqueeze(-1).expand(-1, -1, nodes.shape[-1]),
                )
                edge = self.embedding(edge_types)
                messages = self.edge_projection(torch.cat((source, edge), dim=-1))
                messages = messages * edge_mask.unsqueeze(-1)
                aggregated = torch.zeros_like(nodes)
                degree = torch.zeros(
                    (batch_size, node_count, 1),
                    dtype=nodes.dtype,
                    device=nodes.device,
                )
                aggregated.scatter_add_(
                    1,
                    edge_targets.unsqueeze(-1).expand(-1, -1, nodes.shape[-1]),
                    messages,
                )
                degree.scatter_add_(
                    1,
                    edge_targets.unsqueeze(-1),
                    edge_mask.unsqueeze(-1),
                )
                aggregated = aggregated / degree.clamp_min(1.0)
                nodes = update(torch.cat((nodes, aggregated), dim=-1))
                nodes = nodes * node_mask.unsqueeze(-1)
            pooled_nodes = self._masked_mean(nodes, node_mask, dim=1)
            embedded_global = self.embedding(global_ids)
            pooled_global = self.global_projection(
                self._masked_mean(
                    embedded_global,
                    global_ids != 0,
                    dim=1,
                )
            )
            return self.output(torch.cat((pooled_nodes, pooled_global), dim=-1))

    class SAGEMTModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph_encoder = GraphEncoder()
            transition_width = config.hidden_width * 4
            self.transition_encoder = torch.nn.Sequential(
                torch.nn.Linear(transition_width, config.hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.hidden_width),
                torch.nn.Linear(config.hidden_width, config.latent_width),
            )
            self.query_encoder = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_width, config.hidden_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.hidden_width),
                torch.nn.Linear(config.hidden_width, config.latent_width),
            )
            self.uncertainty_head = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_width, config.hidden_width // 2),
                torch.nn.GELU(),
                torch.nn.Linear(config.hidden_width // 2, 1),
            )
            self.delta_head = torch.nn.Linear(
                config.latent_width,
                config.delta_buckets,
            )
            self.invariant_head = torch.nn.Linear(
                config.latent_width,
                max(1, invariant_width),
            )
            self.identity_head = torch.nn.Linear(
                config.latent_width,
                max(1, game_count),
            )

        @staticmethod
        def _normalize(values: Any) -> Any:
            return torch.nn.functional.normalize(values, p=2, dim=-1)

        def encode_graph(self, graph: Sequence[Any]) -> Any:
            return self.graph_encoder(*graph)

        def forward(
            self,
            before: Sequence[Any],
            after: Sequence[Any],
        ) -> Mapping[str, Any]:
            left = self.encode_graph(before)
            right = self.encode_graph(after)
            teacher = self._normalize(
                self.transition_encoder(
                    torch.cat((left, right, right - left, torch.abs(right - left)), dim=-1)
                )
            )
            query = self._normalize(self.query_encoder(left))
            return {
                "teacher": teacher,
                "query": query,
                "uncertainty": torch.nn.functional.softplus(
                    self.uncertainty_head(left)
                )[:, 0],
                "teacher_delta": self.delta_head(teacher),
                "query_delta": self.delta_head(query),
                "invariants": self.invariant_head(teacher)[:, :invariant_width],
                "identity": self.identity_head(GradientReverse.apply(teacher)),
            }

        def predict(self, before: Sequence[Any]) -> Mapping[str, Any]:
            left = self.encode_graph(before)
            query = self._normalize(self.query_encoder(left))
            return {
                "query": query,
                "uncertainty": torch.nn.functional.softplus(
                    self.uncertainty_head(left)
                )[:, 0],
                "delta": self.delta_head(query),
            }

    return SAGEMTModel()


def _graph_batch_to_torch(
    batch: TensorizedGraphBatch,
    indices: np.ndarray,
    *,
    device: str,
) -> tuple[Any, ...]:
    import torch

    return (
        torch.as_tensor(batch.node_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(batch.node_mask[indices], dtype=torch.float32, device=device),
        torch.as_tensor(batch.edge_sources[indices], dtype=torch.long, device=device),
        torch.as_tensor(batch.edge_targets[indices], dtype=torch.long, device=device),
        torch.as_tensor(batch.edge_types[indices], dtype=torch.long, device=device),
        torch.as_tensor(batch.edge_mask[indices], dtype=torch.float32, device=device),
        torch.as_tensor(batch.global_ids[indices], dtype=torch.long, device=device),
    )


def _supervised_contrastive_loss(
    latent: Any,
    signature_ids: Any,
    game_ids: Any,
    state_ids: Any,
    *,
    temperature: float,
) -> Any:
    import torch

    similarity = latent @ latent.T / max(float(temperature), 1e-4)
    eye = torch.eye(latent.shape[0], dtype=torch.bool, device=latent.device)
    positive = (
        signature_ids[:, None].eq(signature_ids[None, :])
        & game_ids[:, None].ne(game_ids[None, :])
        & ~eye
    )
    # Every transition also receives a dropout view so singleton signatures
    # retain a valid positive without inventing a semantic class.
    noisy = torch.nn.functional.normalize(
        torch.nn.functional.dropout(latent, p=0.10, training=True),
        p=2,
        dim=-1,
    )
    view_similarity = (latent * noisy).sum(dim=-1) / max(float(temperature), 1e-4)
    masked = similarity.masked_fill(eye, -1e9)
    denominator = torch.logsumexp(masked, dim=1)
    cross_game = torch.where(
        positive.any(dim=1),
        -(
            torch.logsumexp(similarity.masked_fill(~positive, -1e9), dim=1)
            - denominator
        ),
        torch.zeros_like(denominator),
    )
    view_loss = -view_similarity + torch.logsumexp(
        torch.cat((masked, view_similarity[:, None]), dim=1),
        dim=1,
    )
    hard_negative = (
        state_ids[:, None].eq(state_ids[None, :])
        & signature_ids[:, None].ne(signature_ids[None, :])
        & ~eye
    )
    hard_penalty = torch.relu(similarity * float(temperature) - 0.20)
    hard_loss = (
        hard_penalty[hard_negative].mean()
        if hard_negative.any()
        else latent.new_tensor(0.0)
    )
    return (cross_game + view_loss).mean() + hard_loss


def _variance_loss(latent: Any) -> Any:
    import torch

    if latent.shape[0] < 2:
        return latent.new_tensor(0.0)
    standard_deviation = torch.sqrt(latent.var(dim=0) + 1e-4)
    return torch.relu(1.0 / math.sqrt(latent.shape[1]) - standard_deviation).mean()


def fit_mt_model(
    records: Sequence[MTTransitionRecord],
    *,
    config: MTModelConfig | None = None,
    device: str = "cpu",
) -> tuple[Any, Mapping[str, Any]]:
    """Fit the teacher and deployable query encoder on observed transitions."""

    import torch

    selected = config or MTModelConfig()
    if not records:
        raise ValueError("SAGE-MT training requires observed transitions")
    random.seed(selected.seed)
    np.random.seed(selected.seed)
    torch.manual_seed(selected.seed)
    tensors = tensorize_transitions(records, selected)
    game_count = int(tensors.game_ids.max()) + 1
    model = _torch_model(
        selected,
        invariant_width=len(tensors.invariant_keys),
        game_count=game_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    losses: list[float] = []
    components: dict[str, list[float]] = {
        "delta": [],
        "contrastive": [],
        "query": [],
        "invariant": [],
        "variance": [],
        "identity": [],
    }
    indices = np.arange(len(records), dtype=np.int64)
    for epoch in range(max(1, int(selected.epochs))):
        rng = np.random.default_rng(selected.seed + epoch)
        game_order = sorted(set(tensors.game_ids.tolist()))
        rng.shuffle(game_order)
        balanced: list[int] = []
        per_game = {
            game: indices[tensors.game_ids == game].copy()
            for game in game_order
        }
        maximum = max(len(values) for values in per_game.values())
        for offset in range(maximum):
            for game in game_order:
                values = per_game[game]
                balanced.append(int(values[offset % len(values)]))
        for start in range(0, len(balanced), max(2, int(selected.batch_size))):
            batch_indices = np.asarray(
                balanced[start : start + max(2, int(selected.batch_size))],
                dtype=np.int64,
            )
            before = _graph_batch_to_torch(
                tensors.before,
                batch_indices,
                device=device,
            )
            after = _graph_batch_to_torch(
                tensors.after,
                batch_indices,
                device=device,
            )
            delta_targets = torch.as_tensor(
                tensors.delta_targets[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            invariant_targets = torch.as_tensor(
                tensors.invariant_targets[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            signature_ids = torch.as_tensor(
                tensors.signature_ids[batch_indices],
                dtype=torch.long,
                device=device,
            )
            game_ids = torch.as_tensor(
                tensors.game_ids[batch_indices],
                dtype=torch.long,
                device=device,
            )
            state_ids = torch.as_tensor(
                tensors.state_ids[batch_indices],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(before, after)
            delta_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["teacher_delta"],
                    delta_targets,
                )
                + 0.5
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    output["query_delta"],
                    delta_targets,
                )
            )
            contrastive = _supervised_contrastive_loss(
                output["teacher"],
                signature_ids,
                game_ids,
                state_ids,
                temperature=selected.temperature,
            )
            query = (1.0 - (output["teacher"].detach() * output["query"]).sum(dim=1)).mean()
            invariant = (
                torch.nn.functional.smooth_l1_loss(
                    output["invariants"],
                    invariant_targets,
                )
                if invariant_targets.shape[1]
                else output["teacher"].new_tensor(0.0)
            )
            variance = _variance_loss(output["teacher"])
            identity = torch.nn.functional.cross_entropy(
                output["identity"],
                game_ids,
            )
            loss = (
                selected.delta_loss_weight * delta_loss
                + selected.contrastive_loss_weight * contrastive
                + selected.query_loss_weight * query
                + selected.invariant_loss_weight * invariant
                + selected.variance_loss_weight * variance
                + selected.identity_loss_weight * identity
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in (
                ("delta", delta_loss),
                ("contrastive", contrastive),
                ("query", query),
                ("invariant", invariant),
                ("variance", variance),
                ("identity", identity),
            ):
                components[name].append(float(value.detach().cpu()))
    model.eval()
    return model, {
        "format_version": MODEL_FORMAT_VERSION,
        "config": asdict(selected),
        "invariant_keys": list(tensors.invariant_keys),
        "games": sorted({record.source_game_id for record in records}),
        "records": len(records),
        "final_loss": losses[-1],
        "losses": {
            key: float(np.mean(values[-max(1, len(records) // selected.batch_size) :]))
            for key, values in components.items()
            if values
        },
    }


def encode_transitions(
    model: Any,
    records: Sequence[MTTransitionRecord],
    *,
    config: MTModelConfig,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[TransformationEmbedding, ...]:
    import torch

    if not records:
        return ()
    tensors = tensorize_transitions(records, config)
    output: list[TransformationEmbedding] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), max(1, int(batch_size))):
            indices = np.arange(start, min(len(records), start + batch_size))
            before = _graph_batch_to_torch(tensors.before, indices, device=device)
            after = _graph_batch_to_torch(tensors.after, indices, device=device)
            values = model(before, after)
            teacher = values["teacher"].cpu().numpy()
            query = values["query"].cpu().numpy()
            uncertainty = values["uncertainty"].cpu().numpy()
            for local_index, record_index in enumerate(indices):
                record = records[int(record_index)]
                output.append(
                    TransformationEmbedding(
                        transition_id=record.transition_id,
                        vector=tuple(float(item) for item in teacher[local_index]),
                        predicted_vector=tuple(
                            float(item) for item in query[local_index]
                        ),
                        uncertainty=float(uncertainty[local_index]),
                        delta_signature=record.delta_signature,
                        source_game_id=record.source_game_id,
                    )
                )
    return tuple(output)


def predict_graphs(
    model: Any,
    graphs: Sequence[MorphoTopologicalGraph],
    *,
    config: MTModelConfig,
    device: str = "cpu",
    histories: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[tuple[float, ...], float], ...]:
    detailed = predict_graph_details(
        model,
        graphs,
        config=config,
        device=device,
        histories=histories,
    )
    return tuple(
        (item.vector, item.uncertainty)
        for item in detailed
    )


def predict_graph_details(
    model: Any,
    graphs: Sequence[MorphoTopologicalGraph],
    *,
    config: MTModelConfig,
    device: str = "cpu",
    histories: Sequence[Sequence[str]] | None = None,
) -> tuple[MTGraphPrediction, ...]:
    import torch

    if not graphs:
        return ()
    batch = _tensorize_graphs(graphs, config, histories=histories)
    indices = np.arange(len(graphs), dtype=np.int64)
    values = model.predict(
        _graph_batch_to_torch(batch, indices, device=device)
    )
    vectors = values["query"].detach().cpu().numpy()
    uncertainty = values["uncertainty"].detach().cpu().numpy()
    delta = torch.sigmoid(values["delta"]).detach().cpu().numpy()
    return tuple(
        MTGraphPrediction(
            vector=tuple(float(item) for item in vectors[index]),
            uncertainty=float(uncertainty[index]),
            delta_probabilities=tuple(float(item) for item in delta[index]),
        )
        for index in range(len(graphs))
    )


def checkpoint_payload(
    model: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": MODEL_FORMAT_VERSION,
        "metadata": dict(metadata),
        "state_dict": model.state_dict(),
    }


def load_mt_model(
    payload: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Any, MTModelConfig, Mapping[str, Any]]:
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("unsupported SAGE-MT checkpoint")
    metadata = dict(payload["metadata"])
    config = MTModelConfig(**dict(metadata["config"]))
    model = _torch_model(
        config,
        invariant_width=len(metadata.get("invariant_keys", ())),
        game_count=max(1, len(metadata.get("games", ()))),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, metadata


__all__ = [
    "DELTA_BUCKETS",
    "MODEL_FORMAT_VERSION",
    "MTGraphPrediction",
    "MTModelConfig",
    "TensorizedGraphBatch",
    "TensorizedTransitions",
    "TransformationEmbedding",
    "checkpoint_payload",
    "encode_transitions",
    "fit_mt_model",
    "load_mt_model",
    "predict_graph_details",
    "predict_graphs",
    "tensorize_transitions",
]
