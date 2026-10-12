"""Unified pooling/query/graph-query model for annotation-guided Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.camelyon_dataset import EvidenceSlideInput
from modules.graph_tokenizer import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN,
    DEFAULT_INPUT_DIM,
    DEFAULT_K,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_QUERIES,
    RegionTokenizer,
    ResidualGCNBlock,
    build_morph_knn,
    build_spatial_knn,
)


MODEL_VARIANTS = ("pooling", "query", "graph_query", "linear_patch")
GRAPH_MODES = ("none", "spatial", "morph", "dual")


@dataclass(frozen=True)
class EvidenceModelConfig:
    variant: str = "graph_query"
    graph_mode: str = "dual"
    input_dim: int = DEFAULT_INPUT_DIM
    hidden_dim: int = DEFAULT_HIDDEN
    k_s: int = DEFAULT_K
    k_m: int = DEFAULT_K
    num_queries: int = DEFAULT_NUM_QUERIES
    num_heads: int = DEFAULT_NUM_HEADS
    dropout: float = DEFAULT_DROPOUT
    num_layers: int = DEFAULT_NUM_LAYERS


@dataclass
class EvidenceModelOutput:
    slide_logits: Tensor
    evidence_scores: Tensor
    region_tokens: Tensor
    query_attention: Tensor
    node_features: Tensor
    gate: Tensor | None
    raw_patch_logits: Tensor | None = None


def _blocks(count: int, dim: int, dropout: float) -> nn.ModuleList:
    return nn.ModuleList(ResidualGCNBlock(dim, dropout) for _ in range(count))


class EvidenceModel(nn.Module):
    """Single WSI to class logits, evidence map and fixed region tokens."""

    def __init__(self, config: EvidenceModelConfig) -> None:
        super().__init__()
        if config.variant not in MODEL_VARIANTS or config.graph_mode not in GRAPH_MODES:
            raise ValueError(f"MODEL_CONFIG {config}")
        if config.variant in {"pooling", "query", "linear_patch"} and config.graph_mode != "none":
            raise ValueError(f"MODEL_CONFIG variant={config.variant} graph={config.graph_mode}")
        if config.variant == "graph_query" and config.graph_mode == "none":
            raise ValueError("MODEL_CONFIG graph_query requires a graph")
        self.config = config

        if config.variant == "linear_patch":
            self.linear_patch_head = nn.Linear(config.input_dim, 1)
            return

        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        needs_spatial = config.graph_mode in {"spatial", "dual"}
        needs_morph = config.graph_mode in {"morph", "dual"}
        self.spatial_blocks = (
            _blocks(config.num_layers, config.hidden_dim, config.dropout)
            if needs_spatial
            else None
        )
        self.morph_blocks = (
            _blocks(config.num_layers, config.hidden_dim, config.dropout)
            if needs_morph
            else None
        )
        self.gate_layer = (
            nn.Linear(2 * config.hidden_dim, config.hidden_dim)
            if config.graph_mode == "dual"
            else None
        )
        self.region = (
            RegionTokenizer(
                config.hidden_dim,
                config.num_queries,
                config.num_heads,
                config.dropout,
            )
            if config.variant != "pooling"
            else None
        )
        self.classifier = nn.Linear(config.hidden_dim, 2)
        self.node_evidence_head = nn.Linear(config.hidden_dim, 1)
        self.query_importance = nn.Linear(config.hidden_dim, 1)
        self.evidence_scale_raw = nn.Parameter(torch.zeros(()))
        self.evidence_bias = nn.Parameter(torch.zeros(()))

    def _encode_nodes(self, slide: EvidenceSlideInput) -> tuple[Tensor, Tensor | None]:
        projected = self.input_projection(slide.features.float())
        if self.config.graph_mode == "none":
            return projected, None

        spatial = projected
        if self.spatial_blocks is not None:
            edges = build_spatial_knn(slide.coords_level0, self.config.k_s)
            for block in self.spatial_blocks:
                spatial = block(spatial, edges)

        morph = projected
        if self.morph_blocks is not None:
            edges = build_morph_knn(slide.features, self.config.k_m)
            for block in self.morph_blocks:
                morph = block(morph, edges)

        if self.config.graph_mode == "spatial":
            return spatial, None
        if self.config.graph_mode == "morph":
            return morph, None

        assert self.gate_layer is not None
        gate = torch.sigmoid(self.gate_layer(torch.cat([spatial, morph], dim=-1)))
        return gate * spatial + (1.0 - gate) * morph, gate

    def _attention_evidence(self, tokens: Tensor, attention: Tensor) -> Tensor:
        importance = self.query_importance(tokens).squeeze(-1).softmax(dim=0)
        mass = torch.sum(importance[:, None] * attention, dim=0)
        relative_to_uniform = mass * attention.shape[1]
        scale = F.softplus(self.evidence_scale_raw)
        return torch.sigmoid(
            scale * torch.log(relative_to_uniform.clamp_min(1e-8)) + self.evidence_bias
        )

    def forward(self, slide: EvidenceSlideInput) -> EvidenceModelOutput:
        if self.config.variant == "linear_patch":
            logits = self.linear_patch_head(slide.features.float()).squeeze(-1)
            slide_logit = logits.max()
            return EvidenceModelOutput(
                slide_logits=torch.stack((-slide_logit / 2, slide_logit / 2)),
                evidence_scores=torch.sigmoid(logits),
                region_tokens=slide.features.new_empty((0, self.config.input_dim)),
                query_attention=logits.new_empty((0, logits.numel())),
                node_features=slide.features,
                gate=None,
                raw_patch_logits=logits,
            )
        node_features, gate = self._encode_nodes(slide)
        if self.region is None:
            region_tokens = node_features.mean(dim=0, keepdim=True)
            query_attention = node_features.new_empty((0, node_features.shape[0]))
            evidence_scores = torch.sigmoid(
                self.node_evidence_head(node_features).squeeze(-1)
            )
        else:
            region_tokens, query_attention = self.region(
                node_features, return_attention=True
            )
            evidence_scores = self._attention_evidence(region_tokens, query_attention)
        return EvidenceModelOutput(
            slide_logits=self.classifier(region_tokens.mean(dim=0)),
            evidence_scores=evidence_scores,
            region_tokens=region_tokens,
            query_attention=query_attention,
            node_features=node_features,
            gate=gate,
        )
