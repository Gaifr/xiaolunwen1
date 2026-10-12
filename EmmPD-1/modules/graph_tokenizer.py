"""双关系图 tokenizer：k-NN 建图 → 双分支残差 GCN → Region Tokenizer。

契约来源：docs/superpowers/plans/2026-09-10-dual-graph-classifier.md Task 2-4。

设计要点：
- 空间图由 patch 坐标的欧氏距离构建（物理相邻组织）；
- 形态图由 UNI 特征余弦相似度构建（形态相似但可能空间远离的组织）；
- 两者都做「加反向边 + 去自环 + 去重」，输出确定性排序的边表，保证可复现；
- 双分支各用两层残差 GCN，节点门控融合（gate * spatial + (1 - gate) * morph）；
- 32 个可学习 query 通过交叉注意力把 N 个节点压成固定长度的潜在区域 token。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

DEFAULT_K = 8
DEFAULT_CHUNK = 512
DEFAULT_HIDDEN = 256
DEFAULT_INPUT_DIM = 1024
DEFAULT_DROPOUT = 0.1
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_QUERIES = 32
DEFAULT_NUM_HEADS = 8


def symmetrize_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """加反向边、去自环、去重；返回按 (src, dst) 确定性排序的 ``[2, E]`` 边表。"""
    edge_index = edge_index.long()
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    both = torch.stack((torch.cat((src, dst)), torch.cat((dst, src))))
    flat = torch.unique(both[0] * num_nodes + both[1])
    return torch.stack((flat // num_nodes, flat % num_nodes))


def build_spatial_knn(coords: Tensor, k: int = DEFAULT_K) -> Tensor:
    """按坐标欧氏距离的对称 k-NN 边表。"""
    coords = coords.float()
    num_nodes = coords.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=coords.device)
    kk = min(k, num_nodes - 1)
    distances = torch.cdist(coords, coords)
    distances.fill_diagonal_(float("inf"))
    dst = distances.topk(kk, dim=1, largest=False).indices
    src = torch.arange(num_nodes, device=coords.device)[:, None].expand_as(dst)
    return symmetrize_edge_index(torch.stack((src.reshape(-1), dst.reshape(-1))), num_nodes)


def build_morph_knn(
    features: Tensor, k: int = DEFAULT_K, chunk_size: int = DEFAULT_CHUNK
) -> Tensor:
    """按特征余弦相似度的对称 k-NN 边表；分块计算以限制显存占用。"""
    z = F.normalize(features.detach().float(), dim=-1)
    num_nodes = z.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=z.device)
    kk = min(k, num_nodes - 1)
    rows: list[Tensor] = []
    for start in range(0, num_nodes, chunk_size):
        scores = z[start : start + chunk_size] @ z.t()
        local = torch.arange(scores.size(0), device=z.device)
        scores[local, local + start] = -torch.inf
        dst = scores.topk(kk, dim=1).indices
        src = torch.arange(
            start, start + scores.size(0), device=z.device
        )[:, None].expand_as(dst)
        rows.append(torch.stack((src.reshape(-1), dst.reshape(-1))))
    return symmetrize_edge_index(torch.cat(rows, dim=1), num_nodes)


class ResidualGCNBlock(nn.Module):
    """单层残差 GCN 块：GCNConv → GELU → Dropout，再与输入残差相加后 LayerNorm。"""

    def __init__(self, dim: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.conv = GCNConv(dim, dim, add_self_loops=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        update = self.dropout(F.gelu(self.conv(x, edge_index)))
        return self.norm(x + update)


class DualRelationEncoder(nn.Module):
    """空间图与形态图的双分支编码器，输出节点级融合表示与门控值。

    返回 ``{"spatial", "morph", "fused", "gate"}``，每个都是 ``[N, hidden_dim]``；
    ``gate`` 是 ``[0, 1]`` 的逐维门控（1 偏空间关系，0 偏形态关系）。
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_blocks = nn.ModuleList(
            ResidualGCNBlock(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.morph_blocks = nn.ModuleList(
            ResidualGCNBlock(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        features: Tensor,
        spatial_edge_index: Tensor,
        morph_edge_index: Tensor,
    ) -> dict[str, Tensor]:
        projected = self.input_projection(features.float())

        spatial = projected
        for block in self.spatial_blocks:
            spatial = block(spatial, spatial_edge_index)

        morph = projected
        for block in self.morph_blocks:
            morph = block(morph, morph_edge_index)

        gate = torch.sigmoid(self.gate(torch.cat([spatial, morph], dim=-1)))
        fused = gate * spatial + (1.0 - gate) * morph
        return {"spatial": spatial, "morph": morph, "fused": fused, "gate": gate}


class RegionTokenizer(nn.Module):
    """用可学习 query 把 N 个节点压成固定数量（默认 32）的潜在区域 token。

    只使用交叉注意力（query 可学习，key/value 是节点），因此对节点顺序不变。
    """

    def __init__(
        self,
        dim: int = DEFAULT_HIDDEN,
        num_queries: int = DEFAULT_NUM_QUERIES,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, nodes: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        queries = self.queries.unsqueeze(0)
        keys = nodes.float().unsqueeze(0)
        tokens, weights = self.cross_attn(
            queries,
            keys,
            keys,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        normalized_tokens = self.norm(queries + tokens).squeeze(0)
        if not return_attention:
            return normalized_tokens
        attention = weights.squeeze(0)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return normalized_tokens, attention


class DualGraphRegionTokenizer(nn.Module):
    """完整切片级图 tokenizer：k-NN 建图 → 双分支编码 → 区域 token。

    返回 ``{"node_features": [N, hidden], "region_tokens": [Q, hidden], "gate": [N, hidden]}``。
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        k_s: int = DEFAULT_K,
        k_m: int = DEFAULT_K,
        num_queries: int = DEFAULT_NUM_QUERIES,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.k_s = k_s
        self.k_m = k_m
        self.encoder = DualRelationEncoder(input_dim, hidden_dim, dropout, num_layers)
        self.region = RegionTokenizer(hidden_dim, num_queries, num_heads, dropout)

    def forward(self, features: Tensor, coords: Tensor) -> dict[str, Tensor]:
        spatial_edges = build_spatial_knn(coords, self.k_s)
        morph_edges = build_morph_knn(features, self.k_m)
        encoded = self.encoder(features, spatial_edges, morph_edges)
        return {
            "node_features": encoded["fused"],
            "region_tokens": self.region(encoded["fused"]),
            "gate": encoded["gate"],
        }
