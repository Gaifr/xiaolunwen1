"""Loss primitives for annotation-guided evidence learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def location_bce(
    evidence_scores: Tensor,
    tumor_fraction: Tensor,
    annotation_mask: Tensor,
) -> Tensor:
    """Continuous patch-level BCE over only reliably annotated tissue."""
    if evidence_scores.shape != tumor_fraction.shape or evidence_scores.shape != annotation_mask.shape:
        raise ValueError(
            "LOCATION_SHAPE "
            f"scores={tuple(evidence_scores.shape)} target={tuple(tumor_fraction.shape)} "
            f"mask={tuple(annotation_mask.shape)}"
        )
    mask = annotation_mask.bool()
    if not bool(mask.any()):
        return evidence_scores.sum() * 0.0
    return F.binary_cross_entropy(
        evidence_scores[mask].float().clamp(1e-6, 1.0 - 1e-6),
        tumor_fraction[mask].float(),
    )


def query_diversity_loss(query_attention: Tensor) -> Tensor:
    """Mean off-diagonal cosine similarity between Query attention maps."""
    if query_attention.ndim != 2:
        raise ValueError(f"QUERY_ATTENTION_SHAPE {tuple(query_attention.shape)}")
    query_count = query_attention.shape[0]
    if query_count < 2:
        return query_attention.sum() * 0.0
    normalized = F.normalize(query_attention.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(query_count, dtype=torch.bool, device=similarity.device)
    return similarity[off_diagonal].mean()
