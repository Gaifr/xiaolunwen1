"""严格的患者级 / 切片级指标：不调阈值、不猜结果，契约不满足就直接报错。

契约来源：docs/superpowers/plans/2026-09-10-dual-graph-classifier.md Task 6。

原则：
- 二分类直接用两路 logits，不做任何阈值搜索；
- 切片级三分类用 one-vs-rest 计算 Macro AUROC / Macro AUPRC；
- 评价集合缺少所需类别时抛 :class:`MetricContractError`，绝不用 0 冒充结果。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

NUM_PATIENT_CLASSES = 2
NUM_SLIDE_CLASSES = 3


class MetricContractError(ValueError):
    """评价集合不满足指标契约（例如缺少某个类别）时抛出。"""


def _as_probabilities(probabilities, expected_classes: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != expected_classes:
        raise ValueError(f"PROB_SHAPE {probs.shape}: 期望 (N, {expected_classes})")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"NON_FINITE PROB {probs.shape}")
    return probs


def compute_binary_metrics(y_true, probabilities) -> dict[str, float]:
    """患者级二分类指标：accuracy / F1 / balanced accuracy / AUROC / AUPRC。"""
    labels = np.asarray(y_true).astype(int)
    probs = _as_probabilities(probabilities, NUM_PATIENT_CLASSES)
    present = sorted(set(labels.tolist()))
    if present != [0, 1]:
        raise MetricContractError(f"Binary metrics 需要 0/1 两类，实际 {present}")
    predicted = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        "auprc": float(average_precision_score(labels, probs[:, 1])),
    }


def compute_multiclass_metrics(
    y_true, probabilities, num_classes: int = NUM_SLIDE_CLASSES
) -> dict[str, float]:
    """切片级多分类指标：Macro AUROC / Macro AUPRC / Macro F1 / balanced accuracy。"""
    labels = np.asarray(y_true).astype(int)
    probs = _as_probabilities(probabilities, num_classes)
    present = sorted(set(labels.tolist()))
    if present != list(range(num_classes)):
        raise MetricContractError(f"Multiclass metrics 需要全部 {num_classes} 类，实际 {present}")
    predicted = probs.argmax(axis=1)
    one_hot = np.eye(num_classes)[labels]
    return {
        "macro_auroc": float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        ),
        "macro_auprc": float(average_precision_score(one_hot, probs, average="macro")),
        "macro_f1": float(
            f1_score(labels, predicted, average="macro")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
