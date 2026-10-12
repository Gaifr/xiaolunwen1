"""End-to-end Stage-1 training and resume smoke tests."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
import pytest

from modules.evidence_model import EvidenceModelConfig
from scripts.train_stage1 import Stage1TrainingConfig, run_stage1


def _write_slide(path: Path, *, tumor: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    node_count = 6
    labels = np.array([1, 1, 1, 0, 0, 0] if tumor else [0] * node_count, dtype=np.int64)
    features = rng.normal(size=(node_count, 16)).astype(np.float32)
    if tumor:
        features[:3, 0] += 3.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset(
            "coords", data=np.stack([np.arange(node_count) * 16, np.zeros(node_count)], axis=1)
        )
        handle.create_dataset("tumor_fraction", data=labels.astype(np.float32))
        handle.create_dataset("patch_labels", data=labels)
        handle.create_dataset("annotation_mask", data=np.ones(node_count, dtype=np.bool_))
        handle.create_dataset("lesion_ids", data=np.where(labels == 1, 0, -1))


def _tiny_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for split in ("train", "val", "test"):
        for tumor in (False, True):
            index += 1
            slide_id = f"{'tumor' if tumor else 'normal'}_{index:03d}"
            h5_path = root / f"{slide_id}.h5"
            _write_slide(h5_path, tumor=tumor, seed=index)
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": slide_id,
                    "split": split,
                    "slide_label": int(tumor),
                    "h5_path": str(h5_path),
                    "patch_size_level0": 16,
                    "annotation_status": "full" if tumor else "normal",
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _config(root: Path, **overrides) -> Stage1TrainingConfig:
    defaults = dict(
        output_root=root / "run",
        manifest=_tiny_manifest(root / "data"),
        model=EvidenceModelConfig(
            variant="pooling",
            graph_mode="none",
            input_dim=16,
            hidden_dim=8,
            num_queries=3,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        epochs=1,
        patience=5,
        lr=1e-3,
        weight_decay=0.0,
        seed=9233,
        device="cpu",
        lambda_loc=1.0,
        lambda_div=0.0,
    )
    defaults.update(overrides)
    return Stage1TrainingConfig(**defaults)


def test_stage1_tiny_run_writes_loss_components_and_resumes(tmp_path: Path):
    """Dropping optimizer/RNG state would make a resumed run a different experiment."""
    config = _config(tmp_path)
    first = run_stage1(config)
    resumed = run_stage1(
        replace(config, epochs=2, resume=first.last_checkpoint)
    )

    assert resumed.start_epoch == 2
    history = list(csv.DictReader((config.output_root / "history.csv").open(encoding="utf-8")))
    assert len(history) == 2
    assert {
        "loss_total",
        "loss_cls",
        "loss_loc",
        "loss_div",
        "weighted_loc",
        "val_auroc",
    } <= history[-1].keys()
    checkpoint = torch.load(resumed.last_checkpoint, map_location="cpu", weights_only=False)
    assert {
        "model_state",
        "optimizer_state",
        "scaler_state",
        "rng_state",
        "epoch",
    } <= checkpoint.keys()
    assert checkpoint["epoch"] == 2


def test_annotation_enabled_graph_query_has_finite_nonzero_location_loss(tmp_path: Path):
    """A disconnected annotation branch could train while reporting a fake zero loc loss."""
    config = _config(
        tmp_path,
        model=EvidenceModelConfig(
            variant="graph_query",
            graph_mode="dual",
            input_dim=16,
            hidden_dim=8,
            num_queries=3,
            num_heads=2,
            num_layers=1,
            k_s=2,
            k_m=2,
            dropout=0.0,
        ),
    )

    result = run_stage1(config)

    assert result.history[0]["loss_loc"] > 0.0
    assert np.isfinite(result.history[0]["loss_total"])


@pytest.mark.parametrize("variant", ["query", "linear_patch"])
def test_equal_supervision_e_and_f_smoke(tmp_path: Path, variant: str):
    config = _config(
        tmp_path,
        model=EvidenceModelConfig(
            variant=variant, graph_mode="none", input_dim=16, hidden_dim=8,
            num_queries=3, num_heads=2, num_layers=1, dropout=0.0,
        ),
    )
    result = run_stage1(config)
    assert result.history[0]["weighted_loc"] > 0
    assert result.history[0]["val_auroc"] >= 0
    checkpoint = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["selection_rule"] == "validation_slide_auroc_round6_earliest"
