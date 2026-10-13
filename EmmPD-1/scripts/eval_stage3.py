"""Independent frozen Stage-3 evaluator. Run validation before test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.camelyon_dataset import CamelyonDataset
from modules.evidence_model import EvidenceModel, EvidenceModelConfig
from modules.stage3_review import assert_repairs_approved
from modules.stage3_metrics import (area_metrics, overlap_matrix, parse_truth, patch_metrics,
                                    peak_froc, validate_grid)
from scripts.freeze_stage3 import verify_lock


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_threshold(labels, scores):
    # Youden J; ties take the higher threshold.
    options = sorted(set(map(float, scores)), reverse=True)
    return max(options, key=lambda t: ((scores[labels == 1] >= t).mean() +
                                      (scores[labels == 0] < t).mean() - 1, t))


def evaluate(checkpoint, manifest, split, output_root, device="cuda", threshold_from=None,
             freeze_file=None, config_path=None, evidence_manifest_path=None):
    if split not in {"train", "val", "test"}:
        raise ValueError("SPLIT_INVALID")
    if split == "test" and threshold_from is None:
        raise ValueError("TEST_REQUIRES_VALIDATION_THRESHOLD_FILE")
    checkpoint, manifest, output_root = map(Path, (checkpoint, manifest, output_root))
    locked = None
    if split == "test" and freeze_file is None:
        raise ValueError("TEST_REQUIRES_FREEZE_LOCK")
    if split == "test" and evidence_manifest_path is None:
        raise ValueError("TEST_REQUIRES_EVIDENCE_MANIFEST")
    if freeze_file is not None:
        if evidence_manifest_path is None:
            raise ValueError("FREEZE_REQUIRES_EVIDENCE_MANIFEST")
        # Verify the bound protocol, artifacts, configs, and code before even
        # opening the supplied manifest or loading any slide/H5/WSI inputs.
        locked = verify_lock(freeze_file, evidence_manifest_path)
        if (manifest.resolve() != Path(locked["manifest_path"]).resolve()
                or _hash(manifest) != locked["manifest_sha256"]):
            raise ValueError("TEST_MANIFEST_LOCK_MISMATCH")
        if (split == "test" and config_path is None) or (
                config_path is not None and
                locked["config_sha256"].get(Path(config_path).name) != _hash(config_path)):
            raise ValueError("TEST_CONFIG_LOCK_MISMATCH")
    elif split == "test":
        raise ValueError("TEST_REQUIRES_FREEZE_LOCK")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"].strip() == split]
    if not rows or len({r["slide_id"] for r in rows}) != len(rows):
        raise ValueError("MANIFEST_EMPTY_OR_DUPLICATE")
    dataset = CamelyonDataset.from_manifest(manifest, split)
    if dataset.audit or len(dataset) != len(rows):
        raise ValueError(f"DATASET_INCOMPLETE {dataset.audit}")
    row_by_id = {r["slide_id"]: r for r in rows}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = EvidenceModelConfig(**payload["model_config"])
    model = EvidenceModel(config)
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()
    spatial = config.variant != "pooling"
    output_root.mkdir(parents=True, exist_ok=True)
    token_dir = output_root / "tokens"
    if config.variant in {"query", "graph_query"}:
        token_dir.mkdir(exist_ok=True)
    patch_rows, slide_rows, lesion_rows, froc_slides = [], [], [], []
    probabilities, labels, per_slide_patch = [], [], []
    repair_review = locked.get("repair_review") if locked else None
    for slide in dataset:
        row = row_by_id[slide.slide_id]
        coords = slide.coords_level0.numpy()
        size = slide.patch_size_level0
        wsi_path = Path(row["wsi_path"])
        if not wsi_path.is_file():
            raise ValueError(f"WSI_MISSING {slide.slide_id}")
        import openslide
        with openslide.OpenSlide(str(wsi_path)) as wsi:
            dimensions = wsi.dimensions
        validate_grid(coords, size, dimensions)
        # The model receives neither XML nor label tensors during inference.
        inference = replace(slide, features=slide.features.to(device),
                            coords_level0=slide.coords_level0.to(device),
                            tumor_fraction=torch.zeros_like(slide.tumor_fraction),
                            patch_labels=torch.zeros_like(slide.patch_labels),
                            annotation_mask=torch.zeros_like(slide.annotation_mask),
                            lesion_ids=torch.full_like(slide.lesion_ids, -1))
        with torch.no_grad():
            output = model(inference)
        if config.variant in {"query", "graph_query"}:
            torch.save({"slide_id": slide.slide_id,
                        "region_tokens": output.region_tokens.float().cpu(),
                        "query_attention": output.query_attention.float().cpu(),
                        "evidence_scores": output.evidence_scores.float().cpu(),
                        "coords_level0": slide.coords_level0},
                       token_dir / f"{slide.slide_id}.pt")
        scores = (output.raw_patch_logits if config.variant == "linear_patch"
                  else output.evidence_scores).float().cpu().numpy()
        probability = float(torch.softmax(output.slide_logits.float(), -1)[1].cpu())
        labels.append(slide.slide_label)
        probabilities.append(probability)
        truth = None
        if slide.slide_label == 1:
            if slide.annotation_status != "full" or not row.get("xml_path"):
                raise ValueError(f"POSITIVE_XML_INCOMPLETE {slide.slide_id}")
            truth = parse_truth(row["xml_path"])
            if truth.repairs:
                assert_repairs_approved(repair_review, slide.slide_id, row["split"],
                                        _hash(row["xml_path"]), truth.repairs)
        elif slide.annotation_status != "normal":
            raise ValueError(f"NORMAL_STATUS_INVALID {slide.slide_id}")
        lesions = truth.lesions if truth else ()
        if spatial:
            froc_slides.append({"slide_id": slide.slide_id, "coords": coords,
                                "size": size, "scores": scores, "lesions": lesions})
        patch_result = patch_metrics(slide.patch_labels.numpy(), scores,
                                     slide.annotation_mask.numpy()) if spatial else None
        if patch_result is not None:
            per_slide_patch.append(patch_result)
        result = {"slide_id": slide.slide_id, "slide_label": slide.slide_label,
                  "slide_probability": probability, "n": len(coords),
                  "patch_auroc": patch_result["auroc"] if patch_result else None,
                  "patch_auprc": patch_result["auprc"] if patch_result else None}
        if truth:
            result["xml_group_counts"] = truth.group_counts
            result["lesion_count"] = len(lesions)
            if spatial:
                overlaps = overlap_matrix(coords, size, lesions)
                fractions = overlaps.sum(axis=1) / (size * size)
                if not np.allclose(fractions, slide.tumor_fraction.numpy(), atol=1e-5):
                    raise ValueError(f"H5_XML_LABEL_MISMATCH {slide.slide_id}")
                if config.variant in {"query", "graph_query"}:
                    attention = output.query_attention.float().cpu().numpy()
                    purity = attention @ fractions
                    lesion_mass = attention @ (overlaps / (size * size))
                    norm = np.linalg.norm(attention, axis=1)
                    cosine = (attention @ attention.T) / np.maximum(norm[:, None] * norm[None, :], 1e-12)
                    off_diag = cosine[~np.eye(len(cosine), dtype=bool)]
                    result["query_diagnostics"] = {
                        "attention_mass_purity": float(purity.mean()),
                        "attention_lesion_coverage_0.1": float(np.mean(lesion_mass.max(axis=0) >= 0.1)),
                        "query_redundancy": float(off_diag.mean()) if len(off_diag) else 0.0,
                    }
                for budget in (0.01, 0.05, 0.10):
                    metrics = area_metrics(coords, size, scores, lesions, budget, dimensions, overlaps)
                    key = f"{budget:.2f}"
                    result[f"area_{key}"] = {k: v for k, v in metrics.items()
                                             if k not in {"per_lesion", "reachable_per_lesion", "selected_indices"}}
                    for j, (covered, reachable) in enumerate(zip(metrics["per_lesion"],
                                                                 metrics["reachable_per_lesion"])):
                        lesion_rows.append({"slide_id": slide.slide_id, "budget": budget,
                                            "lesion_id": j, "covered": covered,
                                            "reachable": reachable,
                                            "lesion_area": lesions[j].area})
        slide_rows.append(result)
        for j, (x, y) in enumerate(coords):
            patch_rows.append({"slide_id": slide.slide_id, "patch_index": j,
                               "x_level0": int(x), "y_level0": int(y), "patch_size_level0": size,
                               "evidence_score": float(scores[j]) if spatial else None,
                               "tumor_fraction": float(slide.tumor_fraction[j]),
                               "patch_label": int(slide.patch_labels[j])})
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("SLIDE_CLASS_MISSING")
    classification = {"auroc": float(roc_auc_score(labels, probabilities)),
                      "auprc": float(average_precision_score(labels, probabilities))}
    if split == "val":
        threshold = _select_threshold(labels, probabilities)
        classification["selected_threshold"] = threshold
    elif threshold_from:
        val = json.loads(Path(threshold_from).read_text(encoding="utf-8"))
        if val["split"] != "val" or val["checkpoint_sha256"] != _hash(checkpoint):
            raise ValueError("VALIDATION_THRESHOLD_CHECKPOINT_MISMATCH")
        threshold = val["classification"]["selected_threshold"]
    else:
        threshold = None
    if threshold is not None:
        classification["sensitivity"] = float((probabilities[labels == 1] >= threshold).mean())
        classification["specificity"] = float((probabilities[labels == 0] < threshold).mean())
        classification["threshold"] = threshold
    metrics = {"split": split, "checkpoint_sha256": _hash(checkpoint),
               "manifest_sha256": _hash(manifest), "evaluator_sha256": _hash(__file__),
               "seed": payload.get("seed"), "model_variant": config.variant,
               "graph_mode": config.graph_mode, "lambda_loc": payload.get("lambda_loc"),
               "classification": classification, "slide_count": len(dataset)}
    if spatial:
        for name in ("auroc", "auprc"):
            valid = [r[name] for r in per_slide_patch if r[name] is not None]
            metrics[f"patch_{name}"] = {"macro": float(np.mean(valid)) if valid else None,
                                       "valid_slides": len(valid), "na_slides": len(dataset) - len(valid)}
        froc = peak_froc(froc_slides)
        metrics["peak_froc"] = {k: v for k, v in froc.items() if k != "detections"}
        positive = [r["area_0.05"]["mlac"] for r in slide_rows if r["slide_label"] == 1]
        metrics["mlac_0.05"] = float(np.mean(positive))
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(slide_rows).to_json(output_root / "slides.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame(patch_rows).to_parquet(output_root / "patch_predictions.parquet", index=False)
    if spatial:
        pd.DataFrame(lesion_rows).to_parquet(output_root / "lesions.parquet", index=False)
        pd.DataFrame(froc["detections"]).to_parquet(output_root / "peak_detections.parquet", index=False)
    (output_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold-from", type=Path)
    parser.add_argument("--freeze-file", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--evidence-manifest", dest="evidence_manifest_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
