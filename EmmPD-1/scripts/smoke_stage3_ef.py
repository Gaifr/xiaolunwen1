"""Four-slide implementation smoke and prediction export for Stage-3 E/F.

The models in this script are freshly initialized. The output records verify
the data/model execution path; they are not trained-model performance results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.camelyon_dataset import CamelyonDataset
from modules.evidence_model import EvidenceModel, EvidenceModelConfig
from modules.evidence_objectives import location_bce


APPROVED_TRAIN_SLIDES = ("normal_001", "tumor_002", "tumor_011", "tumor_031")
VARIANTS = ("query", "linear_patch")
SCORE_SOURCES = {
    "slide_logits": "model_output",
    "slide_probabilities": "softmax(slide_logits)",
    "query_attention": "model_output",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_sha256(model: torch.nn.Module) -> str:
    """Hash initialized weights in a stable, device-independent format."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError(f"SMOKE_NON_TENSOR_MODEL_STATE {name}")
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptor = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "nbytes": len(raw),
        }
        digest.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def _config_payload(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        values = asdict(config)
    elif hasattr(config, "__dict__"):
        values = dict(vars(config))
    else:
        raise TypeError("SMOKE_MODEL_CONFIG_NOT_SERIALIZABLE")
    return {str(key): value for key, value in values.items()}


def _resolve_h5_path(manifest_path: Path, row: Mapping[str, str]) -> Path:
    raw_path = (row.get("h5_path") or "").strip()
    if not raw_path:
        raise ValueError(f"SMOKE_H5_PATH_MISSING {row.get('slide_id', '')}")
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _optional_metadata_int(value: Any, *, name: str, slide_id: str) -> int:
    """Parse optional integer metadata supplied as a number or CSV text."""
    if value is None:
        return 0
    raw_value = str(value).strip()
    if not raw_value:
        return 0
    try:
        parsed = Decimal(raw_value)
    except InvalidOperation as error:
        raise ValueError(f"SMOKE_INVALID_{name.upper()} {slide_id} value={value!r}") from error
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"SMOKE_INVALID_{name.upper()} {slide_id} value={value!r}")
    return int(parsed)


def _manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _validate_requested_slides(slide_ids: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(slide_id) for slide_id in slide_ids)
    if (
        len(requested) != len(APPROVED_TRAIN_SLIDES)
        or len(set(requested)) != len(requested)
        or set(requested) != set(APPROVED_TRAIN_SLIDES)
    ):
        raise ValueError(
            "SMOKE_SLIDE_ALLOWLIST "
            f"expected={list(APPROVED_TRAIN_SLIDES)} requested={list(requested)}"
        )
    # Stable ordering makes report and export inventories reproducible even if
    # the caller supplied the four approved IDs in a different order.
    return APPROVED_TRAIN_SLIDES


def _validate_manifest_selection(
    rows: Sequence[Mapping[str, str]], slide_ids: Sequence[str]
) -> dict[str, Mapping[str, str]]:
    selected: dict[str, Mapping[str, str]] = {}
    for slide_id in slide_ids:
        matches = [row for row in rows if (row.get("slide_id") or "").strip() == slide_id]
        if len(matches) != 1:
            raise ValueError(f"SMOKE_MANIFEST_SLIDE_COUNT {slide_id} count={len(matches)}")
        row = matches[0]
        if (row.get("split") or "").strip() != "train":
            raise ValueError(f"SMOKE_REQUIRES_TRAIN_SLIDE {slide_id}")
        selected[slide_id] = row
    return selected


def _tensor_numpy(value: torch.Tensor, name: str, slide_id: str) -> np.ndarray:
    if not torch.is_tensor(value):
        raise ValueError(f"SMOKE_TENSOR_REQUIRED {name} {slide_id}")
    return value.detach().cpu().contiguous().numpy()


def _validate_h5_alignment(
    slide: Any, row: Mapping[str, str], h5_path: Path
) -> tuple[int, str]:
    slide_id = str(slide.slide_id)
    if not h5_path.is_file():
        raise ValueError(f"SMOKE_H5_MISSING {slide_id} {h5_path}")
    features = slide.features
    coords = slide.coords_level0
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError(f"SMOKE_FEATURE_SHAPE {slide_id}")
    if not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"SMOKE_COORD_SHAPE {slide_id}")
    patch_count = int(features.shape[0])
    if patch_count < 1 or int(coords.shape[0]) != patch_count:
        raise ValueError(
            f"SMOKE_PATCH_COORD_ROW_COUNT {slide_id} "
            f"features={patch_count} coords={tuple(coords.shape)}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"SMOKE_NONFINITE_FEATURES {slide_id}")

    slide_features = _tensor_numpy(features, "features", slide_id).astype(np.float32, copy=False)
    slide_coords = _tensor_numpy(coords, "coords_level0", slide_id).astype(np.int64, copy=False)
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle or "coords" not in handle:
            raise ValueError(f"SMOKE_H5_DATASET_MISSING {slide_id}")
        h5_features = handle["features"]
        h5_coords = handle["coords"]
        if h5_features.ndim != 2 or h5_coords.ndim != 2 or h5_coords.shape[1] != 2:
            raise ValueError(f"SMOKE_H5_ARRAY_SHAPE {slide_id}")
        if int(h5_features.shape[0]) != patch_count or int(h5_coords.shape[0]) != patch_count:
            raise ValueError(
                f"SMOKE_H5_PATCH_COUNT {slide_id} slide={patch_count} "
                f"h5_features={h5_features.shape[0]} h5_coords={h5_coords.shape[0]}"
            )
        if int(h5_features.shape[1]) != int(features.shape[1]):
            raise ValueError(f"SMOKE_H5_FEATURE_DIM {slide_id}")
        # The in-memory dataset and the H5 source must preserve the same row
        # order, so every exported score row maps back to exactly one H5 coord.
        if not np.array_equal(np.asarray(h5_coords, dtype=np.int64), slide_coords):
            raise ValueError(f"SMOKE_COORD_ROW_MAPPING {slide_id}")
        if not np.array_equal(np.asarray(h5_features, dtype=np.float32), slide_features):
            raise ValueError(f"SMOKE_FEATURE_ROW_MAPPING {slide_id}")
        h5_patch_size = int(handle.attrs.get("patch_size_level0", 0))
        manifest_patch_size = _optional_metadata_int(
            row.get("patch_size_level0"), name="patch_size_metadata", slide_id=slide_id
        )
        expected_patch_size = h5_patch_size or manifest_patch_size
        if expected_patch_size < 1 or int(slide.patch_size_level0) != expected_patch_size:
            raise ValueError(
                f"SMOKE_PATCH_SIZE_METADATA {slide_id} "
                f"slide={slide.patch_size_level0} h5={h5_patch_size} manifest={manifest_patch_size}"
            )
    return patch_count, _sha256(h5_path)


def _require_finite(value: torch.Tensor, name: str, slide_id: str) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"SMOKE_NONFINITE_{name.upper()} {slide_id}")


def _gradient_l1(loss: torch.Tensor, model: torch.nn.Module, name: str, slide_id: str) -> float:
    if not torch.is_tensor(loss) or not loss.requires_grad:
        raise RuntimeError(f"SMOKE_NO_{name.upper()}_GRADIENT {slide_id}")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError(f"SMOKE_NO_{name.upper()}_GRADIENT {slide_id}")
    try:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    except RuntimeError as error:
        raise RuntimeError(f"SMOKE_NO_{name.upper()}_GRADIENT {slide_id}: {error}") from error
    values = [gradient.detach().abs().sum() for gradient in gradients if gradient is not None]
    if not values:
        raise RuntimeError(f"SMOKE_NO_{name.upper()}_GRADIENT {slide_id}")
    total = torch.stack(values).sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise RuntimeError(f"SMOKE_NO_{name.upper()}_GRADIENT {slide_id}")
    return float(total.detach().cpu())


def _validate_export(
    *,
    variant: str,
    slide_id: str,
    patch_count: int,
    slide_logits: torch.Tensor,
    slide_probabilities: torch.Tensor,
    patch_scores: torch.Tensor,
    query_attention: torch.Tensor,
) -> None:
    if slide_logits.shape != (2,) or slide_probabilities.shape != (2,):
        raise ValueError(f"SMOKE_CLASS_OUTPUT_SHAPE {variant} {slide_id}")
    if patch_scores.shape != (patch_count,):
        raise ValueError(
            f"SMOKE_PATCH_SCORE_ROW_COUNT {variant} {slide_id} "
            f"scores={tuple(patch_scores.shape)} patches={patch_count}"
        )
    if query_attention.ndim != 2 or query_attention.shape[1] != patch_count:
        raise ValueError(f"SMOKE_QUERY_ATTENTION_SHAPE {variant} {slide_id}")
    if variant == "query" and query_attention.shape[0] < 1:
        raise ValueError(f"SMOKE_QUERY_ATTENTION_SHAPE {variant} {slide_id}")
    if variant == "linear_patch" and query_attention.shape != (0, patch_count):
        raise ValueError(f"SMOKE_LINEAR_ATTENTION_SHAPE {variant} {slide_id}")
    for name, value in (
        ("slide_logits", slide_logits),
        ("slide_probabilities", slide_probabilities),
        ("patch_scores", patch_scores),
        ("query_attention", query_attention),
    ):
        _require_finite(value, name, slide_id)


def smoke(manifest, slide_ids, output, device="cuda") -> dict[str, Any]:
    """Run fresh E/F models on the approved four training slides and export them."""
    manifest_path = Path(manifest).resolve()
    requested_ids = _validate_requested_slides(slide_ids)
    rows = _manifest_rows(manifest_path)
    manifest_selection = _validate_manifest_selection(rows, requested_ids)
    manifest_hash = _sha256(manifest_path)

    dataset = CamelyonDataset.from_manifest(manifest_path, "train")
    if dataset.audit:
        raise ValueError(f"SMOKE_TRAIN_DATASET_AUDIT {dataset.audit}")
    selected_slides = [slide for slide in dataset if slide.slide_id in requested_ids]
    selected_counts = {slide_id: 0 for slide_id in requested_ids}
    for slide in selected_slides:
        selected_counts[slide.slide_id] += 1
    if any(count != 1 for count in selected_counts.values()):
        raise ValueError(f"SMOKE_TRAIN_SLIDE_COUNT {selected_counts}")
    slide_by_id = {slide.slide_id: slide for slide in selected_slides}

    h5_metadata: dict[str, tuple[int, str]] = {}
    for slide_id in requested_ids:
        slide = slide_by_id[slide_id]
        row = manifest_selection[slide_id]
        if int(row.get("slide_label", -1)) != int(slide.slide_label):
            raise ValueError(f"SMOKE_MANIFEST_LABEL_MISMATCH {slide_id}")
        if hasattr(slide, "split") and slide.split != "train":
            raise ValueError(f"SMOKE_DATASET_SLIDE_NOT_TRAIN {slide_id}")
        h5_path = _resolve_h5_path(manifest_path, row)
        h5_metadata[slide_id] = _validate_h5_alignment(slide, row, h5_path)

    torch.manual_seed(9233)
    prepared_exports: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for variant in VARIANTS:
        config = EvidenceModelConfig(variant=variant, graph_mode="none")
        model = EvidenceModel(config).to(device)
        model.train()
        model_hash = _model_sha256(model)
        config_values = _config_payload(config)
        config_hash = _canonical_sha256(config_values)

        for slide_id in requested_ids:
            slide = slide_by_id[slide_id]
            patch_count, h5_hash = h5_metadata[slide_id]
            inference = replace(
                slide,
                features=slide.features.to(device),
                coords_level0=slide.coords_level0.to(device),
            )
            model.zero_grad(set_to_none=True)
            result = model(inference)

            logits = result.slide_logits.detach().float()
            _require_finite(logits, "slide_logits", slide_id)
            if logits.shape != (2,):
                raise ValueError(f"SMOKE_CLASS_OUTPUT_SHAPE {variant} {slide_id}")
            probabilities = torch.softmax(logits, dim=-1)
            localization_scores = result.evidence_scores
            if not torch.is_tensor(localization_scores) or localization_scores.shape != (patch_count,):
                shape = tuple(localization_scores.shape) if torch.is_tensor(localization_scores) else None
                raise ValueError(
                    f"SMOKE_LOCATION_SCORE_ROW_COUNT {variant} {slide_id} "
                    f"scores={shape} patches={patch_count}"
                )
            _require_finite(localization_scores, "localization_scores", slide_id)

            label = torch.tensor([int(slide.slide_label)], dtype=torch.long, device=logits.device)
            loss_cls = F.cross_entropy(result.slide_logits.float()[None], label)
            target = slide.tumor_fraction.to(localization_scores.device)
            annotation_mask = slide.annotation_mask.to(localization_scores.device)
            loss_loc = location_bce(localization_scores, target, annotation_mask)
            _require_finite(loss_cls, "classification_loss", slide_id)
            _require_finite(loss_loc, "localization_loss", slide_id)
            classification_gradient = _gradient_l1(loss_cls, model, "classification", slide_id)
            location_gradient = _gradient_l1(loss_loc, model, "location", slide_id)
            (loss_cls + loss_loc).backward()
            model_gradient_values = [
                parameter.grad.detach().abs().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not model_gradient_values:
                raise RuntimeError(f"SMOKE_NO_MODEL_GRADIENT {variant} {slide_id}")
            model_gradient = torch.stack(model_gradient_values).sum()
            if not bool(torch.isfinite(model_gradient)) or float(model_gradient) <= 0.0:
                raise RuntimeError(f"SMOKE_NO_MODEL_GRADIENT {variant} {slide_id}")

            if variant == "linear_patch":
                if result.raw_patch_logits is None:
                    raise ValueError(f"SMOKE_RAW_PATCH_LOGITS_MISSING {slide_id}")
                patch_scores = result.raw_patch_logits.detach().float().cpu()
                patch_score_source = "raw_patch_logits"
            else:
                patch_scores = result.evidence_scores.detach().float().cpu()
                patch_score_source = "evidence_scores"
            query_attention = result.query_attention.detach().float().cpu()
            slide_logits = logits.detach().cpu()
            slide_probabilities = probabilities.detach().cpu()
            coords_level0 = slide.coords_level0.detach().cpu().long().contiguous()
            _validate_export(
                variant=variant,
                slide_id=slide_id,
                patch_count=patch_count,
                slide_logits=slide_logits,
                slide_probabilities=slide_probabilities,
                patch_scores=patch_scores,
                query_attention=query_attention,
            )
            if coords_level0.shape != (patch_count, 2):
                raise ValueError(f"SMOKE_EXPORT_COORD_ROW_COUNT {variant} {slide_id}")

            score_sources = {
                **SCORE_SOURCES,
                "patch_scores": patch_score_source,
            }
            filename = f"{variant}_{slide_id}.pt"
            record = {
                "slide_id": slide_id,
                "variant": variant,
                "slide_label": int(slide.slide_label),
                "patch_count": patch_count,
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_loc": float(loss_loc.detach().cpu()),
                "classification_gradient_l1": classification_gradient,
                "location_gradient_l1": location_gradient,
                "gradient_l1": float(model_gradient.detach().cpu()),
                "prediction_file": filename,
                "manifest_sha256": manifest_hash,
                "h5_sha256": h5_hash,
                "model_sha256": model_hash,
                "config_sha256": config_hash,
                "model_state_stage": "fresh_initialization",
                "model_config": config_values,
                "score_sources": score_sources,
            }
            export = {
                "slide_id": slide_id,
                "variant": variant,
                "slide_logits": slide_logits,
                "slide_probabilities": slide_probabilities,
                "coords_level0": coords_level0,
                "patch_scores": patch_scores,
                "query_attention": query_attention,
                "provenance": record,
            }
            prepared_exports.append((filename, export, record))

    expected_pairs = {
        (variant, slide_id) for variant in VARIANTS for slide_id in APPROVED_TRAIN_SLIDES
    }
    actual_pairs = {
        (record["variant"], record["slide_id"])
        for _, _, record in prepared_exports
    }
    if len(prepared_exports) != 8 or actual_pairs != expected_pairs:
        raise RuntimeError(f"SMOKE_EXPORT_INVENTORY {sorted(actual_pairs)}")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename, export, record in prepared_exports:
        destination = output_path / filename
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(export, temporary)
        os.replace(temporary, destination)
        records.append(record)

    report = {
        "schema": "stage3-ef-smoke-v2",
        "status": "passed",
        "slide_ids": list(APPROVED_TRAIN_SLIDES),
        "variant_count": len(VARIANTS),
        "records": records,
        "interpretation": (
            "Implementation smoke using freshly initialized models; not a trained-model "
            "performance result."
        ),
    }
    report_path = output_path / "smoke_report.json"
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage-3 E/F implementation smoke on four approved train slides."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--slide-id",
        action="append",
        dest="slide_ids",
        help="Optional repeated IDs; if supplied they must equal the fixed four-slide allowlist.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    slide_ids = args.slide_ids or APPROVED_TRAIN_SLIDES
    report = smoke(args.manifest, slide_ids, args.output, args.device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
