"""Measure deterministic Stage-3 metric processing scale on train/val only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_metrics import (  # noqa: E402
    area_metrics,
    overlap_matrix,
    parse_truth,
    patch_metrics,
    peaks,
    validate_grid,
)


SCHEMA = "stage3-train-val-scale-acceptance-v1"
EXPECTED_MANIFEST_SPLIT_COUNTS = {"train": 60, "val": 20, "test": 20}
EVALUATED_SPLITS = ("train", "val")
PEAK_SCORE_SOURCE = "xml_derived_h5_tumor_fraction"
PEAK_INTERPRETATION = "data_pipeline_scale_only_not_model_performance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(raw_value: Any, manifest_dir: Path) -> Path:
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("MANIFEST_PATH_MISSING")
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _process_peak_rss_bytes() -> int:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        try:
            import psutil

            info = psutil.Process().memory_info()
            return int(getattr(info, "peak_wset", info.rss))
        except ImportError as error:
            raise RuntimeError("PROCESS_RSS_UNAVAILABLE") from error


def _load_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"slide_id", "split", "slide_label", "annotation_status", "xml_path", "h5_path"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"MANIFEST_FIELDS_MISSING {sorted(required - set(reader.fieldnames or ())) }")
        rows = list(reader)

    split_counts = {name: 0 for name in EXPECTED_MANIFEST_SPLIT_COUNTS}
    slide_ids: set[str] = set()
    eligible: list[dict[str, str]] = []
    for row in rows:
        split = str(row.get("split") or "").strip()
        slide_id = str(row.get("slide_id") or "").strip()
        if split not in split_counts:
            raise ValueError(f"MANIFEST_SPLIT_INVALID {split!r}")
        if not slide_id or slide_id in slide_ids:
            raise ValueError(f"MANIFEST_SLIDE_ID_INVALID_OR_DUPLICATE {slide_id!r}")
        slide_ids.add(slide_id)
        split_counts[split] += 1
        # Deliberately do not inspect labels or path columns in test rows.
        if split in EVALUATED_SPLITS:
            eligible.append(row)
    if split_counts != EXPECTED_MANIFEST_SPLIT_COUNTS:
        raise ValueError(f"MANIFEST_COHORT_CHANGED {split_counts}")
    if len(eligible) != 80:
        raise ValueError(f"TRAIN_VAL_SLIDE_COUNT_INVALID {len(eligible)}")
    return eligible, split_counts


def _load_inventory(inventory_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "stage3-h5-inventory-v1":
        raise ValueError("INVENTORY_SCHEMA_INVALID")
    if inventory.get("status") != "passed":
        raise ValueError("INVENTORY_STATUS_NOT_PASSED")
    records: dict[str, dict[str, Any]] = {}
    for item in inventory.get("slides", []):
        # Only inspect train/val inventory records. Test paths and other fields
        # remain opaque and are never resolved, opened, or hashed here.
        split = item.get("split")
        if split not in EVALUATED_SPLITS:
            continue
        slide_id = str(item.get("slide_id") or "")
        if not slide_id or slide_id in records:
            raise ValueError(f"INVENTORY_SLIDE_ID_INVALID_OR_DUPLICATE {slide_id!r}")
        records[slide_id] = item
    return records, inventory


def _inventory_gate(row: dict[str, str], inventory: dict[str, Any],
                    manifest_dir: Path) -> dict[str, Any]:
    slide_id = row["slide_id"]
    if slide_id not in inventory:
        raise ValueError(f"INVENTORY_SLIDE_MISSING {slide_id}")
    item = inventory[slide_id]
    if item.get("split") != row["split"]:
        raise ValueError(f"INVENTORY_SPLIT_MISMATCH {slide_id}")
    if item.get("status") != "passed":
        raise ValueError(f"INVENTORY_STATUS_FAILED {slide_id}")
    h5_path = _resolve_path(row.get("h5_path"), manifest_dir)
    if not h5_path.is_file():
        raise ValueError(f"H5_PATH_NOT_FILE {slide_id}")
    actual_sha = _sha256(h5_path)
    expected_sha = str(item.get("h5_sha256") or "").lower()
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(f"INVENTORY_H5_SHA256_MISMATCH {slide_id}")
    try:
        expected_patches = int(item["patch_count"])
        size = int(item["patch_size_level0"])
        dims_raw = item["wsi_dimensions"]
        dimensions = (int(dims_raw[0]), int(dims_raw[1]))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"INVENTORY_METADATA_INVALID {slide_id}") from error
    if expected_patches < 1 or size < 1 or min(dimensions) < 1:
        raise ValueError(f"INVENTORY_METADATA_NONPOSITIVE {slide_id}")
    return {
        "h5_path": h5_path,
        "h5_sha256": actual_sha,
        "patch_count": expected_patches,
        "patch_size_level0": size,
        "wsi_dimensions": dimensions,
    }


def _metric_summary(row: dict[str, str], verified_input: dict[str, Any],
                    manifest_dir: Path) -> dict[str, Any]:
    slide_id, split = row["slide_id"], row["split"]
    h5_path = verified_input["h5_path"]
    h5_sha = verified_input["h5_sha256"]
    inventory_patch_count = verified_input["patch_count"]
    size = verified_input["patch_size_level0"]
    dimensions = verified_input["wsi_dimensions"]
    label_text = str(row.get("slide_label") or "").strip()
    if label_text not in {"0", "1"}:
        raise ValueError(f"SLIDE_LABEL_INVALID {slide_id}")
    slide_label = int(label_text)
    annotation_status = str(row.get("annotation_status") or "").strip()

    with h5py.File(h5_path, "r") as handle:
        if int(handle.attrs.get("complete", 0)) != 1:
            raise ValueError(f"H5_INCOMPLETE {slide_id}")
        h5_size = int(handle.attrs.get("patch_size_level0", 0))
        if h5_size != size:
            raise ValueError(f"PATCH_SIZE_INVENTORY_MISMATCH {slide_id}")
        required = ("features", "coords", "tumor_fraction", "patch_labels",
                    "annotation_mask", "lesion_ids")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"H5_DATASETS_MISSING {slide_id} {missing}")
        features = handle["features"]
        coords_dataset = handle["coords"]
        if features.ndim != 2 or features.shape[1] < 1:
            raise ValueError(f"FEATURE_SHAPE_INVALID {slide_id}")
        patch_count = int(features.shape[0])
        feature_dim = int(features.shape[1])
        if patch_count != inventory_patch_count:
            raise ValueError(f"INVENTORY_PATCH_COUNT_MISMATCH {slide_id}")
        if coords_dataset.shape != (patch_count, 2):
            raise ValueError(f"COORD_SHAPE_INVALID {slide_id}")
        for name in ("tumor_fraction", "patch_labels", "annotation_mask", "lesion_ids"):
            if handle[name].shape != (patch_count,):
                raise ValueError(f"H5_LENGTH_MISMATCH {slide_id} {name}")
        coords = np.asarray(coords_dataset[...])
        validate_grid(coords, size, dimensions)
        fractions = np.asarray(handle["tumor_fraction"][...], dtype=np.float64)
        labels = np.asarray(handle["patch_labels"][...])
        annotation_mask = np.asarray(handle["annotation_mask"][...], dtype=bool)
        if not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1):
            raise ValueError(f"TUMOR_FRACTION_INVALID {slide_id}")
        if not np.array_equal(labels, fractions >= 0.5):
            raise ValueError(f"PATCH_LABEL_MISMATCH {slide_id}")
        if not np.all(annotation_mask):
            raise ValueError(f"ANNOTATION_MASK_NOT_FULL {slide_id}")

    truth = None
    xml_sha = None
    if slide_label == 1:
        if annotation_status != "full":
            raise ValueError(f"ANNOTATION_STATUS_INVALID {slide_id}")
        xml_path = _resolve_path(row.get("xml_path"), manifest_dir)
        if not xml_path.is_file():
            raise ValueError(f"XML_PATH_NOT_FILE {slide_id}")
        xml_sha = _sha256(xml_path)
        truth = parse_truth(xml_path)
        if _sha256(xml_path) != xml_sha:
            raise ValueError(f"XML_CHANGED_DURING_READ {slide_id}")
        lesions = truth.lesions
        overlaps = overlap_matrix(coords, size, lesions)
        geometric_fraction = overlaps.sum(axis=1) / float(size * size)
        if not np.allclose(geometric_fraction, fractions, atol=1e-5, rtol=0):
            raise ValueError(f"H5_XML_LABEL_MISMATCH {slide_id}")
        area = area_metrics(coords, size, fractions, lesions,
                            dimensions=dimensions, overlaps=overlaps)
    else:
        if annotation_status != "normal" or np.any(fractions) or np.any(labels):
            raise ValueError(f"NORMAL_LABEL_INVALID {slide_id}")
        lesions = ()
        area = None

    patch = patch_metrics(labels, fractions, annotation_mask)
    peak_indices = [int(index) for index in peaks(coords, size, fractions)]
    # Retain only stable metric values in the digest. The score vector is XML
    # derived and is used to exercise the data pipeline, never model inference.
    return {
        "slide_id": slide_id,
        "split": split,
        "h5_sha256": h5_sha,
        "xml_sha256": xml_sha,
        "patch_count": patch_count,
        "feature_dim": feature_dim,
        "patch_size_level0": size,
        "lesion_count": len(lesions),
        "truth_repair_count": len(truth.repairs) if truth is not None else 0,
        "peak_count": len(peak_indices),
        "peak_indices": peak_indices,
        "area_metrics": _canonical_metric_values(area),
        "patch_metrics": _canonical_metric_values(patch),
    }


def _canonical_metric_values(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        elif isinstance(value, tuple):
            output[key] = list(value)
        else:
            output[key] = value
    return output


def _canonical_digest(summaries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(summaries, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_scale_acceptance(manifest_path: Path, inventory_path: Path,
                         output_path: Path, repetitions: int = 2) -> dict[str, Any]:
    """Repeat metric processing on the 60/20 train/validation WSI cohort."""
    started = time.perf_counter()
    manifest_path, inventory_path, output_path = map(Path, (manifest_path, inventory_path, output_path))
    issues: list[dict[str, str]] = []
    summaries_by_run: list[list[dict[str, Any]]] = []
    manifest_counts = {key: 0 for key in EXPECTED_MANIFEST_SPLIT_COUNTS}
    eligible_rows: list[dict[str, str]] = []
    inventory_records: dict[str, dict[str, Any]] = {}
    verified_inputs: dict[str, dict[str, Any]] = {}
    manifest_hash: str | None = None
    inventory_hash: str | None = None

    try:
        manifest_hash = _sha256(manifest_path)
        inventory_hash = _sha256(inventory_path)
        eligible_rows, manifest_counts = _load_manifest(manifest_path)
        inventory_records, _ = _load_inventory(inventory_path)
        if set(inventory_records) != {row["slide_id"] for row in eligible_rows}:
            issues.append({"stage": "inventory", "error": "TRAIN_VAL_INVENTORY_ID_SET_MISMATCH"})
    except Exception as error:
        issues.append({"stage": "inputs", "error": f"{type(error).__name__}: {error}"})

    if repetitions < 2:
        issues.append({"stage": "configuration", "error": "REPETITIONS_MUST_BE_AT_LEAST_TWO"})
    elif not issues:
        rows = sorted(eligible_rows, key=lambda row: (row["split"], row["slide_id"]))
        for row in rows:
            slide_id = row["slide_id"]
            try:
                verified_inputs[slide_id] = _inventory_gate(
                    row, inventory_records[slide_id], manifest_path.parent
                )
            except Exception as error:
                issues.append({
                    "stage": "inventory", "slide_id": slide_id,
                    "error": f"{type(error).__name__}: {error}",
                })
        for repetition in range(repetitions):
            run_summaries: list[dict[str, Any]] = []
            for row in rows:
                slide_id = row["slide_id"]
                if slide_id not in verified_inputs:
                    continue
                try:
                    summary = _metric_summary(
                        row, verified_inputs[slide_id], manifest_path.parent
                    )
                    run_summaries.append(summary)
                except Exception as error:
                    issues.append({
                        "stage": "slide", "slide_id": slide_id,
                        "error": f"{type(error).__name__}: {error}",
                        "repetition": str(repetition + 1),
                    })
            summaries_by_run.append(run_summaries)

    digests = [_canonical_digest(summaries) for summaries in summaries_by_run]
    deterministic = len(digests) == repetitions and len(set(digests)) == 1
    if len(digests) == repetitions and not deterministic:
        issues.append({"stage": "determinism", "error": "REPEATED_OUTPUT_DIGEST_MISMATCH"})

    final_slides = summaries_by_run[0] if summaries_by_run else []
    split_slide_counts = {
        split: sum(row.get("split") == split for row in final_slides)
        for split in EVALUATED_SPLITS
    }
    totals = {
        "patch_count": sum(int(item["patch_count"]) for item in final_slides),
        "lesion_count": sum(int(item["lesion_count"]) for item in final_slides),
        "peak_count": sum(int(item["peak_count"]) for item in final_slides),
    }
    try:
        process_peak_rss = _process_peak_rss_bytes()
    except Exception as error:
        process_peak_rss = 0
        issues.append({"stage": "process_metrics", "error": f"{type(error).__name__}: {error}"})
    elapsed = max(0.0, float(time.perf_counter() - started))
    report = {
        "schema": SCHEMA,
        "status": "passed" if not issues
        and manifest_counts == EXPECTED_MANIFEST_SPLIT_COUNTS
        and split_slide_counts == {"train": 60, "val": 20}
        and len(final_slides) == 80 and deterministic else "failed",
        "split_set": list(EVALUATED_SPLITS),
        "split_slide_counts": split_slide_counts,
        "manifest_split_counts": manifest_counts,
        "slides": final_slides,
        "totals": totals,
        "peak_score_source": PEAK_SCORE_SOURCE,
        "peak_interpretation": PEAK_INTERPRETATION,
        "repetitions_requested": repetitions,
        "repetitions_completed": len(summaries_by_run),
        "repetition_digests": digests,
        "deterministic_digest": digests[0] if digests else None,
        "wall_time_seconds": elapsed,
        "process_peak_rss_bytes": int(process_peak_rss),
        "input_hashes": {
            "manifest_sha256": manifest_hash,
            "inventory_sha256": inventory_hash,
        },
        "issues": issues,
    }
    _atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    report = run_scale_acceptance(args.manifest, args.inventory, args.output, args.repetitions)
    print(json.dumps({key: report[key] for key in (
        "status", "split_slide_counts", "manifest_split_counts", "totals",
        "repetitions_completed", "deterministic_digest",
    )}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
