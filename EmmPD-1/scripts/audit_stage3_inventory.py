"""Auditable, read-only inventory checks for Stage-3 patch-feature H5 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

try:
    import openslide
except ImportError:  # Keep synthetic tests usable where OpenSlide is not installed.
    def _openslide_unavailable(_path: str):
        raise ImportError("OpenSlide Python bindings are required to inspect WSI dimensions")

    openslide = SimpleNamespace(OpenSlide=_openslide_unavailable)


SCHEMA = "stage3-h5-inventory-v1"
REQUIRED_DATASETS = (
    "features",
    "coords",
    "tumor_fraction",
    "patch_labels",
    "annotation_mask",
    "lesion_ids",
)
MANIFEST_FIELDS = ("slide_id", "split", "h5_path", "wsi_path")
FEATURE_CHUNK_ROWS = 2048

CHECK_NAMES = (
    "manifest_fields",
    "slide_id_unique",
    "split_valid",
    "h5_path_present",
    "wsi_path_present",
    "h5_file_readable",
    "wsi_readable",
    "required_datasets_present",
    "feature_shape_valid",
    "feature_dim_positive",
    "feature_dim_consistent",
    "dataset_rows_aligned",
    "features_finite",
    "labels_finite",
    "patch_size_level0_positive",
    "coords_shape_valid",
    "coords_integer",
    "coords_unique",
    "coords_nonnegative",
    "coords_patch_aligned",
    "patch_squares_nonoverlapping",
    "coords_within_wsi",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_sha256(coords: np.ndarray) -> str:
    canonical = np.asarray(coords, dtype="<i8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _failure(row: dict[str, Any], code: str, message: str) -> None:
    if code not in row["failures"]:
        row["failures"].append(code)
    row["error_messages"].append({"code": code, "message": message})


def _mark(row: dict[str, Any], check: str, passed: bool) -> None:
    row["checks"][check] = bool(passed)


def _resolve_path(raw_value: Any, manifest_dir: Path) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


def _base_slide(row: dict[str, Any], manifest_dir: Path, index: int) -> dict[str, Any]:
    slide_id = str(row.get("slide_id") or "").strip()
    h5_path = _resolve_path(row.get("h5_path"), manifest_dir)
    wsi_path = _resolve_path(row.get("wsi_path"), manifest_dir)
    return {
        "manifest_row": index,
        "slide_id": slide_id or None,
        "split": str(row.get("split") or "").strip() or None,
        "h5_path": str(h5_path) if h5_path else None,
        "wsi_path": str(wsi_path) if wsi_path else None,
        "status": "failed",
        "h5_sha256": None,
        "h5_size_bytes": None,
        "dataset_shapes": {name: None for name in REQUIRED_DATASETS},
        "patch_count": None,
        "feature_dim": None,
        "patch_size_level0": None,
        "wsi_dimensions": None,
        "coordinate_sha256": None,
        "coordinate_bounds": None,
        "overlap_check": None,
        "checks": {name: None for name in CHECK_NAMES},
        "failures": [],
        "error_messages": [],
    }


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _has_patch_overlap(coords: np.ndarray, patch_size: int) -> tuple[bool, int]:
    """Check equal-sized integer squares with a spatial hash, without O(N^2) scans."""
    buckets: dict[tuple[int, int], list[int]] = {}
    size = int(patch_size)
    candidate_pair_checks = 0
    for index, (raw_x, raw_y) in enumerate(coords):
        x, y = int(raw_x), int(raw_y)
        x_end, y_end = x + size, y + size
        first_cell_x, last_cell_x = x // size, (x_end - 1) // size
        first_cell_y, last_cell_y = y // size, (y_end - 1) // size
        for cell_x in range(first_cell_x, last_cell_x + 1):
            for cell_y in range(first_cell_y, last_cell_y + 1):
                key = (cell_x, cell_y)
                occupants = buckets.setdefault(key, [])
                for previous_index in occupants:
                    candidate_pair_checks += 1
                    px, py = (int(v) for v in coords[previous_index])
                    if _intervals_overlap((x, x_end), (px, px + size)) and _intervals_overlap(
                        (y, y_end), (py, py + size)
                    ):
                        return True, candidate_pair_checks
                occupants.append(index)
    return False, candidate_pair_checks


def _feature_finite(dataset: h5py.Dataset) -> tuple[bool, str | None]:
    if not np.issubdtype(dataset.dtype, np.number):
        return False, "FEATURE_DTYPE_INVALID"
    rows = int(dataset.shape[0])
    for start in range(0, rows, FEATURE_CHUNK_ROWS):
        values = np.asarray(dataset[start : min(rows, start + FEATURE_CHUNK_ROWS)])
        if not np.isfinite(values).all():
            return False, "FEATURE_NONFINITE"
    return True, None


def _labels_finite(handle: h5py.File, row: dict[str, Any]) -> bool:
    labels_finite = True
    for name in ("tumor_fraction", "patch_labels", "annotation_mask", "lesion_ids"):
        if name not in handle:
            continue
        dataset = handle[name]
        try:
            finite = bool(np.isfinite(np.asarray(dataset[...])).all())
        except (TypeError, ValueError):
            finite = False
        if not finite:
            labels_finite = False
            _failure(row, f"LABEL_NONFINITE:{name}", f"Dataset {name} contains non-finite or non-numeric values")
    return labels_finite


def _inspect_slide(
    row: dict[str, Any],
    item: dict[str, Any],
    duplicate_ids: set[str],
) -> None:
    required_present = all(str(row.get(field) or "").strip() for field in MANIFEST_FIELDS)
    _mark(item, "manifest_fields", required_present)
    if not required_present:
        missing = [field for field in MANIFEST_FIELDS if not str(row.get(field) or "").strip()]
        _failure(item, "MANIFEST_FIELD_MISSING", f"Missing manifest values: {', '.join(missing)}")

    slide_id = item["slide_id"]
    unique = bool(slide_id) and slide_id not in duplicate_ids
    _mark(item, "slide_id_unique", unique)
    if not unique:
        _failure(item, "MANIFEST_SLIDE_ID_INVALID_OR_DUPLICATE", f"Missing or duplicate slide_id: {slide_id!r}")

    split_valid = item["split"] in {"train", "val", "test"}
    _mark(item, "split_valid", split_valid)
    if not split_valid:
        _failure(item, "SPLIT_INVALID", f"Expected train, val, or test; found {item['split']!r}")

    h5_path = Path(item["h5_path"]) if item["h5_path"] else None
    wsi_path = Path(item["wsi_path"]) if item["wsi_path"] else None
    _mark(item, "h5_path_present", h5_path is not None)
    if h5_path is None:
        _failure(item, "H5_PATH_MISSING", "Manifest h5_path is empty")
    _mark(item, "wsi_path_present", wsi_path is not None)
    if wsi_path is None:
        _failure(item, "WSI_PATH_MISSING", "Manifest wsi_path is empty")

    if h5_path is not None:
        try:
            item["h5_sha256"] = _sha256(h5_path)
            item["h5_size_bytes"] = h5_path.stat().st_size
        except OSError as error:
            _failure(item, "H5_FILE_UNREADABLE", f"Could not hash H5 file: {error}")
        else:
            _mark(item, "h5_file_readable", True)
    else:
        _mark(item, "h5_file_readable", False)

    if wsi_path is not None:
        try:
            with openslide.OpenSlide(str(wsi_path)) as wsi:
                dimensions = tuple(int(value) for value in wsi.dimensions)
            if len(dimensions) != 2 or min(dimensions) <= 0:
                raise ValueError(f"Invalid WSI dimensions: {dimensions!r}")
            item["wsi_dimensions"] = list(dimensions)
            _mark(item, "wsi_readable", True)
        except Exception as error:
            _mark(item, "wsi_readable", False)
            _failure(item, "WSI_OPEN_FAILED", f"Could not read WSI dimensions: {type(error).__name__}: {error}")
    else:
        _mark(item, "wsi_readable", False)

    if h5_path is None or item["h5_sha256"] is None:
        _mark(item, "h5_file_readable", False)
        return

    try:
        with h5py.File(h5_path, "r") as handle:
            present = [name in handle for name in REQUIRED_DATASETS]
            _mark(item, "required_datasets_present", all(present))
            for name, exists in zip(REQUIRED_DATASETS, present):
                if exists:
                    item["dataset_shapes"][name] = [int(value) for value in handle[name].shape]
                else:
                    _failure(item, f"DATASET_MISSING:{name}", f"Required H5 dataset {name!r} is missing")

            if not present[0]:
                _mark(item, "feature_shape_valid", False)
                _mark(item, "feature_dim_positive", False)
                _mark(item, "dataset_rows_aligned", False)
                _mark(item, "features_finite", False)
            else:
                features = handle["features"]
                shape_valid = features.ndim == 2
                _mark(item, "feature_shape_valid", shape_valid)
                if not shape_valid:
                    _failure(item, "FEATURE_SHAPE_INVALID", f"Expected features with rank 2; found shape {features.shape}")
                else:
                    item["patch_count"] = int(features.shape[0])
                    item["feature_dim"] = int(features.shape[1])
                    if features.shape[0] == 0:
                        _failure(item, "PATCH_COUNT_INVALID", "Feature dataset has zero patch rows")
                    positive_dim = int(features.shape[1]) > 0
                    _mark(item, "feature_dim_positive", positive_dim)
                    if not positive_dim:
                        _failure(item, "FEATURE_DIM_INVALID", "Feature dimension must be positive")
                    if positive_dim:
                        try:
                            finite, code = _feature_finite(features)
                        except (TypeError, ValueError, OSError) as error:
                            finite, code = False, "FEATURE_DTYPE_INVALID"
                            _failure(item, code, f"Could not validate feature values: {error}")
                        _mark(item, "features_finite", finite)
                        if code and not any(value == code for value in item["failures"]):
                            _failure(item, code, "Feature values must be finite and numeric")

                if shape_valid:
                    patch_count = int(features.shape[0])
                    aligned = True
                    for name, exists in zip(REQUIRED_DATASETS[1:], present[1:]):
                        if not exists:
                            aligned = False
                            continue
                        dataset = handle[name]
                        expected_shape = (patch_count, 2) if name == "coords" else (patch_count,)
                        if tuple(int(value) for value in dataset.shape) != expected_shape:
                            aligned = False
                            _failure(item, f"DATASET_SHAPE_INVALID:{name}", f"Dataset {name} must have shape {expected_shape}; found {dataset.shape}")
                    _mark(item, "dataset_rows_aligned", aligned)
                    labels_finite = _labels_finite(handle, item)
                    _mark(item, "labels_finite", labels_finite)
                else:
                    _mark(item, "dataset_rows_aligned", False)
                    _mark(item, "labels_finite", False)

            patch_size_raw = handle.attrs.get("patch_size_level0")
            valid_patch_size = False
            if patch_size_raw is not None:
                try:
                    patch_size_number = float(patch_size_raw)
                    valid_patch_size = math.isfinite(patch_size_number) and patch_size_number > 0 and patch_size_number.is_integer()
                except (TypeError, ValueError, OverflowError):
                    valid_patch_size = False
            if valid_patch_size:
                item["patch_size_level0"] = int(patch_size_number)
            _mark(item, "patch_size_level0_positive", valid_patch_size)
            if not valid_patch_size:
                _failure(item, "PATCH_SIZE_LEVEL0_INVALID", f"Expected a positive integer patch_size_level0 attribute; found {patch_size_raw!r}")

            coords = None
            if not present[1]:
                for check in ("coords_shape_valid", "coords_integer", "coords_unique", "coords_nonnegative", "coords_patch_aligned", "patch_squares_nonoverlapping", "coords_within_wsi"):
                    _mark(item, check, False)
            else:
                coordinate_data = handle["coords"]
                coords_shape_valid = coordinate_data.ndim == 2 and coordinate_data.shape[1] == 2
                _mark(item, "coords_shape_valid", coords_shape_valid)
                if not coords_shape_valid:
                    _failure(item, "COORD_SHAPE_INVALID", f"Expected coords shape (N, 2); found {coordinate_data.shape}")
                    for check in ("coords_integer", "coords_unique", "coords_nonnegative", "coords_patch_aligned", "patch_squares_nonoverlapping", "coords_within_wsi"):
                        _mark(item, check, False)
                else:
                    raw_coords = np.asarray(coordinate_data[...])
                    integer_coords = np.issubdtype(raw_coords.dtype, np.integer) and not np.issubdtype(raw_coords.dtype, np.bool_)
                    _mark(item, "coords_integer", integer_coords)
                    if not integer_coords:
                        _failure(item, "COORD_NOT_INTEGER", f"Coordinate dtype must be an integer type; found {raw_coords.dtype}")
                    elif raw_coords.size and np.issubdtype(raw_coords.dtype, np.unsignedinteger) and int(raw_coords.max()) > np.iinfo(np.int64).max:
                        integer_coords = False
                        _mark(item, "coords_integer", False)
                        _failure(item, "COORD_NOT_INTEGER", "Coordinate value exceeds the supported signed integer range")

                    if integer_coords:
                        coords = raw_coords.astype(np.int64, copy=False)
                        unique = len(np.unique(coords, axis=0)) == len(coords)
                        _mark(item, "coords_unique", unique)
                        if not unique:
                            _failure(item, "COORD_DUPLICATE", "Coordinate origins are not unique")
                        nonnegative = bool(np.all(coords >= 0))
                        _mark(item, "coords_nonnegative", nonnegative)
                        if not nonnegative:
                            _failure(item, "COORD_NEGATIVE", "Coordinates must be nonnegative")
                        if len(coords):
                            x_values, y_values = coords[:, 0], coords[:, 1]
                            item["coordinate_sha256"] = _coordinate_sha256(coords)
                            item["coordinate_bounds"] = {
                                "min_x": int(x_values.min()),
                                "min_y": int(y_values.min()),
                                "max_x": int(x_values.max()),
                                "max_y": int(y_values.max()),
                            }
                        elif item["patch_count"] == 0:
                            item["coordinate_sha256"] = _coordinate_sha256(coords)
                            item["coordinate_bounds"] = None

                        if valid_patch_size:
                            size = item["patch_size_level0"]
                            aligned = bool(np.all(coords % size == 0))
                            _mark(item, "coords_patch_aligned", aligned)
                            if not aligned:
                                _failure(item, "COORD_MISALIGNED", f"All coordinate origins must be multiples of {size}")
                            overlap_started = time.perf_counter()
                            overlap, candidate_pair_checks = _has_patch_overlap(coords, size)
                            overlap_seconds = time.perf_counter() - overlap_started
                            item["overlap_check"] = {
                                "algorithm": "equal-size spatial hash with exact rectangle checks",
                                "coordinate_count": int(len(coords)),
                                "candidate_pair_checks": int(candidate_pair_checks),
                                "seconds": overlap_seconds,
                            }
                            _mark(item, "patch_squares_nonoverlapping", not overlap)
                            if overlap:
                                _failure(item, "PATCH_OVERLAP", "Two patch squares overlap with positive area")
                            if item["wsi_dimensions"] is not None:
                                width, height = item["wsi_dimensions"]
                                in_bounds = bool(np.all(coords[:, 0] <= width - size) and np.all(coords[:, 1] <= height - size))
                                _mark(item, "coords_within_wsi", in_bounds)
                                if not in_bounds:
                                    _failure(item, "COORD_OUT_OF_BOUNDS", f"At least one patch square exceeds WSI bounds {width}x{height}")
                            else:
                                _mark(item, "coords_within_wsi", None)
                        else:
                            for check in ("coords_patch_aligned", "patch_squares_nonoverlapping", "coords_within_wsi"):
                                _mark(item, check, False)
                    else:
                        for check in ("coords_unique", "coords_nonnegative", "coords_patch_aligned", "patch_squares_nonoverlapping", "coords_within_wsi"):
                            _mark(item, check, False)
        _mark(item, "h5_file_readable", True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        _mark(item, "h5_file_readable", False)
        _failure(item, "H5_READ_FAILED", f"Could not inspect H5 file: {type(error).__name__}: {error}")

    item["status"] = "passed" if not item["failures"] else "failed"


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


def audit_inventory(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Audit each manifest slide and atomically write a complete JSON inventory."""
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    manifest_sha256 = None
    manifest_error = None
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    try:
        manifest_sha256 = _sha256(manifest_path)
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        manifest_error = {"code": "MANIFEST_READ_FAILED", "message": f"{type(error).__name__}: {error}"}

    missing_columns = [field for field in MANIFEST_FIELDS if field not in fieldnames]
    manifest_failures: list[dict[str, str]] = []
    if manifest_error:
        manifest_failures.append(manifest_error)
    if missing_columns:
        manifest_failures.append({
            "code": "MANIFEST_COLUMNS_MISSING",
            "message": f"Required columns are missing: {', '.join(missing_columns)}",
        })
    if not rows and not manifest_error:
        manifest_failures.append({
            "code": "MANIFEST_EMPTY",
            "message": "Manifest contains no slide rows",
        })

    id_counts: dict[str, int] = {}
    for row in rows:
        slide_id = str(row.get("slide_id") or "").strip()
        if slide_id:
            id_counts[slide_id] = id_counts.get(slide_id, 0) + 1
    duplicate_ids = {slide_id for slide_id, count in id_counts.items() if count > 1}

    slides: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows, start=2):
        item = _base_slide(source_row, manifest_path.parent, index)
        if missing_columns:
            for column in missing_columns:
                _failure(item, f"MANIFEST_COLUMN_MISSING:{column}", f"Manifest does not contain required column {column!r}")
        try:
            _inspect_slide(source_row, item, duplicate_ids)
        except Exception as error:
            _failure(item, "SLIDE_AUDIT_UNEXPECTED_ERROR", f"{type(error).__name__}: {error}")
            item["status"] = "failed"
        slides.append(item)

    observed_feature_dimensions = sorted({
        int(item["feature_dim"])
        for item in slides
        if isinstance(item["feature_dim"], int) and int(item["feature_dim"]) > 0
    })
    feature_dim_reference = (
        observed_feature_dimensions[0] if len(observed_feature_dimensions) == 1 else None
    )
    dimensions_are_consistent = len(observed_feature_dimensions) == 1
    for item in slides:
        dimension = item["feature_dim"]
        if dimension is None:
            continue
        consistent = dimensions_are_consistent and int(dimension) == feature_dim_reference
        _mark(item, "feature_dim_consistent", consistent)
        if not consistent:
            if dimensions_are_consistent:
                message = f"Expected {feature_dim_reference}, found {dimension}"
            else:
                message = (
                    "No unique positive feature dimension exists; observed "
                    f"{observed_feature_dimensions}"
                )
            _failure(item, "FEATURE_DIM_MISMATCH", message)
        item["status"] = "passed" if not item["failures"] else "failed"

    failed_count = sum(item["status"] != "passed" for item in slides)
    status = "failed" if failed_count or manifest_failures else "passed"
    report = {
        "schema": SCHEMA,
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_failures": manifest_failures,
        "observed_feature_dimensions": observed_feature_dimensions,
        "feature_dim_reference": feature_dim_reference,
        "coordinate_sha256_spec": "SHA-256 of coords canonicalized to signed little-endian int64, C-order bytes",
        "overlap_check_method": {
            "algorithm": "spatial hash over patch_size_level0-sized bins; exact rectangle checks only within shared bins",
            "expected_complexity": "O(N) time and O(N) auxiliary references for regular patch grids; pathological bucket crowding can increase comparisons",
            "limitation": "coordinates are held in memory for sorting, hashing, and bounds checks; feature values are scanned in 2048-row chunks",
        },
        "slide_count": len(slides),
        "passed_slide_count": len(slides) - failed_count,
        "failed_slide_count": failed_count,
        "slides": slides,
    }
    _atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_inventory(args.manifest, args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "slide_count", "passed_slide_count", "failed_slide_count", "manifest_failures"
    )}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
