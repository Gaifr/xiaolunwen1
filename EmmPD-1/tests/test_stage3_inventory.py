"""Synthetic acceptance tests for the Stage-3 H5 inventory audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest


class FakeWSI:
    """Provide real OpenSlide-shaped dimensions without opening a WSI file."""

    dimensions = (12, 12)

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _write_h5(path: Path, case: str = "valid") -> None:
    features = np.asarray(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
        dtype=np.float32,
    )
    coords = np.asarray([[0, 0], [4, 0], [0, 4]], dtype=np.int64)
    tumor_fraction = np.asarray([0.0, 0.75, 0.0], dtype=np.float32)
    patch_labels = np.asarray([0, 1, 0], dtype=np.uint8)
    annotation_mask = np.ones(3, dtype=np.bool_)
    lesion_ids = np.asarray([-1, 0, -1], dtype=np.int64)
    patch_size = 4
    omitted_dataset = None

    if case == "duplicate":
        coords[1] = coords[0]
    elif case == "misaligned":
        coords[1, 0] = 2
    elif case == "overlap":
        coords[1, 0] = 2
    elif case == "out_of_bounds":
        coords[1, 0] = 10
    elif case == "negative_coordinate":
        coords[1, 0] = -4
    elif case == "noninteger_coordinate":
        coords = coords.astype(np.float32)
        coords[1, 0] = 4.5
    elif case == "missing_dataset":
        omitted_dataset = "lesion_ids"
    elif case == "bad_shape":
        patch_labels = patch_labels.reshape(3, 1)
    elif case == "nonfinite_feature":
        features[1, 2] = np.nan
    elif case == "nonfinite_label":
        tumor_fraction[1] = np.inf
    elif case == "bad_feature_rank":
        features = features.reshape(-1)
    elif case == "zero_feature_dim":
        features = np.empty((3, 0), dtype=np.float32)
    elif case == "feature_dim_mismatch":
        features = np.concatenate(
            [features, np.full((3, 1), 12.0, dtype=np.float32)], axis=1
        )
    elif case == "zero_patch_size":
        patch_size = 0
    elif case != "valid":
        raise AssertionError(f"unknown fixture case: {case}")

    values = {
        "features": features,
        "coords": coords,
        "tumor_fraction": tumor_fraction,
        "patch_labels": patch_labels,
        "annotation_mask": annotation_mask,
        "lesion_ids": lesion_ids,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["patch_size_level0"] = patch_size
        for name, value in values.items():
            if name != omitted_dataset:
                handle.create_dataset(name, data=value)


def _write_manifest(root: Path, cases: list[tuple[str, str]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (slide_id, case) in enumerate(cases):
        h5_path = root / f"{slide_id}.h5"
        _write_h5(h5_path, case)
        rows.append(
            {
                "slide_id": slide_id,
                "split": ("train", "val", "test")[index % 3],
                "h5_path": str(h5_path),
                "wsi_path": str(root / f"{slide_id}.tif"),
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_fake_openslide(monkeypatch) -> None:
    from scripts import audit_stage3_inventory

    monkeypatch.setattr(audit_stage3_inventory.openslide, "OpenSlide", FakeWSI)


def test_valid_inventory_reports_hashes_geometry_and_dataset_shapes(tmp_path, monkeypatch):
    from scripts.audit_stage3_inventory import _coordinate_sha256, audit_inventory

    manifest = _write_manifest(tmp_path / "data", [("slide_ok", "valid")])
    _install_fake_openslide(monkeypatch)
    output = tmp_path / "inventory.json"

    report = audit_inventory(manifest, output)

    row = report["slides"][0]
    h5_path = Path(row["h5_path"])
    expected_coord_bytes = np.asarray(
        [[0, 0], [4, 0], [0, 4]], dtype="<i8", order="C"
    ).tobytes(order="C")
    assert report["schema"] == "stage3-h5-inventory-v1"
    assert report["status"] == "passed"
    assert report["slide_count"] == report["passed_slide_count"] == 1
    assert report["failed_slide_count"] == 0
    assert report["manifest_sha256"] == _sha256(manifest)
    assert row["h5_sha256"] == _sha256(h5_path)
    assert row["h5_size_bytes"] == h5_path.stat().st_size
    assert row["patch_count"] == 3
    assert row["feature_dim"] == 4
    assert row["dataset_shapes"] == {
        "features": [3, 4],
        "coords": [3, 2],
        "tumor_fraction": [3],
        "patch_labels": [3],
        "annotation_mask": [3],
        "lesion_ids": [3],
    }
    assert row["patch_size_level0"] == 4
    assert row["wsi_dimensions"] == [12, 12]
    assert row["coordinate_sha256"] == hashlib.sha256(expected_coord_bytes).hexdigest()
    big_endian_coords = np.asarray([[0, 0], [4, 0], [0, 4]], dtype=">i8")
    assert _coordinate_sha256(big_endian_coords) == hashlib.sha256(expected_coord_bytes).hexdigest()
    assert report["coordinate_sha256_spec"] == (
        "SHA-256 of coords canonicalized to signed little-endian int64, C-order bytes"
    )
    assert row["coordinate_bounds"] == {
        "min_x": 0,
        "min_y": 0,
        "max_x": 4,
        "max_y": 4,
    }
    assert all(row["checks"].values())
    assert row["failures"] == []
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("case", "failure_code"),
    [
        ("duplicate", "COORD_DUPLICATE"),
        ("misaligned", "COORD_MISALIGNED"),
        ("overlap", "PATCH_OVERLAP"),
        ("out_of_bounds", "COORD_OUT_OF_BOUNDS"),
        ("negative_coordinate", "COORD_NEGATIVE"),
        ("noninteger_coordinate", "COORD_NOT_INTEGER"),
        ("missing_dataset", "DATASET_MISSING:lesion_ids"),
        ("bad_shape", "DATASET_SHAPE_INVALID:patch_labels"),
        ("nonfinite_feature", "FEATURE_NONFINITE"),
        ("nonfinite_label", "LABEL_NONFINITE:tumor_fraction"),
        ("bad_feature_rank", "FEATURE_SHAPE_INVALID"),
        ("zero_feature_dim", "FEATURE_DIM_INVALID"),
        ("zero_patch_size", "PATCH_SIZE_LEVEL0_INVALID"),
    ],
)
def test_invalid_h5_fixture_emits_named_slide_diagnostic(
    tmp_path, monkeypatch, case, failure_code
):
    from scripts.audit_stage3_inventory import audit_inventory

    manifest = _write_manifest(tmp_path / "data", [("slide_bad", case)])
    _install_fake_openslide(monkeypatch)
    output = tmp_path / "inventory.json"

    report = audit_inventory(manifest, output)

    assert report["status"] == "failed"
    assert report["slide_count"] == 1
    assert report["passed_slide_count"] == 0
    assert report["failed_slide_count"] == 1
    assert report["slides"][0]["slide_id"] == "slide_bad"
    assert failure_code in report["slides"][0]["failures"]
    assert any(not passed for passed in report["slides"][0]["checks"].values())
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_failed_inventory_writes_complete_atomic_report_for_every_slide(
    tmp_path, monkeypatch
):
    from scripts import audit_stage3_inventory

    manifest = _write_manifest(
        tmp_path / "data", [("slide_ok", "valid"), ("slide_bad", "out_of_bounds")]
    )
    _install_fake_openslide(monkeypatch)
    output = tmp_path / "inventory.json"
    output.write_text('{"previous": true}', encoding="utf-8")
    real_replace = os.replace
    replacements = []

    def observe_atomic_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == output
        assert output.read_text(encoding="utf-8") == '{"previous": true}'
        staged = json.loads(source_path.read_text(encoding="utf-8"))
        assert staged["schema"] == "stage3-h5-inventory-v1"
        assert staged["status"] == "failed"
        assert staged["slide_count"] == 2
        assert [row["slide_id"] for row in staged["slides"]] == ["slide_ok", "slide_bad"]
        replacements.append((source_path, destination_path))
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(audit_stage3_inventory.os, "replace", observe_atomic_replace)

    report = audit_stage3_inventory.audit_inventory(manifest, output)

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == report
    assert report["status"] == "failed"
    assert report["slide_count"] == 2
    assert report["passed_slide_count"] == 1
    assert report["failed_slide_count"] == 1
    assert [row["slide_id"] for row in report["slides"]] == ["slide_ok", "slide_bad"]
    assert "COORD_OUT_OF_BOUNDS" in report["slides"][1]["failures"]
    assert len(replacements) == 1
    assert replacements[0][0].exists() is False


def test_manifest_with_inconsistent_feature_dimensions_fails_every_dimension(
    tmp_path, monkeypatch
):
    from scripts.audit_stage3_inventory import audit_inventory

    manifest = _write_manifest(
        tmp_path / "data",
        [("slide_dim4", "valid"), ("slide_dim5", "feature_dim_mismatch")],
    )
    _install_fake_openslide(monkeypatch)

    report = audit_inventory(manifest, tmp_path / "inventory.json")

    assert report["status"] == "failed"
    assert report["passed_slide_count"] == 0
    assert report["failed_slide_count"] == 2
    assert report["observed_feature_dimensions"] == [4, 5]
    assert report["feature_dim_reference"] is None
    assert report["slides"][0]["feature_dim"] == 4
    assert report["slides"][1]["feature_dim"] == 5
    for slide in report["slides"]:
        assert "FEATURE_DIM_MISMATCH" in slide["failures"]
        assert slide["checks"]["feature_dim_consistent"] is False


def test_feature_dimension_diagnostics_are_independent_of_manifest_order(
    tmp_path, monkeypatch
):
    from scripts.audit_stage3_inventory import audit_inventory

    cases = [("slide_dim4", "valid"), ("slide_dim5", "feature_dim_mismatch")]
    manifest_forward = _write_manifest(tmp_path / "forward", cases)
    manifest_reverse = _write_manifest(tmp_path / "reverse", list(reversed(cases)))
    _install_fake_openslide(monkeypatch)

    forward = audit_inventory(manifest_forward, tmp_path / "forward.json")
    reverse = audit_inventory(manifest_reverse, tmp_path / "reverse.json")

    def diagnostics(report):
        return {
            row["slide_id"]: (
                row["status"],
                sorted(row["failures"]),
                row["checks"]["feature_dim_consistent"],
            )
            for row in report["slides"]
        }

    assert diagnostics(forward) == diagnostics(reverse)
    assert forward["observed_feature_dimensions"] == reverse["observed_feature_dimensions"] == [4, 5]
    assert forward["feature_dim_reference"] is reverse["feature_dim_reference"] is None
    assert diagnostics(forward) == {
        "slide_dim4": ("failed", ["FEATURE_DIM_MISMATCH"], False),
        "slide_dim5": ("failed", ["FEATURE_DIM_MISMATCH"], False),
    }
