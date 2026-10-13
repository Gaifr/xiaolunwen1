"""Numerical and train/validation scale acceptance artifact contracts."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from shapely.geometry import box


SYNTHETIC_CASES = {
    "zero_reachability": {"mlac": 0.0, "reachable_macro": 0.0},
    "full_reachability": {"k": 1, "mlac": 1.0, "reachable_macro": 1.0},
    "partial_reachability": {
        "actual_budget": 1.0,
        "mlac": 0.5,
        "reachable_macro": 0.5,
    },
    "exclusion_hole": {
        "lesion_count": 1,
        "net_tumor_area": 300.0,
        "hole_patch_overlap_area": 75.0,
        "hole_center_covered": False,
    },
    "overlapping_grid_rejection": {
        "rejected": True,
        "error_code": "GRID_OVERLAP_OR_DUPLICATE",
    },
    "budget_floor": {"k": 0, "selected_indices": [], "actual_budget": 0.0},
    "budget_ties": {"k": 1, "selected_indices": [0], "actual_budget": 0.05},
    "small_lesion": {"mlac": 1.0, "purity": 0.01, "tumor_area_covered": 1.0},
    "boundary_lesion": {
        "selected_indices": [1],
        "mlac": 1.0,
        "reachable_macro": 1.0,
    },
    "normal_empty_truth": {
        "empty_truth_slide_count": 1,
        "normal_fp_count": 1,
        "fp_per_slide": 0.5,
        "sensitivity": 1.0,
    },
    "repeated_peaks": {
        "peak_indices": [0, 2, 4],
        "reasons": ["tp", "duplicate", "duplicate"],
        "fp_per_slide": 2.0,
        "sensitivity": 1.0,
    },
    "single_class_patch_metric_na": {
        "auroc": None,
        "auprc": None,
        "valid_patches": 2,
    },
}


def _require_runner(module_name: str, function_name: str):
    """Turn a missing future acceptance API into a clear RED assertion."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"missing acceptance API module {module_name}: {error}")
    runner = getattr(module, function_name, None)
    assert callable(runner), f"missing acceptance API {module_name}.{function_name}"
    return module, runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_reported_values(case: dict, expected: dict) -> None:
    assert case["expected_values"] == expected
    actual = case["actual_values"]
    tolerance = case["tolerance"]
    assert isinstance(tolerance, (int, float)) and tolerance >= 0
    for name, value in expected.items():
        if isinstance(value, float):
            assert actual[name] == pytest.approx(value, abs=tolerance, rel=0)
        else:
            assert actual[name] == value


def test_synthetic_acceptance_writes_all_frozen_cases_and_expected_values(tmp_path):
    _, run_synthetic_acceptance = _require_runner(
        "scripts.accept_stage3_metrics", "run_synthetic_acceptance"
    )
    output = tmp_path / "metric_acceptance.json"

    report = run_synthetic_acceptance(output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["schema"] == "stage3-metric-acceptance-v1"
    assert report["status"] == "passed"
    assert report["case_count"] == len(SYNTHETIC_CASES)
    assert isinstance(report["cases"], list)
    assert len(report["cases"]) == report["case_count"]
    cases = {case["case_id"]: case for case in report["cases"]}
    assert len(cases) == len(report["cases"])
    assert set(cases) == set(SYNTHETIC_CASES)

    for case_id, expected in SYNTHETIC_CASES.items():
        case = cases[case_id]
        assert isinstance(case["input_summary"], dict)
        assert isinstance(case["expected_values"], dict)
        assert isinstance(case["actual_values"], dict)
        assert case["passed"] is True
        _assert_reported_values(case, expected)


def test_synthetic_acceptance_fails_when_a_public_metric_returns_wrong_value(
    tmp_path, monkeypatch
):
    """Prove expected-value checking catches a deliberately corrupted result."""
    from modules import stage3_metrics

    acceptance, run_synthetic_acceptance = _require_runner(
        "scripts.accept_stage3_metrics", "run_synthetic_acceptance"
    )
    real_area_metrics = stage3_metrics.area_metrics

    def wrong_area_metrics(*args, **kwargs):
        result = dict(real_area_metrics(*args, **kwargs))
        result["mlac"] = 123.456
        return result

    monkeypatch.setattr(stage3_metrics, "area_metrics", wrong_area_metrics)
    # Support either a module-qualified call or a directly imported public symbol.
    if hasattr(acceptance, "area_metrics"):
        monkeypatch.setattr(acceptance, "area_metrics", wrong_area_metrics)

    report = run_synthetic_acceptance(tmp_path / "corrupted_acceptance.json")

    full = next(case for case in report["cases"] if case["case_id"] == "full_reachability")
    assert report["status"] == "failed"
    assert full["passed"] is False
    assert full["expected_values"]["mlac"] == 1.0
    assert full["actual_values"]["mlac"] == 123.456


def _write_fake_h5(path: Path, *, tumor: bool) -> None:
    coords = np.asarray([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.int64)
    fractions = np.asarray([1.0, 0.0, 1.0, 0.0] if tumor else [0.0] * 4,
                          dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = 1
        handle.attrs["patch_size_level0"] = 10
        handle.create_dataset("features", data=np.zeros((4, 3), dtype=np.float32))
        handle.create_dataset("coords", data=coords)
        handle.create_dataset("tumor_fraction", data=fractions)
        handle.create_dataset("patch_labels", data=(fractions >= 0.5).astype(np.uint8))
        handle.create_dataset("annotation_mask", data=np.ones(4, dtype=np.uint8))
        handle.create_dataset("lesion_ids", data=np.asarray([0, -1, 1, -1], dtype=np.int64)
                              if tumor else np.full(4, -1, dtype=np.int64))


def _write_scale_inputs(root: Path) -> tuple[Path, Path, dict[str, int]]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    inventory_rows = []
    labels: dict[str, int] = {}
    for index in range(100):
        split = "train" if index < 60 else "val" if index < 80 else "test"
        slide_id = f"{split}_{index:03d}"
        offset = index if split == "train" else index - 60 if split == "val" else index - 80
        positive_limit = {"train": 36, "val": 12, "test": 12}[split]
        tumor = offset < positive_limit
        labels[slide_id] = int(tumor)

        # Test paths are intentionally poison: absent and unsafe to open.
        if split == "test":
            h5_path = root / "poison_test" / f"{slide_id}.h5"
            xml_path = root / "poison_test" / f"{slide_id}.xml" if tumor else ""
        else:
            h5_path = root / f"{slide_id}.h5"
            _write_fake_h5(h5_path, tumor=tumor)
            xml_path = root / f"{slide_id}.xml" if tumor else ""
            if tumor:
                Path(xml_path).write_text("fake XML; parser is patched in this test\n",
                                         encoding="utf-8")

        rows.append({
            "slide_id": slide_id,
            "split": split,
            "slide_label": int(tumor),
            "annotation_status": "full" if tumor else "normal",
            "xml_path": str(xml_path),
            "h5_path": str(h5_path),
        })
        inventory_rows.append({
            "slide_id": slide_id,
            "split": split,
            "h5_path": str(h5_path),
            "h5_sha256": _sha256(h5_path) if split != "test" else "poison-test-hash",
            "status": "passed",
            "patch_count": 4,
            "patch_size_level0": 10,
            "wsi_dimensions": [40, 10],
        })

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({
        "schema": "stage3-h5-inventory-v1",
        "status": "passed",
        "slide_count": 100,
        "passed_slide_count": 100,
        "failed_slide_count": 0,
        "slides": inventory_rows,
    }, sort_keys=True), encoding="utf-8")
    return manifest, inventory, labels


def _patch_scale_inputs(
    monkeypatch, acceptance, labels: dict[str, int]
) -> tuple[list[str], list[str]]:
    opened_h5: list[str] = []
    real_h5_file = h5py.File

    def guarded_h5_file(path, *args, **kwargs):
        path = Path(path)
        assert "poison_test" not in path.parts, f"test H5 was opened: {path}"
        opened_h5.append(path.name)
        return real_h5_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", guarded_h5_file)
    seen_xml: list[str] = []

    def fake_parse_truth(path):
        path = Path(path)
        assert "poison_test" not in path.parts, f"test XML was parsed: {path}"
        slide_id = path.stem
        assert labels[slide_id] == 1
        seen_xml.append(slide_id)
        return SimpleNamespace(
            lesions=(box(0, 0, 10, 10), box(20, 0, 30, 10)),
            group_counts={},
            repairs=(),
        )

    metrics_module = importlib.import_module("modules.stage3_metrics")
    monkeypatch.setattr(metrics_module, "parse_truth", fake_parse_truth)
    if hasattr(acceptance, "parse_truth"):
        monkeypatch.setattr(acceptance, "parse_truth", fake_parse_truth)
    return opened_h5, seen_xml


def test_scale_acceptance_is_train_val_only_and_has_stable_model_free_digest(
    tmp_path, monkeypatch
):
    acceptance, run_scale_acceptance = _require_runner(
        "scripts.accept_stage3_scale", "run_scale_acceptance"
    )
    manifest, inventory, labels = _write_scale_inputs(tmp_path / "inputs")
    opened_h5, seen_xml = _patch_scale_inputs(monkeypatch, acceptance, labels)

    # Make elapsed durations differ across repetitions. They must not enter the digest.
    clock = {"tick": 0}

    def changing_clock():
        clock["tick"] += 1
        return float(clock["tick"] ** 2)

    if hasattr(acceptance, "time"):
        monkeypatch.setattr(acceptance.time, "perf_counter", changing_clock)
    if hasattr(acceptance, "perf_counter"):
        monkeypatch.setattr(acceptance, "perf_counter", changing_clock)

    output = tmp_path / "scale_acceptance.json"
    report = run_scale_acceptance(manifest, inventory, output, repetitions=2)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["status"] == "passed"
    assert set(report["split_set"]) == {"train", "val"}
    assert report["split_slide_counts"] == {"train": 60, "val": 20}
    assert report["manifest_split_counts"] == {"train": 60, "val": 20, "test": 20}
    assert len(report["slides"]) == 80
    assert {slide["split"] for slide in report["slides"]} == {"train", "val"}
    assert all(slide["patch_count"] == 4 for slide in report["slides"])
    assert all(slide["lesion_count"] == (2 if labels[slide["slide_id"]] else 0)
               for slide in report["slides"])
    assert all(slide["peak_count"] == (2 if labels[slide["slide_id"]] else 1)
               for slide in report["slides"])
    assert report["peak_score_source"] == "xml_derived_h5_tumor_fraction"
    assert report["peak_interpretation"] == "data_pipeline_scale_only_not_model_performance"
    assert len(report["repetition_digests"]) == 2
    assert report["repetition_digests"][0] == report["repetition_digests"][1]
    assert report["deterministic_digest"] == report["repetition_digests"][0]
    assert isinstance(report["wall_time_seconds"], (int, float))
    assert report["wall_time_seconds"] > 0
    assert isinstance(report["process_peak_rss_bytes"], (int, float))
    assert report["process_peak_rss_bytes"] > 0
    assert report["input_hashes"] == {
        "manifest_sha256": _sha256(manifest),
        "inventory_sha256": _sha256(inventory),
    }
    eligible_ids = {slide_id for slide_id in labels if not slide_id.startswith("test_")}
    positive_eligible_ids = {slide_id for slide_id in eligible_ids if labels[slide_id]}
    assert set(Path(name).stem for name in opened_h5) == eligible_ids
    assert len(opened_h5) >= len(eligible_ids)
    assert set(seen_xml) == positive_eligible_ids
    assert not any(slide_id.startswith("test_") for slide_id in seen_xml)


def test_scale_acceptance_fails_if_repeated_canonical_outputs_differ(tmp_path, monkeypatch):
    from modules import stage3_metrics

    acceptance, run_scale_acceptance = _require_runner(
        "scripts.accept_stage3_scale", "run_scale_acceptance"
    )
    manifest, inventory, labels = _write_scale_inputs(tmp_path / "inputs")
    _patch_scale_inputs(monkeypatch, acceptance, labels)
    real_peaks = stage3_metrics.peaks
    calls = {"count": 0}

    def changing_peaks(*args, **kwargs):
        calls["count"] += 1
        result = list(real_peaks(*args, **kwargs))
        if calls["count"] > 80:
            result.append(999)  # Deliberately corrupt the second full repetition.
        return result

    monkeypatch.setattr(stage3_metrics, "peaks", changing_peaks)
    if hasattr(acceptance, "peaks"):
        monkeypatch.setattr(acceptance, "peaks", changing_peaks)

    report = run_scale_acceptance(
        manifest, inventory, tmp_path / "nondeterministic_scale.json", repetitions=2
    )

    assert report["status"] == "failed"
    assert len(report["repetition_digests"]) == 2
    assert report["repetition_digests"][0] != report["repetition_digests"][1]
