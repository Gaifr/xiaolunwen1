"""Check the frozen Stage-3 metric definitions on small, fixed geometries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_metrics import (  # noqa: E402
    area_metrics,
    parse_truth,
    patch_metrics,
    peak_froc,
    peaks,
    validate_grid,
)


SCHEMA = "stage3-metric-acceptance-v1"
TOLERANCE = 1e-10


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


def _record(case_id: str, input_summary: dict[str, Any], expected: dict[str, Any],
            actual: dict[str, Any]) -> dict[str, Any]:
    passed = True
    for name, target in expected.items():
        observed = actual.get(name)
        if isinstance(target, float):
            if not isinstance(observed, (int, float)) or not np.isclose(
                observed, target, atol=TOLERANCE, rtol=0
            ):
                passed = False
        elif observed != target:
            passed = False
    return {
        "case_id": case_id,
        "input_summary": input_summary,
        "expected_values": expected,
        "actual_values": actual,
        "tolerance": TOLERANCE,
        "passed": bool(passed),
    }


def _grid(count: int, size: int = 10) -> np.ndarray:
    return np.asarray([[size * index, 0] for index in range(count)], dtype=np.int64)


def _write_truth_xml(path: Path) -> None:
    path.write_text(
        """<ASAP_Annotations>
  <AnnotationGroups>
    <Group Name="tumor"/><Group Name="_2"/>
  </AnnotationGroups>
  <Annotations>
    <Annotation PartOfGroup="tumor"><Coordinates>
      <Coordinate Order="0" X="0" Y="0"/>
      <Coordinate Order="1" X="20" Y="0"/>
      <Coordinate Order="2" X="20" Y="20"/>
      <Coordinate Order="3" X="0" Y="20"/>
    </Coordinates></Annotation>
    <Annotation PartOfGroup="_2"><Coordinates>
      <Coordinate Order="0" X="5" Y="5"/>
      <Coordinate Order="1" X="15" Y="5"/>
      <Coordinate Order="2" X="15" Y="15"/>
      <Coordinate Order="3" X="5" Y="15"/>
    </Coordinates></Annotation>
  </Annotations>
</ASAP_Annotations>\n""",
        encoding="utf-8",
    )


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    coords = _grid(20)
    result = area_metrics(coords, 10, np.zeros(20), [box(200, 0, 210, 10)])
    cases.append(_record(
        "zero_reachability", {"patch_count": 20, "lesion": [200, 0, 210, 10]},
        {"mlac": 0.0, "reachable_macro": 0.0},
        {"mlac": result["mlac"], "reachable_macro": result["reachable_macro"]},
    ))

    scores = np.zeros(20)
    scores[0] = 1
    result = area_metrics(coords, 10, scores, [box(0, 0, 10, 10)])
    cases.append(_record(
        "full_reachability", {"patch_count": 20, "selected_patch": 0},
        {"k": 1, "mlac": 1.0, "reachable_macro": 1.0},
        {"k": result["k"], "mlac": result["mlac"],
         "reachable_macro": result["reachable_macro"]},
    ))

    partial_coords = np.asarray([[0, 0]], dtype=np.int64)
    result = area_metrics(partial_coords, 10, np.asarray([0.5]),
                          [box(0, 0, 20, 10)], budget=1.0)
    cases.append(_record(
        "partial_reachability", {"patch_count": 1, "budget": 1.0,
                                  "lesion": [0, 0, 20, 10]},
        {"actual_budget": 1.0, "mlac": 0.5, "reachable_macro": 0.5},
        {"actual_budget": result["actual_budget"], "mlac": result["mlac"],
         "reachable_macro": result["reachable_macro"]},
    ))

    with tempfile.TemporaryDirectory(prefix="stage3-metric-case-") as temp_dir:
        xml_path = Path(temp_dir) / "exclusion_hole.xml"
        _write_truth_xml(xml_path)
        truth = parse_truth(xml_path)
    hole_overlap = box(0, 0, 10, 10).intersection(truth.lesions[0]).area
    hole_center_covered = truth.lesions[0].covers(Point(7.5, 7.5))
    cases.append(_record(
        "exclusion_hole", {"tumor_bounds": [0, 0, 20, 20],
                            "exclusion_bounds": [5, 5, 15, 15],
                            "probe_patch": [0, 0, 10, 10]},
        {"lesion_count": 1, "net_tumor_area": 300.0,
         "hole_patch_overlap_area": 75.0, "hole_center_covered": False},
        {"lesion_count": len(truth.lesions), "net_tumor_area": float(sum(g.area for g in truth.lesions)),
         "hole_patch_overlap_area": float(hole_overlap),
         "hole_center_covered": bool(hole_center_covered)},
    ))

    overlap_rejected = False
    overlap_error = ""
    try:
        validate_grid(np.asarray([[0, 0], [5, 0]], dtype=np.int64), 10)
    except ValueError as error:
        overlap_rejected = True
        overlap_error = str(error)
    cases.append(_record(
        "overlapping_grid_rejection", {"coords": [[0, 0], [5, 0]], "patch_size": 10},
        {"rejected": True, "error_code": "GRID_OVERLAP_OR_DUPLICATE"},
        {"rejected": overlap_rejected, "error_code": overlap_error},
    ))

    floor_result = area_metrics(_grid(19), 10, np.zeros(19), [box(0, 0, 10, 10)])
    cases.append(_record(
        "budget_floor", {"patch_count": 19, "budget": 0.05},
        {"k": 0, "selected_indices": [], "actual_budget": 0.0},
        {"k": floor_result["k"], "selected_indices": floor_result["selected_indices"],
         "actual_budget": floor_result["actual_budget"]},
    ))

    tie_result = area_metrics(_grid(20), 10, np.zeros(20), [box(0, 0, 10, 10)])
    cases.append(_record(
        "budget_ties", {"patch_count": 20, "all_scores": 0.0},
        {"k": 1, "selected_indices": [0], "actual_budget": 0.05},
        {"k": tie_result["k"], "selected_indices": tie_result["selected_indices"],
         "actual_budget": tie_result["actual_budget"]},
    ))

    small_scores = np.zeros(20)
    small_scores[0] = 1
    small_result = area_metrics(_grid(20), 10, small_scores, [box(0, 0, 1, 1)])
    cases.append(_record(
        "small_lesion", {"patch_count": 20, "lesion": [0, 0, 1, 1]},
        {"mlac": 1.0, "purity": 0.01, "tumor_area_covered": 1.0},
        {"mlac": small_result["mlac"], "purity": small_result["purity"],
         "tumor_area_covered": small_result["tumor_area_covered"]},
    ))

    boundary_coords = _grid(20)
    boundary_scores = np.zeros(20)
    boundary_scores[1] = 1
    boundary_result = area_metrics(boundary_coords, 10, boundary_scores,
                                   [box(10, 2, 12, 8)])
    cases.append(_record(
        "boundary_lesion", {"patch_count": 20, "lesion": [10, 2, 12, 8]},
        {"selected_indices": [1], "mlac": 1.0, "reachable_macro": 1.0},
        {"selected_indices": boundary_result["selected_indices"],
         "mlac": boundary_result["mlac"],
         "reachable_macro": boundary_result["reachable_macro"]},
    ))

    positive = {"slide_id": "positive", "coords": np.asarray([[0, 0]], dtype=np.int64),
                "size": 10, "scores": np.asarray([1.0]), "lesions": [box(0, 0, 10, 10)]}
    normal = {"slide_id": "normal", "coords": np.asarray([[0, 0]], dtype=np.int64),
              "size": 10, "scores": np.asarray([0.0]), "lesions": ()}
    normal_result = peak_froc([positive, normal])
    normal_detections = [item for item in normal_result["detections"]
                         if item["slide_id"] == "normal"]
    cases.append(_record(
        "normal_empty_truth", {"slide_count": 2, "empty_truth_slide": "normal"},
        {"empty_truth_slide_count": 1, "normal_fp_count": 1,
         "fp_per_slide": 0.5, "sensitivity": 1.0},
        {"empty_truth_slide_count": sum(not slide["lesions"] for slide in (positive, normal)),
         "normal_fp_count": sum(item["reason"] == "outside" for item in normal_detections),
         "fp_per_slide": normal_result["curve"][-1]["fp_per_slide"],
         "sensitivity": normal_result["curve"][-1]["sensitivity"]},
    ))

    repeated_coords = np.asarray([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]],
                                 dtype=np.int64)
    repeated_scores = np.asarray([5, 1, 4, 1, 3], dtype=float)
    repeated_peaks = peaks(repeated_coords, 10, repeated_scores)
    repeated = peak_froc([{
        "slide_id": "repeated", "coords": repeated_coords, "size": 10,
        "scores": repeated_scores, "lesions": [box(0, 0, 50, 10)],
    }])
    repeated_detections = repeated["detections"]
    cases.append(_record(
        "repeated_peaks", {"patch_count": 5, "lesion": [0, 0, 50, 10]},
        {"peak_indices": [0, 2, 4], "reasons": ["tp", "duplicate", "duplicate"],
         "fp_per_slide": 2.0, "sensitivity": 1.0},
        {"peak_indices": [int(index) for index in repeated_peaks],
         "reasons": [item["reason"] for item in repeated_detections],
         "fp_per_slide": repeated["curve"][-1]["fp_per_slide"],
         "sensitivity": repeated["curve"][-1]["sensitivity"]},
    ))

    patch_result = patch_metrics(np.asarray([0, 0]), np.asarray([0.2, 0.8]),
                                 np.asarray([True, True]))
    cases.append(_record(
        "single_class_patch_metric_na", {"labels": [0, 0], "mask": [True, True]},
        {"auroc": None, "auprc": None, "valid_patches": 2},
        {"auroc": patch_result["auroc"], "auprc": patch_result["auprc"],
         "valid_patches": patch_result["valid_patches"]},
    ))

    return cases


def run_synthetic_acceptance(output_path: Path) -> dict[str, Any]:
    """Run twelve fixed numerical checks against the public metric functions."""
    try:
        cases = _build_cases()
    except Exception as error:
        # Keep an observable artifact when an unexpected implementation error
        # prevents construction of any or all cases.
        cases = [{
            "case_id": "runner_error", "input_summary": {}, "expected_values": {},
            "actual_values": {"error_type": type(error).__name__, "error": str(error)},
            "tolerance": TOLERANCE, "passed": False,
        }]
    report = {
        "schema": SCHEMA,
        "status": "passed" if cases and all(case["passed"] for case in cases) else "failed",
        "case_count": len(cases),
        "cases": cases,
    }
    _atomic_write_json(Path(output_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_synthetic_acceptance(args.output)
    print(json.dumps({"status": report["status"], "case_count": report["case_count"]}))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
