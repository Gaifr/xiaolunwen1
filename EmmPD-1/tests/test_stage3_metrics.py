"""Synthetic numerical acceptance for the frozen Stage-3 evaluator."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from modules.stage3_metrics import (area_metrics, overlap_matrix, paired_bootstrap,
                                    parse_truth, patch_metrics, peak_froc,
                                    peaks, validate_grid)


def _grid(n=20, size=10):
    return np.asarray([[i * size, 0] for i in range(n)], dtype=np.int64)


def test_zero_full_and_unreachable_lesions():
    coords = _grid()
    scores = np.arange(20, dtype=float)
    full = area_metrics(coords, 10, scores, (box(190, 0, 200, 10),))
    assert full["k"] == 1 and full["actual_budget"] == pytest.approx(0.05)
    assert full["mlac"] == full["reachable_macro"] == full["xml_oracle"] == 1
    assert full["random_expected"] == pytest.approx(0.05)
    missed = area_metrics(coords, 10, scores, (box(0, 0, 10, 10),))
    assert missed["mlac"] == 0 and missed["reachable_macro"] == 1
    unreachable = area_metrics(coords, 10, scores, (box(200, 0, 210, 10),))
    assert unreachable["mlac"] == unreachable["reachable_macro"] == 0


def test_partial_reachability_keeps_full_lesion_denominator():
    result = area_metrics(_grid(), 10, np.zeros(20), (box(0, 0, 20, 10),))
    assert result["reachable_macro"] == 1
    assert result["mlac"] == pytest.approx(0.5)
    partial = area_metrics(_grid(), 10, np.zeros(20), (box(0, 0, 10, 20),))
    assert partial["reachable_macro"] == partial["mlac"] == pytest.approx(0.5)


def test_budget_ties_and_zero_budget():
    tied = area_metrics(_grid(), 10, np.zeros(20), (box(0, 0, 10, 10),))
    assert tied["selected_indices"] == [0]
    assert area_metrics(_grid(19), 10, np.zeros(19), (box(0, 0, 10, 10),))["k"] == 0
    assert area_metrics(_grid(21), 10, np.zeros(21), (box(0, 0, 10, 10),))["actual_budget"] == pytest.approx(1 / 21)


def test_overlapping_grid_rejected_and_union_geometry_not_double_counted():
    with pytest.raises(ValueError, match="GRID_OVERLAP"):
        validate_grid(np.array([[0, 0], [0, 0]]), 10)
    with pytest.raises(ValueError, match="GRID_OVERLAP"):
        validate_grid(np.array([[0, 0], [5, 0]]), 10)
    assert box(0, 0, 10, 10).union(box(5, 0, 15, 10)).area == 150


def test_exclusion_hole_and_point_touch_component(tmp_path: Path):
    def ann(name, points):
        coordinates = "".join(f'<Coordinate Order="{i}" X="{x}" Y="{y}" />'
                              for i, (x, y) in enumerate(points))
        return f'<Annotation PartOfGroup="{name}"><Coordinates>{coordinates}</Coordinates></Annotation>'
    path = tmp_path / "truth.xml"
    path.write_text('<ASAP_Annotations><Annotations>' +
                    ann("Tumor", [(0, 0), (20, 0), (20, 20), (0, 20)]) +
                    ann("Exclusion", [(5, 5), (15, 5), (15, 15), (5, 15)]) +
                    '</Annotations><AnnotationGroups><Group Name="Tumor" />'
                    '<Group Name="Exclusion" /></AnnotationGroups></ASAP_Annotations>',
                    encoding="utf-8")
    truth = parse_truth(path)
    assert len(truth.lesions) == 1 and truth.lesions[0].area == 300
    assert overlap_matrix(np.asarray([[0, 0]]), 10, truth.lesions)[0, 0] == 75
    path.write_text('<ASAP_Annotations><Annotations>' +
                    ann("Tumor", [(0, 0), (10, 0), (10, 10), (0, 10)]) +
                    ann("Tumor", [(10, 10), (20, 10), (20, 20), (10, 20)]) +
                    '</Annotations><AnnotationGroups><Group Name="Tumor" />'
                    '</AnnotationGroups></ASAP_Annotations>', encoding="utf-8")
    assert len(parse_truth(path).lesions) == 1


def test_invalid_xml_polygon_has_reviewable_repair_record(tmp_path: Path):
    path = tmp_path / "self_intersection.xml"
    path.write_text(
        '<ASAP_Annotations><Annotations><Annotation PartOfGroup="Tumor">'
        '<Coordinates><Coordinate Order="0" X="0" Y="0" />'
        '<Coordinate Order="1" X="10" Y="10" />'
        '<Coordinate Order="2" X="0" Y="10" />'
        '<Coordinate Order="3" X="10" Y="0" />'
        '</Coordinates></Annotation></Annotations><AnnotationGroups>'
        '<Group Name="Tumor" /></AnnotationGroups></ASAP_Annotations>',
        encoding="utf-8",
    )
    truth = parse_truth(path)
    assert len(truth.repairs) == 1
    repair = truth.repairs[0]
    assert repair["annotation"] == 0 and repair["group"] == "Tumor"
    assert repair["reason"] and repair["vertex_count"] == 4
    assert repair["bounds_level0"] == [0.0, 0.0, 10.0, 10.0]
    assert repair["polygon_parts"] >= 1 and repair["after_area"] > 0


def test_audit_keeps_repair_inventory_after_other_slide_error(tmp_path: Path, monkeypatch):
    import csv
    from types import SimpleNamespace

    from scripts import audit_stage3

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("slide_id", "split", "slide_label", "xml_path"))
        writer.writeheader()
        for i in range(100):
            writer.writerow({"slide_id": f"slide_{i:03d}",
                             "split": "train" if i < 60 else "val" if i < 80 else "test",
                             "slide_label": int(i < 60), "xml_path": f"slide_{i:03d}.xml"})
    repair = {"annotation": 3, "reason": "Self-intersection"}
    monkeypatch.setattr(audit_stage3, "parse_truth", lambda path: SimpleNamespace(
        repairs=(repair,) if str(path).endswith("slide_000.xml") else ()))
    monkeypatch.setattr(audit_stage3, "_sha256", lambda path: "xml-hash")

    def failing_slide(row, truth):
        if row["slide_id"] == "slide_001":
            raise ValueError("H5_XML_LABEL_MISMATCH")
        return {"slide_id": row["slide_id"]}, None

    monkeypatch.setattr(audit_stage3, "_audit_slide", failing_slide)
    output = tmp_path / "audit.json"
    result = audit_stage3.audit(manifest, output)
    assert result["status"] == "failed" and result["slide_count"] == 99
    assert result["repair_annotation_count"] == 1
    assert result["repair_cases"][0]["xml_sha256"] == "xml-hash"
    assert result["issues"][0]["slide_id"] == "slide_001"
    assert output.exists()

    monkeypatch.setattr(audit_stage3, "_audit_slide",
                        lambda row, truth: ({"slide_id": row["slide_id"]}, None))
    monkeypatch.setattr(audit_stage3, "peak_froc", lambda slides: {"detections": []})
    assert audit_stage3.audit(manifest, output)["status"] == "needs_review"


def test_peak_plateau_and_duplicate_detection_with_normal_slide():
    coords = np.asarray([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]])
    assert peaks(coords, 10, np.ones(5)) == [2]
    slides = [
        {"slide_id": "tumor", "coords": coords, "size": 10,
         "scores": np.asarray([0.9, 0.1, 0.8, 0.1, 0.7]),
         "lesions": (box(0, 0, 50, 10),)},
        {"slide_id": "normal", "coords": np.asarray([[0, 0]]), "size": 10,
         "scores": np.asarray([0.95]), "lesions": ()},
    ]
    result = peak_froc(slides)
    assert [d["reason"] for d in result["detections"]] == ["outside", "tp", "duplicate", "duplicate"]
    assert result["curve"][-1]["fp_per_slide"] == pytest.approx(1.5)
    assert result["curve"][-1]["sensitivity"] == 1


def test_single_class_patch_metrics_are_na():
    result = patch_metrics([0, 0], [0.1, 0.2], [True, True])
    assert result["auroc"] is None and result["auprc"] is None


def test_paired_bootstrap_keeps_seed_slide_pairing():
    c = np.zeros((3, 12))
    d = np.ones((3, 12)) * np.asarray([[0.1], [0.2], [0.3]])
    result = paired_bootstrap(c, d)
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci95_percentile"] == pytest.approx([0.2, 0.2])
    assert result["single_seed"] == pytest.approx([0.1, 0.2, 0.3])
