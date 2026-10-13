"""Acceptance tests for auditable manual decisions on repaired XML polygons."""

import csv
import hashlib
import json

import pytest

from modules.stage3_review import (assert_repairs_approved, create_review_receipt,
                                   validate_review_receipt)
from scripts.freeze_stage3 import lock_payload, verify_lock


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, decision="accept_make_valid"):
    xml = tmp_path / "tumor_000.xml"
    xml.write_text("<annotations />", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("slide_id", "split", "xml_path"))
        writer.writeheader()
        writer.writerow({"slide_id": "tumor_000", "split": "train", "xml_path": str(xml)})

    audit = tmp_path / "audit.json"
    report = {"status": "needs_review", "issues": [], "repair_cases": [{
        "slide_id": "tumor_000", "split": "train", "xml_sha256": _sha256(xml),
        "repairs": [{"annotation": 4, "group": "Tumor", "reason": "Self-intersection",
                     "after_type": "MultiPolygon", "polygon_parts": 2,
                     "before_area": 30.0, "after_area": 25.0}],
    }]}
    audit.write_text(json.dumps(report), encoding="utf-8")

    review_dir = tmp_path / "review"
    review_dir.mkdir(exist_ok=True)
    render_manifest = review_dir / "render_manifest.json"
    render_manifest.write_text(json.dumps({
        "audit_sha256": _sha256(audit), "manifest_sha256": _sha256(manifest),
        "splits": ["train"], "rendered_annotations": 1,
    }), encoding="utf-8")
    queue = review_dir / "review_queue.csv"
    with queue.open("w", newline="", encoding="utf-8") as stream:
        fields = ("slide_id", "split", "xml_sha256", "annotation", "group", "reason",
                  "after_type", "polygon_parts", "before_area", "after_area", "image",
                  "review_decision", "review_note")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"slide_id": "tumor_000", "split": "train", "xml_sha256": _sha256(xml),
                         "annotation": 4, "group": "Tumor", "reason": "Self-intersection",
                         "after_type": "MultiPolygon", "polygon_parts": 2,
                         "before_area": 30.0, "after_area": 25.0,
                         "image": "tumor_000__ann_0004.png", "review_decision": decision,
                         "review_note": "repair follows the drawn contour"})
    return manifest, audit, queue, xml, report


def _receipt(tmp_path, manifest, audit, queue):
    output = tmp_path / "repair_review_receipt.json"
    create_review_receipt(manifest, audit, queue, "Xiaoxiao", output)
    return output


def test_receipt_binds_complete_review_inventory_and_current_xml(tmp_path):
    manifest, audit, queue, xml, report = _fixture(tmp_path)
    receipt_path = _receipt(tmp_path, manifest, audit, queue)
    receipt = validate_review_receipt(receipt_path, manifest, audit)

    assert receipt["reviewer"] == "Xiaoxiao"
    assert receipt["manifest_sha256"] == _sha256(manifest)
    assert receipt["audit_sha256"] == _sha256(audit)
    assert receipt["review_queue_sha256"] == _sha256(queue)
    assert len(receipt["approved_repairs"]) == sum(
        len(case["repairs"]) for case in report["repair_cases"])
    assert_repairs_approved(receipt, "tumor_000", "train", _sha256(xml),
                            report["repair_cases"][0]["repairs"])


def test_receipt_rejects_missing_or_unapproved_rows(tmp_path):
    manifest, audit, queue, _, _ = _fixture(tmp_path)

    with queue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("slide_id", "split", "xml_sha256", "annotation",
                                                    "group", "reason", "after_type", "polygon_parts",
                                                    "before_area", "after_area", "image",
                                                    "review_decision", "review_note"))
        writer.writeheader()
    with pytest.raises(ValueError, match="REPAIR_REVIEW_INVENTORY_MISMATCH"):
        create_review_receipt(manifest, audit, queue, "Xiaoxiao", tmp_path / "empty-receipt.json")

    manifest, audit, queue, _, _ = _fixture(tmp_path, decision="needs_source_review")

    with pytest.raises(ValueError, match="REPAIR_REVIEW_NOT_ACCEPTED"):
        create_review_receipt(manifest, audit, queue, "Xiaoxiao", tmp_path / "receipt.json")


def test_receipt_rejects_stale_audit_and_xml_hashes(tmp_path):
    manifest, audit, queue, xml, _ = _fixture(tmp_path)
    receipt_path = _receipt(tmp_path, manifest, audit, queue)

    with xml.open("a", encoding="utf-8") as stream:
        stream.write("<!-- changed -->")
    with pytest.raises(ValueError, match="REPAIR_REVIEW_XML_HASH_MISMATCH"):
        validate_review_receipt(receipt_path, manifest, audit)

    with xml.open("w", encoding="utf-8") as stream:
        stream.write("<annotations />")
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="REPAIR_REVIEW_AUDIT_MISMATCH"):
        validate_review_receipt(receipt_path, manifest, audit)


def test_freeze_and_evaluator_require_the_same_validated_receipt(tmp_path):
    manifest, audit, queue, xml, report = _fixture(tmp_path)
    receipt = _receipt(tmp_path, manifest, audit, queue)
    config = tmp_path / "config.yaml"
    config.write_text("seed: 9233\n", encoding="utf-8")

    with pytest.raises(ValueError, match="STAGE3_REPAIR_REVIEW_REQUIRED"):
        lock_payload(manifest, audit, [config])

    lock = lock_payload(manifest, audit, [config], repair_review_receipt=receipt)
    assert lock["repair_review_sha256"] == _sha256(receipt)
    lock_path = tmp_path / "freeze.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    verified = verify_lock(lock_path, manifest, repair_review_receipt=receipt)
    assert_repairs_approved(verified["repair_review"], "tumor_000", "train", _sha256(xml),
                            report["repair_cases"][0]["repairs"])
    with pytest.raises(ValueError, match="REPAIR_REVIEW_XML_MISMATCH"):
        assert_repairs_approved(verified["repair_review"], "tumor_000", "train", "0" * 64,
                                report["repair_cases"][0]["repairs"])

    with pytest.raises(ValueError, match="STAGE3_REPAIR_REVIEW_LOCK_MISMATCH"):
        verify_lock(lock_path, manifest, repair_review_receipt=tmp_path / "missing.json")
