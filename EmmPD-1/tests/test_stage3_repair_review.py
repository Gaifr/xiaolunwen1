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


def _stages(tmp_path, queue, geometry_decision="geometry_ok"):
    with queue.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    identity = {field: row[field] for field in ("slide_id", "split", "xml_sha256", "annotation")}
    assistant = tmp_path / "assistant.json"
    assistant.write_text(json.dumps({
        "schema": "stage3-xml-geometry-review-v1", "reviewer": "assistant",
        "reviewed_at_utc": "2026-10-13T01:00:00Z", "queue_sha256": _sha256(queue),
        "results": [{**identity, "geometry_decision": geometry_decision,
                     "geometry_note": "contour checked"}]}))
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema": "stage3-xml-user-confirmation-v1", "reviewer": "Xiaoxiao",
        "confirmed_at_utc": "2026-10-13T02:00:00Z", "queue_sha256": _sha256(queue),
        "assistant_review_sha256": _sha256(assistant),
        "acknowledge_geometry_only_scope": True, "acknowledge_unflagged_review": True,
        "resolutions": [{**identity, "decision": "accept_make_valid", "review_note": "accepted"}]}))
    return assistant, confirmation


def _receipt(tmp_path, manifest, audit, queue):
    assistant, confirmation = _stages(tmp_path, queue)
    output = tmp_path / "repair_review_receipt.json"
    create_review_receipt(manifest, audit, queue, assistant, confirmation, output)
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


@pytest.mark.parametrize("stage, mutation", [
    ("assistant", "missing"), ("assistant", "duplicate"), ("assistant", "extra"),
    ("assistant", "stale"), ("assistant", "undecided"), ("assistant", "cannot_judge"),
    ("confirmation", "missing"), ("confirmation", "duplicate"), ("confirmation", "extra"),
    ("confirmation", "stale"), ("confirmation", "reject"), ("confirmation", "scope"),
    ("confirmation", "acknowledge"), ("assistant", "timestamp"), ("confirmation", "timestamp"),
])
def test_two_stage_receipt_blocks_incomplete_or_invalid_review(tmp_path, stage, mutation):
    manifest, audit, queue, _, _ = _fixture(tmp_path)
    assistant, confirmation = _stages(tmp_path, queue, "needs_user_confirmation")
    path = assistant if stage == "assistant" else confirmation
    data = json.loads(path.read_text())
    field = "results" if stage == "assistant" else "resolutions"
    if mutation == "missing":
        data[field] = []
    elif mutation == "duplicate":
        data[field] *= 2
    elif mutation == "extra":
        data[field].append({**data[field][0], "annotation": 99})
    elif mutation == "stale":
        data["queue_sha256"] = "0" * 64
    elif mutation in {"undecided", "cannot_judge"}:
        data[field][0]["geometry_decision"] = "" if mutation == "undecided" else mutation
    elif mutation == "reject":
        data[field][0]["decision"] = "reject"
    elif mutation == "scope":
        data["acknowledge_geometry_only_scope"] = False
    elif mutation == "acknowledge":
        data["acknowledge_unflagged_review"] = False
    elif mutation == "timestamp":
        data["reviewed_at_utc" if stage == "assistant" else "confirmed_at_utc"] = "2026-10-13"
    path.write_text(json.dumps(data))
    if stage == "assistant":
        data = json.loads(confirmation.read_text())
        data["assistant_review_sha256"] = _sha256(assistant)
        confirmation.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        create_review_receipt(manifest, audit, queue, assistant, confirmation, tmp_path / "receipt.json")


def test_unflagged_review_still_needs_explicit_confirmation(tmp_path):
    manifest, audit, queue, _, _ = _fixture(tmp_path)
    assistant, confirmation = _stages(tmp_path, queue)
    confirmation.unlink()
    with pytest.raises(ValueError):
        create_review_receipt(manifest, audit, queue, assistant, confirmation, tmp_path / "receipt.json")


def test_assistant_validation_and_flagged_accepted_resolution(tmp_path):
    from modules import stage3_review
    manifest, audit, queue, _, _ = _fixture(tmp_path)
    assistant, confirmation = _stages(tmp_path, queue, "needs_user_confirmation")
    reviewed = stage3_review.validate_assistant_review(queue, assistant, manifest, audit)
    assert reviewed["results"][0]["geometry_decision"] == "needs_user_confirmation"
    receipt = create_review_receipt(manifest, audit, queue, assistant, confirmation, tmp_path / "receipt.json")
    assert receipt["schema"] == "stage3-xml-repair-review-v2"
    responsibilities = receipt["review_responsibilities"]
    assert responsibilities["assistant"]["reviewer"] == "assistant"
    assert responsibilities["assistant"]["scope"] == "geometry_only"
    assert responsibilities["assistant"]["reviewed_count"] == 1
    assert responsibilities["assistant"]["flagged_count"] == 1
    assert responsibilities["user"]["reviewer"] == "Xiaoxiao"
    assert responsibilities["user"]["scope"] == "confirm_unflagged_review_and_resolve_flags"
    assert responsibilities["user"]["acknowledge_geometry_only_scope"] is True
    assert responsibilities["user"]["acknowledge_unflagged_review"] is True
    assert responsibilities["user"]["resolution_count"] == 1


@pytest.mark.parametrize("target", ["assistant", "confirmation", "queue"])
def test_receipt_rechecks_live_stage_hashes(tmp_path, target):
    manifest, audit, queue, _, _ = _fixture(tmp_path)
    receipt_path = _receipt(tmp_path, manifest, audit, queue)
    path = {"assistant": tmp_path / "assistant.json", "confirmation": tmp_path / "confirmation.json", "queue": queue}[target]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        validate_review_receipt(receipt_path, manifest, audit)


def test_v1_receipt_cannot_pass_v2_validation(tmp_path):
    manifest, audit, queue, _, _ = _fixture(tmp_path)
    receipt_path = _receipt(tmp_path, manifest, audit, queue)
    receipt = json.loads(receipt_path.read_text())
    receipt["schema"] = "stage3-xml-repair-review-v1"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="SCHEMA_INVALID"):
        validate_review_receipt(receipt_path, manifest, audit)


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
