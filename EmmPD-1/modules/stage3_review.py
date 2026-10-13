"""Validate human decisions for deterministic XML geometry repairs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

RECEIPT_SCHEMA = "stage3-xml-repair-review-v1"
ACCEPT_DECISION = "accept_make_valid"
REPAIR_FIELDS = ("group", "reason", "after_type", "polygon_parts", "before_area", "after_area")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"REPAIR_REVIEW_JSON_INVALID {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"REPAIR_REVIEW_JSON_INVALID {path}")
    return value


def _key(record: dict) -> tuple[str, str, str, int]:
    try:
        annotation = int(record["annotation"])
        slide_id = str(record["slide_id"]).strip()
        split = str(record["split"]).strip()
        xml_sha = str(record["xml_sha256"]).lower().strip()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("REPAIR_REVIEW_IDENTITY_INVALID") from error
    if not slide_id or split not in {"train", "val", "test"} or annotation < 0:
        raise ValueError("REPAIR_REVIEW_IDENTITY_INVALID")
    if not re.fullmatch(r"[0-9a-f]{64}", xml_sha):
        raise ValueError(f"REPAIR_REVIEW_XML_SHA_INVALID {slide_id}")
    return slide_id, split, xml_sha, annotation


def _same_value(left, right) -> bool:
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(right, int) and not isinstance(right, bool):
        try:
            return int(left) == right
        except (TypeError, ValueError):
            return False
    return str(left).strip() == str(right).strip()


def _expected_inventory(manifest: Path, audit_report: Path):
    report = _load_json(audit_report)
    if report.get("issues"):
        raise ValueError("STAGE3_AUDIT_HAS_ISSUES")
    if report.get("status") not in {"needs_review", "passed"}:
        raise ValueError("STAGE3_AUDIT_NOT_ACCEPTABLE")
    try:
        with Path(manifest).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise ValueError(f"REPAIR_REVIEW_MANIFEST_UNREADABLE {manifest}") from error
    manifest_rows = {str(row.get("slide_id", "")).strip(): row for row in rows}
    if len(manifest_rows) != len(rows) or "" in manifest_rows:
        raise ValueError("REPAIR_REVIEW_MANIFEST_DUPLICATE_OR_EMPTY_ID")

    expected = {}
    for case in report.get("repair_cases", []):
        slide_id, split = str(case.get("slide_id", "")).strip(), str(case.get("split", "")).strip()
        row = manifest_rows.get(slide_id)
        if row is None or row.get("split", "").strip() != split:
            raise ValueError(f"REPAIR_REVIEW_MANIFEST_AUDIT_MISMATCH {slide_id}")
        xml_path = Path(row.get("xml_path", ""))
        expected_xml_sha = str(case.get("xml_sha256", "")).lower()
        if not xml_path.is_file() or sha256(xml_path) != expected_xml_sha:
            raise ValueError(f"REPAIR_REVIEW_XML_HASH_MISMATCH {slide_id}")
        for repair in case.get("repairs", []):
            identity = {"slide_id": slide_id, "split": split,
                        "xml_sha256": expected_xml_sha, "annotation": int(repair["annotation"])}
            key = _key(identity)
            if key in expected:
                raise ValueError(f"REPAIR_REVIEW_AUDIT_DUPLICATE {slide_id}")
            expected[key] = {**identity, **{field: repair[field] for field in REPAIR_FIELDS}}
    if report.get("status") == "needs_review" and not expected:
        raise ValueError("REPAIR_REVIEW_AUDIT_INVENTORY_EMPTY")
    if report.get("status") == "passed" and expected:
        raise ValueError("REPAIR_REVIEW_AUDIT_STATUS_MISMATCH")
    return report, expected


def _validate_render_manifest(queue_path: Path, manifest: Path, audit_report: Path,
                              expected_count: int, expected_splits: set[str]) -> None:
    render_path = Path(queue_path).parent / "render_manifest.json"
    metadata = _load_json(render_path)
    if metadata.get("audit_sha256") != sha256(audit_report):
        raise ValueError("REPAIR_REVIEW_RENDER_AUDIT_MISMATCH")
    if metadata.get("manifest_sha256") != sha256(manifest):
        raise ValueError("REPAIR_REVIEW_RENDER_MANIFEST_MISMATCH")
    if int(metadata.get("rendered_annotations", -1)) != expected_count:
        raise ValueError("REPAIR_REVIEW_RENDER_COUNT_MISMATCH")
    if not expected_splits.issubset(set(metadata.get("splits", []))):
        raise ValueError("REPAIR_REVIEW_RENDER_SPLITS_INCOMPLETE")


def _read_and_validate_queue(manifest: Path, audit_report: Path, queue_path: Path):
    report, expected = _expected_inventory(manifest, audit_report)
    if not expected:
        raise ValueError("REPAIR_REVIEW_NOT_REQUIRED")
    try:
        with Path(queue_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = set(reader.fieldnames or [])
    except OSError as error:
        raise ValueError(f"REPAIR_REVIEW_QUEUE_UNREADABLE {queue_path}") from error
    required = {"slide_id", "split", "xml_sha256", "annotation", *REPAIR_FIELDS,
                "review_decision", "review_note"}
    if not required.issubset(fields):
        raise ValueError("REPAIR_REVIEW_QUEUE_COLUMNS_INVALID")

    observed = {}
    for row in rows:
        key = _key(row)
        if key in observed:
            raise ValueError(f"REPAIR_REVIEW_QUEUE_DUPLICATE {key[0]} annotation={key[3]}")
        observed[key] = row
    if set(observed) != set(expected):
        missing = len(set(expected) - set(observed))
        extra = len(set(observed) - set(expected))
        raise ValueError(f"REPAIR_REVIEW_INVENTORY_MISMATCH missing={missing} extra={extra}")

    for key, expected_row in expected.items():
        row = observed[key]
        slide_id, _, _, annotation = key
        for field in REPAIR_FIELDS:
            if not _same_value(row.get(field), expected_row[field]):
                raise ValueError(f"REPAIR_REVIEW_QUEUE_METADATA_MISMATCH {slide_id} annotation={annotation} {field}")
        decision = str(row.get("review_decision", "")).strip()
        if decision != ACCEPT_DECISION:
            note = str(row.get("review_note", "")).strip()
            raise ValueError(f"REPAIR_REVIEW_NOT_ACCEPTED {slide_id} annotation={annotation} decision={decision!r} note={note!r}")

    expected_splits = {key[1] for key in expected}
    _validate_render_manifest(queue_path, manifest, audit_report, len(expected), expected_splits)
    approved = []
    for key in sorted(expected):
        row = observed[key]
        approved.append({**expected[key], "decision": ACCEPT_DECISION,
                         "review_note": str(row.get("review_note", "")).strip()})
    return report, approved


def create_review_receipt(manifest: Path, audit_report: Path, review_queue: Path,
                          reviewer: str, output: Path) -> dict:
    reviewer = str(reviewer).strip()
    if not reviewer:
        raise ValueError("REPAIR_REVIEWER_REQUIRED")
    _, approved = _read_and_validate_queue(manifest, audit_report, review_queue)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "reviewer": reviewer,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest_sha256": sha256(manifest),
        "audit_sha256": sha256(audit_report),
        "review_queue_sha256": sha256(review_queue),
        "approved_count": len(approved),
        "approved_repairs": approved,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return receipt


def validate_review_receipt(receipt_path: Path, manifest: Path, audit_report: Path) -> dict:
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("REPAIR_REVIEW_RECEIPT_SCHEMA_INVALID")
    if not str(receipt.get("reviewer", "")).strip():
        raise ValueError("REPAIR_REVIEWER_REQUIRED")
    timestamp = str(receipt.get("reviewed_at_utc", ""))
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("REPAIR_REVIEW_TIMESTAMP_INVALID") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("REPAIR_REVIEW_TIMESTAMP_INVALID")
    if receipt.get("manifest_sha256") != sha256(manifest):
        raise ValueError("REPAIR_REVIEW_MANIFEST_MISMATCH")
    if receipt.get("audit_sha256") != sha256(audit_report):
        raise ValueError("REPAIR_REVIEW_AUDIT_MISMATCH")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("review_queue_sha256", ""))):
        raise ValueError("REPAIR_REVIEW_QUEUE_SHA_INVALID")

    _, expected = _expected_inventory(manifest, audit_report)
    approved = receipt.get("approved_repairs")
    if not isinstance(approved, list) or len(approved) != len(expected):
        raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
    observed = {}
    for item in approved:
        key = _key(item)
        if key in observed or item.get("decision") != ACCEPT_DECISION:
            raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
        observed[key] = item
    if set(observed) != set(expected):
        raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
    for key, expected_row in expected.items():
        item = observed[key]
        for field in REPAIR_FIELDS:
            if not _same_value(item.get(field), expected_row[field]):
                raise ValueError(f"REPAIR_REVIEW_METADATA_MISMATCH {key[0]} annotation={key[3]} {field}")
    if int(receipt.get("approved_count", -1)) != len(expected):
        raise ValueError("REPAIR_REVIEW_COUNT_MISMATCH")
    return receipt


def assert_repairs_approved(receipt: dict | None, slide_id: str, split: str,
                            xml_sha256: str, repairs: tuple[dict, ...] | list[dict]) -> bool:
    if not repairs:
        return True
    if not isinstance(receipt, dict):
        raise ValueError("REPAIR_REVIEW_RECEIPT_REQUIRED")
    expected = {int(item["annotation"]): item for item in repairs}
    approved = [item for item in receipt.get("approved_repairs", [])
                if item.get("slide_id") == slide_id and item.get("split") == split]
    if any(item.get("xml_sha256") != xml_sha256 for item in approved):
        raise ValueError(f"REPAIR_REVIEW_XML_MISMATCH {slide_id}")
    observed = {int(item["annotation"]): item for item in approved}
    if set(observed) != set(expected):
        raise ValueError(f"REPAIR_REVIEW_REPAIR_INVENTORY_MISMATCH {slide_id}")
    for annotation, repair in expected.items():
        item = observed[annotation]
        if item.get("decision") != ACCEPT_DECISION:
            raise ValueError(f"REPAIR_REVIEW_NOT_ACCEPTED {slide_id} annotation={annotation}")
        for field in REPAIR_FIELDS:
            if not _same_value(item.get(field), repair[field]):
                raise ValueError(f"REPAIR_REVIEW_METADATA_MISMATCH {slide_id} annotation={annotation} {field}")
    return True
