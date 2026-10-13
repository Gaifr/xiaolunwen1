"""Validate the two-stage human review for deterministic XML geometry repairs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

RECEIPT_SCHEMA = "stage3-xml-repair-review-v2"
ASSISTANT_REVIEW_SCHEMA = "stage3-xml-geometry-review-v1"
USER_CONFIRMATION_SCHEMA = "stage3-xml-user-confirmation-v1"
ACCEPT_DECISION = "accept_make_valid"
GEOMETRY_DECISIONS = {"geometry_ok", "needs_user_confirmation", "cannot_judge"}
REPAIR_FIELDS = ("group", "reason", "after_type", "polygon_parts", "before_area", "after_area")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)\Z"
)


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
        annotation_value = record["annotation"]
        if isinstance(annotation_value, bool):
            raise ValueError
        annotation = int(annotation_value)
        if str(annotation_value).strip() != str(annotation):
            raise ValueError
        slide_id = str(record["slide_id"]).strip()
        split = str(record["split"]).strip()
        xml_sha = str(record["xml_sha256"]).lower().strip()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("REPAIR_REVIEW_IDENTITY_INVALID") from error
    if not slide_id or split not in {"train", "val", "test"} or annotation < 0:
        raise ValueError("REPAIR_REVIEW_IDENTITY_INVALID")
    if not _SHA256_RE.fullmatch(xml_sha):
        raise ValueError(f"REPAIR_REVIEW_XML_SHA_INVALID {slide_id}")
    return slide_id, split, xml_sha, annotation


def _same_value(left, right) -> bool:
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return str(left).strip() == str(right).strip()


def _required_text(value, error_code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_code)
    return value.strip()


def _validate_utc_timestamp(value, error_code: str) -> str:
    if not isinstance(value, str) or not _UTC_TIMESTAMP_RE.fullmatch(value):
        raise ValueError(error_code)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ValueError(error_code) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(error_code)
    return value


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

    cases = report.get("repair_cases", [])
    if not isinstance(cases, list):
        raise ValueError("REPAIR_REVIEW_AUDIT_INVENTORY_INVALID")
    expected = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("REPAIR_REVIEW_AUDIT_INVENTORY_INVALID")
        slide_id, split = str(case.get("slide_id", "")).strip(), str(case.get("split", "")).strip()
        row = manifest_rows.get(slide_id)
        if row is None or str(row.get("split", "")).strip() != split:
            raise ValueError(f"REPAIR_REVIEW_MANIFEST_AUDIT_MISMATCH {slide_id}")
        xml_path = Path(row.get("xml_path", ""))
        expected_xml_sha = str(case.get("xml_sha256", "")).lower()
        if not _SHA256_RE.fullmatch(expected_xml_sha):
            raise ValueError(f"REPAIR_REVIEW_XML_SHA_INVALID {slide_id}")
        if not xml_path.is_file() or sha256(xml_path) != expected_xml_sha:
            raise ValueError(f"REPAIR_REVIEW_XML_HASH_MISMATCH {slide_id}")
        repairs = case.get("repairs", [])
        if not isinstance(repairs, list):
            raise ValueError(f"REPAIR_REVIEW_AUDIT_INVENTORY_INVALID {slide_id}")
        for repair in repairs:
            if not isinstance(repair, dict):
                raise ValueError(f"REPAIR_REVIEW_AUDIT_INVENTORY_INVALID {slide_id}")
            try:
                identity = {"slide_id": slide_id, "split": split,
                            "xml_sha256": expected_xml_sha, "annotation": repair["annotation"]}
                key = _key(identity)
                metadata = {field: repair[field] for field in REPAIR_FIELDS}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"REPAIR_REVIEW_AUDIT_INVENTORY_INVALID {slide_id}") from error
            if key in expected:
                raise ValueError(f"REPAIR_REVIEW_AUDIT_DUPLICATE {slide_id}")
            expected[key] = {**identity, **metadata}
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
    try:
        rendered_count = int(metadata.get("rendered_annotations", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("REPAIR_REVIEW_RENDER_COUNT_MISMATCH") from error
    if rendered_count != expected_count:
        raise ValueError("REPAIR_REVIEW_RENDER_COUNT_MISMATCH")
    splits = metadata.get("splits")
    if not isinstance(splits, list) or not expected_splits.issubset(set(splits)):
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
    required = {"slide_id", "split", "xml_sha256", "annotation", *REPAIR_FIELDS}
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

    expected_splits = {key[1] for key in expected}
    _validate_render_manifest(queue_path, manifest, audit_report, len(expected), expected_splits)
    return report, expected, observed


def _identity_from_key(key: tuple[str, str, str, int]) -> dict:
    slide_id, split, xml_sha, annotation = key
    return {"slide_id": slide_id, "split": split,
            "xml_sha256": xml_sha, "annotation": annotation}


def _read_stage_rows(rows, expected_keys: set[tuple[str, str, str, int]],
                     prefix: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"{prefix}_INVENTORY_INVALID")
    observed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{prefix}_IDENTITY_INVALID")
        key = _key(row)
        if key in observed:
            raise ValueError(f"{prefix}_DUPLICATE {key[0]} annotation={key[3]}")
        observed[key] = row
    if set(observed) != expected_keys:
        missing = len(expected_keys - set(observed))
        extra = len(set(observed) - expected_keys)
        raise ValueError(f"{prefix}_INVENTORY_MISMATCH missing={missing} extra={extra}")
    return observed


def validate_assistant_review(queue_path: Path, assistant_review_path: Path,
                              manifest: Path, audit_report: Path) -> dict:
    """Validate the assistant's geometry-only review against the complete queue."""
    _, expected, _ = _read_and_validate_queue(manifest, audit_report, queue_path)
    review = _load_json(assistant_review_path)
    if review.get("schema") != ASSISTANT_REVIEW_SCHEMA:
        raise ValueError("REPAIR_ASSISTANT_REVIEW_SCHEMA_INVALID")
    reviewer = _required_text(review.get("reviewer"), "REPAIR_ASSISTANT_REVIEWER_REQUIRED")
    reviewed_at = _validate_utc_timestamp(review.get("reviewed_at_utc"),
                                          "REPAIR_ASSISTANT_TIMESTAMP_INVALID")
    queue_digest = sha256(queue_path)
    if review.get("queue_sha256") != queue_digest:
        raise ValueError("REPAIR_ASSISTANT_QUEUE_MISMATCH")
    observed = _read_stage_rows(review.get("results"), set(expected),
                                "REPAIR_ASSISTANT_REVIEW")
    results = []
    for key in sorted(expected):
        row = observed[key]
        decision = row.get("geometry_decision")
        if not isinstance(decision, str) or decision not in GEOMETRY_DECISIONS:
            raise ValueError(f"REPAIR_ASSISTANT_DECISION_INVALID {key[0]} annotation={key[3]}")
        note = _required_text(row.get("geometry_note"), "REPAIR_ASSISTANT_NOTE_REQUIRED")
        results.append({**_identity_from_key(key), "geometry_decision": decision,
                        "geometry_note": note})
    return {"schema": ASSISTANT_REVIEW_SCHEMA, "reviewer": reviewer,
            "reviewed_at_utc": reviewed_at, "queue_sha256": queue_digest,
            "assistant_review_sha256": sha256(assistant_review_path),
            "results": results}


def _validate_user_confirmation(queue_path: Path, assistant_review: dict,
                                user_confirmation_path: Path) -> dict:
    confirmation = _load_json(user_confirmation_path)
    if confirmation.get("schema") != USER_CONFIRMATION_SCHEMA:
        raise ValueError("REPAIR_USER_CONFIRMATION_SCHEMA_INVALID")
    reviewer = _required_text(confirmation.get("reviewer"), "REPAIR_USER_REVIEWER_REQUIRED")
    confirmed_at = _validate_utc_timestamp(confirmation.get("confirmed_at_utc"),
                                           "REPAIR_USER_TIMESTAMP_INVALID")
    if reviewer.casefold() == assistant_review["reviewer"].casefold():
        raise ValueError("REPAIR_REVIEWERS_NOT_DISTINCT")
    assistant_time = datetime.fromisoformat(
        assistant_review["reviewed_at_utc"].replace("Z", "+00:00"))
    confirmation_time = datetime.fromisoformat(confirmed_at.replace("Z", "+00:00"))
    if confirmation_time < assistant_time:
        raise ValueError("REPAIR_USER_CONFIRMATION_PRECEDES_ASSISTANT_REVIEW")
    queue_digest = sha256(queue_path)
    if confirmation.get("queue_sha256") != queue_digest:
        raise ValueError("REPAIR_USER_QUEUE_MISMATCH")
    if confirmation.get("assistant_review_sha256") != assistant_review["assistant_review_sha256"]:
        raise ValueError("REPAIR_USER_ASSISTANT_REVIEW_MISMATCH")
    if confirmation.get("acknowledge_geometry_only_scope") is not True:
        raise ValueError("REPAIR_USER_GEOMETRY_SCOPE_NOT_ACKNOWLEDGED")
    if confirmation.get("acknowledge_unflagged_review") is not True:
        raise ValueError("REPAIR_USER_UNFLAGGED_REVIEW_NOT_ACKNOWLEDGED")

    assistant_rows = {_key(row): row for row in assistant_review["results"]}
    observed = _read_stage_rows(confirmation.get("resolutions"), set(assistant_rows),
                                "REPAIR_USER_CONFIRMATION")
    resolutions = []
    for key in sorted(assistant_rows):
        row = observed[key]
        if row.get("decision") != ACCEPT_DECISION:
            raise ValueError(f"REPAIR_USER_RESOLUTION_NOT_ACCEPTED {key[0]} annotation={key[3]}")
        note = _required_text(row.get("review_note"), "REPAIR_USER_REVIEW_NOTE_REQUIRED")
        if (assistant_rows[key]["geometry_decision"] == "needs_user_confirmation"
                and not note):
            raise ValueError(f"REPAIR_USER_FLAG_UNRESOLVED {key[0]} annotation={key[3]}")
        resolutions.append({**_identity_from_key(key), "decision": ACCEPT_DECISION,
                            "review_note": note})
    return {"schema": USER_CONFIRMATION_SCHEMA, "reviewer": reviewer,
            "confirmed_at_utc": confirmed_at, "queue_sha256": queue_digest,
            "assistant_review_sha256": assistant_review["assistant_review_sha256"],
            "user_confirmation_sha256": sha256(user_confirmation_path),
            "acknowledge_geometry_only_scope": True,
            "acknowledge_unflagged_review": True, "resolutions": resolutions}


def _approved_repairs(expected: dict, assistant_review: dict, confirmation: dict) -> list[dict]:
    assistant_rows = {_key(row): row for row in assistant_review["results"]}
    resolution_rows = {_key(row): row for row in confirmation["resolutions"]}
    if any(row["geometry_decision"] == "cannot_judge" for row in assistant_rows.values()):
        raise ValueError("REPAIR_ASSISTANT_CANNOT_JUDGE")
    if set(assistant_rows) != set(expected) or set(resolution_rows) != set(expected):
        raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
    approved = []
    for key in sorted(expected):
        geometry = assistant_rows[key]
        resolution = resolution_rows[key]
        if geometry["geometry_decision"] not in {"geometry_ok", "needs_user_confirmation"}:
            raise ValueError("REPAIR_ASSISTANT_DECISION_INVALID")
        if resolution["decision"] != ACCEPT_DECISION:
            raise ValueError("REPAIR_USER_RESOLUTION_NOT_ACCEPTED")
        approved.append({**expected[key], "decision": ACCEPT_DECISION,
                         "assistant_geometry_decision": geometry["geometry_decision"],
                         "assistant_geometry_note": geometry["geometry_note"],
                         "review_note": resolution["review_note"]})
    return approved


def _responsibilities(assistant_review: dict, confirmation: dict) -> dict:
    return {
        "assistant": {
            "reviewer": assistant_review["reviewer"],
            "scope": "geometry_only",
            "reviewed_at_utc": assistant_review["reviewed_at_utc"],
            "reviewed_count": len(assistant_review["results"]),
            "flagged_count": sum(row["geometry_decision"] == "needs_user_confirmation"
                                  for row in assistant_review["results"]),
        },
        "user": {
            "reviewer": confirmation["reviewer"],
            "scope": "confirm_unflagged_review_and_resolve_flags",
            "confirmed_at_utc": confirmation["confirmed_at_utc"],
            "acknowledge_geometry_only_scope": True,
            "acknowledge_unflagged_review": True,
            "resolution_count": len(confirmation["resolutions"]),
        },
    }


def create_review_receipt(manifest: Path, audit_report: Path, queue_path: Path,
                          assistant_review_path: Path, user_confirmation_path: Path,
                          output: Path) -> dict:
    """Create a receipt only after both exact-inventory review stages pass."""
    _, expected, _ = _read_and_validate_queue(manifest, audit_report, queue_path)
    assistant_review = validate_assistant_review(queue_path, assistant_review_path,
                                                 manifest, audit_report)
    confirmation = _validate_user_confirmation(queue_path, assistant_review,
                                              user_confirmation_path)
    approved = _approved_repairs(expected, assistant_review, confirmation)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "reviewer": confirmation["reviewer"],
        "reviewed_at_utc": confirmation["confirmed_at_utc"],
        "manifest_sha256": sha256(manifest),
        "audit_sha256": sha256(audit_report),
        "review_queue_sha256": sha256(queue_path),
        "review_queue_path": str(Path(queue_path).resolve()),
        "assistant_review_sha256": assistant_review["assistant_review_sha256"],
        "user_confirmation_sha256": confirmation["user_confirmation_sha256"],
        "assistant_review_path": str(Path(assistant_review_path).resolve()),
        "user_confirmation_path": str(Path(user_confirmation_path).resolve()),
        "approved_count": len(approved),
        "approved_repairs": approved,
        "review_responsibilities": _responsibilities(assistant_review, confirmation),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return receipt


def _same_approved_inventory(observed_rows, expected_rows) -> bool:
    if not isinstance(observed_rows, list) or len(observed_rows) != len(expected_rows):
        return False
    observed = {}
    for item in observed_rows:
        if not isinstance(item, dict):
            return False
        try:
            key = _key(item)
        except ValueError:
            return False
        if key in observed:
            return False
        observed[key] = item
    expected_by_key = {_key(item): item for item in expected_rows}
    if set(observed) != set(expected_by_key):
        return False
    for key, expected in expected_by_key.items():
        item = observed[key]
        if item.get("decision") != ACCEPT_DECISION:
            return False
        for field in (*REPAIR_FIELDS, "assistant_geometry_decision",
                      "assistant_geometry_note", "review_note"):
            if field not in item or not _same_value(item[field], expected[field]):
                return False
    return True


def validate_review_receipt(receipt_path: Path, manifest: Path, audit_report: Path) -> dict:
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("REPAIR_REVIEW_RECEIPT_SCHEMA_INVALID")
    if receipt.get("manifest_sha256") != sha256(manifest):
        raise ValueError("REPAIR_REVIEW_MANIFEST_MISMATCH")
    if receipt.get("audit_sha256") != sha256(audit_report):
        raise ValueError("REPAIR_REVIEW_AUDIT_MISMATCH")
    for field, error_code in (("review_queue_sha256", "REPAIR_REVIEW_QUEUE_SHA_INVALID"),
                              ("assistant_review_sha256", "REPAIR_ASSISTANT_REVIEW_SHA_INVALID"),
                              ("user_confirmation_sha256", "REPAIR_USER_CONFIRMATION_SHA_INVALID")):
        if not isinstance(receipt.get(field), str) or not _SHA256_RE.fullmatch(receipt[field]):
            raise ValueError(error_code)
    _required_text(receipt.get("reviewer"), "REPAIR_USER_REVIEWER_REQUIRED")
    _validate_utc_timestamp(receipt.get("reviewed_at_utc"), "REPAIR_USER_TIMESTAMP_INVALID")
    try:
        queue_path = Path(receipt["review_queue_path"])
    except (KeyError, TypeError) as error:
        raise ValueError("REPAIR_REVIEW_QUEUE_PATH_REQUIRED") from error
    try:
        assistant_path = Path(receipt["assistant_review_path"])
        confirmation_path = Path(receipt["user_confirmation_path"])
    except (KeyError, TypeError) as error:
        raise ValueError("REPAIR_REVIEW_STAGE_PATH_REQUIRED") from error
    if sha256(queue_path) != receipt["review_queue_sha256"]:
        raise ValueError("REPAIR_REVIEW_QUEUE_MISMATCH")
    _, expected, _ = _read_and_validate_queue(manifest, audit_report, queue_path)
    assistant_review = validate_assistant_review(queue_path, assistant_path, manifest, audit_report)
    confirmation = _validate_user_confirmation(queue_path, assistant_review, confirmation_path)
    if assistant_review["assistant_review_sha256"] != receipt["assistant_review_sha256"]:
        raise ValueError("REPAIR_ASSISTANT_REVIEW_MISMATCH")
    if confirmation["user_confirmation_sha256"] != receipt["user_confirmation_sha256"]:
        raise ValueError("REPAIR_USER_CONFIRMATION_MISMATCH")
    if (confirmation["reviewer"] != receipt["reviewer"]
            or confirmation["confirmed_at_utc"] != receipt["reviewed_at_utc"]):
        raise ValueError("REPAIR_REVIEW_RESPONSIBILITY_MISMATCH")
    if receipt.get("review_responsibilities") != _responsibilities(assistant_review, confirmation):
        raise ValueError("REPAIR_REVIEW_RESPONSIBILITY_MISMATCH")

    approved = _approved_repairs(expected, assistant_review, confirmation)
    if not _same_approved_inventory(receipt.get("approved_repairs"), approved):
        raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
    try:
        approved_count = int(receipt.get("approved_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("REPAIR_REVIEW_COUNT_MISMATCH") from error
    if approved_count != len(approved):
        raise ValueError("REPAIR_REVIEW_COUNT_MISMATCH")
    return receipt


def assert_repairs_approved(receipt: dict | None, slide_id: str, split: str,
                            xml_sha256: str, repairs: tuple[dict, ...] | list[dict]) -> bool:
    if not repairs:
        return True
    if not isinstance(receipt, dict):
        raise ValueError("REPAIR_REVIEW_RECEIPT_REQUIRED")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("REPAIR_REVIEW_RECEIPT_SCHEMA_INVALID")
    expected = {}
    for item in repairs:
        try:
            annotation = int(item["annotation"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"REPAIR_REVIEW_REPAIR_INVENTORY_INVALID {slide_id}") from error
        if annotation in expected:
            raise ValueError(f"REPAIR_REVIEW_REPAIR_INVENTORY_INVALID {slide_id}")
        expected[annotation] = item
    approved_rows = receipt.get("approved_repairs")
    if not isinstance(approved_rows, list):
        raise ValueError("REPAIR_REVIEW_INVENTORY_MISMATCH")
    approved = [item for item in approved_rows
                if isinstance(item, dict)
                and item.get("slide_id") == slide_id and item.get("split") == split]
    if any(item.get("xml_sha256") != xml_sha256 for item in approved):
        raise ValueError(f"REPAIR_REVIEW_XML_MISMATCH {slide_id}")
    observed = {}
    for item in approved:
        annotation = int(item["annotation"])
        if annotation in observed:
            raise ValueError(f"REPAIR_REVIEW_REPAIR_INVENTORY_MISMATCH {slide_id}")
        observed[annotation] = item
    if set(observed) != set(expected):
        raise ValueError(f"REPAIR_REVIEW_REPAIR_INVENTORY_MISMATCH {slide_id}")
    for annotation, repair in expected.items():
        item = observed[annotation]
        assistant_decision = item.get("assistant_geometry_decision")
        if (item.get("decision") != ACCEPT_DECISION
                or not isinstance(assistant_decision, str)
                or assistant_decision not in {"geometry_ok", "needs_user_confirmation"}):
            raise ValueError(f"REPAIR_REVIEW_NOT_ACCEPTED {slide_id} annotation={annotation}")
        if (not str(item.get("assistant_geometry_note", "")).strip()
                or not str(item.get("review_note", "")).strip()):
            raise ValueError(f"REPAIR_REVIEW_RESPONSIBILITY_MISMATCH {slide_id} annotation={annotation}")
        for field in REPAIR_FIELDS:
            if not _same_value(item.get(field), repair[field]):
                raise ValueError(f"REPAIR_REVIEW_METADATA_MISMATCH {slide_id} annotation={annotation} {field}")
    return True
