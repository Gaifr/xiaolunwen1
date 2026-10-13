"""Write a source/data hash lock after the user has accepted Stage-3 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCKED_SOURCES = (
    "modules/stage3_metrics.py", "modules/evidence_model.py",
    "modules/evidence_objectives.py", "modules/camelyon_dataset.py",
    "modules/stage3_review.py",
    "scripts/train_stage1.py", "scripts/eval_stage3.py",
    "scripts/audit_stage3.py", "scripts/smoke_stage3_ef.py",
    "scripts/summarize_stage3.py", "scripts/freeze_stage3.py",
    "scripts/render_stage3_xml_repairs.py", "scripts/accept_stage3_repairs.py",
    "docs/camelyon16-stage3-evaluation-freeze.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_payload(manifest, audit, configs, repair_review_receipt=None):
    report = json.loads(Path(audit).read_text(encoding="utf-8"))
    review = None
    if report.get("status") == "failed" or report.get("issues"):
        raise ValueError("STAGE3_AUDIT_FAILED")
    if report.get("status") == "needs_review":
        if repair_review_receipt is None:
            raise ValueError("STAGE3_REPAIR_REVIEW_REQUIRED")
        from modules.stage3_review import validate_review_receipt
        review = validate_review_receipt(repair_review_receipt, manifest, audit)
    elif report.get("status") != "passed":
        raise ValueError("STAGE3_AUDIT_STATUS_INVALID")
    elif repair_review_receipt is not None:
        raise ValueError("STAGE3_UNEXPECTED_REPAIR_REVIEW_RECEIPT")
    return {"protocol": "camelyon16-stage3-mlac5-v1",
            "manifest_sha256": sha256(manifest), "audit_sha256": sha256(audit),
            "source_sha256": {name: sha256(ROOT / name) for name in LOCKED_SOURCES},
            "config_sha256": {Path(name).name: sha256(name) for name in configs},
            "repair_review_sha256": (sha256(repair_review_receipt)
                                     if repair_review_receipt is not None else None),
            "repair_review": review}


def verify_lock(lock_file, manifest, repair_review_receipt=None):
    locked = json.loads(Path(lock_file).read_text(encoding="utf-8"))
    if locked["protocol"] != "camelyon16-stage3-mlac5-v1" or locked["manifest_sha256"] != sha256(manifest):
        raise ValueError("STAGE3_MANIFEST_LOCK_MISMATCH")
    for name, expected in locked["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"STAGE3_SOURCE_LOCK_MISMATCH {name}")
    expected_review_sha = locked.get("repair_review_sha256")
    if expected_review_sha is not None:
        if repair_review_receipt is None:
            raise ValueError("STAGE3_REPAIR_REVIEW_LOCK_MISMATCH")
        try:
            actual_review_sha = sha256(repair_review_receipt)
            receipt = json.loads(Path(repair_review_receipt).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("STAGE3_REPAIR_REVIEW_LOCK_MISMATCH") from error
        if (actual_review_sha != expected_review_sha or
                receipt != locked.get("repair_review") or
                receipt.get("manifest_sha256") != locked["manifest_sha256"]):
            raise ValueError("STAGE3_REPAIR_REVIEW_LOCK_MISMATCH")
    elif repair_review_receipt is not None or locked.get("repair_review") is not None:
        raise ValueError("STAGE3_REPAIR_REVIEW_LOCK_MISMATCH")
    return locked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--repair-review-receipt", type=Path)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-confirmed", action="store_true",
                        help="Assert synthetic tests and train/val scale acceptance have passed")
    args = parser.parse_args()
    if not args.acceptance_confirmed:
        raise ValueError("ACCEPTANCE_NOT_CONFIRMED")
    payload = lock_payload(args.manifest, args.audit_report, args.config,
                           repair_review_receipt=args.repair_review_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
