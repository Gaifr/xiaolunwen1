"""Write a source/data hash lock after the user has accepted Stage-3 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_SOURCES = (
    "modules/stage3_metrics.py", "modules/evidence_model.py",
    "modules/evidence_objectives.py", "modules/camelyon_dataset.py",
    "scripts/train_stage1.py", "scripts/eval_stage3.py",
    "scripts/audit_stage3.py", "scripts/smoke_stage3_ef.py",
    "scripts/summarize_stage3.py",
    "docs/camelyon16-stage3-evaluation-freeze.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_payload(manifest, audit, configs):
    report = json.loads(Path(audit).read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("STAGE3_AUDIT_NOT_PASSED")
    return {"protocol": "camelyon16-stage3-mlac5-v1",
            "manifest_sha256": sha256(manifest), "audit_sha256": sha256(audit),
            "source_sha256": {name: sha256(ROOT / name) for name in LOCKED_SOURCES},
            "config_sha256": {Path(name).name: sha256(name) for name in configs}}


def verify_lock(lock_file, manifest):
    locked = json.loads(Path(lock_file).read_text(encoding="utf-8"))
    if locked["protocol"] != "camelyon16-stage3-mlac5-v1" or locked["manifest_sha256"] != sha256(manifest):
        raise ValueError("STAGE3_MANIFEST_LOCK_MISMATCH")
    for name, expected in locked["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"STAGE3_SOURCE_LOCK_MISMATCH {name}")
    return locked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-confirmed", action="store_true",
                        help="Assert synthetic tests and train/val scale acceptance have passed")
    args = parser.parse_args()
    if not args.acceptance_confirmed:
        raise ValueError("ACCEPTANCE_NOT_CONFIRMED")
    payload = lock_payload(args.manifest, args.audit_report, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
