"""Write and verify a Stage-3 lock bound to the complete evidence manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_evidence import PROTOCOL, sha256, validate_evidence_manifest


def lock_payload(evidence_manifest_path: Path) -> dict:
    """Validate and describe the exact evidence state to be frozen."""
    evidence_path = Path(evidence_manifest_path).expanduser().resolve()
    evidence = validate_evidence_manifest(evidence_path)
    config_hashes = {
        row["path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]: row["sha256"]
        for row in evidence["configs"]["templates"]
    }
    run_config_hashes = {
        f"{row['group']}:{row['seed']}": row["config_sha256"]
        for row in evidence["configs"]["runs"]
    }
    artifact_hashes = {
        name: item["sha256"]
        for name, item in evidence["artifacts"].items()
    }
    review_entry = evidence["artifacts"]["repair_review_receipt"]
    review_receipt = None
    if review_entry["path"] is not None:
        review_receipt = json.loads(Path(review_entry["path"]).read_text(encoding="utf-8"))
    return {
        "schema": "stage3-freeze-lock-v1",
        "protocol": PROTOCOL,
        "evidence_manifest_path": str(evidence_path),
        "evidence_manifest_sha256": sha256(evidence_path),
        "manifest_path": evidence["artifacts"]["manifest"]["path"],
        "manifest_sha256": evidence["artifacts"]["manifest"]["sha256"],
        "artifact_sha256": artifact_hashes,
        "locked_sources": evidence["locked_sources"],
        "config_sha256": config_hashes,
        "run_config_sha256": run_config_hashes,
        "seeds": evidence["seeds"],
        "repair_review": review_receipt,
    }


def verify_lock(lock_path: Path, evidence_manifest_path: Path) -> dict:
    """Revalidate all evidence and reject a stale or mismatched lock."""
    try:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("STAGE3_FREEZE_LOCK_INVALID") from error
    if not isinstance(lock, dict):
        raise ValueError("STAGE3_FREEZE_LOCK_INVALID")
    expected = lock_payload(evidence_manifest_path)
    if lock != expected:
        raise ValueError("STAGE3_FREEZE_LOCK_MISMATCH")
    return lock


def _write_lock_atomic(path: Path, payload: dict) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Stage-3 only after the complete evidence manifest validates."
    )
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence_manifest is None:
        parser.error("the following arguments are required: --evidence-manifest")
    payload = lock_payload(args.evidence_manifest)
    _write_lock_atomic(args.output, payload)
    print(json.dumps({
        "schema": payload["schema"],
        "protocol": payload["protocol"],
        "evidence_manifest_sha256": payload["evidence_manifest_sha256"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
