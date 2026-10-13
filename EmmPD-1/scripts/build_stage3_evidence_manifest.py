"""Validate Stage-3 acceptance artifacts and write their evidence manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_evidence import CONFIG_FILES, SEEDS, build_evidence_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the frozen cohort, human XML review, H5 inventory, numerical and scale "
            "acceptance, E/F smoke, A-F configurations, and source hashes."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--repair-review-receipt", type=Path)
    parser.add_argument("--h5-inventory", type=Path, required=True)
    parser.add_argument("--metric-acceptance", type=Path, required=True)
    parser.add_argument("--scale-acceptance", type=Path, required=True)
    parser.add_argument("--ef-report", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs" / "stage1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_paths = {
        group: args.config_dir / filename for group, filename in CONFIG_FILES.items()
    }
    evidence = build_evidence_manifest(
        manifest_path=args.manifest,
        audit_path=args.audit_report,
        receipt_path=args.repair_review_receipt,
        inventory_path=args.h5_inventory,
        metric_acceptance_path=args.metric_acceptance,
        scale_path=args.scale_acceptance,
        ef_report_path=args.ef_report,
        config_paths=config_paths,
        seeds=SEEDS,
        output_path=args.output,
    )
    print(json.dumps({
        "status": "passed",
        "schema": evidence["schema"],
        "artifact_count": len(evidence["artifacts"]),
        "template_count": len(evidence["configs"]["templates"]),
        "resolved_run_count": len(evidence["configs"]["runs"]),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
