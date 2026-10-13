"""Create a hash-bound receipt after all XML repair rows are reviewed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_review import create_review_receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--reviewer", required=True,
                        help="Name or identifier attesting that every queue row was reviewed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = create_review_receipt(args.manifest, args.audit_report, args.review_queue,
                                    args.reviewer, args.output)
    print(json.dumps({"status": "accepted", "reviewer": receipt["reviewer"],
                      "approved_count": receipt["approved_count"],
                      "receipt": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
