"""Create a hash-bound receipt after two-stage XML geometry review."""

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
    parser.add_argument("--assistant-review", type=Path, required=True,
                        help="JSON record of the assistant's geometry-only review")
    parser.add_argument("--user-confirmation", type=Path, required=True,
                        help="JSON record of the user's confirmation and flagged resolutions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = create_review_receipt(args.manifest, args.audit_report, args.review_queue,
                                    args.assistant_review, args.user_confirmation, args.output)
    print(json.dumps({"status": "accepted", "reviewer": receipt["reviewer"],
                      "approved_count": receipt["approved_count"],
                      "receipt": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
