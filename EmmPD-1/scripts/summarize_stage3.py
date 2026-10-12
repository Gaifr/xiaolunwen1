"""Paired three-seed D-C MLAC@5% summary from frozen test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_metrics import paired_bootstrap


def _read(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics["split"] != "test":
        raise ValueError("SUMMARY_REQUIRES_TEST_EVALUATION")
    rows = [json.loads(line) for line in (directory / "slides.jsonl").read_text(encoding="utf-8").splitlines()]
    positives = {r["slide_id"]: r["area_0.05"]["mlac"] for r in rows if r["slide_label"] == 1}
    if len(positives) != 12:
        raise ValueError(f"POSITIVE_TEST_COUNT {len(positives)}")
    return metrics, positives


def summarize(c_dirs, d_dirs):
    if len(c_dirs) != 3 or len(d_dirs) != 3:
        raise ValueError("THREE_SEEDS_REQUIRED")
    c_runs = [_read(path) for path in c_dirs]
    d_runs = [_read(path) for path in d_dirs]
    for expected_seed, ((cm, _), (dm, _)) in zip((9233, 9234, 9235), zip(c_runs, d_runs)):
        if (cm["seed"], dm["seed"]) != (expected_seed, expected_seed):
            raise ValueError("SEED_ORDER_MISMATCH")
        if cm["model_variant"] != "graph_query" or dm["model_variant"] != "graph_query" or cm["graph_mode"] != "dual" or dm["graph_mode"] != "dual":
            raise ValueError("MAIN_COMPARISON_MODEL_MISMATCH")
        if cm["lambda_loc"] != 0 or dm["lambda_loc"] != 1:
            raise ValueError("MAIN_COMPARISON_SUPERVISION_MISMATCH")
    manifest_hashes = {m["manifest_sha256"] for m, _ in c_runs + d_runs}
    evaluator_hashes = {m["evaluator_sha256"] for m, _ in c_runs + d_runs}
    if len(manifest_hashes) != 1 or len(evaluator_hashes) != 1:
        raise ValueError("RUN_PROVENANCE_MISMATCH")
    ids = sorted(c_runs[0][1])
    if any(sorted(values) != ids for _, values in c_runs + d_runs):
        raise ValueError("PAIRED_SLIDE_IDS_MISMATCH")
    c = [[values[slide_id] for slide_id in ids] for _, values in c_runs]
    d = [[values[slide_id] for slide_id in ids] for _, values in d_runs]
    result = paired_bootstrap(c, d)
    result.update(slide_ids=ids, seeds=[9233, 9234, 9235],
                  manifest_sha256=next(iter(manifest_hashes)),
                  evaluator_sha256=next(iter(evaluator_hashes)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=Path, action="append", required=True)
    parser.add_argument("--d", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.c, args.d)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
