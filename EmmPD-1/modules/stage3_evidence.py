"""Build and validate the fail-closed Stage-3 evidence manifest.

The manifest binds the frozen cohort, data-quality reports, human XML review,
implementation smoke, effective A--F run configurations, and protocol sources.
It contains hashes and paths only; it never creates model predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "stage3-evidence-manifest-v1"
PROTOCOL = "camelyon16-stage3-mlac5-v1"
GROUPS = ("A", "B", "C", "D", "E", "F")
SEEDS = (9233, 9234, 9235)
CONFIG_FILES = {
    "A": "a_pooling.yaml",
    "B": "b_query.yaml",
    "C": "c_graph_query.yaml",
    "D": "d_graph_query_annotation.yaml",
    "E": "e_query_annotation.yaml",
    "F": "f_linear_patch_annotation.yaml",
}
METRIC_CASE_IDS = {
    "zero_reachability", "full_reachability", "partial_reachability",
    "exclusion_hole", "overlapping_grid_rejection", "budget_floor",
    "budget_ties", "small_lesion", "boundary_lesion", "normal_empty_truth",
    "repeated_peaks", "single_class_patch_metric_na",
}
SMOKE_SLIDES = ("normal_001", "tumor_002", "tumor_011", "tumor_031")
SMOKE_VARIANTS = ("query", "linear_patch")
LOCKED_SOURCE_FILES = (
    "modules/stage3_metrics.py",
    "modules/stage3_review.py",
    "modules/stage3_evidence.py",
    "modules/evidence_model.py",
    "modules/evidence_objectives.py",
    "modules/camelyon_dataset.py",
    "modules/graph_metrics.py",
    "scripts/train_stage1.py",
    "scripts/eval_stage3.py",
    "scripts/freeze_stage3.py",
    "scripts/build_stage3_evidence_manifest.py",
    "scripts/audit_stage3.py",
    "scripts/audit_stage3_inventory.py",
    "scripts/accept_stage3_metrics.py",
    "scripts/accept_stage3_scale.py",
    "scripts/smoke_stage3_ef.py",
    "scripts/render_stage3_xml_repairs.py",
    "scripts/accept_stage3_repairs.py",
    "scripts/summarize_stage3.py",
    "docs/camelyon16-stage3-evaluation-freeze.md",
)
ARTIFACT_KEYS = (
    "manifest", "audit", "repair_review_receipt", "h5_inventory",
    "metric_acceptance", "scale_acceptance", "ef_report",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{code} {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{code} {path}")
    return value


def _required_file(path: Path, code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{code} {resolved}")
    return resolved


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"slide_id", "split", "slide_label", "annotation_status", "xml_path", "h5_path"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError("STAGE3_MANIFEST_COLUMNS_INVALID")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"STAGE3_MANIFEST_UNREADABLE {path}") from error
    if len(rows) != 100:
        raise ValueError(f"STAGE3_MANIFEST_COHORT_COUNT {len(rows)}")
    slide_ids = [row.get("slide_id", "").strip() for row in rows]
    if not all(slide_ids) or len(set(slide_ids)) != 100:
        raise ValueError("STAGE3_MANIFEST_DUPLICATE_OR_EMPTY_SLIDE_ID")
    split_counts = {split: sum(row.get("split", "").strip() == split for row in rows)
                    for split in ("train", "val", "test")}
    if split_counts != {"train": 60, "val": 20, "test": 20}:
        raise ValueError(f"STAGE3_MANIFEST_SPLIT_COUNTS {split_counts}")
    labels: dict[int, int] = {0: 0, 1: 0}
    for row in rows:
        try:
            label = int(row.get("slide_label", ""))
        except (TypeError, ValueError) as error:
            raise ValueError("STAGE3_MANIFEST_LABEL_INVALID") from error
        if label not in labels:
            raise ValueError("STAGE3_MANIFEST_LABEL_INVALID")
        labels[label] += 1
        expected_status = "normal" if label == 0 else "full"
        if row.get("annotation_status", "").strip() != expected_status:
            raise ValueError(f"STAGE3_MANIFEST_ANNOTATION_STATUS {row['slide_id']}")
        xml = (row.get("xml_path") or "").strip()
        if (label == 1) != bool(xml):
            raise ValueError(f"STAGE3_MANIFEST_XML_SCOPE {row['slide_id']}")
        if not (row.get("h5_path") or "").strip():
            raise ValueError(f"STAGE3_MANIFEST_H5_PATH {row['slide_id']}")
    if labels != {0: 40, 1: 60}:
        raise ValueError(f"STAGE3_MANIFEST_LABEL_COUNTS {labels}")
    return rows


def _index_unique(rows: Any, name: str, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{name}_ROWS_INVALID")
    indexed: dict[tuple[str, ...], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name}_ROW_INVALID")
        try:
            key = tuple(str(row[field]).strip() for field in key_fields)
        except KeyError as error:
            raise ValueError(f"{name}_ROW_FIELDS_INVALID") from error
        if any(not part for part in key) or key in indexed:
            raise ValueError(f"{name}_DUPLICATE_OR_EMPTY_KEY {key}")
        indexed[key] = row
    return indexed


def _validate_audit(report: dict, rows: list[dict[str, str]]) -> None:
    if report.get("status") not in {"passed", "needs_review"} or report.get("issues"):
        raise ValueError("STAGE3_AUDIT_NOT_ACCEPTED")
    if report.get("slide_count") != 100 or report.get("cohort_slide_count") != 100:
        raise ValueError("STAGE3_AUDIT_SLIDE_COUNT")
    expected_splits = {split: sum(row["split"].strip() == split for row in rows)
                       for split in ("train", "val", "test")}
    if report.get("split_counts") != expected_splits:
        raise ValueError("STAGE3_AUDIT_SPLIT_COUNTS")
    expected = {(row["slide_id"].strip(), row["split"].strip()) for row in rows}
    observed = _index_unique(report.get("slides"), "STAGE3_AUDIT", ("slide_id", "split"))
    if set(observed) != expected or len(observed) != 100:
        raise ValueError("STAGE3_AUDIT_INVENTORY_MISMATCH")
    cases = report.get("repair_cases")
    if not isinstance(cases, list):
        raise ValueError("STAGE3_AUDIT_REPAIR_INVENTORY_INVALID")
    repairs = 0
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("repairs"), list) or not case["repairs"]:
            raise ValueError("STAGE3_AUDIT_REPAIR_INVENTORY_INVALID")
        repairs += len(case["repairs"])
    if int(report.get("repair_slide_count", -1)) != len(cases):
        raise ValueError("STAGE3_AUDIT_REPAIR_SLIDE_COUNT")
    if int(report.get("repair_annotation_count", -1)) != repairs:
        raise ValueError("STAGE3_AUDIT_REPAIR_ANNOTATION_COUNT")
    if bool(cases) != (report.get("status") == "needs_review"):
        raise ValueError("STAGE3_AUDIT_REPAIR_STATUS_MISMATCH")


def _validate_human_review(receipt_path: Path | None, manifest: Path, audit_path: Path,
                           report: dict) -> dict | None:
    if report["status"] == "needs_review":
        if receipt_path is None:
            raise ValueError("STAGE3_NEEDS_REVIEW_HUMAN_CONFIRMATION_REQUIRED")
        receipt_path = _required_file(receipt_path, "STAGE3_REVIEW_RECEIPT_MISSING")
        from modules.stage3_review import validate_review_receipt
        receipt = validate_review_receipt(receipt_path, manifest, audit_path)
        return receipt
    if receipt_path is not None:
        raise ValueError("STAGE3_UNEXPECTED_REPAIR_REVIEW_RECEIPT")
    return None


def _validate_inventory(report: dict, manifest_path: Path, rows: list[dict[str, str]]) -> None:
    if report.get("schema") != "stage3-h5-inventory-v1" or report.get("status") != "passed":
        raise ValueError("STAGE3_H5_INVENTORY_STATUS")
    if (report.get("slide_count") != 100 or report.get("passed_slide_count") != 100
            or report.get("failed_slide_count") != 0):
        raise ValueError("STAGE3_H5_INVENTORY_COUNTS")
    observed = _index_unique(report.get("slides"), "STAGE3_H5_INVENTORY", ("slide_id",))
    if len(observed) != 100:
        raise ValueError("STAGE3_H5_INVENTORY_COUNT")
    row_by_id = {row["slide_id"].strip(): row for row in rows}
    if {key[0] for key in observed} != set(row_by_id):
        raise ValueError("STAGE3_H5_INVENTORY_MANIFEST_MISMATCH")
    seen_h5_paths: set[Path] = set()
    for (slide_id,), item in observed.items():
        manifest_row = row_by_id[slide_id]
        if item.get("status") != "passed" or item.get("split") != manifest_row["split"].strip():
            raise ValueError(f"STAGE3_H5_INVENTORY_SCOPE {slide_id}")
        if item.get("failures", []) not in ([], None) or item.get("error_messages", []) not in ([], None):
            raise ValueError(f"STAGE3_H5_INVENTORY_FAILURES {slide_id}")
        checks = item.get("checks")
        if checks is not None and (not isinstance(checks, dict) or not checks
                                   or any(value is not True for value in checks.values())):
            raise ValueError(f"STAGE3_H5_INVENTORY_CHECKS {slide_id}")
        manifest_h5 = _resolve_path(manifest_row["h5_path"].strip(), manifest_path.parent)
        inventory_h5 = _resolve_path(str(item.get("h5_path", "")), manifest_path.parent)
        if manifest_h5 != inventory_h5 or not manifest_h5.is_file():
            raise ValueError(f"STAGE3_H5_INVENTORY_PATH {slide_id}")
        if manifest_h5 in seen_h5_paths:
            raise ValueError(f"STAGE3_H5_INVENTORY_PATH_DUPLICATE {slide_id}")
        seen_h5_paths.add(manifest_h5)
        expected_sha = item.get("h5_sha256")
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or sha256(manifest_h5) != expected_sha):
            raise ValueError(f"STAGE3_H5_HASH_MISMATCH {slide_id}")
        try:
            patch_count = int(item["patch_count"])
            patch_size = int(item["patch_size_level0"])
            dimensions = tuple(int(value) for value in item["wsi_dimensions"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"STAGE3_H5_INVENTORY_GEOMETRY {slide_id}") from error
        if patch_count < 1 or patch_size < 1 or len(dimensions) != 2 or min(dimensions) < 1:
            raise ValueError(f"STAGE3_H5_INVENTORY_GEOMETRY {slide_id}")


def _validate_metric_acceptance(report: dict) -> None:
    if report.get("schema") != "stage3-metric-acceptance-v1" or report.get("status") != "passed":
        raise ValueError("STAGE3_METRIC_ACCEPTANCE_STATUS")
    cases = _index_unique(report.get("cases"), "STAGE3_METRIC_CASES", ("case_id",))
    if (report.get("case_count") != 12 or len(cases) != 12
            or set(key[0] for key in cases) != METRIC_CASE_IDS
            or any(case.get("passed") is not True for case in cases.values())):
        raise ValueError("STAGE3_METRIC_ACCEPTANCE_CASES")
    if (report.get("required_case_count", 12) != 12
            or report.get("case_id_integrity_passed", True) is not True):
        raise ValueError("STAGE3_METRIC_ACCEPTANCE_INTEGRITY")


def _validate_scale(report: dict, rows: list[dict[str, str]]) -> None:
    if report.get("schema") != "stage3-train-val-scale-acceptance-v1" or report.get("status") != "passed":
        raise ValueError("STAGE3_SCALE_ACCEPTANCE_STATUS")
    if report.get("issues", []) not in ([], None):
        raise ValueError("STAGE3_SCALE_ACCEPTANCE_ISSUES")
    split_set = report.get("split_set")
    if not isinstance(split_set, list) or len(split_set) != 2 or set(split_set) != {"train", "val"}:
        raise ValueError("STAGE3_SCALE_ACCEPTANCE_SCOPE")
    expected_counts = {split: sum(row["split"].strip() == split for row in rows)
                       for split in ("train", "val", "test")}
    if report.get("manifest_split_counts") != expected_counts:
        raise ValueError("STAGE3_SCALE_MANIFEST_COUNTS")
    if report.get("split_slide_counts") != {"train": 60, "val": 20}:
        raise ValueError("STAGE3_SCALE_SPLIT_COUNTS")
    expected = {(row["slide_id"].strip(), row["split"].strip()) for row in rows
                if row["split"].strip() in {"train", "val"}}
    observed = _index_unique(report.get("slides"), "STAGE3_SCALE", ("slide_id", "split"))
    if set(observed) != expected or len(observed) != 80:
        raise ValueError("STAGE3_SCALE_INVENTORY_MISMATCH")
    if any(item.get("status", "passed") != "passed" for item in observed.values()):
        raise ValueError("STAGE3_SCALE_SLIDE_FAILED")
    try:
        requested = int(report["repetitions_requested"])
        completed = int(report["repetitions_completed"])
        counts = [int(value) for value in report["repetition_summary_slide_counts"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("STAGE3_SCALE_REPETITIONS_INVALID") from error
    digests = report.get("repetition_digests")
    if (requested < 2 or completed != requested or len(counts) != completed
            or any(value != 80 for value in counts)
            or not isinstance(digests, list) or len(digests) != completed
            or any(not _valid_sha(value) for value in digests)
            or len(set(digests)) != 1 or report.get("deterministic_digest") != digests[0]):
        raise ValueError("STAGE3_SCALE_REPETITIONS_MISMATCH")


def _resolve_prediction_file(report_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = (report_path.parent / candidate,
                  report_path.parent / "ef_exports" / candidate)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"STAGE3_EF_PREDICTION_MISSING {value}")


def _valid_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_ef_report(report: dict, report_path: Path, manifest_path: Path,
                        rows: list[dict[str, str]], inventory_report: dict) -> None:
    if report.get("schema") != "stage3-ef-smoke-v2" or report.get("status") != "passed":
        raise ValueError("STAGE3_EF_SMOKE_STATUS")
    if (report.get("slide_ids") != list(SMOKE_SLIDES) or report.get("variant_count") != 2):
        raise ValueError("STAGE3_EF_SMOKE_ALLOWLIST")
    expected_pairs = {(variant, slide_id) for variant in SMOKE_VARIANTS for slide_id in SMOKE_SLIDES}
    records = _index_unique(report.get("records"), "STAGE3_EF_RECORDS", ("variant", "slide_id"))
    if set(records) != expected_pairs or len(records) != 8:
        raise ValueError("STAGE3_EF_EXPORT_INVENTORY")
    row_by_id = {row["slide_id"].strip(): row for row in rows}
    inventory_by_id = {item["slide_id"]: item for item in inventory_report["slides"]}
    manifest_hash = sha256(manifest_path)
    prediction_paths: set[Path] = set()
    variant_identity: dict[str, tuple[str, str]] = {}
    for (variant, slide_id), record in records.items():
        # The manifest hash and exact slide ID bind older v2 reports that omit
        # the redundant per-record split; if present, the field must agree.
        record_split = record.get("split", row_by_id[slide_id]["split"].strip())
        if record_split != "train" or row_by_id[slide_id]["split"].strip() != "train":
            raise ValueError(f"STAGE3_EF_SPLIT {variant} {slide_id}")
        if record.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"STAGE3_EF_MANIFEST_HASH {variant} {slide_id}")
        if record.get("h5_sha256") != inventory_by_id[slide_id].get("h5_sha256"):
            raise ValueError(f"STAGE3_EF_H5_HASH {variant} {slide_id}")
        if ("patch_count" in record
                and record["patch_count"] != inventory_by_id[slide_id].get("patch_count")):
            raise ValueError(f"STAGE3_EF_PATCH_COUNT {variant} {slide_id}")
        if ("model_state_stage" in record
                and record["model_state_stage"] != "fresh_initialization"):
            raise ValueError(f"STAGE3_EF_MODEL_STATE_STAGE {variant} {slide_id}")
        if not _valid_sha(record.get("model_sha256")) or not _valid_sha(record.get("config_sha256")):
            raise ValueError(f"STAGE3_EF_MODEL_CONFIG_HASH {variant} {slide_id}")
        sources = record.get("score_sources")
        expected_patch_source = "evidence_scores" if variant == "query" else "raw_patch_logits"
        if not isinstance(sources, dict) or sources != {
                "slide_logits": "model_output",
                "slide_probabilities": "softmax(slide_logits)",
                "patch_scores": expected_patch_source,
                "query_attention": "model_output",
        }:
            raise ValueError(f"STAGE3_EF_SCORE_SOURCE {variant} {slide_id}")
        prediction_name = record.get("prediction_file")
        prediction_hash = record.get("prediction_sha256")
        if not isinstance(prediction_name, str) or not _valid_sha(prediction_hash):
            raise ValueError(f"STAGE3_EF_PREDICTION_METADATA {variant} {slide_id}")
        prediction_path = _resolve_prediction_file(report_path, prediction_name)
        if sha256(prediction_path) != prediction_hash:
            raise ValueError(f"STAGE3_EF_PREDICTION_HASH {variant} {slide_id}")
        if prediction_path in prediction_paths:
            raise ValueError(f"STAGE3_EF_PREDICTION_PATH_DUPLICATE {variant} {slide_id}")
        prediction_paths.add(prediction_path)
        identity = (record["model_sha256"], record["config_sha256"])
        previous = variant_identity.setdefault(variant, identity)
        if previous != identity:
            raise ValueError(f"STAGE3_EF_VARIANT_IDENTITY_MISMATCH {variant}")


def _validate_config_paths(config_paths: dict[str, Path]) -> dict[str, Path]:
    if not isinstance(config_paths, dict) or set(config_paths) != set(GROUPS):
        raise ValueError("STAGE3_CONFIG_GROUPS_INVALID")
    resolved: dict[str, Path] = {}
    for group in GROUPS:
        path = _required_file(Path(config_paths[group]), f"STAGE3_CONFIG_MISSING {group}")
        if path.name != CONFIG_FILES[group]:
            raise ValueError(f"STAGE3_CONFIG_NAME_INVALID {group}")
        resolved[group] = path
    if len({str(path) for path in resolved.values()}) != len(GROUPS):
        raise ValueError("STAGE3_CONFIG_PATH_DUPLICATE")
    return resolved


def _effective_configs(config_paths: dict[str, Path], seeds: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    if tuple(seeds) != SEEDS:
        raise ValueError(f"STAGE3_SEEDS_INVALID expected={SEEDS} observed={tuple(seeds)}")
    from scripts.train_stage1 import _config_payload, load_config

    templates = []
    runs = []
    for group in GROUPS:
        path = config_paths[group]
        templates.append({"group": group, "path": str(path), "sha256": sha256(path)})
        for seed in SEEDS:
            effective = _config_payload(load_config(path, seed=seed))
            runs.append({
                "group": group,
                "seed": seed,
                "template_sha256": sha256(path),
                "effective_config": effective,
                "config_sha256": _canonical_json_sha256(effective),
            })
    return templates, runs


def _validate_configs(configs: Any) -> None:
    if not isinstance(configs, dict) or set(configs) != {"templates", "runs"}:
        raise ValueError("STAGE3_CONFIGS_INVALID")
    templates = _index_unique(configs.get("templates"), "STAGE3_CONFIG_TEMPLATES", ("group",))
    if len(templates) != 6 or {key[0] for key in templates} != set(GROUPS):
        raise ValueError("STAGE3_CONFIG_TEMPLATES_INCOMPLETE")
    template_paths: set[str] = set()
    template_hashes: dict[str, str] = {}
    for group in GROUPS:
        template = templates[(group,)]
        try:
            path = _required_file(Path(template["path"]), f"STAGE3_CONFIG_MISSING {group}")
        except (KeyError, TypeError) as error:
            raise ValueError(f"STAGE3_CONFIG_TEMPLATE_INVALID {group}") from error
        if path.name != CONFIG_FILES[group] or str(path) in template_paths:
            raise ValueError(f"STAGE3_CONFIG_TEMPLATE_PATH_INVALID {group}")
        template_paths.add(str(path))
        digest = sha256(path)
        if template.get("sha256") != digest:
            raise ValueError(f"STAGE3_CONFIG_TEMPLATE_HASH {group}")
        template_hashes[group] = digest

    runs = _index_unique(configs.get("runs"), "STAGE3_CONFIG_RUNS", ("group", "seed"))
    expected_pairs = {(group, str(seed)) for group in GROUPS for seed in SEEDS}
    if set(runs) != expected_pairs or len(runs) != 18:
        raise ValueError("STAGE3_CONFIG_RUN_MATRIX")
    from scripts.train_stage1 import _config_payload, load_config
    for group in GROUPS:
        path = Path(templates[(group,)]["path"])
        for seed in SEEDS:
            row = runs[(group, str(seed))]
            effective = row.get("effective_config")
            if not isinstance(effective, dict):
                raise ValueError(f"STAGE3_CONFIG_EFFECTIVE_MISSING {group} {seed}")
            if row.get("template_sha256") != template_hashes[group]:
                raise ValueError(f"STAGE3_CONFIG_RUN_TEMPLATE_HASH {group} {seed}")
            canonical_hash = _canonical_json_sha256(effective)
            if row.get("config_sha256") != canonical_hash:
                raise ValueError(f"STAGE3_CONFIG_RUN_HASH {group} {seed}")
            resolved = _config_payload(load_config(path, seed=seed))
            if effective != resolved or canonical_hash != _canonical_json_sha256(resolved):
                raise ValueError(f"STAGE3_CONFIG_RUN_STALE {group} {seed}")


def _validate_locked_sources(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(LOCKED_SOURCE_FILES):
        raise ValueError("STAGE3_LOCKED_SOURCE_INVENTORY")
    for name in LOCKED_SOURCE_FILES:
        path = (ROOT / name).resolve()
        if not path.is_file() or not _valid_sha(value.get(name)) or sha256(path) != value[name]:
            raise ValueError(f"STAGE3_LOCKED_SOURCE_HASH {name}")


def _artifact_entry(path: Path | None, *, optional: bool = False) -> dict[str, Any]:
    if path is None:
        if not optional:
            raise ValueError("STAGE3_REQUIRED_ARTIFACT_MISSING")
        return {"path": None, "sha256": None, "status": "not_required"}
    resolved = _required_file(path, "STAGE3_ARTIFACT_MISSING")
    return {"path": str(resolved), "sha256": sha256(resolved), "status": "present"}


def _validate_artifacts(artifacts: Any) -> dict[str, Path | None]:
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_KEYS):
        raise ValueError("STAGE3_ARTIFACT_INVENTORY")
    paths: dict[str, Path | None] = {}
    for key in ARTIFACT_KEYS:
        entry = artifacts.get(key)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "status"}:
            raise ValueError(f"STAGE3_ARTIFACT_ENTRY_INVALID {key}")
        if key == "repair_review_receipt" and entry.get("status") == "not_required":
            if entry != {"path": None, "sha256": None, "status": "not_required"}:
                raise ValueError("STAGE3_ARTIFACT_OPTIONAL_ENTRY_INVALID")
            paths[key] = None
            continue
        if entry.get("status") != "present" or not isinstance(entry.get("path"), str):
            raise ValueError(f"STAGE3_ARTIFACT_ENTRY_MISSING {key}")
        path = _required_file(Path(entry["path"]), f"STAGE3_ARTIFACT_MISSING {key}")
        if not _valid_sha(entry.get("sha256")) or sha256(path) != entry["sha256"]:
            raise ValueError(f"STAGE3_ARTIFACT_SHA_MISMATCH {key}")
        paths[key] = path
    return paths


def build_evidence_manifest(
    manifest_path: Path,
    audit_path: Path,
    receipt_path: Path | None,
    inventory_path: Path,
    metric_acceptance_path: Path,
    scale_path: Path,
    ef_report_path: Path,
    config_paths: dict[str, Path],
    seeds: tuple[int, ...],
    output_path: Path,
) -> dict:
    """Validate real inputs and atomically write a canonical evidence manifest."""
    manifest = _required_file(manifest_path, "STAGE3_MANIFEST_MISSING")
    audit_file = _required_file(audit_path, "STAGE3_AUDIT_MISSING")
    inventory_file = _required_file(inventory_path, "STAGE3_H5_INVENTORY_MISSING")
    metric_file = _required_file(metric_acceptance_path, "STAGE3_METRIC_ACCEPTANCE_MISSING")
    scale_file = _required_file(scale_path, "STAGE3_SCALE_ACCEPTANCE_MISSING")
    ef_file = _required_file(ef_report_path, "STAGE3_EF_REPORT_MISSING")
    receipt = _required_file(receipt_path, "STAGE3_REVIEW_RECEIPT_MISSING") if receipt_path else None
    config_map = _validate_config_paths(config_paths)
    rows = _manifest_rows(manifest)
    audit_report = _read_json(audit_file, "STAGE3_AUDIT_UNREADABLE")
    _validate_audit(audit_report, rows)
    _validate_human_review(receipt, manifest, audit_file, audit_report)
    inventory_report = _read_json(inventory_file, "STAGE3_H5_INVENTORY_UNREADABLE")
    _validate_inventory(inventory_report, manifest, rows)
    metric_report = _read_json(metric_file, "STAGE3_METRIC_ACCEPTANCE_UNREADABLE")
    _validate_metric_acceptance(metric_report)
    scale_report = _read_json(scale_file, "STAGE3_SCALE_ACCEPTANCE_UNREADABLE")
    _validate_scale(scale_report, rows)
    ef_report = _read_json(ef_file, "STAGE3_EF_REPORT_UNREADABLE")
    _validate_ef_report(ef_report, ef_file, manifest, rows, inventory_report)
    templates, runs = _effective_configs(config_map, seeds)
    locked_sources = {name: sha256(ROOT / name) for name in LOCKED_SOURCE_FILES}
    receipt_entry = _artifact_entry(receipt, optional=True)
    payload = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "artifacts": {
            "manifest": _artifact_entry(manifest),
            "audit": _artifact_entry(audit_file),
            "repair_review_receipt": receipt_entry,
            "h5_inventory": _artifact_entry(inventory_file),
            "metric_acceptance": _artifact_entry(metric_file),
            "scale_acceptance": _artifact_entry(scale_file),
            "ef_report": _artifact_entry(ef_file),
        },
        "configs": {"templates": templates, "runs": runs},
        "seeds": list(seeds),
        "cohort": {
            "slide_count": 100,
            "split_counts": {"train": 60, "val": 20, "test": 20},
            "label_counts": {"negative": 40, "positive": 60},
        },
        "locked_sources": locked_sources,
    }
    output = Path(output_path).expanduser().resolve()
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
    return validate_evidence_manifest(output)


def validate_evidence_manifest(path: Path) -> dict:
    """Fail closed if any bound source, artifact, config, or scope has drifted."""
    evidence_path = _required_file(path, "STAGE3_EVIDENCE_MANIFEST_MISSING")
    evidence = _read_json(evidence_path, "STAGE3_EVIDENCE_MANIFEST_INVALID")
    if evidence.get("schema") != SCHEMA or evidence.get("protocol") != PROTOCOL:
        raise ValueError("STAGE3_EVIDENCE_SCHEMA_OR_PROTOCOL_INVALID")
    artifact_paths = _validate_artifacts(evidence.get("artifacts"))
    manifest = artifact_paths["manifest"]
    audit_file = artifact_paths["audit"]
    inventory_file = artifact_paths["h5_inventory"]
    metric_file = artifact_paths["metric_acceptance"]
    scale_file = artifact_paths["scale_acceptance"]
    ef_file = artifact_paths["ef_report"]
    if any(item is None for item in (manifest, audit_file, inventory_file, metric_file, scale_file, ef_file)):
        raise ValueError("STAGE3_EVIDENCE_REQUIRED_ARTIFACT_MISSING")
    rows = _manifest_rows(manifest)
    audit_report = _read_json(audit_file, "STAGE3_AUDIT_UNREADABLE")
    _validate_audit(audit_report, rows)
    receipt = artifact_paths["repair_review_receipt"]
    _validate_human_review(receipt, manifest, audit_file, audit_report)
    inventory_report = _read_json(inventory_file, "STAGE3_H5_INVENTORY_UNREADABLE")
    _validate_inventory(inventory_report, manifest, rows)
    _validate_metric_acceptance(_read_json(metric_file, "STAGE3_METRIC_ACCEPTANCE_UNREADABLE"))
    _validate_scale(_read_json(scale_file, "STAGE3_SCALE_ACCEPTANCE_UNREADABLE"), rows)
    _validate_ef_report(_read_json(ef_file, "STAGE3_EF_REPORT_UNREADABLE"), ef_file,
                        manifest, rows, inventory_report)
    if evidence.get("seeds") != list(SEEDS):
        raise ValueError("STAGE3_EVIDENCE_SEED_MATRIX")
    _validate_configs(evidence.get("configs"))
    _validate_locked_sources(evidence.get("locked_sources"))
    expected_cohort = {
        "slide_count": 100,
        "split_counts": {"train": 60, "val": 20, "test": 20},
        "label_counts": {"negative": 40, "positive": 60},
    }
    if evidence.get("cohort") != expected_cohort:
        raise ValueError("STAGE3_EVIDENCE_COHORT_INVALID")
    return evidence


__all__ = ["build_evidence_manifest", "validate_evidence_manifest", "sha256"]
