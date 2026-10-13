"""Fail-closed evidence contracts for Stage-3 freeze and future evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml


SEEDS = (9233, 9234, 9235)
GROUPS = ("A", "B", "C", "D", "E", "F")
CONFIG_FILES = {
    "A": "a_pooling.yaml",
    "B": "b_query.yaml",
    "C": "c_graph_query.yaml",
    "D": "d_graph_query_annotation.yaml",
    "E": "e_query_annotation.yaml",
    "F": "f_linear_patch_annotation.yaml",
}
METRIC_CASES = (
    "zero_reachability", "full_reachability", "partial_reachability",
    "exclusion_hole", "overlapping_grid_rejection", "budget_floor",
    "budget_ties", "small_lesion", "boundary_lesion", "normal_empty_truth",
    "repeated_peaks", "single_class_patch_metric_na",
)
APPROVED_SMOKE_SLIDES = ("normal_001", "tumor_002", "tumor_011", "tumor_031")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _require_api(module_name: str, function_name: str):
    """Report an actionable RED assertion when the planned module is absent."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Stage-3 evidence API module {module_name}: {error}")
    function = getattr(module, function_name, None)
    assert callable(function), f"missing Stage-3 evidence API {module_name}.{function_name}"
    return module, function


def _build(case: dict, *, output_path: Path | None = None,
           receipt_path=Ellipsis, config_paths: dict[str, Path] | None = None):
    _, build_evidence_manifest = _require_api(
        "modules.stage3_evidence", "build_evidence_manifest"
    )
    if receipt_path is Ellipsis:
        receipt_path = case["receipt"]
    return build_evidence_manifest(
        manifest_path=case["manifest"],
        audit_path=case["audit"],
        receipt_path=receipt_path,
        inventory_path=case["inventory"],
        metric_acceptance_path=case["metric_acceptance"],
        scale_path=case["scale"],
        ef_report_path=case["ef_report"],
        config_paths=config_paths if config_paths is not None else case["configs"],
        seeds=SEEDS,
        output_path=output_path or case["evidence"],
    )


def _validate(path: Path) -> dict:
    _, validate_evidence_manifest = _require_api(
        "modules.stage3_evidence", "validate_evidence_manifest"
    )
    return validate_evidence_manifest(path)


def _load_evaluator_for_order_test():
    """Load the evaluator with heavyweight slide/geometry collaborators stubbed."""
    dataset_module = types.ModuleType("modules.camelyon_dataset")

    class Dataset:
        pass

    dataset_module.CamelyonDataset = Dataset
    dataset_module.EvidenceSlideInput = object
    model_module = types.ModuleType("modules.evidence_model")
    model_module.EvidenceModel = object
    model_module.EvidenceModelConfig = type("EvidenceModelConfig", (), {})
    review_module = types.ModuleType("modules.stage3_review")
    review_module.assert_repairs_approved = lambda *args, **kwargs: True
    metric_module = types.ModuleType("modules.stage3_metrics")
    for name in ("area_metrics", "overlap_matrix", "parse_truth", "patch_metrics",
                 "peak_froc", "validate_grid"):
        setattr(metric_module, name, lambda *args, **kwargs: None)
    evidence_module = types.ModuleType("modules.stage3_evidence")
    evidence_module.validate_evidence_manifest = lambda path: (_ for _ in ()).throw(
        ValueError("synthetic evidence rejected")
    )
    replacements = {
        "modules.camelyon_dataset": dataset_module,
        "modules.evidence_model": model_module,
        "modules.stage3_review": review_module,
        "modules.stage3_metrics": metric_module,
        "modules.stage3_evidence": evidence_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        source = Path(__file__).resolve().parents[1] / "scripts" / "eval_stage3.py"
        spec = importlib.util.spec_from_file_location("stage3_eval_order_under_test", source)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _valid_receipt(root: Path, manifest: Path, audit: Path, xml_path: Path) -> Path:
    """Create a synthetic, temporary v2 two-stage receipt, never a real one."""
    from modules.stage3_review import create_review_receipt

    repair_case = {
        "slide_id": "tumor_001",
        "split": "train",
        "xml_sha256": _sha256(xml_path),
        "repairs": [{
            "annotation": 4,
            "group": "Tumor",
            "reason": "Self-intersection",
            "after_type": "MultiPolygon",
            "polygon_parts": 2,
            "before_area": 30.0,
            "after_area": 25.0,
        }],
    }
    audit_report = json.loads(audit.read_text(encoding="utf-8"))
    audit_report["repair_cases"] = [repair_case]
    audit_report["repair_slide_count"] = 1
    audit_report["repair_annotation_count"] = 1
    audit_report["status"] = "needs_review"
    next(row for row in audit_report["slides"] if row["slide_id"] == "tumor_001")["repairs"] = repair_case["repairs"]
    _write_json(audit, audit_report)

    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / "render_manifest.json", {
        "audit_sha256": _sha256(audit),
        "manifest_sha256": _sha256(manifest),
        "splits": ["train"],
        "rendered_annotations": 1,
    })
    queue = review_dir / "review_queue.csv"
    fields = (
        "slide_id", "split", "xml_sha256", "annotation", "group", "reason",
        "after_type", "polygon_parts", "before_area", "after_area", "image",
        "review_decision", "review_note",
    )
    with queue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            **{key: repair_case[key] for key in ("slide_id", "split", "xml_sha256")},
            **repair_case["repairs"][0],
            "image": "tumor_001__ann_0004.png",
            "review_decision": "accept_make_valid",
            "review_note": "synthetic temporary geometry fixture",
        })

    identity = {
        "slide_id": "tumor_001", "split": "train",
        "xml_sha256": _sha256(xml_path), "annotation": 4,
    }
    assistant = _write_json(root / "assistant_review.json", {
        "schema": "stage3-xml-geometry-review-v1",
        "reviewer": "assistant geometry reviewer",
        "reviewed_at_utc": "2026-10-13T01:00:00Z",
        "queue_sha256": _sha256(queue),
        "results": [{**identity, "geometry_decision": "needs_user_confirmation",
                     "geometry_note": "synthetic temporary geometry check"}],
    })
    confirmation = _write_json(root / "user_confirmation.json", {
        "schema": "stage3-xml-user-confirmation-v1",
        "reviewer": "Xiaoxiao",
        "confirmed_at_utc": "2026-10-13T02:00:00Z",
        "queue_sha256": _sha256(queue),
        "assistant_review_sha256": _sha256(assistant),
        "acknowledge_geometry_only_scope": True,
        "acknowledge_unflagged_review": True,
        "resolutions": [{**identity, "decision": "accept_make_valid",
                         "review_note": "synthetic temporary confirmation"}],
    })
    receipt = root / "repair_review_receipt.json"
    create_review_receipt(manifest, audit, queue, assistant, confirmation, receipt)
    return receipt


@pytest.fixture
def evidence_case(tmp_path: Path):
    """Small synthetic files with exact cohort/scope counts and no real data."""
    root = tmp_path
    rows = []
    inventory_rows = []
    split_plan = (
        ("train", 60, 36, 1),
        ("val", 20, 12, 61),
        ("test", 20, 12, 81),
    )
    h5_dir = root / "h5"
    xml_dir = root / "xml"
    h5_dir.mkdir()
    xml_dir.mkdir()

    for split, count, positive_count, offset in split_plan:
        negative_count = count - positive_count
        for index in range(negative_count):
            slide_id = f"normal_{offset + index:03d}"
            h5_path = h5_dir / f"{slide_id}.h5"
            h5_path.write_bytes(f"synthetic h5 {slide_id}\n".encode())
            rows.append({
                "slide_id": slide_id, "split": split, "slide_label": 0,
                "annotation_status": "normal", "xml_path": "", "h5_path": str(h5_path),
                "wsi_path": str(root / "wsi" / f"{slide_id}.tif"),
            })
            inventory_rows.append({
                "slide_id": slide_id, "split": split, "h5_path": str(h5_path),
                "h5_sha256": _sha256(h5_path), "status": "passed", "patch_count": 1,
                "patch_size_level0": 10, "wsi_dimensions": [100, 100],
            })
        for index in range(positive_count):
            slide_id = f"tumor_{offset + index:03d}"
            h5_path = h5_dir / f"{slide_id}.h5"
            h5_path.write_bytes(f"synthetic h5 {slide_id}\n".encode())
            xml_path = xml_dir / f"{slide_id}.xml"
            xml_path.write_text("<annotations />\n", encoding="utf-8")
            rows.append({
                "slide_id": slide_id, "split": split, "slide_label": 1,
                "annotation_status": "full", "xml_path": str(xml_path), "h5_path": str(h5_path),
                "wsi_path": str(root / "wsi" / f"{slide_id}.tif"),
            })
            inventory_rows.append({
                "slide_id": slide_id, "split": split, "h5_path": str(h5_path),
                "h5_sha256": _sha256(h5_path), "status": "passed", "patch_count": 1,
                "patch_size_level0": 10, "wsi_dimensions": [100, 100],
            })

    # Keep the fixed four training smoke slides in the manifest/inventory.
    # The offsets above include normal_001 and tumor_001..tumor_036.
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit_slides = [{
        "slide_id": row["slide_id"], "split": row["split"], "n": 1,
        "patch_size_level0": 10, "slide_dimensions": [100, 100],
        "encoder_name": "synthetic", "feature_dim": 1024,
        "lesion_count": int(row["slide_label"]), "xml_groups": {}, "repairs": [],
    } for row in rows]
    audit = _write_json(root / "audit.json", {
        "status": "needs_review", "slide_count": 100, "cohort_slide_count": 100,
        "split_counts": {"train": 60, "val": 20, "test": 20},
        "label_counts": {"0": 40, "1": 60}, "issues": [],
        "repair_slide_count": 1, "repair_annotation_count": 1,
        "repair_cases": [], "slides": audit_slides,
    })
    inventory = _write_json(root / "inventory.json", {
        "schema": "stage3-h5-inventory-v1", "status": "passed", "slide_count": 100,
        "passed_slide_count": 100, "failed_slide_count": 0, "slides": inventory_rows,
    })
    metric_acceptance = _write_json(root / "metric_acceptance.json", {
        "schema": "stage3-metric-acceptance-v1", "status": "passed", "case_count": 12,
        "cases": [{
            "case_id": name, "input_summary": {}, "expected_values": {},
            "actual_values": {}, "tolerance": 1e-10, "passed": True,
        } for name in METRIC_CASES],
    })
    train_val = [row for row in rows if row["split"] in {"train", "val"}]
    scale_slides = [{"slide_id": row["slide_id"], "split": row["split"], "status": "passed"}
                    for row in train_val]
    digest = "a" * 64
    scale = _write_json(root / "scale.json", {
        "schema": "stage3-train-val-scale-acceptance-v1", "status": "passed",
        "split_set": ["train", "val"], "split_slide_counts": {"train": 60, "val": 20},
        "manifest_split_counts": {"train": 60, "val": 20, "test": 20},
        "slides": scale_slides, "repetitions_requested": 2, "repetitions_completed": 2,
        "repetition_summary_slide_counts": [80, 80], "repetition_digests": [digest, digest],
        "deterministic_digest": digest,
    })

    ef_dir = root / "ef_exports"
    ef_dir.mkdir()
    records = []
    for variant in ("query", "linear_patch"):
        for slide_id in APPROVED_SMOKE_SLIDES:
            prediction_file = f"{variant}_{slide_id}.pt"
            prediction_path = ef_dir / prediction_file
            prediction_path.write_bytes(f"synthetic export {variant} {slide_id}\n".encode())
            h5_record = next(item for item in inventory_rows if item["slide_id"] == slide_id)
            records.append({
                "slide_id": slide_id, "split": "train", "variant": variant,
                "prediction_file": prediction_file, "prediction_sha256": _sha256(prediction_path),
                "manifest_sha256": _sha256(manifest), "h5_sha256": h5_record["h5_sha256"],
                "model_sha256": "b" * 64 if variant == "query" else "d" * 64,
                "config_sha256": "c" * 64 if variant == "query" else "e" * 64,
                "score_sources": {
                    "slide_logits": "model_output",
                    "slide_probabilities": "softmax(slide_logits)",
                    "patch_scores": "evidence_scores" if variant == "query" else "raw_patch_logits",
                    "query_attention": "model_output",
                },
            })
    ef_report = _write_json(root / "ef_smoke.json", {
        "schema": "stage3-ef-smoke-v2", "status": "passed",
        "slide_ids": list(APPROVED_SMOKE_SLIDES), "variant_count": 2,
        "records": records,
        "interpretation": "Synthetic temporary exports for gate tests; not model performance.",
    })

    configs = {}
    config_root = root / "configs" / "stage1"
    config_root.mkdir(parents=True)
    repo_configs = Path(__file__).resolve().parents[1] / "configs" / "stage1"
    for group, filename in CONFIG_FILES.items():
        source = repo_configs / filename
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["manifest"] = str(manifest)
        payload["output_root"] = str(root / "runs" / group)
        destination = config_root / filename
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        configs[group] = destination

    # A synthetic single-case audit/receipt is tied to actual temporary files.
    repair_xml = xml_dir / "tumor_001.xml"
    receipt = _valid_receipt(root, manifest, audit, repair_xml)
    return {
        "root": root, "manifest": manifest, "audit": audit, "receipt": receipt,
        "inventory": inventory, "metric_acceptance": metric_acceptance,
        "scale": scale, "ef_report": ef_report, "ef_dir": ef_dir,
        "configs": configs, "evidence": root / "evidence_manifest.json",
    }


def test_builds_exact_complete_manifest_with_temporary_two_stage_review(evidence_case):
    manifest = _build(evidence_case)
    validated = _validate(evidence_case["evidence"])

    assert validated["schema"] == "stage3-evidence-manifest-v1"
    assert set(validated["artifacts"]) == {
        "manifest", "audit", "repair_review_receipt", "h5_inventory",
        "metric_acceptance", "scale_acceptance", "ef_report",
    }
    assert len(validated["configs"]["templates"]) == 6
    assert {row["group"] for row in validated["configs"]["templates"]} == set(GROUPS)
    runs = validated["configs"]["runs"]
    assert len(runs) == 18
    assert {(row["group"], row["seed"]) for row in runs} == {
        (group, seed) for group in GROUPS for seed in SEEDS
    }
    for group in GROUPS:
        group_hashes = {row["config_sha256"] for row in runs if row["group"] == group}
        assert len(group_hashes) == len(SEEDS), "seed override must affect each effective run config"
        assert all(len(value) == 64 for value in group_hashes)
    assert validated["locked_sources"]
    assert all(len(value) == 64 for value in validated["locked_sources"].values())
    assert manifest["schema"] == validated["schema"]

    receipt = json.loads(evidence_case["receipt"].read_text(encoding="utf-8"))
    responsibilities = receipt["review_responsibilities"]
    assert responsibilities["assistant"]["scope"] == "geometry_only"
    assert responsibilities["assistant"]["reviewer"] != responsibilities["user"]["reviewer"]
    assert responsibilities["user"]["scope"] == "confirm_unflagged_review_and_resolve_flags"
    assert responsibilities["assistant"]["reviewed_count"] == 1
    assert responsibilities["user"]["resolution_count"] == 1


def test_build_rejects_absent_required_artifact(evidence_case):
    missing_case = dict(evidence_case)
    missing_case["inventory"] = evidence_case["root"] / "missing_inventory.json"
    with pytest.raises((FileNotFoundError, ValueError)):
        _build(missing_case, output_path=evidence_case["root"] / "missing-build.json")


def test_validation_rejects_artifact_sha_drift(evidence_case):
    _build(evidence_case)
    evidence = json.loads(evidence_case["evidence"].read_text(encoding="utf-8"))
    evidence["artifacts"]["manifest"]["sha256"] = "0" * 64
    _write_json(evidence_case["evidence"], evidence)

    with pytest.raises(ValueError):
        _validate(evidence_case["evidence"])


def test_validation_rejects_locked_source_sha_drift(evidence_case):
    _build(evidence_case)
    evidence = json.loads(evidence_case["evidence"].read_text(encoding="utf-8"))
    source_name = next(iter(evidence["locked_sources"]))
    evidence["locked_sources"][source_name] = "0" * 64
    _write_json(evidence_case["evidence"], evidence)

    with pytest.raises(ValueError):
        _validate(evidence_case["evidence"])


@pytest.mark.parametrize("report_key", ["metric_acceptance", "scale"])
def test_build_rejects_nonpassing_acceptance_report(evidence_case, report_key):
    report_path = evidence_case[report_key]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    _write_json(report_path, report)

    with pytest.raises(ValueError):
        _build(evidence_case)


def test_build_rejects_failed_h5_inventory_status(evidence_case):
    report = json.loads(evidence_case["inventory"].read_text(encoding="utf-8"))
    report["status"] = "failed"
    report["failed_slide_count"] = 1
    _write_json(evidence_case["inventory"], report)

    with pytest.raises(ValueError):
        _build(evidence_case)


def test_validation_rejects_manifest_content_changed_after_build(evidence_case):
    _build(evidence_case)
    with evidence_case["manifest"].open("a", encoding="utf-8") as stream:
        stream.write("tampered,manifest,row\n")

    with pytest.raises(ValueError):
        _validate(evidence_case["evidence"])


def test_build_rejects_test_rows_in_train_validation_scale_report(evidence_case):
    report = json.loads(evidence_case["scale"].read_text(encoding="utf-8"))
    report["slides"].append({"slide_id": "test_001", "split": "test"})
    report["split_slide_counts"]["test"] = 1
    _write_json(evidence_case["scale"], report)

    with pytest.raises(ValueError):
        _build(evidence_case)


def test_build_rejects_incomplete_train_validation_scale_rows(evidence_case):
    report = json.loads(evidence_case["scale"].read_text(encoding="utf-8"))
    report["slides"].pop()
    _write_json(evidence_case["scale"], report)

    with pytest.raises(ValueError):
        _build(evidence_case)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_path"])
def test_build_rejects_nonbijective_a_to_f_template_map(evidence_case, mutation):
    configs = dict(evidence_case["configs"])
    if mutation == "missing":
        configs.pop("F")
    elif mutation == "extra":
        configs["G"] = next(iter(configs.values()))
    else:
        configs["B"] = configs["A"]

    with pytest.raises(ValueError):
        _build(evidence_case, config_paths=configs)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_validation_rejects_missing_extra_or_duplicate_template_rows(evidence_case, mutation):
    _build(evidence_case)
    evidence = json.loads(evidence_case["evidence"].read_text(encoding="utf-8"))
    templates = evidence["configs"]["templates"]
    if mutation == "missing":
        templates.pop()
    elif mutation == "extra":
        templates.append({**templates[0], "group": "G"})
    else:
        templates.append(dict(templates[0]))
    _write_json(evidence_case["evidence"], evidence)

    with pytest.raises(ValueError):
        _validate(evidence_case["evidence"])


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_seed"])
def test_validation_rejects_any_run_matrix_other_than_exact_18_pairs(evidence_case, mutation):
    _build(evidence_case)
    evidence = json.loads(evidence_case["evidence"].read_text(encoding="utf-8"))
    runs = evidence["configs"]["runs"]
    if mutation == "missing":
        runs.pop()
    elif mutation == "extra":
        runs.append({**runs[0], "group": "G", "seed": 9999})
    elif mutation == "duplicate":
        runs.append(dict(runs[0]))
    else:
        runs[0]["seed"] = 9999
    _write_json(evidence_case["evidence"], evidence)

    with pytest.raises(ValueError):
        _validate(evidence_case["evidence"])


@pytest.mark.parametrize("mutation", ["missing_row", "wrong_split", "bad_h5_hash"])
def test_build_rejects_incomplete_or_wrong_scope_100_slide_inventory(evidence_case, mutation):
    report = json.loads(evidence_case["inventory"].read_text(encoding="utf-8"))
    if mutation == "missing_row":
        report["slides"].pop()
        report["slide_count"] = 99
    elif mutation == "wrong_split":
        report["slides"][0]["split"] = "other"
    else:
        report["slides"][0]["h5_sha256"] = "0" * 64
    _write_json(evidence_case["inventory"], report)

    with pytest.raises(ValueError):
        _build(evidence_case)


@pytest.mark.parametrize("mutation", [
    "missing_row", "wrong_split", "duplicate_pair", "bad_prediction_hash",
])
def test_build_rejects_incomplete_or_wrong_scope_ef_exports_and_file_hashes(
    evidence_case, mutation
):
    report = json.loads(evidence_case["ef_report"].read_text(encoding="utf-8"))
    if mutation == "missing_row":
        report["records"].pop()
    elif mutation == "wrong_split":
        report["records"][0]["split"] = "val"
    elif mutation == "duplicate_pair":
        report["records"][0] = dict(report["records"][1])
    else:
        report["records"][0]["prediction_sha256"] = "0" * 64
    _write_json(evidence_case["ef_report"], report)

    with pytest.raises(ValueError):
        _build(evidence_case)


@pytest.mark.parametrize("missing_stage", ["receipt", "user_confirmation"])
def test_build_blocks_until_both_human_review_stages_exist(evidence_case, missing_stage):
    if missing_stage == "receipt":
        with pytest.raises(ValueError):
            _build(evidence_case, receipt_path=None)
    else:
        receipt = json.loads(evidence_case["receipt"].read_text(encoding="utf-8"))
        Path(receipt["user_confirmation_path"]).unlink()
        with pytest.raises((FileNotFoundError, ValueError)):
            _build(evidence_case)


def test_lock_rejects_evidence_or_artifact_digest_drift(evidence_case):
    _, lock_payload = _require_api("scripts.freeze_stage3", "lock_payload")
    _, verify_lock = _require_api("scripts.freeze_stage3", "verify_lock")
    _build(evidence_case)
    lock_path = evidence_case["root"] / "freeze.json"
    _write_json(lock_path, lock_payload(evidence_case["evidence"]))

    verified = verify_lock(lock_path, evidence_case["evidence"])
    assert verified

    evidence_text = evidence_case["evidence"].read_text(encoding="utf-8")
    evidence_case["evidence"].write_text(evidence_text + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_lock(lock_path, evidence_case["evidence"])
    evidence_case["evidence"].write_text(evidence_text, encoding="utf-8")

    with evidence_case["metric_acceptance"].open("a", encoding="utf-8") as stream:
        stream.write("changed after lock\n")
    with pytest.raises(ValueError):
        verify_lock(lock_path, evidence_case["evidence"])


def test_evaluator_rejects_evidence_before_loading_test_dataset(evidence_case, monkeypatch):
    import inspect
    evaluator = _load_evaluator_for_order_test()

    loaded = []
    evidence_checks = []
    monkeypatch.setattr(
        evaluator.CamelyonDataset, "from_manifest",
        classmethod(lambda cls, manifest, split: loaded.append((manifest, split))),
        raising=False,
    )
    assert "evidence_manifest_path" in inspect.signature(evaluator.evaluate).parameters

    if hasattr(evaluator, "validate_evidence_manifest"):
        def reject_direct_validation(path):
            evidence_checks.append(("manifest", Path(path)))
            raise ValueError("synthetic evidence rejected")

        monkeypatch.setattr(evaluator, "validate_evidence_manifest", reject_direct_validation)

    def reject_evidence_before_data(*args, **kwargs):
        evidence_checks.append(("lock", kwargs.get("evidence_manifest_path")))
        raise ValueError("synthetic evidence rejected")

    monkeypatch.setattr(evaluator, "verify_lock", reject_evidence_before_data)

    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.evaluate(
            checkpoint=evidence_case["root"] / "not-loaded.pt",
            manifest=evidence_case["manifest"], split="test",
            output_root=evidence_case["root"] / "eval",
            threshold_from=evidence_case["root"] / "threshold.json",
            freeze_file=evidence_case["root"] / "freeze.json",
            evidence_manifest_path=evidence_case["evidence"],
        )
    assert evidence_checks, "evaluator must verify the evidence manifest or its bound lock"
    assert loaded == [], "test dataset must not load before evidence verification"


def test_freeze_cli_requires_evidence_and_has_no_bypass_or_repeated_config():
    script = Path(__file__).resolve().parents[1] / "scripts" / "freeze_stage3.py"
    evidence_args = subprocess.run(
        [sys.executable, str(script), "--evidence-manifest", "missing.json", "--output", "lock.json"],
        capture_output=True, text=True,
    )
    assert evidence_args.returncode != 2, "new evidence CLI arguments must parse before file validation"
    assert "unrecognized arguments" not in evidence_args.stderr

    bypass_args = subprocess.run(
        [sys.executable, str(script), "--manifest", "m.csv", "--audit-report", "a.json",
         "--config", "a.yaml", "--config", "b.yaml", "--output", "lock.json",
         "--acceptance-confirmed"], capture_output=True, text=True,
    )
    assert "unrecognized arguments" in bypass_args.stderr

