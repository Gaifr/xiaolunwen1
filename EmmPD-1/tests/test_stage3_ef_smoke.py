"""Contract tests for the E/F Stage-3 smoke without opening real WSIs or using a GPU."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


def _load_smoke_module_without_real_slide_dependencies():
    """Import the CLI module with only its data/model collaborators stubbed."""
    camelyon_dataset = types.ModuleType("modules.camelyon_dataset")
    camelyon_dataset.CamelyonDataset = type("CamelyonDataset", (), {})
    camelyon_dataset.EvidenceSlideInput = object
    evidence_model = types.ModuleType("modules.evidence_model")
    evidence_model.EvidenceModel = object

    class EvidenceModelConfig:
        def __init__(self, **values):
            self.__dict__.update(values)

    evidence_model.EvidenceModelConfig = EvidenceModelConfig
    objectives = types.ModuleType("modules.evidence_objectives")

    def location_bce(scores, target, mask):
        return torch.nn.functional.binary_cross_entropy(
            scores[mask].float().clamp(1e-6, 1.0 - 1e-6), target[mask].float()
        )

    objectives.location_bce = location_bce
    replacements = {
        "modules.camelyon_dataset": camelyon_dataset,
        "modules.evidence_model": evidence_model,
        "modules.evidence_objectives": objectives,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        source = Path(__file__).resolve().parents[1] / "scripts" / "smoke_stage3_ef.py"
        spec = importlib.util.spec_from_file_location("stage3_ef_smoke_under_test", source)
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


smoke_stage3_ef = _load_smoke_module_without_real_slide_dependencies()


# Fixed by the user's prior approval for the Stage-3 E/F acceptance smoke.
APPROVED_SLIDES = ("normal_001", "tumor_002", "tumor_011", "tumor_031")
PATCH_COUNT = 6
FEATURE_DIM = 1024


@dataclass
class FakeSlide:
    slide_id: str
    slide_label: int
    features: torch.Tensor
    coords_level0: torch.Tensor
    patch_size_level0: int = 16
    tumor_fraction: torch.Tensor | None = None
    annotation_mask: torch.Tensor | None = None

    def __post_init__(self):
        if self.tumor_fraction is None:
            self.tumor_fraction = torch.tensor(
                [0.15, 0.25, 0.40, 0.00, 0.50, 0.10], dtype=torch.float32
            )
        if self.annotation_mask is None:
            self.annotation_mask = torch.ones(PATCH_COUNT, dtype=torch.bool)


class FakeDataset:
    def __init__(self, slides):
        self._slides = list(slides)
        self.audit = []

    def __iter__(self):
        return iter(self._slides)


class FakeEvidenceModel(torch.nn.Module):
    """Small differentiable stand-in whose patch outputs encode their input row."""

    behavior = "valid"
    output_bias = 0.0

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.loc_scale = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, slide):
        row_ids = slide.features[:, 0]
        raw_patch_logits = self.loc_scale * (row_ids / 10.0) + self.output_bias
        evidence_scores = torch.sigmoid(raw_patch_logits)
        slide_logits = torch.stack((self.cls_scale * -0.2 + self.output_bias,
                                    self.cls_scale * 0.3 - self.output_bias))

        if self.behavior == "classification_missing":
            slide_logits = torch.tensor([0.1, 0.4], device=row_ids.device)
        elif self.behavior == "classification_zero":
            slide_logits = torch.stack((self.cls_scale * 0.0 + 0.1,
                                        self.cls_scale * 0.0 + 0.4))
        elif self.behavior == "location_missing":
            evidence_scores = torch.full_like(row_ids, 0.6)
        elif self.behavior == "location_zero":
            evidence_scores = torch.sigmoid(self.loc_scale * 0.0 + row_ids / 20.0)
        elif self.behavior == "logits_nonfinite":
            slide_logits = torch.stack((self.cls_scale * torch.nan,
                                        self.cls_scale * 0.3))
        elif self.behavior == "scores_nonfinite":
            evidence_scores = evidence_scores.clone()
            evidence_scores[2] = torch.inf
            raw_patch_logits = raw_patch_logits.clone()
            raw_patch_logits[2] = torch.inf
        elif self.behavior == "patch_scores_short" and self.config.variant == "query":
            evidence_scores = evidence_scores[:-1]

        if self.config.variant == "linear_patch":
            attention = row_ids.new_empty((0, row_ids.numel()))
        else:
            attention = torch.stack((row_ids / 10.0, (row_ids + 1.0) / 10.0))

        if self.behavior == "query_attention_transposed" and self.config.variant == "query":
            attention = attention.transpose(0, 1)
        elif self.behavior == "query_attention_nonfinite" and self.config.variant == "query":
            attention = attention.clone()
            attention[0, 0] = torch.nan
        elif self.behavior == "linear_attention_nonempty" and self.config.variant == "linear_patch":
            attention = row_ids.new_zeros((1, row_ids.numel()))

        if self.config.variant == "linear_patch":
            # The E/F protocol reports raw patch logits for F and probabilities as
            # the localization scores used by the shared loss.
            raw_patch_logits = self.loc_scale * (row_ids / 10.0) + self.output_bias
        else:
            raw_patch_logits = None
        return SimpleNamespace(
            slide_logits=slide_logits,
            evidence_scores=evidence_scores,
            query_attention=attention,
            raw_patch_logits=raw_patch_logits,
        )


def _slide(slide_id: str, *, coords_rows: int = PATCH_COUNT) -> FakeSlide:
    # Feature column 0 is a row key; each exported score/attention column is
    # generated from that key, allowing the tests to catch coordinate drift.
    features = torch.zeros((PATCH_COUNT, FEATURE_DIM), dtype=torch.float32)
    features[:, 0] = torch.arange(PATCH_COUNT, dtype=torch.float32)
    coords = torch.stack(
        (1000 + torch.arange(coords_rows), 2000 + 3 * torch.arange(coords_rows)),
        dim=1,
    ).long()
    return FakeSlide(
        slide_id=slide_id,
        slide_label=int(slide_id.startswith("tumor_")),
        features=features,
        coords_level0=coords,
    )


def _write_h5(path: Path, *, feature_rows: int = PATCH_COUNT) -> None:
    features = np.zeros((feature_rows, FEATURE_DIM), dtype=np.float32)
    features[:, 0] = np.arange(feature_rows, dtype=np.float32)
    coords = np.stack((1000 + np.arange(feature_rows), 2000 + 3 * np.arange(feature_rows)), axis=1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)
        handle.attrs["patch_size_level0"] = 16


@pytest.fixture
def smoke_case(tmp_path, monkeypatch):
    rows = []
    slides = []
    for slide_id in APPROVED_SLIDES:
        h5_path = tmp_path / f"{slide_id}.h5"
        _write_h5(h5_path)
        rows.append({
            "slide_id": slide_id,
            "split": "train",
            "slide_label": int(slide_id.startswith("tumor_")),
            "h5_path": str(h5_path),
        })
        slides.append(_slide(slide_id))
    # A manifest row that is intentionally ineligible for the smoke profile.
    val_h5 = tmp_path / "tumor_999.h5"
    _write_h5(val_h5)
    rows.append({
        "slide_id": "tumor_999",
        "split": "val",
        "slide_label": 1,
        "h5_path": str(val_h5),
    })
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        smoke_stage3_ef.CamelyonDataset,
        "from_manifest",
        classmethod(lambda cls, manifest_path, split: FakeDataset(slides)),
        raising=False,
    )
    monkeypatch.setattr(smoke_stage3_ef, "EvidenceModel", FakeEvidenceModel)
    FakeEvidenceModel.behavior = "valid"
    FakeEvidenceModel.output_bias = 0.0
    return SimpleNamespace(
        manifest=manifest,
        rows=rows,
        slides=slides,
        output=tmp_path / "smoke",
    )


def _run(case, slide_ids=APPROVED_SLIDES):
    return smoke_stage3_ef.smoke(case.manifest, slide_ids, case.output, device="cpu")


@pytest.mark.parametrize(
    "slide_ids",
    [
        ("normal_001", "tumor_002", "tumor_011", "tumor_777"),
        ("normal_001", "tumor_002", "tumor_011", "tumor_011"),
        ("normal_001", "tumor_002", "tumor_011", "tumor_999"),
    ],
    ids=("wrong-allowlist", "duplicate-id", "non-training-slide"),
)
def test_smoke_rejects_any_slide_set_other_than_the_approved_train_allowlist(smoke_case, slide_ids):
    with pytest.raises((ValueError, RuntimeError), match="(?i)(slide|train|allowlist|smoke)"):
        _run(smoke_case, slide_ids)


@pytest.mark.parametrize(
    "mismatch",
    ["h5_feature_rows", "slide_coords_rows", "output_score_rows"],
    ids=("patch-count-mismatch", "coords-row-mismatch", "score-row-mismatch"),
)
def test_smoke_rejects_patch_count_or_coordinate_row_mismatch(smoke_case, mismatch):
    slide = smoke_case.slides[0]
    if mismatch == "h5_feature_rows":
        row = next(item for item in smoke_case.rows if item["slide_id"] == slide.slide_id)
        _write_h5(Path(row["h5_path"]), feature_rows=PATCH_COUNT + 1)
    elif mismatch == "slide_coords_rows":
        slide.coords_level0 = slide.coords_level0[:-1]
    else:
        FakeEvidenceModel.behavior = "patch_scores_short"

    rejected_errors = (ValueError, RuntimeError, IndexError) if mismatch == "output_score_rows" else (
        ValueError,
        RuntimeError,
    )
    with pytest.raises(rejected_errors, match="(?i)(patch|coord|row|shape|count|smoke|mask|tensor)"):
        _run(smoke_case)


@pytest.mark.parametrize(
    "behavior, message",
    [
        ("query_attention_transposed", "attention"),
        ("query_attention_nonfinite", "attention"),
        ("linear_attention_nonempty", "attention"),
    ],
)
def test_smoke_rejects_attention_with_invalid_shape_or_nonfinite_values(smoke_case, behavior, message):
    FakeEvidenceModel.behavior = behavior
    with pytest.raises((ValueError, RuntimeError), match=f"(?i){message}"):
        _run(smoke_case)


@pytest.mark.parametrize(
    "behavior, message",
    [("logits_nonfinite", "finite|nonfinite|loss"), ("scores_nonfinite", "finite|nonfinite|loss|score")],
)
def test_smoke_rejects_nonfinite_logits_or_patch_scores(smoke_case, behavior, message):
    FakeEvidenceModel.behavior = behavior
    with pytest.raises((ValueError, RuntimeError), match=f"(?i)({message})"):
        _run(smoke_case)


@pytest.mark.parametrize(
    "behavior, message",
    [
        ("classification_missing", "classification|class|gradient"),
        ("classification_zero", "classification|class|gradient"),
        ("location_missing", "location|gradient|require grad"),
        ("location_zero", "location|gradient"),
    ],
)
def test_smoke_rejects_missing_or_zero_classification_and_localization_gradients(
    smoke_case, behavior, message
):
    FakeEvidenceModel.behavior = behavior
    with pytest.raises((ValueError, RuntimeError), match=f"(?i)({message})"):
        _run(smoke_case)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_success_writes_exact_eight_v2_records_hashes_sources_and_row_aligned_tensors(smoke_case):
    result = _run(smoke_case)

    assert isinstance(result, dict)
    assert result["schema"] == "stage3-ef-smoke-v2"
    assert result["status"] == "passed"
    assert set(result["slide_ids"]) == set(APPROVED_SLIDES)
    assert len(result["slide_ids"]) == len(APPROVED_SLIDES)
    assert result["variant_count"] == 2
    records = result["records"]
    assert len(records) == 2 * len(APPROVED_SLIDES)
    pairs = {(row["variant"], row["slide_id"]) for row in records}
    assert pairs == {
        (variant, slide_id)
        for variant in ("query", "linear_patch")
        for slide_id in APPROVED_SLIDES
    }

    by_slide = {slide.slide_id: slide for slide in smoke_case.slides}
    manifest_hash = _sha256(smoke_case.manifest)
    for variant in ("query", "linear_patch"):
        variant_records = [row for row in records if row["variant"] == variant]
        assert {row["manifest_sha256"] for row in variant_records} == {manifest_hash}
        assert len({row["model_sha256"] for row in variant_records}) == 1
        assert len({row["config_sha256"] for row in variant_records}) == 1
    for record in records:
        variant = record["variant"]
        slide_id = record["slide_id"]
        slide = by_slide[slide_id]
        expected_sources = {
            "slide_logits": "model_output",
            "slide_probabilities": "softmax(slide_logits)",
            "patch_scores": "evidence_scores" if variant == "query" else "raw_patch_logits",
            "query_attention": "model_output",
        }
        assert record["score_sources"] == expected_sources
        for name in ("manifest_sha256", "h5_sha256", "model_sha256", "config_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", record[name]), name
        assert record["manifest_sha256"] == manifest_hash
        h5_path = Path(next(row["h5_path"] for row in smoke_case.rows if row["slide_id"] == slide_id))
        assert record["h5_sha256"] == _sha256(h5_path)
        assert np.isfinite(record["loss_cls"])
        assert np.isfinite(record["loss_loc"])

        prediction_path = smoke_case.output / record["prediction_file"]
        assert prediction_path.is_file()
        assert re.fullmatch(r"[0-9a-f]{64}", record["prediction_sha256"])
        assert record["prediction_sha256"] == _sha256(prediction_path)
        prediction = torch.load(prediction_path, map_location="cpu", weights_only=False)
        assert {
            "slide_id", "variant", "slide_logits", "slide_probabilities",
            "coords_level0", "patch_scores", "query_attention",
        } <= prediction.keys()
        assert prediction["slide_id"] == slide_id
        assert prediction["variant"] == variant
        assert torch.equal(prediction["coords_level0"], slide.coords_level0)
        assert prediction["patch_scores"].shape == (PATCH_COUNT,)
        assert torch.isfinite(prediction["patch_scores"]).all()
        row_keys = slide.features[:, 0]
        expected_patch_scores = (
            torch.sigmoid(torch.tensor(0.8) * (row_keys / 10.0))
            if variant == "query"
            else torch.tensor(0.8) * (row_keys / 10.0)
        )
        assert torch.allclose(prediction["patch_scores"], expected_patch_scores)
        expected_attention = (
            torch.stack((row_keys / 10.0, (row_keys + 1.0) / 10.0))
            if variant == "query"
            else torch.empty((0, PATCH_COUNT))
        )
        assert prediction["query_attention"].shape == expected_attention.shape
        assert torch.allclose(prediction["query_attention"], expected_attention)
        assert torch.isfinite(prediction["slide_logits"]).all()
        assert torch.isfinite(prediction["slide_probabilities"]).all()
        assert torch.allclose(prediction["slide_probabilities"], prediction["slide_logits"].softmax(dim=-1))

    report_path = smoke_case.output / "smoke_report.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == result


def _assert_failed_rerun_invalidated_success(output: Path) -> None:
    report_path = output / "smoke_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report.get("status") != "passed", "failed rerun left a stale passed report"
    else:
        assert not list(output.glob("*.pt")), "failed rerun left orphaned old prediction exports"


def test_pre_export_validation_failure_invalidates_a_previous_success(smoke_case):
    first = _run(smoke_case)
    assert first["status"] == "passed"
    assert len(list(smoke_case.output.glob("*.pt"))) == 8

    FakeEvidenceModel.behavior = "query_attention_nonfinite"
    with pytest.raises(ValueError, match="(?i)attention"):
        _run(smoke_case)

    _assert_failed_rerun_invalidated_success(smoke_case.output)


def test_save_failure_cleans_temporary_file_and_invalidates_previous_success(smoke_case, monkeypatch):
    first = _run(smoke_case)
    assert first["status"] == "passed"
    assert len(list(smoke_case.output.glob("*.pt"))) == 8

    real_torch_save = torch.save
    save_calls = 0

    def save_then_fail_on_second_export(value, path, *args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        real_torch_save(value, path, *args, **kwargs)
        if save_calls == 2:
            raise OSError("INJECTED_STAGE3_EXPORT_WRITE_FAILURE")

    monkeypatch.setattr(smoke_stage3_ef.torch, "save", save_then_fail_on_second_export)
    FakeEvidenceModel.output_bias = 0.2
    with pytest.raises(OSError, match="INJECTED_STAGE3_EXPORT_WRITE_FAILURE"):
        _run(smoke_case)

    assert not list(smoke_case.output.rglob("*.tmp")), "failed save left a temporary export"
    _assert_failed_rerun_invalidated_success(smoke_case.output)
