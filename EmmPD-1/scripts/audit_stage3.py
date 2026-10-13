"""Read-only Stage-3 H5/XML/WSI geometry acceptance; run before freezing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import openslide

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.stage3_metrics import area_metrics, overlap_matrix, parse_truth, peak_froc, validate_grid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_slide(row, truth=None):
    slide_id, split = row["slide_id"], row["split"]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"SPLIT_INVALID {slide_id}")
    with h5py.File(row["h5_path"], "r") as h5, openslide.OpenSlide(row["wsi_path"]) as wsi:
        if int(h5.attrs.get("complete", 0)) != 1:
            raise ValueError(f"H5_INCOMPLETE {slide_id}")
        size = int(h5.attrs["patch_size_level0"])
        coords = validate_grid(h5["coords"][:], size, wsi.dimensions)
        n = len(coords)
        if h5["features"].shape != (n, 1024):
            raise ValueError(f"FEATURE_SHAPE {slide_id}")
        for key in ("tumor_fraction", "patch_labels", "annotation_mask", "lesion_ids"):
            if h5[key].shape != (n,):
                raise ValueError(f"H5_LENGTH {slide_id} {key}")
        fractions = h5["tumor_fraction"][:]
        if not np.all(h5["annotation_mask"][:]):
            raise ValueError(f"ANNOTATION_NOT_FULL {slide_id}")
        if not np.array_equal(h5["patch_labels"][:], fractions >= 0.5):
            raise ValueError(f"PATCH_LABEL_MISMATCH {slide_id}")
        item = {"slide_id": slide_id, "split": split, "n": n,
                "patch_size_level0": size, "slide_dimensions": wsi.dimensions,
                "encoder_name": str(h5.attrs.get("encoder_name", "unknown"))}
        if int(row["slide_label"]) == 1:
            if row["annotation_status"] != "full":
                raise ValueError(f"ANNOTATION_STATUS {slide_id}")
            if truth is None:
                raise ValueError(f"XML_TRUTH_MISSING {slide_id}")
            item.update(lesion_count=len(truth.lesions), xml_groups=truth.group_counts,
                        repairs=list(truth.repairs))
            if split != "test":
                overlaps = overlap_matrix(coords, size, truth.lesions)
                if not np.allclose(overlaps.sum(axis=1) / (size * size), fractions, atol=1e-5):
                    raise ValueError(f"H5_XML_LABEL_MISMATCH {slide_id}")
                stats = area_metrics(coords, size, np.zeros(n), truth.lesions,
                                     dimensions=wsi.dimensions, overlaps=overlaps)
                item.update(reachable_macro=stats["reachable_macro"],
                            unreachable_lesions=sum(x == 0 for x in stats["reachable_per_lesion"]))
                scale_slide = {"slide_id": slide_id, "coords": coords, "size": size,
                               "scores": fractions, "lesions": truth.lesions}
            else:
                scale_slide = None
        else:
            if row["annotation_status"] != "normal" or np.any(fractions):
                raise ValueError(f"NORMAL_LABEL_INVALID {slide_id}")
            scale_slide = ({"slide_id": slide_id, "coords": coords, "size": size,
                            "scores": fractions, "lesions": ()} if split != "test" else None)
        return item, scale_slide


def audit(manifest: Path, output: Path):
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["slide_id"] for r in rows}) != len(rows):
        raise ValueError("MANIFEST_DUPLICATE_SLIDE")
    split_counts = {name: sum(r["split"] == name for r in rows) for name in ("train", "val", "test")}
    label_counts = {name: sum(int(r["slide_label"]) == name for r in rows) for name in (0, 1)}
    if len(rows) != 100 or split_counts != {"train": 60, "val": 20, "test": 20} or label_counts != {0: 40, 1: 60}:
        raise ValueError(f"MANIFEST_COHORT_CHANGED splits={split_counts} labels={label_counts}")
    results, scale_slides, issues, repair_cases = [], [], [], []
    started = time.perf_counter()
    for row in rows:
        truth = None
        if int(row["slide_label"]) == 1:
            try:
                xml_path = Path(row["xml_path"])
                truth = parse_truth(xml_path)
                if truth.repairs:
                    repair_cases.append({"slide_id": row["slide_id"], "split": row["split"],
                                         "xml_sha256": _sha256(xml_path),
                                         "lesion_count_after_repair": len(truth.lesions),
                                         "repairs": list(truth.repairs)})
            except Exception as error:
                issues.append({"slide_id": row["slide_id"], "split": row["split"],
                               "stage": "xml", "error_type": type(error).__name__,
                               "message": str(error)})
                continue
        try:
            item, scale_slide = _audit_slide(row, truth)
            results.append(item)
            if scale_slide is not None:
                scale_slides.append(scale_slide)
        except Exception as error:
            issues.append({"slide_id": row["slide_id"], "split": row["split"],
                           "stage": "h5_wsi", "error_type": type(error).__name__,
                           "message": str(error)})
    report = {
        "status": "failed" if issues else "needs_review" if repair_cases else "passed",
        "slide_count": len(results), "cohort_slide_count": len(rows),
        "split_counts": split_counts, "label_counts": label_counts,
        "patient_isolation_verified": False, "issues": issues,
        "repair_slide_count": len(repair_cases), "repair_cases": repair_cases,
        "repair_annotation_count": sum(len(item["repairs"]) for item in repair_cases),
        "total_seconds": time.perf_counter() - started, "slides": results,
    }
    if not issues:
        froc_started = time.perf_counter()
        try:
            froc = peak_froc(scale_slides)
            report.update(train_val_peak_count=len(froc["detections"]),
                          train_val_peak_seconds=time.perf_counter() - froc_started)
        except Exception as error:
            issues.append({"slide_id": "<train_val_peak_scale>", "split": "train_val",
                           "error_type": type(error).__name__, "message": str(error)})
            report["status"] = "failed"
    report["total_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    print(json.dumps({key: result[key] for key in
                      ("status", "slide_count", "repair_slide_count", "repair_annotation_count")},
                     ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2 if result["status"] == "failed" else 3)


if __name__ == "__main__":
    main()
