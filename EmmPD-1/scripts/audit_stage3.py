"""Read-only Stage-3 H5/XML/WSI geometry acceptance; run before freezing."""

from __future__ import annotations

import argparse
import csv
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


def audit(manifest: Path, output: Path):
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["slide_id"] for r in rows}) != len(rows):
        raise ValueError("MANIFEST_DUPLICATE_SLIDE")
    split_counts = {name: sum(r["split"] == name for r in rows) for name in ("train", "val", "test")}
    label_counts = {name: sum(int(r["slide_label"]) == name for r in rows) for name in (0, 1)}
    if len(rows) != 100 or split_counts != {"train": 60, "val": 20, "test": 20} or label_counts != {0: 40, 1: 60}:
        raise ValueError(f"MANIFEST_COHORT_CHANGED splits={split_counts} labels={label_counts}")
    results = []
    scale_slides = []
    started = time.perf_counter()
    for row in rows:
        slide_id, split = row["slide_id"], row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"SPLIT_INVALID {slide_id}")
        h5_path = Path(row["h5_path"])
        wsi_path = Path(row["wsi_path"])
        with h5py.File(h5_path, "r") as h5, openslide.OpenSlide(str(wsi_path)) as wsi:
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
                truth = parse_truth(row["xml_path"])
                if truth.repairs:
                    raise ValueError(f"XML_REPAIR_REQUIRES_REVIEW {slide_id} {truth.repairs}")
                item.update(lesion_count=len(truth.lesions), xml_groups=truth.group_counts)
                if split != "test":
                    overlaps = overlap_matrix(coords, size, truth.lesions)
                    if not np.allclose(overlaps.sum(axis=1) / (size * size), fractions, atol=1e-5):
                        raise ValueError(f"H5_XML_LABEL_MISMATCH {slide_id}")
                    stats = area_metrics(coords, size, np.zeros(n), truth.lesions,
                                         dimensions=wsi.dimensions, overlaps=overlaps)
                    item.update(reachable_macro=stats["reachable_macro"],
                                unreachable_lesions=sum(x == 0 for x in stats["reachable_per_lesion"]))
                    scale_slides.append({"slide_id": slide_id, "coords": coords, "size": size,
                                         "scores": fractions, "lesions": truth.lesions})
            else:
                if row["annotation_status"] != "normal" or np.any(fractions):
                    raise ValueError(f"NORMAL_LABEL_INVALID {slide_id}")
                if split != "test":
                    scale_slides.append({"slide_id": slide_id, "coords": coords, "size": size,
                                         "scores": fractions, "lesions": ()})
            results.append(item)
    froc_started = time.perf_counter()
    froc = peak_froc(scale_slides)
    report = {"status": "passed", "slide_count": len(results),
              "split_counts": split_counts, "label_counts": label_counts,
              "patient_isolation_verified": False,
              "train_val_peak_count": len(froc["detections"]),
              "train_val_peak_seconds": time.perf_counter() - froc_started,
              "total_seconds": time.perf_counter() - started,
              "slides": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "slide_count": result["slide_count"]}))


if __name__ == "__main__":
    main()
