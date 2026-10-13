"""CAMELYON slide-level evidence data contract and validated HDF5 loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree

import h5py
import numpy as np
import shapely
import torch
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from torch import Tensor


ANNOTATION_STATUSES = frozenset({"full", "normal", "partial", "missing"})
REQUIRED_H5_DATASETS = (
    "features",
    "coords",
    "tumor_fraction",
    "patch_labels",
    "annotation_mask",
    "lesion_ids",
)


@dataclass(frozen=True)
class EvidenceSlideInput:
    slide_id: str
    patient_id: str
    split: str
    slide_label: int
    features: Tensor
    coords_level0: Tensor
    patch_size_level0: int
    tumor_fraction: Tensor
    patch_labels: Tensor
    annotation_mask: Tensor
    lesion_ids: Tensor
    annotation_status: str


@dataclass(frozen=True)
class LesionPolygon:
    lesion_id: int
    geometry: BaseGeometry


@dataclass(frozen=True)
class PatchAnnotations:
    tumor_fraction: Tensor
    patch_labels: Tensor
    annotation_mask: Tensor
    lesion_ids: Tensor


def _valid_polygon(points: list[tuple[float, float]]) -> BaseGeometry | None:
    if len(points) < 3:
        return None
    geometry: BaseGeometry = Polygon(points)
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    if geometry.is_empty:
        return None
    polygons = [part for part in _polygon_parts(geometry) if part.area > 0]
    return unary_union(polygons) if polygons else None


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [
            part
            for child in geometry.geoms
            for part in _polygon_parts(child)
        ]
    return []


def _annotation_role(annotation: ElementTree.Element) -> str | None:
    group = (annotation.attrib.get("PartOfGroup") or "").strip().lower()
    name = (annotation.attrib.get("Name") or "").strip().lower()
    combined = f"{group} {name}"
    if "exclusion" in combined:
        return "exclusion"
    if "tumor" in combined or "tumour" in combined:
        return "tumor"
    return None


def parse_camelyon_xml(path: Path | str) -> list[LesionPolygon]:
    """Parse ASAP XML and return deterministic, exclusion-subtracted lesions."""
    root = ElementTree.parse(path).getroot()
    tumors: list[BaseGeometry] = []
    exclusions: list[BaseGeometry] = []
    for annotation in root.findall(".//Annotation"):
        role = _annotation_role(annotation)
        if role is None:
            continue
        coordinates = sorted(
            annotation.findall("./Coordinates/Coordinate"),
            key=lambda item: int(item.attrib.get("Order", "0")),
        )
        points = [
            (float(item.attrib["X"]), float(item.attrib["Y"]))
            for item in coordinates
        ]
        geometry = _valid_polygon(points)
        if geometry is None:
            continue
        (tumors if role == "tumor" else exclusions).append(geometry)

    if not tumors:
        return []
    tumor_geometry = unary_union(tumors)
    if exclusions:
        tumor_geometry = tumor_geometry.difference(unary_union(exclusions))
    if not tumor_geometry.is_valid:
        tumor_geometry = shapely.make_valid(tumor_geometry)

    parts = [part for part in _polygon_parts(tumor_geometry) if part.area > 0]
    parts.sort(key=lambda part: (part.bounds[1], part.bounds[0], part.area))
    return [LesionPolygon(index, part) for index, part in enumerate(parts)]


def label_patch_grid(
    lesions: Sequence[LesionPolygon],
    coords_level0: Tensor,
    patch_size_level0: int,
    positive_threshold: float,
) -> PatchAnnotations:
    """Compute continuous tumor overlap and lesion identity for each patch."""
    if not 0.0 <= positive_threshold <= 1.0:
        raise ValueError(f"POSITIVE_THRESHOLD {positive_threshold}")
    if patch_size_level0 < 1:
        raise ValueError(f"PATCH_SIZE_LEVEL0 {patch_size_level0}")
    if coords_level0.ndim != 2 or coords_level0.shape[1] != 2:
        raise ValueError(f"COORD_SHAPE {tuple(coords_level0.shape)}")

    fractions: list[float] = []
    instance_ids: list[int] = []
    patch_area = float(patch_size_level0 * patch_size_level0)
    for x_value, y_value in coords_level0.detach().cpu().tolist():
        patch = shapely.box(
            float(x_value),
            float(y_value),
            float(x_value + patch_size_level0),
            float(y_value + patch_size_level0),
        )
        overlaps = [patch.intersection(lesion.geometry).area for lesion in lesions]
        total = min(1.0, sum(overlaps) / patch_area)
        fractions.append(total)
        if overlaps and max(overlaps) > 0:
            instance_ids.append(lesions[int(np.argmax(overlaps))].lesion_id)
        else:
            instance_ids.append(-1)

    tumor_fraction = torch.tensor(fractions, dtype=torch.float32)
    return PatchAnnotations(
        tumor_fraction=tumor_fraction,
        patch_labels=(tumor_fraction >= positive_threshold).long(),
        annotation_mask=torch.ones(len(fractions), dtype=torch.bool),
        lesion_ids=torch.tensor(instance_ids, dtype=torch.long),
    )


def validate_slide(slide: EvidenceSlideInput) -> None:
    """Reject inconsistent bags before they can reach a model or metric."""
    if slide.annotation_status not in ANNOTATION_STATUSES:
        raise ValueError(f"ANNOTATION_STATUS {slide.annotation_status}")
    if slide.slide_label not in (0, 1):
        raise ValueError(f"SLIDE_LABEL {slide.slide_label}")
    if slide.patch_size_level0 < 1:
        raise ValueError(f"PATCH_SIZE_LEVEL0 {slide.patch_size_level0}")

    if slide.features.ndim != 2:
        raise ValueError(f"FEATURE_SHAPE {tuple(slide.features.shape)}")
    node_count = slide.features.shape[0]
    if slide.coords_level0.shape != (node_count, 2):
        raise ValueError(f"COORD_SHAPE {tuple(slide.coords_level0.shape)}")
    fields = {
        "tumor_fraction": slide.tumor_fraction,
        "patch_labels": slide.patch_labels,
        "annotation_mask": slide.annotation_mask,
        "lesion_ids": slide.lesion_ids,
    }
    for name, value in fields.items():
        if value.shape != (node_count,):
            raise ValueError(f"EVIDENCE_LENGTH {name}={tuple(value.shape)} nodes={node_count}")

    if not bool(torch.isfinite(slide.features).all()):
        raise ValueError("NON_FINITE features")
    if not bool(torch.isfinite(slide.tumor_fraction).all()):
        raise ValueError("NON_FINITE tumor_fraction")
    if bool(((slide.tumor_fraction < 0) | (slide.tumor_fraction > 1)).any()):
        raise ValueError("TUMOR_FRACTION_RANGE")
    if not set(slide.patch_labels.unique().tolist()).issubset({0, 1}):
        raise ValueError("PATCH_LABEL_RANGE")
    if bool((slide.coords_level0 < 0).any()):
        raise ValueError("NEGATIVE_COORD")

    if slide.annotation_status in {"partial", "missing"} and bool(
        slide.annotation_mask.any()
    ):
        raise ValueError("PARTIAL_ANNOTATION_MASK")
    if slide.annotation_status == "normal":
        if bool(slide.patch_labels.any()) or bool((slide.tumor_fraction > 0).any()):
            raise ValueError("NORMAL_POSITIVE_PATCH")
        if not bool(slide.annotation_mask.all()):
            raise ValueError("NORMAL_ANNOTATION_MASK")


def _read_h5(path: Path, row: dict[str, str]) -> EvidenceSlideInput:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_H5_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"H5_DATASET_MISSING {','.join(missing)}")
        stored_patch_size = int(handle.attrs.get("patch_size_level0", 0))
        manifest_patch_size = int(row.get("patch_size_level0") or 0)
        patch_size_level0 = stored_patch_size or manifest_patch_size
        slide = EvidenceSlideInput(
            slide_id=row["slide_id"].strip(),
            patient_id=(row.get("patient_id") or row["slide_id"]).strip(),
            split=row["split"].strip(),
            slide_label=int(row["slide_label"]),
            features=torch.from_numpy(np.asarray(handle["features"], dtype=np.float32)),
            coords_level0=torch.from_numpy(np.asarray(handle["coords"], dtype=np.int64)),
            patch_size_level0=patch_size_level0,
            tumor_fraction=torch.from_numpy(
                np.asarray(handle["tumor_fraction"], dtype=np.float32)
            ),
            patch_labels=torch.from_numpy(np.asarray(handle["patch_labels"], dtype=np.int64)),
            annotation_mask=torch.from_numpy(
                np.asarray(handle["annotation_mask"], dtype=np.bool_)
            ),
            lesion_ids=torch.from_numpy(np.asarray(handle["lesion_ids"], dtype=np.int64)),
            annotation_status=row["annotation_status"].strip(),
        )
    validate_slide(slide)
    return slide


class CamelyonDataset:
    """Validated in-memory slide bags selected from a manifest split."""

    def __init__(
        self,
        items: Sequence[EvidenceSlideInput],
        audit: Sequence[dict[str, str]],
    ) -> None:
        self._items = list(items)
        self.audit = list(audit)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> EvidenceSlideInput:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    @property
    def labels(self) -> list[int]:
        return [item.slide_label for item in self._items]

    def class_counts(self) -> dict[int, int]:
        return {label: self.labels.count(label) for label in (0, 1)}

    @classmethod
    def from_manifest(cls, manifest: Path | str, split: str) -> "CamelyonDataset":
        manifest = Path(manifest)
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        items: list[EvidenceSlideInput] = []
        audit: list[dict[str, str]] = []
        for row in rows:
            if (row.get("split") or "").strip() != split:
                continue
            slide_id = (row.get("slide_id") or "").strip()
            raw_path = Path((row.get("h5_path") or "").strip())
            path = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
            if not path.exists():
                audit.append(
                    {
                        "code": "FEATURE_MISSING",
                        "slide_id": slide_id,
                        "message": f"missing H5 {path}",
                    }
                )
                continue
            try:
                items.append(_read_h5(path, row))
            except ValueError as error:
                code = str(error).split(maxsplit=1)[0]
                if code == "PARTIAL_ANNOTATION_MASK":
                    raise
                audit.append(
                    {"code": code, "slide_id": slide_id, "message": str(error)}
                )
            except (KeyError, OSError, TypeError) as error:
                audit.append(
                    {
                        "code": "H5_INVALID",
                        "slide_id": slide_id,
                        "message": str(error),
                    }
                )
        return cls(items, audit)


def collate_evidence_batch(
    batch: Sequence[EvidenceSlideInput],
) -> list[EvidenceSlideInput]:
    """Keep each WSI as an independent variable-size graph."""
    return list(batch)
