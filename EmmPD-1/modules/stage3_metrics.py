"""Frozen CAMELYON16 Stage-3 geometry and detection definitions.

All coordinates are WSI level-0 pixels. XML is evaluation truth, never an
input to candidate selection or model inference.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Truth:
    lesions: tuple
    group_counts: dict[str, int]
    repairs: tuple[dict, ...]


def _parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 0 else []
    return [part for child in getattr(geometry, "geoms", ()) for part in _parts(child)]


def _components(geometry):
    """Merge polygon parts that share even a single boundary point."""
    parts = _parts(geometry)
    if not parts:
        return ()
    tree = STRtree(parts)
    parent = list(range(len(parts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, part in enumerate(parts):
        for j in tree.query(part, predicate="intersects"):
            a, b = root(i), root(int(j))
            parent[b] = a
    groups = defaultdict(list)
    for i, part in enumerate(parts):
        groups[root(i)].append(part)
    result = [unary_union(group) for group in groups.values()]
    result.sort(key=lambda g: (g.bounds[1], g.bounds[0], g.area, g.normalize().wkb_hex))
    return tuple(result)


def parse_truth(path: str | Path) -> Truth:
    root = ElementTree.parse(path).getroot()
    declared = {g.get("Name") for g in root.findall(".//AnnotationGroups/Group")}
    declared.discard(None)
    if not declared:
        raise ValueError("XML_GROUP_DECLARATIONS_MISSING")
    # ASAP names are semantic in the local files; numeric IDs follow README16.
    roles = {"tumor": "tumor", "tumour": "tumor", "exclusion": "exclusion",
             "_0": "tumor", "_1": "tumor", "_2": "exclusion"}
    unknown = {group for group in declared if group.casefold() not in roles}
    if unknown:
        raise ValueError(f"XML_UNKNOWN_GROUP {sorted(unknown)}")
    tumors, exclusions, repairs = [], [], []
    counts = {group: 0 for group in sorted(declared)}
    for index, ann in enumerate(root.findall(".//Annotations/Annotation")):
        group = ann.get("PartOfGroup")
        if group not in declared:
            raise ValueError(f"XML_UNDECLARED_GROUP annotation={index} group={group}")
        counts[group] += 1
        coords = sorted(ann.findall("./Coordinates/Coordinate"),
                        key=lambda c: int(c.get("Order", "0")))
        points = [(float(c.attrib["X"]), float(c.attrib["Y"])) for c in coords]
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"XML_INVALID_COORDINATES annotation={index}")
        raw = Polygon(points)
        geom = raw if raw.is_valid else shapely.make_valid(raw)
        if not raw.is_valid:
            repairs.append({"annotation": index, "before_area": raw.area,
                            "after_area": geom.area, "after_type": geom.geom_type})
        polygons = _parts(geom)
        if not polygons:
            raise ValueError(f"XML_NO_POSITIVE_AREA annotation={index}")
        (tumors if roles[group.casefold()] == "tumor" else exclusions).extend(polygons)
    if not tumors:
        raise ValueError("XML_NO_TUMOR")
    net = unary_union(tumors).difference(unary_union(exclusions) if exclusions else shapely.GeometryCollection())
    if not net.is_valid:
        raise ValueError("XML_NET_INVALID")
    lesions = _components(net)
    if not lesions:
        raise ValueError("XML_NO_NET_LESIONS")
    return Truth(lesions, counts, tuple(repairs))


def validate_grid(coords, size: int, dimensions=None):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0 or size < 1:
        raise ValueError("GRID_SHAPE_OR_SIZE")
    if not np.issubdtype(coords.dtype, np.integer) or np.any(coords < 0):
        raise ValueError("GRID_COORDINATES")
    if np.any(coords % size) or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("GRID_OVERLAP_OR_DUPLICATE")
    if dimensions is not None and (np.any(coords[:, 0] + size > dimensions[0]) or
                                   np.any(coords[:, 1] + size > dimensions[1])):
        raise ValueError("GRID_OUTSIDE_WSI")
    return coords


def overlap_matrix(coords, size, lesions):
    tree = STRtree(lesions)
    overlaps = np.zeros((len(coords), len(lesions)), dtype=np.float64)
    for i, (x, y) in enumerate(coords):
        patch = box(int(x), int(y), int(x) + size, int(y) + size)
        for j in tree.query(patch, predicate="intersects"):
            overlaps[i, int(j)] = patch.intersection(lesions[int(j)]).area
    return overlaps


def area_metrics(coords, size, scores, lesions, budget=0.05, dimensions=None, overlaps=None):
    coords = validate_grid(coords, size, dimensions)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(coords),) or not np.isfinite(scores).all():
        raise ValueError("SCORES_INVALID")
    if not 0 < budget <= 1 or not lesions:
        raise ValueError("AREA_INPUT_INVALID")
    n, k = len(coords), int(np.floor(budget * len(coords)))
    rank = np.lexsort((np.arange(n), coords[:, 0], coords[:, 1], -scores))
    selected = rank[:k]
    overlaps = overlap_matrix(coords, size, lesions) if overlaps is None else np.asarray(overlaps)
    if overlaps.shape != (n, len(lesions)):
        raise ValueError("OVERLAP_SHAPE")
    lesion_areas = np.asarray([g.area for g in lesions])
    if np.any(lesion_areas <= 0):
        raise ValueError("LESION_NONPOSITIVE_AREA")
    normalized = overlaps / lesion_areas
    per_lesion = normalized[selected].sum(axis=0) if k else np.zeros(len(lesions))
    reachable = normalized.sum(axis=0)
    gains = normalized.mean(axis=1)
    oracle_order = np.lexsort((np.arange(n), coords[:, 0], coords[:, 1], -gains))
    oracle = float(gains[oracle_order[:k]].sum())
    return {
        "n": n, "k": k, "actual_budget": k / n,
        "selected_indices": selected.tolist(),
        "per_lesion": per_lesion.tolist(),
        "mlac": float(per_lesion.mean()),
        "reachable_per_lesion": reachable.tolist(),
        "reachable_macro": float(reachable.mean()),
        "random_expected": float(k / n * reachable.mean()),
        "xml_oracle": oracle,
        "purity": float(overlaps[selected].sum() / (k * size * size)) if k else None,
        "tumor_area_covered": float(overlaps[selected].sum()),
        "tumor_area": float(lesion_areas.sum()),
    }


def patch_metrics(labels, scores, mask):
    labels, scores, mask = np.asarray(labels), np.asarray(scores), np.asarray(mask, bool)
    if labels.shape != scores.shape or labels.shape != mask.shape or labels.ndim != 1:
        raise ValueError("PATCH_METRIC_SHAPE")
    y, p = labels[mask], scores[mask]
    if not np.isfinite(p).all():
        raise ValueError("PATCH_SCORE_NONFINITE")
    if len(np.unique(y)) != 2:
        return {"auroc": None, "auprc": None, "valid_patches": len(y)}
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "valid_patches": len(y)}


def peaks(coords, size, scores):
    coords = validate_grid(coords, size)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(coords),) or not np.isfinite(scores).all():
        raise ValueError("PEAK_SCORE")
    lookup = {tuple(xy): i for i, xy in enumerate(coords.tolist())}
    neighbors = []
    for x, y in coords:
        neighbors.append([lookup[(int(x + dx * size), int(y + dy * size))]
                          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          if (dx or dy) and (int(x + dx * size), int(y + dy * size)) in lookup])
    maxima = {i for i, adj in enumerate(neighbors) if all(scores[i] >= scores[j] for j in adj)}
    output = []
    while maxima:
        seed = maxima.pop()
        plateau, stack = {seed}, [seed]
        while stack:
            for j in neighbors[stack.pop()]:
                if j in maxima and scores[j] == scores[seed]:
                    maxima.remove(j)
                    plateau.add(j)
                    stack.append(j)
        center = coords[list(plateau)].mean(axis=0)
        chosen = min(plateau, key=lambda j: (float(np.sum((coords[j] - center) ** 2)),
                                             int(coords[j, 1]), int(coords[j, 0]), j))
        output.append(chosen)
    return sorted(output, key=lambda j: (-scores[j], int(coords[j, 1]), int(coords[j, 0]), j))


def peak_froc(slides, fp_grid=(0.25, 0.5, 1, 2, 4, 8)):
    """slides: dicts with coords, size, scores and exact lesion geometries."""
    if not slides or not any(s["lesions"] for s in slides):
        raise ValueError("FROC_EMPTY_OR_NO_LESIONS")
    detections = []
    total_lesions = sum(len(s["lesions"]) for s in slides)
    for slide in slides:
        for j in peaks(slide["coords"], slide["size"], slide["scores"]):
            x, y = slide["coords"][j]
            point = Point(float(x) + slide["size"] / 2, float(y) + slide["size"] / 2)
            matches = [i for i, g in enumerate(slide["lesions"]) if g.covers(point)]
            detections.append({"slide_id": slide["slide_id"], "patch_index": int(j),
                               "x": point.x, "y": point.y,
                               "score": float(slide["scores"][j]),
                               "lesion_id": min(matches) if matches else None})
    detections.sort(key=lambda d: (-d["score"], d["slide_id"],
                                    -1 if d["lesion_id"] is None else d["lesion_id"],
                                    d["y"], d["x"]))
    seen, tp, fp = set(), 0, 0
    curve = [{"threshold": None, "fp_per_slide": 0.0, "sensitivity": 0.0}]
    for i, d in enumerate(detections):
        key = (d["slide_id"], d["lesion_id"])
        if d["lesion_id"] is None:
            d["reason"] = "outside"
            fp += 1
        elif key in seen:
            d["reason"] = "duplicate"
            fp += 1
        else:
            d["reason"] = "tp"
            tp += 1
            seen.add(key)
        if i == len(detections) - 1 or detections[i + 1]["score"] != d["score"]:
            curve.append({"threshold": d["score"], "fp_per_slide": fp / len(slides),
                          "sensitivity": tp / total_lesions})
    grid = {str(target): max(p["sensitivity"] for p in curve
                             if p["fp_per_slide"] <= target) for target in fp_grid}
    return {"detections": detections, "curve": curve,
            "grid_sensitivity": grid, "mean_grid_sensitivity": float(np.mean(list(grid.values()))),
            "target_reached": {str(target): curve[-1]["fp_per_slide"] >= target
                               for target in fp_grid},
            "max_observed_fp_per_slide": curve[-1]["fp_per_slide"]}


def paired_bootstrap(c_by_seed, d_by_seed, seed=9233, repetitions=10000):
    """Arrays have shape [seed, positive_slide] in identical slide order."""
    c, d = np.asarray(c_by_seed, float), np.asarray(d_by_seed, float)
    if c.shape != d.shape or c.ndim != 2 or c.shape[0] != 3 or c.shape[1] < 2:
        raise ValueError("PAIRED_SHAPE")
    per_slide = (d - c).mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_slide), size=(repetitions, len(per_slide)))
    estimates = per_slide[draws].mean(axis=1)
    single_seed = (d - c).mean(axis=1)
    return {"delta": float(per_slide.mean()),
            "ci95_percentile": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "single_seed": single_seed.tolist(),
            "seed_sd": float(single_seed.std(ddof=1)),
            "per_slide": per_slide.tolist()}
