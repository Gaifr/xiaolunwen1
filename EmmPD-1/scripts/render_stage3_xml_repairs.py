"""Render read-only WSI crops for manual review of Stage-3 XML repairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree

import openslide
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 0 else []
    return [part for child in getattr(geometry, "geoms", ())
            for part in polygon_parts(child)]


def annotation_polygon(annotation):
    coordinates = sorted(annotation.findall("./Coordinates/Coordinate"),
                         key=lambda item: int(item.get("Order", "0")))
    points = [(float(item.attrib["X"]), float(item.attrib["Y"]))
              for item in coordinates]
    if len(points) < 3 or not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("XML_INVALID_COORDINATES")
    return Polygon(points)


def crop_bounds(bounds, dimensions):
    left, top, right, bottom = bounds
    width = max(256.0, (right - left) * 1.2)
    height = max(256.0, (bottom - top) * 1.2)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, y0 = max(0, math.floor(cx - width / 2)), max(0, math.floor(cy - height / 2))
    x1, y1 = min(dimensions[0], math.ceil(cx + width / 2)), min(dimensions[1], math.ceil(cy + height / 2))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("REPAIR_BOUNDS_OUTSIDE_WSI")
    return x0, y0, x1, y1


def wsi_crop(wsi, bounds):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    desired_downsample = max(1.0, max(width, height) / 900)
    level = next((i for i, factor in enumerate(wsi.level_downsamples)
                  if factor >= desired_downsample), wsi.level_count - 1)
    downsample = float(wsi.level_downsamples[level])
    level_width = math.ceil(width / downsample)
    level_height = math.ceil(height / downsample)
    if max(level_width, level_height) <= 2048:
        image = wsi.read_region((x0, y0), level, (level_width, level_height)).convert("RGB")
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        view_bounds = (x0, y0, x0 + level_width * downsample,
                       y0 + level_height * downsample)
    else:
        thumbnail = wsi.get_thumbnail((4096, 4096)).convert("RGB")
        sx = thumbnail.width / wsi.dimensions[0]
        sy = thumbnail.height / wsi.dimensions[1]
        px0, py0 = math.floor(x0 * sx), math.floor(y0 * sy)
        px1, py1 = math.ceil(x1 * sx), math.ceil(y1 * sy)
        image = thumbnail.crop((px0, py0, px1, py1))
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        view_bounds = (px0 / sx, py0 / sy, px1 / sx, py1 / sy)
    return image, view_bounds


def project(ring, bounds, image):
    x0, y0, x1, y1 = bounds
    return [((x - x0) * image.width / (x1 - x0),
             (y - y0) * image.height / (y1 - y0)) for x, y in ring.coords]


def draw_polygon(image, polygon, bounds, color):
    draw = ImageDraw.Draw(image)
    draw.line(project(polygon.exterior, bounds, image), fill=color, width=3, joint="curve")
    for interior in polygon.interiors:
        draw.line(project(interior, bounds, image), fill=color, width=2, joint="curve")


def render_case(wsi, annotation, repair, path, slide_id):
    raw = annotation_polygon(annotation)
    fixed = shapely.make_valid(raw)
    if raw.is_valid or explain_validity(raw) != repair["reason"]:
        raise ValueError(f"REPAIR_REASON_CHANGED {slide_id} annotation={repair['annotation']}")
    if fixed.geom_type != repair["after_type"] or len(polygon_parts(fixed)) != repair["polygon_parts"]:
        raise ValueError(f"REPAIR_GEOMETRY_CHANGED {slide_id} annotation={repair['annotation']}")
    if not math.isclose(fixed.area, repair["after_area"], rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f"REPAIR_AREA_CHANGED {slide_id} annotation={repair['annotation']}")
    bounds = crop_bounds(raw.bounds, wsi.dimensions)
    base, view_bounds = wsi_crop(wsi, bounds)
    before, after = base.copy(), base.copy()
    draw_polygon(before, raw, view_bounds, "#ff3030")
    for polygon in polygon_parts(fixed):
        draw_polygon(after, polygon, view_bounds, "#22e07c")
    header = 32
    canvas = Image.new("RGB", (base.width * 2, base.height + header), "white")
    canvas.paste(before, (0, header))
    canvas.paste(after, (base.width, header))
    ImageDraw.Draw(canvas).text((8, 9), f"{slide_id} ann={repair['annotation']} raw outline",
                                fill="black")
    ImageDraw.Draw(canvas).text((base.width + 8, 9), "make_valid polygon boundaries",
                                fill="black")
    canvas.save(path)
    return bounds


def contact_sheets(image_names, output, slide_id):
    sheet_names = []
    for start in range(0, len(image_names), 8):
        page = start // 8 + 1
        name = f"{slide_id}__sheet_{page:02d}.png"
        sheet = Image.new("RGB", (1200, 1440), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, image_name in enumerate(image_names[start:start + 8]):
            with Image.open(output / image_name) as image:
                thumbnail = image.copy()
            thumbnail.thumbnail((580, 330), Image.Resampling.LANCZOS)
            x = (offset % 2) * 600 + (600 - thumbnail.width) // 2
            y = (offset // 2) * 360 + 20
            sheet.paste(thumbnail, (x, y))
            draw.text((x, y + 332), image_name, fill="black")
        sheet.save(output / name)
        sheet_names.append(name)
    return sheet_names


def render(manifest: Path, audit_report: Path, output: Path, splits: set[str]):
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    if report.get("status") != "needs_review" or report.get("issues"):
        raise ValueError("AUDIT_MUST_HAVE_REVIEWABLE_REPAIRS_AND_NO_ISSUES")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["slide_id"]: row for row in csv.DictReader(stream)}
    selected = [case for case in report["repair_cases"] if case["split"] in splits]
    if not selected:
        raise ValueError("NO_REPAIRS_FOR_SELECTED_SPLITS")
    output.mkdir(parents=True, exist_ok=True)
    queue = []
    for case in selected:
        slide_id = case["slide_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", slide_id):
            raise ValueError(f"SLIDE_ID_UNSAFE {slide_id}")
        row = rows.get(slide_id)
        if row is None or row["split"] != case["split"]:
            raise ValueError(f"MANIFEST_AUDIT_MISMATCH {slide_id}")
        xml_path = Path(row["xml_path"])
        if sha256(xml_path) != case["xml_sha256"]:
            raise ValueError(f"XML_SHA256_CHANGED {slide_id}")
        annotations = ElementTree.parse(xml_path).getroot().findall(".//Annotations/Annotation")
        case_start = len(queue)
        image_names = []
        with openslide.OpenSlide(row["wsi_path"]) as wsi:
            for repair in case["repairs"]:
                index = int(repair["annotation"])
                if index < 0 or index >= len(annotations):
                    raise ValueError(f"REPAIR_ANNOTATION_MISSING {slide_id} {index}")
                image_name = f"{slide_id}__ann_{index:04d}.png"
                crop = render_case(wsi, annotations[index], repair, output / image_name, slide_id)
                image_names.append(image_name)
                queue.append({"slide_id": slide_id, "split": case["split"],
                              "xml_sha256": case["xml_sha256"], "annotation": index,
                              "group": repair["group"], "reason": repair["reason"],
                              "after_type": repair["after_type"],
                              "polygon_parts": repair["polygon_parts"],
                              "before_area": repair["before_area"],
                              "after_area": repair["after_area"],
                              "absolute_area_delta": abs(repair["after_area"] - repair["before_area"]),
                              "relative_area_delta": (abs(repair["after_area"] - repair["before_area"])
                                                      / repair["before_area"] if repair["before_area"] > 0
                                                      else ""),
                              "crop_level0": list(crop), "image": image_name,
                              "contact_sheet": "",
                              "review_decision": "", "review_note": ""})
        sheets = contact_sheets(image_names, output, slide_id)
        for offset, item in enumerate(queue[case_start:]):
            item["contact_sheet"] = sheets[offset // 8]
    queue_path = output / "review_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    metadata = {"audit_sha256": sha256(audit_report), "manifest_sha256": sha256(manifest),
                "renderer_sha256": sha256(Path(__file__)),
                "splits": sorted(splits), "rendered_annotations": len(queue),
                "review_queue": str(queue_path)}
    (output / "render_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    args = parser.parse_args()
    result = render(args.manifest, args.audit_report, args.output,
                    set(args.split or ("train", "val")))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
