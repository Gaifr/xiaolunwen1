"""Synthetic acceptance for read-only XML repair review images."""

import csv
import json

import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from scripts import render_stage3_xml_repairs as renderer


def test_repair_renderer_checks_xml_and_writes_review_queue(tmp_path, monkeypatch):
    xml = tmp_path / "tumor_000.xml"
    xml.write_text(
        '<ASAP_Annotations><Annotations><Annotation PartOfGroup="Tumor">'
        '<Coordinates><Coordinate Order="0" X="100" Y="100" />'
        '<Coordinate Order="1" X="110" Y="110" />'
        '<Coordinate Order="2" X="100" Y="110" />'
        '<Coordinate Order="3" X="110" Y="100" />'
        '</Coordinates></Annotation></Annotations></ASAP_Annotations>',
        encoding="utf-8",
    )
    raw = Polygon([(100, 100), (110, 110), (100, 110), (110, 100)])
    fixed = shapely.make_valid(raw)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("slide_id", "split", "xml_path", "wsi_path"))
        writer.writeheader()
        writer.writerow({"slide_id": "tumor_000", "split": "train",
                         "xml_path": str(xml), "wsi_path": str(tmp_path / "fake.tif")})
    report_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps({
        "status": "needs_review", "issues": [], "repair_cases": [{
            "slide_id": "tumor_000", "split": "train", "xml_sha256": renderer.sha256(xml),
            "repairs": [{"annotation": 0, "group": "Tumor", "reason": explain_validity(raw),
                         "after_type": fixed.geom_type,
                         "polygon_parts": len(renderer.polygon_parts(fixed)),
                         "before_area": raw.area, "after_area": fixed.area}],
        }],
    }), encoding="utf-8")

    class FakeWSI:
        dimensions = (1000, 1000)
        level_downsamples = (1.0,)
        level_count = 1

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read_region(self, location, level, size):
            return Image.new("RGBA", size, "white")

    monkeypatch.setattr(renderer.openslide, "OpenSlide", FakeWSI)
    output = tmp_path / "review"
    result = renderer.render(manifest, report_path, output, {"train"})
    assert result["rendered_annotations"] == 1
    assert (output / "tumor_000__ann_0000.png").is_file()
    assert (output / "tumor_000__sheet_01.png").is_file()
    with (output / "review_queue.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["review_decision"] == ""
    assert rows[0]["contact_sheet"] == "tumor_000__sheet_01.png"


def test_renderer_cli_defaults_all_splits_and_preserves_explicit_split(monkeypatch, tmp_path):
    import sys
    calls = []
    monkeypatch.setattr(renderer, "render", lambda manifest, audit, output, splits: calls.append(splits) or {})
    args = ["renderer", "--manifest", "manifest.csv", "--audit-report", "audit.json", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    renderer.main()
    assert calls[-1] == {"train", "val", "test"}
    monkeypatch.setattr(sys, "argv", args + ["--split", "val"])
    renderer.main()
    assert calls[-1] == {"val"}
