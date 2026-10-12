"""Real training-slide forward/backward and prediction export for E and F."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.camelyon_dataset import CamelyonDataset
from modules.evidence_model import EvidenceModel, EvidenceModelConfig
from modules.evidence_objectives import location_bce


def smoke(manifest, slide_ids, output, device="cuda"):
    with Path(manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["slide_id"] in slide_ids]
    if len(selected) != len(slide_ids) or any(r["split"] != "train" for r in selected):
        raise ValueError("SMOKE_REQUIRES_EXISTING_TRAIN_SLIDES")
    dataset = CamelyonDataset.from_manifest(manifest, "train")
    if dataset.audit:
        raise ValueError(f"TRAIN_DATASET_AUDIT {dataset.audit}")
    chosen = {slide.slide_id: slide for slide in dataset if slide.slide_id in slide_ids}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(9233)
    report = []
    for variant in ("query", "linear_patch"):
        model = EvidenceModel(EvidenceModelConfig(variant=variant, graph_mode="none")).to(device)
        for slide_id in slide_ids:
            slide = chosen[slide_id]
            features = slide.features.to(device)
            inference = replace(slide, features=features, coords_level0=slide.coords_level0.to(device))
            model.zero_grad(set_to_none=True)
            result = model(inference)
            loss_cls = F.cross_entropy(result.slide_logits[None],
                                       torch.tensor([slide.slide_label], device=device))
            loss_loc = location_bce(result.evidence_scores,
                                    slide.tumor_fraction.to(device),
                                    slide.annotation_mask.to(device))
            loc_grads = torch.autograd.grad(loss_loc, tuple(model.parameters()),
                                            retain_graph=True, allow_unused=True)
            loc_gradient = sum(g.detach().abs().sum().item() for g in loc_grads if g is not None)
            if loc_gradient <= 0:
                raise RuntimeError(f"SMOKE_NO_LOCATION_GRADIENT {variant} {slide_id}")
            (loss_cls + loss_loc).backward()
            gradients = [p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None]
            if not gradients or not any(g > 0 for g in gradients):
                raise RuntimeError(f"SMOKE_NO_GRADIENT {variant} {slide_id}")
            if not torch.isfinite(loss_cls + loss_loc):
                raise RuntimeError(f"SMOKE_NONFINITE_LOSS {variant} {slide_id}")
            evidence = result.raw_patch_logits if variant == "linear_patch" else result.evidence_scores
            values = evidence.detach().float().cpu()
            torch.save({"slide_id": slide_id, "variant": variant, "coords_level0": slide.coords_level0,
                        "patch_scores": values,
                        "query_attention": result.query_attention.detach().float().cpu()},
                       output / f"{variant}_{slide_id}.pt")
            report.append({"variant": variant, "slide_id": slide_id, "slide_label": slide.slide_label,
                           "loss_cls": float(loss_cls.detach()), "loss_loc": float(loss_loc.detach()),
                           "gradient_l1": float(sum(gradients)),
                           "location_gradient_l1": float(loc_gradient), "patch_count": len(values)})
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slide-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(smoke(args.manifest, args.slide_id, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
