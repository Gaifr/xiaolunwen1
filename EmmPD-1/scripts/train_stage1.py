"""Train the annotation-guided CAMELYON Stage-1 evidence model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.camelyon_dataset import CamelyonDataset, EvidenceSlideInput
from modules.evidence_model import EvidenceModel, EvidenceModelConfig
from modules.evidence_objectives import location_bce, query_diversity_loss
from modules.graph_metrics import compute_binary_metrics


@dataclass(frozen=True)
class Stage1TrainingConfig:
    output_root: Path
    manifest: Path
    model: EvidenceModelConfig
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 9233
    device: str = "cuda"
    grad_clip: float = 5.0
    lambda_loc: float = 0.0
    lambda_div: float = 0.0
    resume: Path | None = None


@dataclass
class Stage1RunResult:
    output_root: Path
    best_checkpoint: Path
    last_checkpoint: Path
    start_epoch: int
    best_epoch: int
    best_metric_value: float
    history: list[dict]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_slide(slide: EvidenceSlideInput, device: torch.device) -> EvidenceSlideInput:
    return replace(
        slide,
        features=slide.features.to(device),
        coords_level0=slide.coords_level0.to(device),
        tumor_fraction=slide.tumor_fraction.to(device),
        patch_labels=slide.patch_labels.to(device),
        annotation_mask=slide.annotation_mask.to(device),
        lesion_ids=slide.lesion_ids.to(device),
    )


def _classification_weights(slides: Sequence[EvidenceSlideInput], device) -> torch.Tensor:
    counts = torch.bincount(
        torch.tensor([slide.slide_label for slide in slides], dtype=torch.long),
        minlength=2,
    ).float()
    if bool((counts == 0).any()):
        raise ValueError(f"TRAIN_CLASS_MISSING {counts.tolist()}")
    return (counts.sum() / (2 * counts)).to(device)


def _evaluate(model: nn.Module, slides: Sequence[EvidenceSlideInput]) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for slide in slides:
            output = model(slide)
            labels.append(slide.slide_label)
            probabilities.append(torch.softmax(output.slide_logits.float(), -1).cpu().numpy())
    classification = compute_binary_metrics(np.asarray(labels), np.stack(probabilities))
    return {
        "auroc": float(classification["auroc"]),
        "auprc": float(classification["auprc"]),
    }


def _history_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict] = []
    for row in rows:
        result.append(
            {
                key: int(value) if key == "epoch" else float(value)
                for key, value in row.items()
            }
        )
    return result


def _write_history(path: Path, history: Sequence[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _config_payload(config: Stage1TrainingConfig) -> dict:
    payload = asdict(config)
    payload["output_root"] = str(config.output_root)
    payload["manifest"] = str(config.manifest)
    payload["resume"] = str(config.resume) if config.resume else None
    return payload


def run_stage1(config: Stage1TrainingConfig) -> Stage1RunResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(config.output_root)
    checkpoint_dir = output_root / "checkpoints"
    if config.resume is None and ((output_root / "history.csv").exists() or
                                  (checkpoint_dir / "best.pt").exists()):
        raise FileExistsError(f"RUN_ALREADY_EXISTS {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    history_path = output_root / "history.csv"
    started_at = time.time()

    datasets = {
        split: CamelyonDataset.from_manifest(config.manifest, split)
        for split in ("train", "val")
    }
    train_slides = [_move_slide(slide, device) for slide in datasets["train"]]
    val_slides = [_move_slide(slide, device) for slide in datasets["val"]]
    if not train_slides or not val_slides:
        raise ValueError("STAGE1_EMPTY_SPLIT")

    model = EvidenceModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    class_loss = nn.CrossEntropyLoss(
        weight=_classification_weights(train_slides, device)
    )

    history = _history_from_csv(history_path) if config.resume else []
    start_epoch = 1
    best_epoch = 0
    best_value = float("-inf")
    bad_epochs = 0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        if checkpoint.get("selection_rule") != "validation_slide_auroc_round6_earliest":
            raise ValueError("RESUME_SELECTION_RULE_MISMATCH")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        _restore_rng(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_value = float(checkpoint.get("best_metric_value", float("-inf")))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        losses = {name: [] for name in ("total", "cls", "loc", "div", "weighted_loc")}
        for slide in train_slides:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(slide)
                target = torch.tensor([slide.slide_label], device=device)
                loss_cls = class_loss(output.slide_logits.unsqueeze(0), target)
                loss_loc = location_bce(
                    output.evidence_scores,
                    slide.tumor_fraction,
                    slide.annotation_mask,
                )
                loss_div = query_diversity_loss(output.query_attention)
                weighted_loc = config.lambda_loc * loss_loc
                loss = loss_cls + weighted_loc + config.lambda_div * loss_div
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"LOSS_DIVERGED epoch={epoch} slide={slide.slide_id}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name, value in (
                ("total", loss),
                ("cls", loss_cls),
                ("loc", loss_loc),
                ("div", loss_div),
                ("weighted_loc", weighted_loc),
            ):
                losses[name].append(float(value.detach().cpu()))

        validation = _evaluate(model, val_slides)
        value = float(Decimal(str(validation["auroc"])).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP))
        row = {
            "epoch": epoch,
            "loss_total": float(np.mean(losses["total"])),
            "loss_cls": float(np.mean(losses["cls"])),
            "loss_loc": float(np.mean(losses["loc"])),
            "loss_div": float(np.mean(losses["div"])),
            "weighted_loc": float(np.mean(losses["weighted_loc"])),
            "val_auroc": validation["auroc"],
            "val_auroc_round6": value,
            "val_auprc": validation["auprc"],
        }
        history.append(row)

        # Strict improvement keeps the earliest epoch on rounded ties.
        if value > best_value:
            best_value = value
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_metric_value": best_value,
                    "model_config": asdict(config.model),
                    "seed": config.seed,
                    "lambda_loc": config.lambda_loc,
                    "lambda_div": config.lambda_div,
                    "selection_rule": "validation_slide_auroc_round6_earliest",
                },
                best_checkpoint,
            )
        else:
            bad_epochs += 1

        _atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": _rng_state(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_metric_value": best_value,
                "bad_epochs": bad_epochs,
                "model_config": asdict(config.model),
                "seed": config.seed,
                "lambda_loc": config.lambda_loc,
                "lambda_div": config.lambda_div,
                "selection_rule": "validation_slide_auroc_round6_earliest",
            },
            last_checkpoint,
        )
        _write_history(history_path, history)
        if bad_epochs >= config.patience:
            break

    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_config_payload(config), sort_keys=False), encoding="utf-8"
    )
    (output_root / "input_manifest.json").write_text(
        json.dumps(
            {
                "manifest": str(config.manifest),
                "sha256": _sha256(Path(config.manifest)),
                "counts": {split: len(dataset) for split, dataset in datasets.items()},
                "audit": {split: dataset.audit for split, dataset in datasets.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "runtime.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.time() - started_at,
                "device": config.device,
                "start_epoch": start_epoch,
                "last_epoch": history[-1]["epoch"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Stage1RunResult(
        output_root=output_root,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        start_epoch=start_epoch,
        best_epoch=best_epoch,
        best_metric_value=best_value,
        history=history,
    )


def load_config(path: Path, *, resume: Path | None = None, device: str | None = None,
                seed: int | None = None, output_root: Path | None = None):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    def expand(value: str) -> Path:
        resolved = os.path.expandvars(value)
        if "${" in resolved:
            raise ValueError(f"CONFIG_ENV_UNSET {value}")
        return Path(resolved)
    training = payload["training"]
    objective = payload["objective"]
    model = EvidenceModelConfig(**payload["model"])
    return Stage1TrainingConfig(
        output_root=output_root or expand(payload["output_root"]),
        manifest=expand(payload["manifest"]),
        model=model,
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        seed=seed if seed is not None else int(payload.get("seed", 9233)),
        device=device or str(training.get("device", "cuda")),
        grad_clip=float(training.get("grad_clip", 5.0)),
        lambda_loc=float(objective["lambda_loc"]),
        lambda_div=float(objective["lambda_div"]),
        resume=resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CAMELYON Stage-1 evidence model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run_stage1(load_config(args.config, resume=args.resume, device=args.device,
                                    seed=args.seed, output_root=args.output_root))
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.best_checkpoint),
                "best_epoch": result.best_epoch,
                "best_metric_value": result.best_metric_value,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
