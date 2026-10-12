# CAMELYON16 Stage-3 evaluation implementation

This branch contains the Stage-3 protocol, A–F training entry point, geometry
evaluator, read-only data audit, and synthetic acceptance cases. The protocol is
**not frozen** until the owner runs the acceptance commands on the real server
and reviews the outputs. No WSI, XML, H5, model weight, or prediction is stored
in this repository.

From `EmmPD-1`, set `CAMELYON16_ROOT` to the local data directory and use the
existing Python environment with `requirements-stage3.txt` installed.

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
python -m pytest -q tests/test_stage3_metrics.py tests/test_stage3_repair_renderer.py tests/test_stage1_training_smoke.py
python scripts/audit_stage3.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --output "$CAMELYON16_ROOT/runs/stage3/audit.json"
```

The audit checks all 100 H5 patch grids and WSI bounds. It uses XML for
train/validation truth checks and skips test performance computation. It writes
`audit.json` even when an XML polygon needs `make_valid` review or another
slide fails. Exit code 3 means `needs_review`: inspect `repair_cases`, including
the XML hash, annotation index, validity reason, bounds, and geometry changes.
Exit code 2 means `failed`: inspect `issues` as well. Neither status permits
`freeze_stage3.py` to create a lock. Generate train/validation repair images
from the audit report:

```bash
python scripts/render_stage3_xml_repairs.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --audit-report "$CAMELYON16_ROOT/runs/stage3/audit.json" \
  --output "$CAMELYON16_ROOT/runs/stage3/xml_repair_review_train_val"
```

This renders train/validation WSI crops and contact sheets and writes a blank
`review_queue.csv`. It checks each XML hash against the audit report. Inspect
the images and record a decision for every repair; generating images does not
approve any repair or change the audit status. Test XML repairs can be rendered
separately with `--split test` after the review rules are fixed.

For the real E/F smoke, give selected **training** slide IDs:

```bash
python scripts/smoke_stage3_ef.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --slide-id TRAIN_NORMAL_ID --slide-id TRAIN_TUMOR_ID \
  --output "$CAMELYON16_ROOT/runs/stage3/ef_smoke"
```

After checking the synthetic results, full audit, E/F smoke, and train/val
scale output, create the lock with all six A–F config files:

```bash
python scripts/freeze_stage3.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --audit-report "$CAMELYON16_ROOT/runs/stage3/audit.json" \
  --config configs/stage1/a_pooling.yaml \
  --config configs/stage1/b_query.yaml \
  --config configs/stage1/c_graph_query.yaml \
  --config configs/stage1/d_graph_query_annotation.yaml \
  --config configs/stage1/e_query_annotation.yaml \
  --config configs/stage1/f_linear_patch_annotation.yaml \
  --output "$CAMELYON16_ROOT/runs/stage3/freeze.json" \
  --acceptance-confirmed
```

Only then run formal baselines. `scripts/train_stage1.py` accepts `--seed` and
`--output-root`; use one distinct output directory for each group and seed.
It loads train/validation only and selects by validation slide AUROC. Run
`scripts/eval_stage3.py` on validation for the chosen checkpoint. Its test
mode requires the freeze lock and validation threshold file for that same
checkpoint. Keep every seed's checkpoint and predictions.

The `Exclusion` group is treated as subtracted non-tumor area. Local XML uses
semantic `Tumor`/`Exclusion` names; provenance of its transformation from the
original numeric group names still needs review before final freeze.
