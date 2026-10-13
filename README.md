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
python -m pytest -q tests/test_stage3_metrics.py tests/test_stage3_repair_renderer.py tests/test_stage3_repair_review.py tests/test_stage1_training_smoke.py
python scripts/audit_stage3.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --output "$CAMELYON16_ROOT/runs/stage3/audit.json"
```

The audit checks all 100 H5 patch grids and WSI bounds. It uses XML for
train/validation truth checks and skips test performance computation. It writes
`audit.json` even when an XML polygon needs `make_valid` review or another
slide fails. Exit code 3 means `needs_review`: inspect `repair_cases`, including
the XML hash, annotation index, validity reason, bounds, and geometry changes.
Exit code 2 means `failed`: inspect `issues` as well. A failed audit never
permits a lock; `needs_review` can proceed only with a complete, hash-matched
human review receipt. Generate repair images for every affected split before
looking at any model predictions:

```bash
python scripts/render_stage3_xml_repairs.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --audit-report "$CAMELYON16_ROOT/runs/stage3/audit.json" \
  --split train --split val --split test \
  --output "$CAMELYON16_ROOT/runs/stage3/xml_repair_review_all"
```

This writes one WSI crop per repaired annotation, contact sheets, a blank
`review_queue.csv`, and a `render_manifest.json`. Inspect every row's crop.
The decision `accept_make_valid` means only that the repaired geometry follows
the drawn contour; it does not certify tumor biology or turn multipart geometry
parts into separate clinical lesions. Do not edit source XML. If a row needs a
source correction or cannot be judged, enter a different decision and a note;
receipt creation will stop until the source issue is resolved and the audit is
repeated. Preserve all queue columns and the render manifest.

After every repair row has been reviewed, create the receipt. Replace the
example reviewer value with the reviewer name or identifier:

```bash
python scripts/accept_stage3_repairs.py \
  --manifest "$CAMELYON16_ROOT/derived/manifest_stage1.csv" \
  --audit-report "$CAMELYON16_ROOT/runs/stage3/audit.json" \
  --review-queue "$CAMELYON16_ROOT/runs/stage3/xml_repair_review_all/review_queue.csv" \
  --reviewer "REVIEWER_NAME" \
  --output "$CAMELYON16_ROOT/runs/stage3/repair_review_receipt.json"
```

The receipt is refused if rows are missing, duplicated, stale, or anything
other than `accept_make_valid`. It binds the approved annotations to the
manifest, audit report, and reviewed CSV hashes. It is a review attestation,
not a cryptographic signature.

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
  --repair-review-receipt "$CAMELYON16_ROOT/runs/stage3/repair_review_receipt.json" \
  --config configs/stage1/a_pooling.yaml \
  --config configs/stage1/b_query.yaml \
  --config configs/stage1/c_graph_query.yaml \
  --config configs/stage1/d_graph_query_annotation.yaml \
  --config configs/stage1/e_query_annotation.yaml \
  --config configs/stage1/f_linear_patch_annotation.yaml \
  --output "$CAMELYON16_ROOT/runs/stage3/freeze.json" \
  --acceptance-confirmed
```

The lock includes the review receipt hash and approved annotation inventory.
Evaluation of any repaired slide verifies the same receipt and current XML
hash. Only then run formal baselines. `scripts/train_stage1.py` accepts `--seed` and
`--output-root`; use one distinct output directory for each group and seed.
It loads train/validation only and selects by validation slide AUROC. Run
`scripts/eval_stage3.py` on validation for the chosen checkpoint, passing the
freeze lock and repair receipt. Test mode also requires the validation
threshold file for that same checkpoint and the matching config file. Keep
every seed's checkpoint and predictions.

The user confirmed that the local `Exclusion` label corresponds to source group
`_2`; the protocol treats `_2` as non-tumor area subtracted from tumor. Record
that mapping as user-confirmed provenance. A geometry review checks the
annotation repair, not whether tissue is biologically tumor.
