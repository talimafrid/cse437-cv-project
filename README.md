# Retail Shelf Planogram Compliance with Open-Set SKU Recognition

CSE437 Computer Vision, Project 5, Daffodil International University.

A shelf photo goes through four stages:

- **Detection:** a class-agnostic YOLO11s detector finds every product.
- **Recognition:** each crop is matched against a gallery with one studio image per product (frozen SigLIP2 embeddings, SIFT re-ranking). Products not in the gallery are flagged as unknown.
- **Alignment:** the detections are grouped into shelf rows and aligned against the planogram.
- **Report:** each slot is reported as correct, misplaced, missing, unknown or extra.

No per-SKU classifier is trained, so a new product can be added by indexing one image.

## Results (test split, `outputs/metrics.json`)

All numbers come from `scripts/evaluate.py`. Splits were frozen with SHA-256 hashes before any tuning, and every threshold was tuned on the 20 validation images only.

| Item | Result | Target |
|---|---|---|
| A1: SKU-110K mAP@0.5 (500 test images, 640 px) | 0.937 | 0.80 |
| A2: GP-180 proposal recall (50 images, 590 boxes) | 0.985 | 0.85 |
| A3: top-1 / top-5 on GT crops (SigLIP2 + SIFT re-rank) | 0.861 / 0.981 | 0.65 / 0.85 |
| A4: unknown rejected / known top-1 drop | 0.132 / 1.6 pp | 0.70 / at most 5 pp (**missed**) |
| A5: violations detected / false-alarm rate (50 images x 5 seeds, 480 violations) | 0.848 / 0.094 (oracle 0.933 / 0.002; original protocol 0.570 / 0.134) | 0.80 / at most 0.10 |
| A6: latency per image, RTX 2060 (mean / p95) | 1.58 s / 2.84 s (CPU only: 10.4 s / 17.2 s) | at most 5 s |
| A6: add a SKU (model load + augment + embed + SIFT + save) | 6.2 s | at most 30 s |

Latency was measured on an RTX 2060, not the T4 named in the brief. Recognition dominates it (1.11 s of 1.58 s), because each crop is embedded and SIFT-verified. The two-level row alignment matters: naive row k to row k matching gives 0.476 false alarms.

**A5 protocol.** The main A5 number does not use the GroZi graph planograms directly. Planogram rows are rebuilt from the ground-truth box layout with the same row rule the pipeline uses. Only violations that can be pinned to one slot are injected: no remove, swap or insert next to an identical product.

The original protocol uses the graph rows and unrestricted edits. It is reported as `secondary_original_protocol`, because it caps an oracle with perfect recognition at about 0.67 detected. The reasons are in `findings.md`: graph rows are shelf levels with stacked items, and an edit inside a run of identical products cannot be localised.

## Data

| Dataset | Source | Local path |
|---|---|---|
| SKU-110K (fixed), 11,743 images, tarball 12,196,152,718 bytes | http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz (from https://github.com/eg4000/SKU110K_CVPR19), downloaded 2026-11-23 | `data/SKU110K_fixed/`, YOLO copy in `data/sku110k_yolo/` |
| GroZi-3.2k gallery (3,286 Food products) and GP-180 boxes/planograms | copied from a local GroZi-3.2k release [CHECK source URL and version] | `data/grozi/` |
| GP-180 planogram fixes | https://gist.github.com/laitalaj/09778eab24c0d16b8447d6ca3360c7b2 | `data/gp180/` |

Splits (`splits/`, checked against `splits/SHA256SUMS` at every run):
- GP-180 is split 20 validation / 50 test, stratified by store. 8 planograms are excluded as broken.
- 657 products (20%) are held out as unknown for A4.
- A1 uses a 500-image subset of SKU-110K test.

## Environment and hardware

The reported numbers come from one machine: an NVIDIA RTX 2060 (6 GB) with CUDA, running Windows 11 and Python 3.13.14.

| Package | Version |
|---|---|
| torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 |
| ultralytics | 8.4.216 |
| transformers | 5.24.0 |
| opencv-python | 5.1.0 |
| gradio | 6.32.0 |

The full pinned list is in `requirements.txt`.

Backbones (frozen, from Hugging Face or torchvision):
- `google/siglip2-base-patch16-384`, the backbone used
- `facebook/dinov2-base`
- `openai/clip-vit-base-patch16`
- ResNet-50 (IMAGENET1K_V2)

Detector training used 1.55 GPU-hours, under the 2 h cap: 1.5 h with batch 8, plus a 3-minute batch-16 run that was aborted after it ran out of VRAM.

## Commands

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/prepare_data.py           # splits + hashes, row planograms, SKU-110K -> YOLO
python scripts/train_detector.py         # YOLO11s on SKU-110K, 1.5 h time cap
python scripts/evaluate.py --task tune   # validation only; writes tuned values to configs/config.yaml
python scripts/evaluate.py --task all    # A1-A6 + ablations on test -> outputs/metrics.json
python scripts/evaluate.py --task latency_cpu
python -m pytest -q tests                # 19 tests
python -m src.app                        # Gradio demo: audit, add/remove SKU, inject violations
```

All settings and thresholds live in `configs/config.yaml`. Galleries are built on first use into `outputs/index/<backbone>_v8/`.

## Layout

```
configs/   config.yaml (all settings), sku110k.yaml (generated)
scripts/   prepare_data.py, train_detector.py, evaluate.py
src/       data, detector, recognizer, gallery, planogram, violations, pipeline, app, utils
tests/     test_core.py
splits/    frozen split lists + SHA256SUMS
outputs/   metrics.json, planograms/, index/, errors/, train/
```

`PLAN.md` is the pre-implementation plan. Its problem list describes the code before the rewrite and is out of date.
