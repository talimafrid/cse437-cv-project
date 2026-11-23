# Retail Shelf Planogram Compliance with Open-Set SKU Recognition

CSE437 Computer Vision, Project 5, Daffodil International University.

Give the app a photo of a supermarket shelf and the planogram (the list of products that should sit on each shelf row). It finds every product, recognises it, lines the shelf up against the planogram and reports each slot as **correct**, **misplaced**, **missing**, **unknown** or **extra**, with a compliance score.

- **Detection:** a class-agnostic YOLO11s detector, fine-tuned on SKU-110K, finds every product.
- **Recognition:** each crop is matched against a gallery with one studio image per product (frozen SigLIP2 embeddings, SIFT re-ranking). Products not in the gallery are flagged as unknown.
- **Alignment:** detections are grouped into shelf rows and aligned to the planogram rows.
- **Report:** an annotated image, a per-slot table and a JSON/CSV report.

No per-SKU classifier is trained, so a new product is added by indexing one image (about 6 s).

---

## Contents

1. [Quick start (Windows, one click)](#1-quick-start-windows-one-click)
2. [Quick start with Docker (any OS)](#2-quick-start-with-docker-any-os)
3. [Manual setup (Windows, Linux, macOS)](#3-manual-setup-windows-linux-macos)
4. [GPU or CPU](#4-gpu-or-cpu)
5. [Using the demo](#5-using-the-demo)
6. [Command-line tools](#6-command-line-tools)
7. [Reproducing the results](#7-reproducing-the-results)
8. [Results](#8-results)
9. [Data](#9-data)
10. [Troubleshooting](#10-troubleshooting)
11. [Project layout](#11-project-layout)

---

## What you need first

The trained detector, the gallery index and the GroZi-3.2k images are stored with **Git LFS**. Install Git LFS before cloning, or you get small pointer files instead of the real data:

```bash
git lfs install
git clone https://github.com/talimafrid/cse437-cv-project.git
cd cse437-cv-project
git lfs pull          # only needed if you cloned before installing Git LFS
```

After the clone these must exist (the scripts check them):

| Path | What it is | Size |
|---|---|---|
| `models/sku110k_yolo11s.pt` | trained product detector | 19 MB |
| `outputs/index/siglip2_b_v8/` | gallery index (3,286 products x 8 views) | 300 MB |
| `data/grozi/Testing/` | GP-180 shelf photos and annotations | 1.4 GB |
| `data/grozi/Training/` | studio product images (only needed to rebuild the index) | 310 MB |

SKU-110K is **not** in the repo (28 GB, and the brief forbids redistributing it). You only need it to retrain the detector or re-measure A1, see [Data](#9-data).

---

## 1. Quick start (Windows, one click)

Requirements: Windows 10/11, [Python 3.13](https://www.python.org/downloads/) with "Add to PATH" ticked, about 8 GB free disk.

Double-click **`run_demo.bat`**. It:

1. creates a virtual environment in `.venv` (first run only),
2. installs the pinned packages from `requirements.txt` (first run only, about 5 GB download),
3. checks that the model, gallery index and shelf images are present, and builds the index if only the studio images are there,
4. starts the demo and opens **http://127.0.0.1:7860** in your browser.

Later runs skip steps 1 and 2 and start in a few seconds. Close the console window to stop the app. The first audit after start takes a little longer because the models load then.

## 2. Quick start with Docker (any OS)

The Docker image is self-contained: code, pinned packages, trained detector, gallery index, SigLIP2 weights and shelf photos. It needs no internet once built, and it runs the same on any machine with Docker.

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux), about 15 GB free disk, and the LFS data from [What you need first](#what-you-need-first) (the image is built from it).

**Windows, one click:** start Docker Desktop, then double-click **`run_docker.bat`**. It uses the GPU when `nvidia-smi` works and the CPU otherwise, builds the image the first time, and opens the browser.

**Any OS, by hand:**

```bash
# CPU (works everywhere)
docker compose up --build -d

# NVIDIA GPU (needs the NVIDIA driver, plus Docker Desktop with WSL2 on Windows
# or nvidia-container-toolkit on Linux)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Open **http://localhost:7860**. Useful commands:

```bash
docker compose logs -f      # watch the app
docker compose down         # stop it
```

Audit images and reports are written to `outputs/demo/` on your machine (mounted into the container). Gallery edits made in the demo live inside the container and are lost when it is removed.

Without Compose:

```bash
docker build -t planogram-demo .
docker run --rm -p 7860:7860 planogram-demo              # CPU
docker run --rm -p 7860:7860 --gpus all planogram-demo   # GPU
```

## 3. Manual setup (Windows, Linux, macOS)

```bash
python3.13 -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m pytest -q tests              # 19 tests, about 1 s
python -m src.app                      # http://127.0.0.1:7860
```

On macOS there is no CUDA build of PyTorch: drop the `+cu126` suffix from the two `torch` lines in `requirements.txt` and install without `--extra-index-url`. The app then runs on CPU.

If `outputs/index/siglip2_b_v8/` is missing, build it from the studio images (about 10 min on a GPU, more than an hour on CPU):

```bash
python -m src.gallery build --backbone siglip2_b
```

## 4. GPU or CPU

The app picks the device by itself (`device: auto` in `configs/config.yaml`):

- it uses the NVIDIA GPU when PyTorch sees one **and** a test operation on it succeeds,
- otherwise (no GPU, AMD/Intel GPU, too-old driver, broken CUDA install) it prints a note and runs on CPU.

To force a device, set the environment variable `PLANOGRAM_DEVICE`:

```bash
set PLANOGRAM_DEVICE=cpu      # Windows cmd
export PLANOGRAM_DEVICE=cpu   # Linux/macOS
```

or change `device:` in `configs/config.yaml`. Expect about 1.6 s per shelf image on an RTX 2060 and about 10 s on CPU. The CUDA build of PyTorch in `requirements.txt` also runs fine on machines without a GPU.

## 5. Using the demo

The demo has three tabs.

**Audit**
1. Pick a test shelf from **GP-180 test image**. The photo and its planogram (JSON rows of product ids) load.
2. Click **Run audit**.
3. You get the photo with each slot boxed by status, the compliance %, a table of every slot (expected product, detected product, status, score), the time taken and a downloadable `report.json`.

You can also upload your own shelf photo and type a planogram as JSON rows, for example `[["Food/Rice/17", "Food/Rice/11"], ["Food/Tea/3"]]`.

**Inject violations**
1. Paste a planogram (copy it from the Audit tab) and choose a seed.
2. Click **Inject 10-20% edits**. You get an edited planogram plus a log of the swaps, removals, insertions and replacements.
3. Paste the edited planogram back into the Audit tab and run the audit again: the edited slots should now be flagged.

This is the A5 use case: the shelf stays the same, the planogram changes, and the system has to find the differences.

**Gallery (add / remove SKU)**
- Upload one studio photo of a new product, enter an id such as `Food/Tea/999` and click **Add and index** (about 6 s). The product is recognised from then on.
- Enter an id and click **Remove** to take it out.
- This rewrites the index in `outputs/index/siglip2_b_v8/`. Back that folder up first if you want to undo later.

**A five-minute presentation script**

| Step | Do | What it shows |
|---|---|---|
| 1 | Audit **s5_72** | 100% compliance in about 1.5 s on GPU (detection, recognition, speed) |
| 2 | Copy its planogram, Inject with seed **0** | 2 replaces and 2 inserts |
| 3 | Audit s5_72 again with the edited planogram | all 4 edits flagged, compliance 82.6% |
| 4 | Repeat with **s5_37**, seed **2** | a swap is caught |
| 5 | Add a product in the Gallery tab | a new SKU in about 6 s, no retraining |

Avoid **s1_28** (the first entry in the list) for a live demo: it scores 67.7% against its own layout, while most of the 50 test shelves score 100%. Some edits between near-identical variants (for example two soup flavours of the same brand) are not caught. That is the main recognition weakness, see [Results](#8-results).

## 6. Command-line tools

```bash
# Audit one image against a row planogram, writes image + JSON + CSV to outputs/audit/
python -m src.pipeline --image data/grozi/Testing/store5/images/72.jpg --planogram outputs/planograms/s5_72.json

# Gallery management
python -m src.gallery build  --backbone siglip2_b
python -m src.gallery add    --sku Food/Tea/999 --image my_product.jpg
python -m src.gallery remove --sku Food/Tea/999
```

All settings and thresholds live in `configs/config.yaml`.

## 7. Reproducing the results

You need SKU-110K for detector training and A1 (see [Data](#9-data)).

```bash
python scripts/prepare_data.py           # splits + hashes, row planograms, SKU-110K -> YOLO
python scripts/train_detector.py         # YOLO11s on SKU-110K, 1.5 h time cap
python scripts/evaluate.py --task tune   # validation only; writes tuned values to configs/config.yaml
python scripts/evaluate.py --task all    # A1-A6 + ablations on test -> outputs/metrics.json
python scripts/evaluate.py --task latency_cpu
python -m pytest -q tests
```

Splits in `splits/` are checked against `splits/SHA256SUMS` on every run.

## 8. Results

All numbers are on the test split and come from `scripts/evaluate.py` (`outputs/metrics.json`). Splits were frozen with SHA-256 hashes before any tuning, and every threshold was tuned on the 20 validation images only.

| Item | Result | Target |
|---|---|---|
| A1: SKU-110K mAP@0.5 (500 test images, 640 px) | 0.937 | 0.80 |
| A2: GP-180 proposal recall (50 images, 590 boxes) | 0.985 | 0.85 |
| A3: top-1 / top-5 on GT crops (SigLIP2 + SIFT re-rank) | 0.861 / 0.981 | 0.65 / 0.85 |
| A4: unknown rejected / known top-1 drop | 0.132 / 1.6 pp | 0.70 / at most 5 pp (**missed**) |
| A5: violations detected / false-alarm rate (50 images x 5 seeds, 480 violations) | 0.848 / 0.094 (oracle 0.933 / 0.002; original protocol 0.570 / 0.134) | 0.80 / at most 0.10 |
| A6: latency per image, RTX 2060 (mean / p95) | 1.58 s / 2.84 s (CPU only: 10.4 s / 17.2 s) | at most 5 s |
| A6: add a SKU (model load + augment + embed + SIFT + save) | 6.2 s | at most 30 s |

- Latency was measured on an RTX 2060, not the T4 named in the brief. Recognition takes most of it (1.11 s of 1.58 s).
- The two-level row alignment matters: naive row-k-to-row-k matching gives 0.476 false alarms.
- Of 590 test crops, 82 are misrecognised; 60 of those are near-identical package variants.

**A5 protocol.** The main A5 number does not use the GroZi graph planograms directly. Planogram rows are rebuilt from the ground-truth box layout with the pipeline's own row rule, and only violations that can be pinned to one slot are injected (no remove, swap or insert next to an identical product). The original protocol (graph rows, unrestricted edits) is reported as `secondary_original_protocol`; it caps even perfect recognition at about 0.67 detected, for the reasons in `findings.md`. The test-split oracle was looked at while diagnosing this; no tuned value came from it.

## 9. Data

| Dataset | Source | Local path | In the repo |
|---|---|---|---|
| SKU-110K (fixed), 11,743 images | http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz (from https://github.com/eg4000/SKU110K_CVPR19) | `data/SKU110K_fixed/`, YOLO copy in `data/sku110k_yolo/` | no (download it) |
| GroZi-3.2k gallery (3,286 Food products) and GP-180 shelves | GroZi-3.2k release [CHECK source URL and version] | `data/grozi/` | yes, Git LFS |
| GP-180 planogram fixes | https://gist.github.com/laitalaj/09778eab24c0d16b8447d6ca3360c7b2 | `data/gp180/` | yes |

To get SKU-110K: download the tarball (12.2 GB) and extract it into `data/` so that `data/SKU110K_fixed/images` and `data/SKU110K_fixed/annotations` exist, then run `python scripts/prepare_data.py`.

Splits (`splits/`):
- GP-180 is split 20 validation / 50 test, stratified by store. 8 planograms are excluded as broken.
- 657 products (20%) are held out as unknown for A4.
- A1 uses a 500-image subset of SKU-110K test.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `run_demo.bat` says Python 3.13 not found | Install Python 3.13 and tick "Add to PATH", then run it again. |
| "models\sku110k_yolo11s.pt is missing" or files of ~130 bytes | Git LFS data not downloaded: `git lfs install` then `git lfs pull`. |
| `[device] CUDA present but unusable` | Your NVIDIA driver is too old for CUDA 12.6. The app runs on CPU; update the driver to use the GPU. |
| Audit takes 10-20 s | You are on CPU. That is expected. |
| Port 7860 already in use | Close the other app, or set `GRADIO_SERVER_PORT=7861` before starting. |
| Docker: `could not select device driver "nvidia"` | No GPU support in Docker. Use the CPU command (`docker compose up --build -d`). |
| Docker build is slow the first time | It downloads PyTorch with CUDA (about 5 GB). Later builds reuse the cache. |
| `opencv` or `libGL` error on Linux without Docker | `sudo apt install libgl1 libglib2.0-0` |

## 11. Project layout

```
run_demo.bat         one-click setup + launch (Windows)
run_docker.bat       one-click Docker launch (Windows)
Dockerfile, docker-compose*.yml, .dockerignore
configs/   config.yaml (all settings), sku110k.yaml (generated)
scripts/   prepare_data.py, train_detector.py, evaluate.py
src/       app (Gradio demo), pipeline, detector, recognizer, gallery, planogram, violations, data, utils
tests/     test_core.py
splits/    frozen split lists + SHA256SUMS
models/    sku110k_yolo11s.pt (trained detector)
outputs/   metrics.json, planograms/, index/, errors/, train/, logs
data/      grozi/, gp180/ (SKU-110K goes here after download)
```

## Environment used for the reported numbers

NVIDIA RTX 2060 (6 GB), Windows 11, Python 3.13.14, torch 2.14.1+cu126, torchvision 0.29.1+cu126, ultralytics 8.4.216, transformers 5.24.0, opencv-python 5.1.0, gradio 6.32.0. The full pinned list is in `requirements.txt`.

Backbones (frozen): `google/siglip2-base-patch16-384` (used), `facebook/dinov2-base`, `openai/clip-vit-base-patch16`, ResNet-50 (IMAGENET1K_V2).

Detector training used 1.55 GPU-hours (1.5 h with batch 8, plus a 3-minute batch-16 run aborted after it ran out of VRAM).

`PLAN.md` is the pre-implementation plan and is out of date.
