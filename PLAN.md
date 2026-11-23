# Project 5 Plan: Retail Shelf Planogram Compliance with Open-Set SKU Recognition

Status: superseded (2026-11-24). Section 1 describes the code before the rewrite; see README.md and outputs/metrics.json for the current state.
Requirement codes (F1-F7, A1-A6, C1-C14) refer to the CSE437 project brief.

## 1. Where the current code stands

The repo has a sensible module layout (detector, recognizer, gallery, planogram,
violations, pipeline, app) and the planogram alignment and violation injection have
unit tests. But the pipeline as written cannot produce any of the numbers the brief
asks for. The problems, most serious first:

| # | Problem | Where | Effect |
|---|---------|-------|--------|
| 1 | `detector.py` is empty. No code ever runs the detector. | `src/detector.py`, `src/pipeline.py` | F1, A1, A2 not implemented. |
| 2 | The demo loads ground-truth boxes from `store1/annotation/132.csv` for any uploaded image, or a fake 2x2 grid. | `src/app.py` lines ~110-130 | Demo output is meaningless for every image except 132.jpg, and uses GT at inference. |
| 3 | `evaluate.py` takes the first 10 CSVs, falls back to a hard-coded `Food/Snacks/19,...` planogram, and uses GT boxes. | `scripts/evaluate.py` | Reported compliance numbers would be invalid (C8). No A1-A6 metric is computed. |
| 4 | No trained weights; `models/` does not exist. | n/a | Nothing to evaluate. |
| 5 | `max_detections: 300`. SKU-110K images carry up to several hundred objects each. | `configs/config.yaml` | Caps recall and mAP on A1. Needs ~1000. |
| 6 | Notebook trains `yolov8n` at 640 px for 30 epochs with no time cap. | `colab_train_yolo.ipynb` | Likely misses A1 (0.80 mAP@0.5) on dense small boxes; no guard on the 2 GPU-hour limit. |
| 7 | Only one recognizer (ResNet-50 global pool, image squashed to 224x224). F2 requires three: local features, CNN, ViT/VLM. | `src/recognizer.py` | F2 incomplete; no classical baseline (C5). |
| 8 | No gallery augmentation (blur, jitter, perspective, glare). | `src/gallery.py` | Studio-to-shelf gap left open; A3 at risk. |
| 9 | App indexes only `Food/Snacks`; build script indexes all of `Training/` including non-food classes. | `src/app.py`, `scripts/build_gallery.py` | Wrong gallery either way. Test shelves are Food only. |
| 10 | No open-set split (20% of products held out, seeded). Thresholds 0.65/0.40 are hard-coded guesses, and `LOW_CONFIDENCE` is treated as a match. | `src/recognizer.py`, `src/app.py` | F3/A4 not measurable; thresholds not tuned on validation (C4). |
| 11 | Planogram parser guesses the `ogg` indexing (0 vs 1 based) and the node meaning (one facing vs one product block) is never checked. | `src/planogram.py::parse_gp180_json` | Alignment may compare against a wrong reference sequence. |
| 12 | Row alignment assumes detected row k matches planogram row k. One spurious or missed row shifts every row below it. | `PlanogramAligner.align_shelf` | Cascading false alarms on A5. |
| 13 | Violation log records only one column for swaps and no per-slot ground truth. Insert pool is hard-coded Snacks/Tea IDs. | `src/violations.py` | A5 (detected rate, false alarm rate) cannot be scored. |
| 14 | Thresholds and paths are hard-coded in app, evaluate and pipeline; `config.yaml` is never read. No `requirements.txt` although the README refers to one. | several | Fails C10. |
| 15 | No split files, no SHA-256 hashes, no seeds. | n/a | Fails C4. |
| 16 | `data/gp180/` holds only the two planogram fix files from the gist. The actual GP-180 boxes and planograms are in `data/grozi/bb` and `data/grozi/planos` (123 images each). The brief says 70 images (20 val / 50 test). | `data/` | Must reconcile which 70 images are GP-180 before freezing splits. [CHECK] |
| 17 | `load_image_rgb` assumes RGB arrays while `extract_batch` assumes BGR arrays. | `src/utils.py`, `src/recognizer.py` | Silent channel swap when numpy arrays are passed. |
| 18 | README environment table lists versions that do not match the machine (local Python is 3.15). | `README.md` | C2/C10 record must be real. |
| 19 | The notebook clones with the GitHub token inside the URL; it ends up in the Colab clone's `.git/config`. | `colab_train_yolo.ipynb` | Token exposure if the runtime is shared. Use Colab Secrets or a credential helper. |

Environment blocker found while checking: importing `torch` on this PC fails with
`An Application Control policy has blocked this file` (`shm.dll`). Local CPU runs are
blocked until that is fixed (Windows Smart App Control / WDAC policy, or a different
Python install). Colab or Kaggle T4 is needed anyway for A6.

Parts worth keeping: `GalleryIndex` (FAISS inner product, save/load, add), the edit
distance core of `PlanogramAligner`, `ShelfRowClusterer` as a starting point, the
violation injector skeleton, and the existing tests.

## 2. Approach in one paragraph

A YOLO detector fine-tuned on SKU-110K proposes class-agnostic product boxes. Each
box is embedded with a frozen pretrained model (DINOv2 expected best; CLIP and
ResNet-50 as comparisons; SIFT+RANSAC as the classical baseline) and matched by
k-NN against a FAISS gallery built from one studio image per product plus synthetic
augmentations of it. A crop is labelled unknown when its match score falls below a
threshold tuned on the 20 validation images, optionally after SIFT geometric
re-ranking of the top-k. Detections are grouped into shelf rows, rows are aligned
to planogram rows, and each row is aligned with a Needleman-Wunsch style DP to give
per-slot {correct, misplaced, missing, unknown} and a compliance percentage. No
classifier is trained per SKU at any point.

## 3. Pretrained models

| Role | Model | Source | Notes |
|------|-------|--------|-------|
| Detector start point | YOLOv8s or YOLO11s, COCO weights | Ultralytics | Fine-tune on SKU-110K (brief requires fine-tuning). `s` over `n` for small dense boxes. |
| Recognizer (main) | DINOv2 ViT-S/14 and ViT-B/14 | `torch.hub` `facebookresearch/dinov2` or HF `facebook/dinov2-*` | Strong instance-level retrieval without fine-tuning. Use CLS token, or CLS + mean patch token. |
| Recognizer (VLM) | CLIP ViT-B/32 or ViT-B/16 | OpenCLIP / HF | Image encoder only. |
| Recognizer (CNN) | ResNet-50 ImageNet | torchvision (already in repo) | Add pad-to-square instead of squash; optional GeM pooling. |
| Classical | SIFT + Lowe ratio test + RANSAC homography | OpenCV | Baseline (C5) and geometric re-ranker (F3). |

Embedding extraction is inference only, so it does not count toward the 3 GPU-hour
training budget. Only detector fine-tuning does.

## 4. Data and splits (do first, commit before any tuning: C4)

1. SKU-110K: use Ultralytics' built-in `SKU-110K.yaml`, which downloads and converts
   the dataset itself. That can replace `scripts/convert_sku110k.py` and the hand
   written `configs/sku110k.yaml`. Keep the committed label files only if needed.
   Test subset: seeded sample of 500 (or all 2,936) test images, list committed.
2. GroZi gallery: Food classes only (test shelves contain Food only). Count the
   reference images per product and confirm the 3,235 figure; `Training/Food` has
   5,430 files on disk, so check for duplicates or extra formats. [CHECK]
3. GP-180: identify the 70 annotated images from Tonioni's page and the `cvpce`
   reference code, then map them to `data/grozi/bb/*.csv` and
   `data/grozi/planos/*.json`. Apply the two gist fixes (already in `data/gp180/`).
   Seeded split 20 val / 50 test.
4. Open-set: seeded 20% of gallery products removed from the index. Test crops whose
   ground-truth product is in that set are "unknown".
5. Write `splits/*.txt` and `splits/SHA256SUMS`, commit, and do not touch thresholds
   on test from here on.

Annotation quirks to handle: Tonioni boxes are in full-resolution pixels (3264x2448)
and some coordinates are negative, so clamp. Image paths inside CSV/JSON look like
`X:/Grocery stores/.../Training//Food/Rice/17.jpg` and
`.../TrainingFood/Snacks/25.jpg`; the existing `normalize_sku_id` regex covers both.

## 5. Work plan by phase

Estimated 16-20 hours for the team, as the brief states. Each phase ends with a
check that can be run.

### Phase 0: Environment and repo hygiene (1.5 h)
- Fix the local torch block or move all runs to Colab/Kaggle; record exact hardware.
- `requirements.txt` with pinned versions from the actual runtime (`pip freeze`).
- Make every script and the app read `configs/config.yaml`; remove hard-coded
  thresholds and paths. Add `seed` handling in one place.
- Notebook: use Colab Secrets for the token, add `time=` cap to training.
- Check: `pytest` passes on Colab; `python -c "import src.pipeline"` works locally.

### Phase 1: Splits and data loaders (2 h)
- Section 4 above. One loader that returns, per GP-180 image: image path, GT boxes,
  GT SKU per box, parsed planogram rows.
- Verify planogram semantics against `cvpce`: is a graph node one facing or one
  product block, and is `ogg` 0 or 1 based. Add a test that parses one known
  planogram and compares to a hand-checked row list.
- Check: split hashes committed; loader test passes.

### Phase 2: Detector, F1 / A1 / A2 (3 h, about 1.5 GPU-h)
- Fine-tune YOLOv8s/YOLO11s on SKU-110K train, val on SKU-110K val. Settings to
  start from: `imgsz=640`, `max_det=1000`, `batch=16`, `time=1.5` (hours), cosine LR,
  mosaic on. If time allows, a short second stage at `imgsz=1024`.
- Implement `src/detector.py`: load weights, predict, return `[x1,y1,x2,y2,score]`.
- High-resolution / tiled inference for GroZi shelves (3264x2448): try full image at
  `imgsz=1280` vs 2x2 tiles with overlap + NMS merge. Pick on val.
- A1: `model.val()` on the committed SKU-110K test subset, `max_det=1000`.
- A2: proposal recall at IoU 0.5 against GP-180 test GT boxes, low confidence
  threshold (pick on val for recall vs box count).
- Domain gap note: GroZi products are much larger than SKU-110K boxes. If A2 falls
  short, scale the image down so products match SKU-110K size, or tile less.
- Check: two numbers (A1, A2) written to `outputs/metrics.json`.

### Phase 3: Recognizer, F2 / A3 (4 h)
- Extend `FeatureExtractor` with `dinov2_s`, `dinov2_b`, `clip_b32`, `resnet50`.
  Pad crops to square before resize.
- Gallery augmentation: for each reference image, K views (for example K=8:
  Gaussian blur, colour jitter, perspective warp, synthetic glare, random crop).
  Store all views and score a product by its max view similarity (or mean
  embedding; compare on val).
- SIFT baseline: per-product keypoints cached; query crop matched with ratio test
  0.75 and RANSAC homography; score = inlier count. Restrict to top-k from a fast
  embedding shortlist only for re-ranking; run full SIFT k-NN once for the baseline
  table (it is slow, report latency honestly).
- Evaluate on GT crops of GP-180 test images: top-1, top-5, ms per crop, index
  memory (MB). One table, four approaches (C5 baseline included).
- Check: `python scripts/evaluate.py --task recognition` reproduces the table.

### Phase 4: Open-set rejection, F3 / A4 (2 h)
- Score options: top-1 cosine, top-1 minus top-2 margin, and SIFT inlier count after
  re-ranking. Choose score and threshold on val only, under the rule "known top-1
  drops at most 5 pp".
- Replace the two-threshold `CONFIRMED / LOW_CONFIDENCE / UNKNOWN` logic with one
  tuned threshold (keep the low-confidence flag only for display).
- Report on test: unknown rejection rate and known top-1 before/after rejection.
- Check: A4 numbers in `metrics.json`; threshold value saved to config.

### Phase 5: Shelf structure and compliance, F4 (2.5 h)
- Row grouping: cluster by box bottom edge or y-centre with a 1D gap rule; handle
  slant by fitting a line per row (RANSAC on centres) before ordering left to right.
- Row-to-row alignment: DP over rows where the cost of pairing a detected row with a
  planogram row is that pair's edit distance, so a spurious row costs one gap
  instead of shifting all rows.
- Within a row: keep the existing DP. If planogram nodes are product blocks
  (Phase 1 finding), collapse consecutive identical detected SKUs into runs first.
- Statuses: correct, misplaced (wrong known product in the slot, or extra product),
  missing (planogram slot with no detection), unknown (rejected crop).
- Check: existing `test_planogram.py` still passes, plus a test with an extra row.

### Phase 6: Violation injection and A5 (2 h)
- Change the injector so it returns per-slot ground truth: for every slot in the
  edited planogram, `changed: bool` and the operation. Log both positions for swaps.
  Insert pool = other gallery products, not hard-coded IDs.
- Protocol: the shelf image is unchanged; the planogram is edited at 10-20% (seeded,
  several seeds per test image). A slot is a true violation if the injector changed
  it. Run the full pipeline (real detector + recognizer). Detected rate = flagged
  true violations / all true violations. False alarm rate = flagged unchanged slots
  / all unchanged slots.
- Note: recognition and detection errors show up as false alarms here, so A5 depends
  on Phases 2-4. Report A5 also with GT boxes + GT labels as an upper bound (clearly
  labelled as an oracle, not the test result).
- Check: A5 numbers in `metrics.json`.

### Phase 7: Pipeline, outputs, add-a-SKU, latency: F6 / F7 / A6 (1.5 h)
- `pipeline.py` calls the detector when no boxes are passed. GT boxes only through
  an explicit `--oracle-boxes` evaluation flag.
- F7: annotated image (labels + violation colours) and `report.json` / `report.csv`
  per shelf.
- F6: `python -m src.gallery add --image path.jpg --sku Food/Tea/99` and
  `python -m src.gallery remove --sku ...`, both updating the saved index. Time it.
- A6 and C9: 100 runs after warm-up, mean and p95, T4 and CPU, model loading
  excluded.
- Check: `metrics.json` has latency block; add-a-SKU under 30 s.

### Phase 8: Demo (1 h)
- Gradio app: upload shelf or pick a GP-180 test image, pick or edit a planogram,
  run, show annotated image + slot table + JSON download. Add-a-SKU and remove tabs
  call the same functions as the CLI.
- Rehearse the viva change: remove two products, add them back, hand-edit a
  planogram, produce the report in under 15 minutes.

### Phase 9: Ablations, error analysis, report (3 h)
- Ablations (C6), at least two of: backbone (ResNet vs CLIP vs DINOv2), gallery
  augmentation on/off, SIFT re-ranking on/off, detector tiling on/off, row-to-row
  alignment on/off.
- Error analysis (C7): 10+ failures grouped (near-identical variants, small/occluded
  crops, missed detections, row slant, reflections), one fix per group.
- IEEE 6-8 page report in LaTeX with related work (at least 5 references, all
  verified by DOI), ethics and licences (GroZi and SKU-110K are research only),
  limitations, AI-usage log (C13).

## 6. Acceptance criteria and how each is measured

| Code | Target | Measured on | Main risk |
|------|--------|-------------|-----------|
| A1 | mAP@0.5 >= 0.80 | SKU-110K test subset (>= 500 imgs) | Too few epochs or 300 max_det cap |
| A2 | Recall >= 85% at IoU 0.5 | GP-180 test boxes | Scale gap between SKU-110K and GroZi |
| A3 | Top-1 >= 65%, top-5 >= 85% | GT crops, GP-180 test | Near-identical variants |
| A4 | >= 70% unknowns rejected, known top-1 drop <= 5 pp | GT crops, GP-180 test | Hardest target; similarity overlap between known and unknown |
| A5 | >= 80% injected violations found, <= 10% false alarms | Injected planograms, GP-180 test | Recognition errors become false alarms |
| A6 | <= 5 s per shelf on T4; new SKU <= 30 s | 100 runs, p95 | SIFT re-ranking cost |

If a target is missed, the report gives a gap analysis rather than validation or
oracle numbers in its place (C8).

## 7. Single commands (C10)

```
python scripts/prepare_data.py          # download/check data, write splits + hashes
python scripts/train_detector.py        # SKU-110K fine-tune (Colab T4)
python scripts/build_gallery.py         # FAISS index (+ augmentations, open-set removal)
python scripts/evaluate.py --all        # A1-A6, ablations, latency -> outputs/metrics.json
python -m src.app                       # Gradio demo
python -m src.gallery add --image X --sku Y
```

## 8. Open questions for the team

1. Which 70 of the 123 Tonioni-annotated images form GP-180? (Phase 1)
2. Local machine: can the Application Control block on torch be lifted, or do we run
   everything on Colab/Kaggle?
3. Is the GitHub repo `talimafrid/cse437-cv-project` the one to push to, and who owns
   the Colab training run?
