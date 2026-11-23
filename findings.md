# Findings

## State on 2026-11-23 (start of this session)
- Previous session rewrote src/ and scripts/ (uncommitted): detector.py, data.py, prepare_data.py, train_detector.py, evaluate.py (565 lines, all tasks A1-A6 + tuning). `PLAN.md` still says "nothing implemented", which is now stale.
- Old tests (tests/test_*.py) are staged as deleted; no new tests exist.
- No `splits/`, `models/`, `outputs/`: nothing has been run.
- `.venv` (Python 3.13.14) has torch 2.14.1+cu126, torchvision, numpy only. Missing: ultralytics, cv2, faiss?, sklearn, transformers, gradio, pandas, yaml.
- torch imports fine now and sees CUDA on an RTX 2060 (the earlier Application Control block is gone).
- GroZi copied into `data/grozi` (10,619 Training / 4,105 Testing files, bb/ and planos/ 123 each). Gist fixes in `data/gp180/`.
- SKU-110K not downloaded. S3 URL is live (HTTP 200).
- Method reasoning lives in `../reports/Planogram compliance method plan.md` and `../research_notes/`; config.yaml already follows it (gallery Food only, excluded broken planograms, cvpce val images forced).
- Disk: 137 GB free on C:.

## Alignment ceiling (synthetic, oracle recognition, 3x8 rows, 300 seeds)
| setting | swap | remove | replace | insert | ALL det / FA |
|---|---|---|---|---|---|
| original (gap 0.6, free ends) | 0.73 / 0.08 | 0.77 / 0.04 | 0.995 | 0.994 | 0.83 / 0.026 |
| + swap step 0.75 | 0.80 / 0.07 | 0.77 / 0.04 | | | 0.86 / 0.025 |
| + end_cost 0.3 | 0.93 / 0.03 | 1.00 / 0.00 | 0.996 | 0.993 | 0.95 / 0.005 |
- Higher end_cost may hurt real shelves where unplanned products sit at row ends; tune on val.
- GP portrait images: 27 of 122 have boxes beyond y=2448 (portrait 2448x3264).

## A5 on real shelves: two protocol problems (found 2026-11-23, after tune_compliance)
- tune_compliance val: no setting reached FA <= 0.08 (best 0.143); fallback picked max detection (FA 0.51) and wrote verify.delta 0.0, min_inliers 1000, gap 0.4, end 0.3 to config. Those values are not usable.
- Oracle (GT boxes + GT labels), Tonioni row planograms: val 0.674 det / 0.116 FA, test 0.673 / 0.042. Synthetic oracle was 0.95 / 0.005. So the floor is not recognition.
- Cause 1, row structure: Tonioni graph rows are shelf levels; stacked products (s4_74 yoghurt, 3 plan rows vs 8 box rows), side-by-side groups (s3_179) and row order (s5_55) differ from bottom-edge grouping. 3/20 val, 9/50 test images. With planogram rows rebuilt from GT boxes by group_rows: FA 0.003 val / 0.005 test, base compliance 1.0.
- Cause 2, identical-product runs: per-op oracle on test with GT-layout rows: insert 0.996, replace 1.000, swap 0.56, remove 0.20. Removing one of N identical adjacent products or swapping into a run is not localisable; the aligner flags an equivalent slot, the per-slot scorer counts it as a miss.
- Pipeline vs oracle on val (Tonioni rows, verify delta 0.1 / mi 12, gap 0.8, end 0.1): pipe 0.472/0.139, GT boxes + real recogniser 0.523/0.116, oracle 0.674/0.116.

## Test results so far (2026-11-24)
| Item | Result | Target |
|---|---|---|
| A1 SKU-110K mAP@0.5 (500 test, imgsz 640 / 1024) | 0.937 / 0.946 | 0.80 |
| A2 GP-180 proposal recall (590 GT boxes) | 0.985 (single-scale 640: 0.931) | 0.85 |
| A3 top-1 / top-5, SigLIP2 + SIFT rerank (590 crops) | 0.861 / 0.981 | 0.65 / 0.85 |
| A3 others top-1: resnet50 0.702, dinov2 0.758, clip 0.739, siglip2 0.834; SIFT kNN 0.839 (85 ms/crop) | | |
| A3 ablation no gallery augmentation (siglip2) | 0.829 / 0.990 | |
| A4 unknown rejected / known drop | 0.132 / 1.6 pp (AUROC 0.790); cosine-only 0.220 / 3.8 pp | 0.70 / <=5 pp |
- A4 is a clear miss: the GroZi products are similar packaging; known-vs-unknown AUROC 0.79 means no threshold gets near 70% rejection within 5 pp.
| A5 det / FA (new protocol, 480 violations) | 0.848 / 0.094; oracle 0.933 / 0.002; naive rows 0.858 / 0.476; original protocol 0.570 / 0.134 (oracle 0.681 / 0.043) | 0.80 / <=0.10 |
| A6 latency RTX 2060 mean / p95 | 1.58 / 2.84 s (detect 0.12, recognize 1.11, align 0.35) | <=5 s |
| A6 latency CPU (i7 family 6 model 165) | 10.4 / 17.2 s | <=5 s (missed on CPU) |
| A6 add-a-SKU | 6.2 s (4.9 s model load) | <=30 s |
| C7 recognition errors | 82 of 590: 60 near-identical variants, 12 same subcategory, 8 partial/aspect, 2 cross-category | |
- A5 FA is at the cap (0.094 vs 0.10); val was 0.068. Most remaining FA comes from verification rejecting true matches (base compliance 0.898 on unedited planograms).
- Demo audit of s1_28 still 67.7% on its own GT-layout planogram (10 missing): detector/recognition misses on that shelf, not checked further.
