# Progress Log

## Session: 2026-11-23

### Current Status
- **Phase:** 1 - Requirements & Discovery
- **Started:** 2026-11-23

### Actions Taken
-

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|

### Errors
| Error | Resolution |
|-------|------------|

## Session 2026-11-23
- Read brief (docx), PLAN.md, reports/ method plan; inspected git state (uncommitted rewrite, tests deleted, nothing run).
- Created task_plan.md / findings.md (legacy root mode), validated.
- Phase 1 done: installed ultralytics 8.4.216, opencv, scikit-learn, transformers 5.24.0, gradio 6.32.0, pandas, pyyaml, pytest; requirements.txt pinned (90 lines). faiss not used by code, not kept.
- Smoke import: all src modules + scripts OK; src/app.py fails (FeatureExtractor gone).
- SKU-110K download started in background.
- Phase 3 done: tests/test_core.py (17 pass). Found + fixed portrait-image clipping default in load_gp_boxes; added swap step + end_cost to row alignment (see findings table). prepare_data.py SKU annotation path now uses data/SKU110K_fixed/annotations.
- Rewrote src/app.py (Audit / Gallery add-remove / Inject tabs, config thresholds only). UI builds. Started dinov2_b gallery build in background.
- dinov2_b gallery build crashed (cv2 float32 in _perspective). Fixed, test added (18 pass), rebuild restarted.
- Phase 2 done: SKU-110K downloaded+extracted; splits frozen and committed (6a39dff2, branch project5-run). Unknown crops 15.3%/15.4% (not reseeded). YOLO conversion running.
- YOLO conversion done (8215/588/2935; 5 corrupt skipped, test_274 not in A1 subset). 122/122 row planograms written. Waiting on gallery build before training (VRAM).
- dinov2_b gallery built (3286x8, 77 MB, 11 min). Detector training: batch 16 spilled VRAM (1.0 s/it), aborted at ~3 min; restarted batch 8 (5.1 it/s, 4.3 GB), 1.5 h cap.
- Detector trained: 29 epochs in 1.5 h, val mAP50 0.928, mAP50-95 0.578. Weights at models/sku110k_yolo11s.pt. Phase 5 started.
- tune_detector: picked scales [640], shrink [1.0,0.5], conf 0.05: val recall 0.987 (301 GT boxes, 39 boxes/img, 0.16 s/img). Single-scale 1024/1280 without shrink collapses (0.62/0.39): GP images need downscaling.
- tune_recognizer (val GT crops, top1/top5): resnet50 0.618/0.814, dinov2_b 0.744/0.944, siglip2_b 0.854/0.980, clip_b16 0.728/0.937. Picked siglip2_b; SIFT rerank margin 0.02, lambda 0.1 -> 0.870 top1.
- tune_openset: threshold 0.513, val unknown rejected 0.304 at known top1 drop 0.039. tune_compliance: no setting under FA 0.08; diagnosis in findings.md (row structure + identical runs). Compliance config values from the tuner are unusable.
- A1-A4 test done (see metrics.json). A5 protocol changed with user approval: GT-layout rows + localisable injection; original protocol kept as secondary. Val oracle new 0.934/0.003 vs original 0.688/0.121. 19 tests pass. Retuning compliance.
- tune_compliance (new protocol) picked delta 0.05, min_inliers 20, gap 0.6, end 0.1: val 0.861 det / 0.068 FA, base compliance 0.888.
- README rewritten (real results, versions, data URLs, commands; GroZi source marked [CHECK]). PLAN.md marked superseded.
- Test A5 0.848/0.094, A6 GPU 1.58 s, CPU 10.4 s, add-SKU 6.2 s, C7 82 errors. Demo audit verified headless; demo planogram switched to GT layout.
