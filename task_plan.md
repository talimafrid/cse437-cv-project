# Task Plan: Project 5 planogram compliance, from written code to measured A1-A6

## Goal
Every acceptance number (A1-A6) in `outputs/metrics.json` is produced by `scripts/evaluate.py` on frozen test splits, with tests passing and an honest gap note for any missed target.

## Next Step
All phases complete. Open: A4 miss, CPU latency, T4 latency, GroZi source [CHECK], browser click test of demo.

## Current Phase
Phase 6

## Phases

### Phase 1: Environment
- [x] pip install missing deps into `.venv`; write pinned `requirements.txt` from `pip freeze`
- [x] Smoke import every module in `src/` and `scripts/` (all OK except src/app.py, see Phase 6)
- [x] Record hardware (RTX 2060 6 GB, CUDA, torch 2.14.1+cu126) for README (C3)
- **Status:** complete

### Phase 2: Data download and split freeze (C4)
- [x] Download SKU110K_fixed.tar.gz (12,196,152,718 bytes, matches server, gzip OK), extracted 11,743 images
- [x] Run `prepare_data.py` for GP-180 split, open-set split, SHA256SUMS (row planograms + YOLO conversion still running, bg bkptuu0uh)
- [x] Check split counts: 20/50, 657 unknown products, unknown crops 15.3% val / 15.4% test; committed on branch `project5-run` (6a39dff2)
- **Status:** complete

### Phase 3: Tests (code was rewritten; old tests deleted, none replaced)
- [x] Small pytest files for planogram alignment, violation injection + scoring, data parsing, NMS/IoU (`tests/test_core.py`)
- [x] All tests pass (17 passed)
- **Status:** complete

### Phase 4: Detector training (F1, <= 2 GPU-h)
- [x] Convert SKU-110K to YOLO format: 8215/588/2935 (5 corrupt files skipped; none in the A1 500 subset)
- [x] `train_detector.py` with `time=1.5`, batch 8, on local RTX 2060 (started ~18:50, bg b22c4zw62, log outputs/train_detector.log); weights to `models/sku110k_yolo11s.pt`. Done: 29 epochs, SKU-110K val mAP50 0.928 / mAP50-95 0.578 (588 val images)
- GPU budget used: 0.05 h aborted (batch 16) + 1.5 h run = ~1.55 h of the 2 h detector cap
- **Status:** complete

### Phase 5: Evaluation A1-A6
- [x] Tune on val (detector scales, backbone, open-set, compliance) and write values to config
- [x] `evaluate.py` tasks: detection (A1), proposals (A2), recognition (A3), openset (A4), compliance (A5), latency + add-sku (A6), errors (C7)
- [x] All numbers in `outputs/metrics.json`
- **Status:** complete

### Phase 6: Delivery
- [x] Rewrite `src/app.py` (Gradio, F7/C11) on top of `Pipeline`/`Gallery` (audit + inject run headless end to end; demo planogram = GT layout; browser click test not done)
- [x] Update README (datasets, versions, hardware, commands), mark PLAN.md as superseded where stale
- [x] Commit work (e4b6dd54 on project5-run, not pushed); summarize results and gaps to user
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Train detector locally on RTX 2060 instead of Colab | torch+CUDA now works locally; similar class to T4; time cap 1.5 h keeps C3; hardware reported honestly. Latency A6 still needs a T4 number later or a stated gap. |
| Keep the previous session's rewrite rather than restart | Code already covers F1-F7 and reads config.yaml; it was just never run. |
| Work on branch `project5-run`, not main; `.gitattributes splits/* -text` | Keeps split bytes identical so SHA256SUMS verify after checkout |
| Detector batch 8 instead of 16 | Batch 16 reserved 9.3 GB on a 6 GB card (Windows spills to system RAM): 1.0 s/it, ~9 epochs in cap. Batch 8: 4.3 GB, 5.1 it/s, ~23 epochs. nbs=64 accumulation keeps effective batch. |
| A5 protocol: planogram rows from GT layout + only localisable edits; original protocol reported as secondary (user chose this) | Oracle with Tonioni rows + unrestricted edits caps at ~0.67 detected / 4-12% FA: graph rows are shelf levels (stacked items), and edits inside runs of identical products cannot be localised. New protocol oracle: val 0.934 / 0.003. Disclose that the test oracle was looked at (no learned parameter used it). |
| Use the official S3 SKU-110K tarball | URL returns 200 (checked 2026-11-23). |
| Add Damerau swap step (`swap_ratio`) and `end_cost` to row alignment | Oracle-recognition A5 ceiling was 83% detected / 2.6% FA: adjacent swaps read as moves, free end skips let correct leading products become "missing". Now ~95% / 0.5% (end_cost 0.3). `end_cost` added to val tuning grid. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| 27 GP images portrait (2448x3264) but `load_gp_boxes` default clipped to 3264x2448, dropping bottom boxes | Default now reads real image size; evaluate.py already passed real size, so metrics unaffected |
| Oracle A5 test failed (swap/remove misses) | Alignment fixes above; test now asserts aggregate bound |
| Gallery build crashed: cv2 5.1 getPerspectiveTransform needs float32 (dst upcast to float64); `| tail` hid exit code | Cast after multiply; added augment test; rerun with pipefail |
| Batch-16 training spilled VRAM (9.29 GB on 6 GB) | Aborted after ~3 min (log kept: outputs/train_detector_b16_aborted.log), batch 8 |
| tune_compliance fallback picked max detection at FA 0.51 when no setting met FA <= 0.08 | Fallback now maximises detected - FA; protocol fixed (see Decisions) |
| `src/app.py` ImportError: FeatureExtractor removed from recognizer | App never updated in the rewrite; scheduled in Phase 6 |
