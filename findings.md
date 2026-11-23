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
