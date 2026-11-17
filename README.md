# Retail Shelf Planogram Compliance with Open-Set SKU Recognition

> **CSE437 — Computer Vision** | Daffodil International University

An AI pipeline that audits supermarket shelves for planogram compliance by detecting every product facing, identifying it from a single-reference gallery (or flagging unknowns), and reporting violations against a reference planogram.

---

## Project Structure

```
Computer Vision Assignment/
├── configs/
│   ├── config.yaml          # Master pipeline configuration
│   └── sku110k.yaml         # YOLOv8 dataset definition
│
├── data/                    # Datasets (not tracked in git)
│   ├── sku110k/             # SKU-110K: 11,743 images + annotations
│   │   ├── images/          #   train_*.jpg, val_*.jpg, test_*.jpg
│   │   └── annotations/     #   annotations_{train,val,test}.csv
│   ├── grozi/               # GroZi-3.2k: gallery + shelf images
│   │   ├── Training/        #   3,235 studio reference images (gallery)
│   │   ├── Testing/         #   680 in-store shelf images (5 stores)
│   │   ├── bb/              #   Bounding box annotations
│   │   └── planos/          #   Planogram definitions
│   └── gp180/               # GP-180: planogram ground truth
│       └── *.json           #   Instance-level boxes + planograms
│
├── src/                     # Core Python modules
│   ├── __init__.py          # Package metadata
│   ├── detector.py          # F1: YOLOv8 product detection
│   ├── recognizer.py        # F2/F3: Embedding retrieval + open-set rejection
│   ├── planogram.py         # F4: Row clustering + edit-distance alignment
│   ├── violations.py        # F5: Synthetic violation injection
│   ├── gallery.py           # F6: Dynamic gallery management
│   ├── pipeline.py          # End-to-end orchestrator
│   ├── app.py               # Gradio web UI
│   └── utils.py             # Shared utilities
│
├── scripts/                 # Data processing & training scripts
│   └── convert_sku110k.py   # SKU-110K CSV → YOLO format converter
│
├── tests/                   # Unit tests (SQA)
│   ├── test_planogram.py    # F4 alignment tests
│   ├── test_recognizer.py   # F2/F3 recognition tests
│   └── test_violations.py   # F5 violation injection tests
│
├── models/                  # Trained model weights (not tracked)
├── outputs/                 # Inference outputs & gallery index
├── venv/                    # Python virtual environment
└── README.md
```

## Datasets

| Dataset | Contents | Local Path | Status |
|---------|----------|------------|--------|
| SKU-110K | 11,743 shelf images + bbox annotations | `data/sku110k/` | ✅ Complete |
| GroZi-3.2k | 10,619 gallery + 4,105 test images | `data/grozi/` | ✅ Complete |
| GP-180 | Planogram JSON files | `data/gp180/` | ✅ Complete |

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.13.14 |
| PyTorch | 2.15.0 (CPU) |
| Ultralytics | 8.4.215 |
| Gradio | 6.32.0 |
| torchvision | 0.29.2 |

### Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline Modules

| Module | Function | Description |
|--------|----------|-------------|
| **F1** | Detection | YOLOv8 class-agnostic product detector (SKU-110K) |
| **F2** | Recognition | CNN/ViT embedding retrieval against GroZi gallery |
| **F3** | Open-Set Rejection | Similarity threshold to flag unknown SKUs |
| **F4** | Planogram Alignment | Row clustering + edit distance compliance check |
| **F5** | Violation Injection | Synthetic non-compliant planogram generation |
| **F6** | Add-a-SKU | Dynamic gallery extension without retraining |

## Quick Start

```powershell
# 1. Convert SKU-110K to YOLO format
python scripts/convert_sku110k.py

# 2. Train detector (you handle this)
yolo detect train data=configs/sku110k.yaml model=yolov8m.pt epochs=50 imgsz=640

# 3. Run the pipeline (after all modules are implemented)
python -m src.app
```