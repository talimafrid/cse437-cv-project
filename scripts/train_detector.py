"""F1: fine-tune a COCO-pretrained YOLO on SKU-110K (single class), capped at the time budget.

  python scripts/train_detector.py            # settings from configs/config.yaml -> models/sku110k_yolo11s.pt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, get_device, load_config, seed_everything


def main():
    from ultralytics import YOLO
    cfg = load_config()
    seed_everything(cfg["seed"])
    t = cfg["detector"]["train"]
    model = YOLO(cfg["detector"]["base_model"])
    model.train(data=str(ROOT / "configs" / "sku110k.yaml"), imgsz=t["imgsz"], batch=t["batch"],
                time=t["time_hours"],              # hard wall-clock cap; LR schedule fits the cap
                epochs=100, max_det=t["max_det"], cos_lr=True, close_mosaic=2, seed=cfg["seed"], deterministic=False,
                device=0 if get_device(cfg) == "cuda" else "cpu", workers=6, cache=False, plots=True,
                project=str(Path(cfg["paths"]["outputs"]) / "train"), name="sku110k", exist_ok=True)
    best = Path(model.trainer.best)
    dst = Path(cfg["detector"]["weights"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dst)
    print(f"best weights -> {dst}")


if __name__ == "__main__":
    main()
