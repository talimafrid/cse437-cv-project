"""F1: class-agnostic product detector (YOLO fine-tuned on SKU-110K) with multi-scale inference.

SKU-110K products are small and dense; GP-180 products are large (about 13 per 3264x2448 photo).
To bridge the scale gap the whole image is run at several input sizes, and optionally shrunk into a
larger grey canvas ("shrink" < 1) so products look SKU-110K sized. Passes are merged by NMS.
Tiling is not used because it cuts large products in half.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils import nms


class Detector:
    def __init__(self, weights: str, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = 0 if device.startswith("cuda") else "cpu"

    def _run(self, img: Image.Image, imgsz: int, conf: float, iou: float, max_det: int):
        r = self.model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=self.device,
                               half=self.device == 0, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def predict(self, img: Image.Image, scales=(640,), shrink=(1.0,), conf=0.05, iou=0.5, max_det=1000):
        """Returns (boxes xyxy in original pixels, scores)."""
        W, H = img.size
        all_b, all_s = [], []
        for f in shrink:
            if f >= 1.0:
                canvas, off = img, (0, 0)
            else:                                   # paste the image, shrunk, in the canvas centre
                w, h = round(W * f), round(H * f)
                canvas = Image.new("RGB", (W, H), (114, 114, 114))
                off = ((W - w) // 2, (H - h) // 2)
                canvas.paste(img.resize((w, h), Image.BILINEAR), off)
            for s in scales:
                b, sc = self._run(canvas, s, conf, iou, max_det)
                if f < 1.0:
                    b = (b - [off[0], off[1], off[0], off[1]]) / f
                all_b.append(b)
                all_s.append(sc)
        boxes = np.concatenate(all_b) if all_b else np.zeros((0, 4))
        scores = np.concatenate(all_s) if all_s else np.zeros(0)
        if len(all_b) > 1 and len(boxes):
            keep = nms(boxes, scores, iou)
            boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return boxes, scores
