"""Config, seeding, image helpers, box maths and the audit overlay."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

STATUS_COLORS = {
    "correct": (46, 204, 113),
    "misplaced": (231, 76, 60),
    "unknown": (52, 152, 219),
    "missing": (241, 196, 15),
    "extra": (155, 89, 182),
}


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    """Loads the YAML config and makes every relative path absolute against the repo root."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def save_config_values(updates: dict, path: str | Path = ROOT / "configs" / "config.yaml"):
    """Writes tuned values back into config.yaml (keys as dotted paths)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for dotted, value in updates.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = value
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict) -> str:
    if cfg.get("device", "auto") != "auto":
        return cfg["device"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_rgb(path_or_img) -> Image.Image:
    """Path, PIL image, or RGB numpy array -> PIL RGB. Numpy input must already be RGB."""
    if isinstance(path_or_img, (str, Path)):
        return Image.open(path_or_img).convert("RGB")
    if isinstance(path_or_img, np.ndarray):
        return Image.fromarray(path_or_img).convert("RGB")
    return path_or_img.convert("RGB")


def crop(img: Image.Image, box) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, max(x1 + 1, int(round(x2)))), min(h, max(y1 + 1, int(round(y2))))
    return img.crop((x1, y1, x2, y2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS, returns kept indices (sorted by score)."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou]
    return np.array(keep, dtype=int)


def draw_audit(img: Image.Image, slots: list[dict], extras: list[dict] = ()) -> Image.Image:
    """Draws each slot box coloured by status with its SKU label. Missing slots have no box."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(3, out.size[0] // 600)
    for s in list(slots) + list(extras):
        if s.get("box") is None:
            continue
        status = s["status"]
        color = STATUS_COLORS.get(status, (150, 150, 150))
        x1, y1, x2, y2 = s["box"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{status}: {(s.get('detected') or '?').split('/')[-1]}"
        if status != "correct" and s.get("expected"):
            label += f" (want {s['expected'].split('/')[-1]})"
        tw = 7 * len(label) * lw // 3
        d.rectangle([x1, max(0, y1 - 6 * lw), x1 + tw, y1], fill=color)
        d.text((x1 + lw, max(0, y1 - 6 * lw)), label, fill=(255, 255, 255))
    return out


def dump_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
