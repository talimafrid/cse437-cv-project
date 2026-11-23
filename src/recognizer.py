"""F2/F3: SKU recognition by retrieval, with open-set rejection. No per-SKU classifier is trained.

Embedders (frozen, pretrained): ResNet-50 (GeM pooling), DINOv2, SigLIP2, CLIP.
Classical: RootSIFT + Lowe ratio test + RANSAC homography (baseline and geometric re-ranker).
Crops are padded to square (not squashed) before resizing.
"""
from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np
from PIL import Image, ImageFilter

PAD_COLOR = (124, 116, 104)


def pad_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return canvas


class Embedder:
    """Frozen backbone -> L2-normalised global descriptor."""

    def __init__(self, name: str, spec: dict, device: str = "cpu"):
        import torch
        self.torch, self.name, self.kind, self.size, self.device = torch, name, spec["kind"], spec["size"], device
        if self.kind == "cnn":
            from torchvision.models import ResNet50_Weights, resnet50
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*list(m.children())[:-2])      # keep the feature map
            self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(spec["hf"])
            self.mean, self.std = list(proc.image_mean), list(proc.image_std)
            self.model = AutoModel.from_pretrained(spec["hf"])
        self.model.eval().to(device)
        if device.startswith("cuda"):
            self.model.half()

    def _tensor(self, imgs):
        t = self.torch
        x = np.stack([np.asarray(pad_square(i.convert("RGB"), self.size), dtype=np.float32) / 255.0 for i in imgs])
        x = (x - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)
        x = t.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        return x.half() if self.device.startswith("cuda") else x

    def embed(self, imgs: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        t = self.torch
        out = []
        with t.no_grad():
            for i in range(0, len(imgs), batch_size):
                x = self._tensor(imgs[i:i + batch_size])
                if self.kind == "cnn":
                    f = self.model(x).float().clamp(min=1e-6)
                    f = f.pow(3).mean(dim=(2, 3)).pow(1 / 3)                    # GeM, p=3
                elif self.kind == "dinov2":
                    h = self.model(pixel_values=x).last_hidden_state.float()
                    cls = t.nn.functional.normalize(h[:, 0], dim=1)
                    patch = t.nn.functional.normalize(h[:, 1:].mean(1), dim=1)
                    f = t.cat([cls, patch], dim=1)
                else:                                                              # siglip / clip
                    f = self.model.get_image_features(pixel_values=x)
                    f = (f if isinstance(f, t.Tensor) else f.pooler_output).float()
                out.append(t.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- gallery augmentation

def _rng_for(key: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5(f"{key}|{seed}".encode()).hexdigest()[:8], 16))


def _perspective(img, rng, strength=0.12):
    w, h = img.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-strength, strength, (4, 2)) * [w, h]).astype(np.float32)   # cv2 needs float32
    m = cv2.getPerspectiveTransform(src, dst)
    a = cv2.warpPerspective(np.asarray(img), m, (w, h), borderValue=PAD_COLOR)
    return Image.fromarray(a)


def _jitter(img, rng):
    a = np.asarray(img, np.float32)
    a = a * rng.uniform(0.6, 1.3) + rng.uniform(-25, 25)                 # brightness / offset
    mean = a.mean()
    a = (a - mean) * rng.uniform(0.7, 1.2) + mean                         # contrast
    gray = a.mean(axis=2, keepdims=True)
    a = gray + (a - gray) * rng.uniform(0.6, 1.3)                         # saturation
    a = a * rng.uniform(0.9, 1.1, 3)                                       # colour cast
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _glare(img, rng):
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    sy, sx = rng.uniform(0.1, 0.35) * h, rng.uniform(0.1, 0.35) * w
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))[..., None]
    return Image.fromarray(np.clip(a + g * rng.uniform(80, 200), 0, 255).astype(np.uint8))


def _blur_lowres(img, rng):
    w, h = img.size
    s = rng.uniform(0.15, 0.4)
    img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))


def _crop(img, rng):
    w, h = img.size
    f = rng.uniform(0.75, 0.95)
    cw, ch = int(w * f), int(h * f)
    x, y = int(rng.uniform(0, w - cw)), int(rng.uniform(0, h - ch))
    return img.crop((x, y, x + cw, y + ch))


def _jpeg_noise_rot(img, rng):
    img = img.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, expand=True, fillcolor=PAD_COLOR)
    a = np.asarray(img, np.float32) + rng.normal(0, 6, (img.size[1], img.size[0], 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=int(rng.uniform(20, 50)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def augment_views(img: Image.Image, n: int, key: str, seed: int = 0) -> list[Image.Image]:
    """View 0 is the studio image itself; the rest are seeded synthetic shelf-like versions."""
    img = img.convert("RGB")
    if max(img.size) > 640:                                                # speed: studio images are large
        s = 640 / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BICUBIC)
    rng = _rng_for(key, seed)
    ops = [
        lambda i: _blur_lowres(i, rng),
        lambda i: _jitter(i, rng),
        lambda i: _perspective(i, rng),
        lambda i: _glare(i, rng),
        lambda i: _crop(i, rng),
        lambda i: _blur_lowres(_jitter(_perspective(i, rng), rng), rng),
        lambda i: _jpeg_noise_rot(_glare(i, rng), rng),
    ]
    views = [img]
    for k in range(n - 1):
        views.append(ops[k % len(ops)](img))
    return views


# ---------------------------------------------------------------- SIFT

class Sift:
    """RootSIFT features, ratio-test matching and RANSAC homography verification."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.det = cv2.SIFT_create(nfeatures=spec["max_features"])

    def features(self, img: Image.Image):
        side = self.spec["crop_side"]
        g = np.asarray(img.convert("L"))
        s = side / max(g.shape)
        g = cv2.resize(g, (max(8, round(g.shape[1] * s)), max(8, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
        kp, des = self.det.detectAndCompute(g, None)
        if des is None or len(kp) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), g.shape
        des = des / (des.sum(axis=1, keepdims=True) + 1e-7)
        des = np.sqrt(des).astype(np.float32)                              # RootSIFT
        return np.float32([k.pt for k in kp]), des, g.shape

    def match(self, q, r) -> tuple[int, float]:
        """Returns (RANSAC inliers, inlier convex-hull area / query area)."""
        (qp, qd, qshape), (rp, rd, _) = q, r
        if len(qd) < 4 or len(rd) < 4:
            return 0, 0.0
        sims = qd @ rd.T
        idx = np.argsort(-sims, axis=1)[:, :2]
        d1 = np.sqrt(np.clip(2 - 2 * sims[np.arange(len(qd)), idx[:, 0]], 0, None))
        d2 = np.sqrt(np.clip(2 - 2 * sims[np.arange(len(qd)), idx[:, 1]], 0, None))
        good = d1 < self.spec["ratio"] * d2
        if good.sum() < 4:
            return 0, 0.0
        src, dst = qp[good], rp[idx[good, 0]]
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.spec["ransac_px"])
        if mask is None:
            return 0, 0.0
        inl = src[mask.ravel().astype(bool)]
        if len(inl) < 3:
            return int(len(inl)), 0.0
        hull = cv2.contourArea(cv2.convexHull(inl.astype(np.float32)))
        return int(len(inl)), float(hull / (qshape[0] * qshape[1]))


# ---------------------------------------------------------------- open-set scoring

def openset_features(top_scores: np.ndarray, inliers: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """[c1, c1 - c2, min(inliers, 50) / 50, hull fraction] per crop."""
    c1, c2 = top_scores[:, 0], top_scores[:, 1]
    return np.stack([c1, c1 - c2, np.minimum(inliers, 50) / 50.0, hull], axis=1)


def known_probability(feats: np.ndarray, coef: list[float]) -> np.ndarray:
    """Logistic model fitted on validation crops: coef = [w1..w4, bias]."""
    w, b = np.asarray(coef[:-1]), coef[-1]
    return 1.0 / (1.0 + np.exp(-(feats @ w + b)))
