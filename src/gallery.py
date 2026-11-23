"""F6: gallery index of one studio image per product (+ synthetic views), with add / remove.

Index layout on disk (outputs/index/<backbone>/):
  emb.npy     float16 (P, V, D)  L2-normalised embeddings, V views per product
  skus.json   list of P product ids, e.g. "Food/Rice/17"
  sift.npz    RootSIFT keypoints/descriptors of each studio image (for verification)

Usage:
  python -m src.gallery build [--backbone dinov2_b]
  python -m src.gallery add --image path/to/new.jpg --sku Food/Tea/999
  python -m src.gallery remove --sku Food/Tea/999
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import gallery_products
from src.recognizer import Embedder, Sift, augment_views
from src.utils import get_device, load_config


class Gallery:
    def __init__(self, skus: list[str], emb: np.ndarray, sift: dict | None = None):
        assert emb.ndim == 3 and len(skus) == emb.shape[0]
        self.skus, self.emb, self.sift = list(skus), emb, sift or {}
        self._pos = {s: i for i, s in enumerate(self.skus)}

    def __len__(self):
        return len(self.skus)

    @property
    def nbytes(self) -> int:
        return int(self.emb.nbytes)

    def scores(self, q: np.ndarray) -> np.ndarray:
        """(N, D) query embeddings -> (N, P) product scores = max cosine over the product's views."""
        p, v, d = self.emb.shape
        s = q.astype(np.float32) @ self.emb.reshape(p * v, d).astype(np.float32).T
        return s.reshape(len(q), p, v).max(axis=2)

    def subset(self, exclude: set[str]) -> "Gallery":
        keep = [i for i, s in enumerate(self.skus) if s not in exclude]
        return Gallery([self.skus[i] for i in keep], self.emb[keep],
                       {s: self.sift[s] for s in (self.skus[i] for i in keep) if s in self.sift})

    def index_of(self, sku: str) -> int | None:
        return self._pos.get(sku)

    def add(self, sku: str, emb_views: np.ndarray, sift_feat=None):
        if sku in self._pos:                                     # re-adding replaces the old entry
            self.remove(sku)
        self.skus.append(sku)
        self.emb = np.concatenate([self.emb, emb_views[None].astype(self.emb.dtype)])
        self._pos[sku] = len(self.skus) - 1
        if sift_feat is not None:
            self.sift[sku] = sift_feat

    def remove(self, sku: str):
        i = self._pos.pop(sku)
        del self.skus[i]
        self.emb = np.delete(self.emb, i, axis=0)
        self.sift.pop(sku, None)
        self._pos = {s: k for k, s in enumerate(self.skus)}

    # -------------------------------------------------------- persistence
    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "emb.npy", self.emb.astype(np.float16))
        (folder / "skus.json").write_text(json.dumps(self.skus), encoding="utf-8")
        if self.sift:
            ks = sorted(self.sift)
            np.savez(folder / "sift.npz", skus=np.array(ks),
                     pts=np.array([self.sift[k][0] for k in ks], dtype=object),
                     des=np.array([self.sift[k][1].astype(np.float16) for k in ks], dtype=object),
                     shape=np.array([self.sift[k][2] for k in ks]))

    @classmethod
    def load(cls, folder: Path, with_sift: bool = True) -> "Gallery":
        skus = json.loads((folder / "skus.json").read_text(encoding="utf-8"))
        emb = np.load(folder / "emb.npy")
        sift = {}
        if with_sift and (folder / "sift.npz").exists():
            z = np.load(folder / "sift.npz", allow_pickle=True)
            for k, p, d, sh in zip(z["skus"], z["pts"], z["des"], z["shape"]):
                sift[str(k)] = (p.astype(np.float32), d.astype(np.float32), tuple(sh))
        return cls(skus, emb, sift)


def embed_product(img: Image.Image, sku: str, embedder: Embedder, cfg: dict, views: int | None = None) -> np.ndarray:
    v = views or cfg["recognizer"]["gallery_views"]
    return embedder.embed(augment_views(img, v, sku, cfg["seed"]), cfg["recognizer"]["batch_size"])


def build(cfg: dict, backbone: str, views: int | None = None, out: Path | None = None, with_sift: bool = True) -> Gallery:
    products = gallery_products(Path(cfg["paths"]["grozi"]))
    spec = cfg["recognizer"]["backbones"][backbone]
    emb = Embedder(backbone, spec, get_device(cfg))
    sift = Sift(cfg["recognizer"]["sift"]) if with_sift else None
    skus, embs, feats = [], [], {}
    t0 = time.time()
    for k, (sku, path) in enumerate(products.items()):
        img = Image.open(path).convert("RGB")
        embs.append(embed_product(img, sku, emb, cfg, views).astype(np.float16))
        skus.append(sku)
        if sift:
            feats[sku] = sift.features(img)
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(products)} products, {time.time() - t0:.0f}s", flush=True)
    g = Gallery(skus, np.stack(embs), feats)
    if out:
        g.save(out)
    return g


def index_dir(cfg: dict, backbone: str, views: int | None = None) -> Path:
    v = views or cfg["recognizer"]["gallery_views"]
    return Path(cfg["paths"]["index"]) / f"{backbone}_v{v}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "add", "remove"])
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--image")
    ap.add_argument("--sku")
    a = ap.parse_args()
    cfg = load_config()
    bb = a.backbone or cfg["recognizer"]["backbone"]
    folder = index_dir(cfg, bb, a.views)
    t0 = time.time()
    if a.cmd == "build":
        g = build(cfg, bb, a.views, folder)
        print(f"built {len(g)} products x {g.emb.shape[1]} views -> {folder} ({g.nbytes / 2**20:.1f} MB)")
        return
    if not a.sku:
        ap.error("--sku is required")
    g = Gallery.load(folder)
    if a.cmd == "add":
        if not a.image or not Path(a.image).exists():
            ap.error("--image must be an existing file")
        img = Image.open(a.image).convert("RGB")
        emb = Embedder(bb, cfg["recognizer"]["backbones"][bb], get_device(cfg))
        g.add(a.sku, embed_product(img, a.sku, emb, cfg, a.views), Sift(cfg["recognizer"]["sift"]).features(img))
    else:
        if g.index_of(a.sku) is None:
            raise SystemExit(f"{a.sku} is not in the gallery")
        g.remove(a.sku)
    g.save(folder)
    print(f"{a.cmd} {a.sku}: gallery now {len(g)} products, {time.time() - t0:.1f}s (includes model load)")


if __name__ == "__main__":
    main()
