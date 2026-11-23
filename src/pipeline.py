"""End-to-end audit: detect -> recognise (+ open-set) -> group rows -> align with planogram -> report.

CLI (F7):
  python -m src.pipeline --image shelf.jpg --planogram outputs/planograms/s1_14.json [--out outputs/audit]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.gallery import Gallery, index_dir
from src.planogram import align_shelf, compliance, group_rows
from src.recognizer import Embedder, Sift, known_probability, openset_features
from src.utils import crop, draw_audit, dump_json, get_device, load_config, load_rgb


class Recognizer:
    """Retrieval + SIFT re-ranking + open-set decision + 1:1 verification for one batch of crops."""

    def __init__(self, cfg: dict, gallery: Gallery, embedder: Embedder | None, sift: Sift | None):
        self.cfg, self.g, self.emb, self.sift = cfg, gallery, embedder, sift
        self.rc = cfg["recognizer"]

    def recognize(self, crops, rerank: bool = True, use_sift_features: bool = True, q=None, qf=None) -> dict:
        """q: precomputed embeddings, qf: precomputed SIFT features of the crops (evaluation caches)."""
        rc, k = self.rc, self.rc["top_k"]
        if q is None:
            q = self.emb.embed(crops, rc["batch_size"])
        S = self.g.scores(q)
        top = np.argsort(-S, axis=1)[:, :k]
        top_s = np.take_along_axis(S, top, axis=1)
        n = len(crops)
        inl, hull = np.zeros(n), np.zeros(n)
        qf = list(qf) if qf is not None else [None] * n
        if self.sift is not None and use_sift_features:
            sc = rc["sift"]
            for i in range(n):
                if qf[i] is None:
                    qf[i] = self.sift.features(crops[i])
                ambiguous = rerank and (top_s[i, 0] - top_s[i, 1]) < sc["rerank_margin"]
                cand = range(k) if ambiguous else range(1)
                res = {c: self.sift.match(qf[i], self.g.sift[self.g.skus[top[i, c]]]) for c in cand
                       if self.g.skus[top[i, c]] in self.g.sift}
                if ambiguous and res:
                    new = top_s[i] + np.array([sc["rerank_lambda"] * min(res.get(c, (0, 0))[0], 50) / 50 for c in range(k)])
                    order = np.argsort(-new)
                    top[i], top_s[i] = top[i, order], top_s[i, order]
                    res = {list(order).index(c): v for c, v in res.items()}
                inl[i], hull[i] = res.get(0, (0, 0.0))
        feats = openset_features(top_s, inl, hull)
        os_cfg = rc["openset"]
        p_known = known_probability(feats, os_cfg["coef"]) if os_cfg.get("coef") else feats[:, 0]
        unknown = p_known < os_cfg["threshold"]
        self._last = dict(S=S, qf=qf, crops=crops)
        return dict(top=top, top_scores=top_s, inliers=inl, hull=hull, feats=feats, p_known=p_known,
                    unknown=unknown, labels=[("unknown" if u else self.g.skus[t[0]]) for u, t in zip(unknown, top)])

    def verify(self, i: int, sku: str) -> bool:
        """Is crop i the product `sku`? Embedding near the top, or enough SIFT inliers."""
        j = self.g.index_of(sku)
        if j is None:
            return False
        S, vc = self._last["S"], self.rc["verify"]
        if S[i, j] >= S[i].max() - vc["delta"]:
            return True
        if self.sift is None or sku not in self.g.sift:
            return False
        key = (i, sku)
        cache = self._last.setdefault("vcache", {})
        if key not in cache:
            qf = self._last["qf"][i] or self.sift.features(self._last["crops"][i])
            cache[key] = self.sift.match(qf, self.g.sift[sku])[0]
        return cache[key] >= vc["min_inliers"]


class Pipeline:
    def __init__(self, cfg: dict | None = None, backbone: str | None = None, detector=True, sift=True,
                 gallery: Gallery | None = None):
        self.cfg = cfg or load_config()
        dev = get_device(self.cfg)
        bb = backbone or self.cfg["recognizer"]["backbone"]
        self.gallery = gallery or Gallery.load(index_dir(self.cfg, bb), with_sift=sift)
        emb = Embedder(bb, self.cfg["recognizer"]["backbones"][bb], dev)
        self.rec = Recognizer(self.cfg, self.gallery, emb, Sift(self.cfg["recognizer"]["sift"]) if sift else None)
        self.det = None
        if detector:
            from src.detector import Detector
            self.det = Detector(self.cfg["detector"]["weights"], dev)

    def run(self, image, plan_rows: list[list[str]], boxes=None, two_level=True) -> dict:
        t = {}
        t0 = time.perf_counter()
        img = load_rgb(image)
        if boxes is None:
            p = self.cfg["detector"]["predict"]
            boxes, scores = self.det.predict(img, p["scales"], p["shrink"], p["conf"], p["iou"], p["max_det"])
        else:
            boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.ones(len(boxes))
        t["detect"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        crops = [crop(img, b) for b in boxes]
        rec = self.rec.recognize(crops) if crops else None
        t["recognize"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        pc = self.cfg["planogram"]
        rows = group_rows(boxes, pc["row_gap"])
        unknown = (lambda d: bool(rec["unknown"][d])) if rec else (lambda d: True)
        slots, extras = align_shelf(plan_rows, rows, self.rec.verify, unknown, pc["gap_cost"],
                                    pc["unknown_cost"], two_level, pc["swap_ratio"],
                                    pc["end_cost"])
        t["align"] = time.perf_counter() - t2
        t["total"] = time.perf_counter() - t0
        for s in slots + extras:
            d = s["det"]
            s["box"] = None if d is None else [round(float(v), 1) for v in boxes[d]]
            s["detected"] = None if d is None else rec["labels"][d]
            s["score"] = None if d is None else round(float(rec["top_scores"][d, 0]), 4)
        return dict(boxes=boxes, scores=scores, rows=rows, rec=rec, slots=slots, extras=extras,
                    compliance=compliance(slots), timing=t, image=img)


def write_report(res: dict, out_dir: Path, stem: str):
    """F7: annotated image, report JSON and per-slot CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_audit(res["image"], res["slots"], res["extras"]).save(out_dir / f"{stem}_audit.jpg", quality=90)
    keys = ["row", "col", "expected", "detected", "status", "score", "box"]
    counts = {k: sum(s["status"] == k for s in res["slots"]) for k in ("correct", "misplaced", "missing", "unknown")}
    dump_json(dict(image=stem, compliance=round(res["compliance"], 4), counts=counts, n_extra=len(res["extras"]),
                   timing_s={k: round(v, 3) for k, v in res["timing"].items()},
                   slots=[{k: s.get(k) for k in keys} for s in res["slots"]],
                   extras=[{k: s.get(k) for k in ("row", "gap_pos", "detected", "box")} for s in res["extras"]]),
              out_dir / f"{stem}_report.json")
    with open(out_dir / f"{stem}_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for s in res["slots"]:
            w.writerow({k: s.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--planogram", required=True, help="row planogram JSON (outputs/planograms/*.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    pipe = Pipeline(cfg)
    rows = json.loads(Path(a.planogram).read_text())["rows"]
    res = pipe.run(a.image, rows)
    out = Path(a.out or Path(cfg["paths"]["outputs"]) / "audit")
    write_report(res, out, Path(a.image).stem)
    print(f"compliance {res['compliance']:.1%}  timing {res['timing']['total']:.2f}s  -> {out}")


if __name__ == "__main__":
    main()
