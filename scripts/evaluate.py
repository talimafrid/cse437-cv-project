"""Tuning (validation split only) and evaluation (test split) for every number in the report (C8, C10).

  python scripts/evaluate.py --task tune        # writes tuned values to configs/config.yaml (val only)
  python scripts/evaluate.py --task all         # A1-A6 + ablations on test -> outputs/metrics.json
  python scripts/evaluate.py --task recognition # or: detection, proposals, openset, compliance, latency, errors

Every task reads the frozen splits (hash-checked) and merges its results into outputs/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from src.data import gp_image_path, load_gp_boxes, read_list, verify_hashes
from src.gallery import Gallery, build, embed_product, index_dir
from src.planogram import align_shelf, compliance, group_rows
from src.pipeline import Pipeline, Recognizer
from src.recognizer import Embedder, Sift, openset_features
from src.utils import (crop, dump_json, get_device, iou_matrix, load_config, save_config_values,
                       seed_everything)
from src.violations import inject, score as score_violations


class Ctx:
    """Caches GT crops, embeddings and SIFT features so each task stays fast."""

    def __init__(self, cfg):
        self.cfg, self.dev = cfg, get_device(cfg)
        self.out = Path(cfg["paths"]["outputs"])
        self.cache = self.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.splits = Path(cfg["paths"]["splits"])
        verify_hashes(self.splits)
        self.grozi = Path(cfg["paths"]["grozi"])
        self._crops, self._emb, self._gal = {}, {}, {}
        self.unknown = set(read_list(self.splits / "openset_unknown.txt"))

    def names(self, split):
        return read_list(self.splits / f"gp_{split}.txt")

    def crops(self, split):
        """Flat list of GT crops with their SKU and source image."""
        if split not in self._crops:
            crops, skus, src = [], [], []
            for n in self.names(split):
                img = Image.open(gp_image_path(self.grozi, n)).convert("RGB")
                boxes, s = load_gp_boxes(self.grozi, n, *img.size)
                crops += [crop(img, b) for b in boxes]
                skus += s
                src += [n] * len(s)
            self._crops[split] = (crops, skus, src)
        return self._crops[split]

    def gallery(self, bb, views=None):
        key = (bb, views)
        if key not in self._gal:
            folder = index_dir(self.cfg, bb, views)
            if not (folder / "emb.npy").exists():
                print(f"building gallery {folder.name} ...", flush=True)
                build(self.cfg, bb, views, folder)
            self._gal[key] = Gallery.load(folder)
        return self._gal[key]

    def embedder(self, bb):
        if bb not in self._emb:
            self._emb[bb] = Embedder(bb, self.cfg["recognizer"]["backbones"][bb], self.dev)
        return self._emb[bb]

    def crop_emb(self, bb, split):
        f = self.cache / f"emb_{bb}_{split}.npy"
        if not f.exists():
            np.save(f, self.embedder(bb).embed(self.crops(split)[0], self.cfg["recognizer"]["batch_size"]))
        return np.load(f)

    def crop_sift(self, split):
        f = self.cache / f"sift_{split}.pkl"
        if not f.exists():
            s = Sift(self.cfg["recognizer"]["sift"])
            pickle.dump([s.features(c) for c in self.crops(split)[0]], open(f, "wb"))
        return pickle.load(open(f, "rb"))

    def recognizer(self, bb, gallery=None):
        return Recognizer(self.cfg, gallery or self.gallery(bb), None, Sift(self.cfg["recognizer"]["sift"]))


def merge_metrics(cfg, section, value):
    p = Path(cfg["paths"]["outputs"]) / "metrics.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m[section] = value
    dump_json(m, p)
    print(json.dumps({section: value}, indent=1)[:3000])


def ranks(S, gallery, skus):
    """Rank (0 = top-1) of the true product per crop; -1 if not in the gallery."""
    out = []
    for i, s in enumerate(skus):
        j = gallery.index_of(s)
        out.append(-1 if j is None else int((S[i] > S[i, j]).sum()))
    return np.array(out)


def topk_acc(r, k):
    return float(np.mean((r >= 0) & (r < k)))


# ---------------------------------------------------------------- A1: SKU-110K detection

def task_detection(ctx):
    from ultralytics import YOLO
    cfg = ctx.cfg
    root = Path(cfg["paths"]["sku110k_yolo"])
    lst = ctx.out / "sku110k_test_subset_paths.txt"
    lst.write_text("\n".join(str(root / "images" / "test" / n) for n in read_list(ctx.splits / "sku110k_test_subset.txt")
                             if (root / "images" / "test" / n).exists()) + "\n")
    yml = ctx.out / "sku110k_test_subset.yaml"
    yml.write_text(f"path: {root.as_posix()}\ntrain: {lst.as_posix()}\nval: {lst.as_posix()}\ntest: {lst.as_posix()}\n"
                   f"nc: 1\nnames:\n  0: object\n")
    m = YOLO(cfg["detector"]["weights"])
    res = {}
    for imgsz in (640, 1024):
        r = m.val(data=str(yml), split="test", imgsz=imgsz, batch=8, conf=0.001, iou=0.7,
                  max_det=cfg["detector"]["predict"]["max_det"], device=0 if ctx.dev == "cuda" else "cpu",
                  plots=False, verbose=False, project=str(ctx.out / "yolo_val"), name=f"test{imgsz}", exist_ok=True)
        res[f"imgsz{imgsz}"] = dict(map50=round(float(r.box.map50), 4), map50_95=round(float(r.box.map), 4),
                                    precision=round(float(r.box.mp), 4), recall=round(float(r.box.mr), 4))
    res["n_images"] = len(read_list(lst))
    res["target_map50"] = 0.80
    merge_metrics(cfg, "A1_sku110k_detection", res)


# ---------------------------------------------------------------- A2: proposal recall on GP-180

def proposal_recall(ctx, det, split, scales, shrink, conf):
    hits = total = nbox = 0
    t = []
    for n in ctx.names(split):
        img = Image.open(gp_image_path(ctx.grozi, n)).convert("RGB")
        gt, _ = load_gp_boxes(ctx.grozi, n, *img.size)
        t0 = time.perf_counter()
        b, _ = det.predict(img, scales, shrink, conf, ctx.cfg["detector"]["predict"]["iou"], 1000)
        t.append(time.perf_counter() - t0)
        if len(b):
            hits += int((iou_matrix(np.array(gt), b).max(axis=1) >= 0.5).sum())
        total += len(gt)
        nbox += len(b)
    return dict(recall=round(hits / total, 4), gt_boxes=total, mean_boxes=round(nbox / len(ctx.names(split)), 1),
                sec_per_image=round(float(np.mean(t)), 3))


def tune_detector(ctx):
    from src.detector import Detector
    det = Detector(ctx.cfg["detector"]["weights"], ctx.dev)
    grid = []
    for scales in ([640], [1024], [1280], [640, 1024], [640, 1024, 1280]):
        for shrink in ([1.0], [1.0, 0.5], [1.0, 0.6, 0.35]):
            for conf in (0.05, 0.15):
                r = proposal_recall(ctx, det, "val", scales, shrink, conf)
                grid.append(dict(scales=scales, shrink=shrink, conf=conf, **r))
                print(grid[-1], flush=True)
    best_recall = max(g["recall"] for g in grid)
    # cheapest setting within 1 pp of the best recall
    pick = min((g for g in grid if g["recall"] >= best_recall - 0.01), key=lambda g: g["sec_per_image"])
    save_config_values({"detector.predict.scales": pick["scales"], "detector.predict.shrink": pick["shrink"],
                        "detector.predict.conf": pick["conf"]})
    merge_metrics(ctx.cfg, "tune_detector_val", dict(grid=grid, picked=pick))


def task_proposals(ctx):
    from src.detector import Detector
    p = ctx.cfg["detector"]["predict"]
    det = Detector(ctx.cfg["detector"]["weights"], ctx.dev)
    res = dict(tuned=proposal_recall(ctx, det, "test", p["scales"], p["shrink"], p["conf"]),
               ablation_single_scale_640=proposal_recall(ctx, det, "test", [640], [1.0], p["conf"]),
               target_recall=0.85, settings=p)
    merge_metrics(ctx.cfg, "A2_gp180_proposal_recall", res)


# ---------------------------------------------------------------- A3: recognition on GT crops

def sift_knn_baseline(ctx, split, max_crops=None):
    """Classical baseline: RootSIFT k-NN voting over all gallery descriptors, RANSAC on top 10."""
    import torch
    g = ctx.gallery(ctx.cfg["recognizer"]["backbone"])
    sift = Sift(ctx.cfg["recognizer"]["sift"])
    skus = [s for s in g.skus if s in g.sift and len(g.sift[s][1])]
    des = np.concatenate([g.sift[s][1] for s in skus])
    owner = np.concatenate([np.full(len(g.sift[s][1]), k) for k, s in enumerate(skus)])
    dev = "cuda" if ctx.dev == "cuda" else "cpu"
    D = torch.from_numpy(des).to(dev).half() if dev == "cuda" else torch.from_numpy(des)
    owner_t = torch.from_numpy(owner).to(dev)
    crops, true, _ = ctx.crops(split)
    feats = ctx.crop_sift(split)
    n = len(crops) if max_crops is None else min(max_crops, len(crops))
    r, t = [], []
    for i in range(n):
        t0 = time.perf_counter()
        qp, qd, _ = feats[i]
        if len(qd) < 2:
            r.append(-1); t.append(time.perf_counter() - t0); continue
        q = torch.from_numpy(qd).to(dev)
        q = q.half() if dev == "cuda" else q
        best2 = []
        for c in range(0, len(D), 400_000):
            s = q @ D[c:c + 400_000].T
            v, ix = s.float().topk(2, dim=1)
            best2.append((v, ix + c))
        v = torch.cat([b[0] for b in best2], 1)
        ix = torch.cat([b[1] for b in best2], 1)
        v, o = v.topk(2, dim=1)
        ix = torch.gather(ix, 1, o)
        d1, d2 = (2 - 2 * v[:, 0]).clamp(min=0).sqrt(), (2 - 2 * v[:, 1]).clamp(min=0).sqrt()
        good = d1 < sift.spec["ratio"] * d2
        votes = torch.bincount(owner_t[ix[good, 0]], minlength=len(skus)).cpu().numpy()
        cand = np.argsort(-votes)[:10]
        inl = np.array([sift.match(feats[i], g.sift[skus[c]])[0] for c in cand])
        order = cand[np.lexsort((-votes[cand], -inl))]
        t.append(time.perf_counter() - t0)
        r.append(int(np.where(np.array(skus)[order] == true[i])[0][0]) if true[i] in np.array(skus)[order] else 99)
    r = np.array(r)
    return dict(top1=round(topk_acc(r, 1), 4), top5=round(topk_acc(r, 5), 4), n_crops=n,
                ms_per_crop=round(1000 * float(np.mean(t)), 1), index_mb=round((des.nbytes / 2 + owner.nbytes) / 2**20, 1))


def embed_speed(ctx, bb, n=64):
    crops = ctx.crops("val")[0][:n]
    e = ctx.embedder(bb)
    e.embed(crops[:8])
    if ctx.dev == "cuda":
        import torch; torch.cuda.synchronize()
    t0 = time.perf_counter()
    e.embed(crops, ctx.cfg["recognizer"]["batch_size"])
    if ctx.dev == "cuda":
        import torch; torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / len(crops)


def eval_backbone(ctx, bb, split, views=None):
    g = ctx.gallery(bb, views)
    S = g.scores(ctx.crop_emb(bb, split))
    r = ranks(S, g, ctx.crops(split)[1])
    return dict(top1=round(topk_acc(r, 1), 4), top5=round(topk_acc(r, 5), 4), index_mb=round(g.nbytes / 2**20, 1))


def eval_rerank(ctx, bb, split, gallery=None):
    rec = ctx.recognizer(bb, gallery)
    crops, skus, _ = ctx.crops(split)
    out = rec.recognize(crops, rerank=True, q=ctx.crop_emb(bb, split), qf=ctx.crop_sift(split))
    g = rec.g
    r = []
    for i, s in enumerate(skus):
        hit = np.where(out["top"][i] == g.index_of(s))[0] if g.index_of(s) is not None else []
        r.append(int(hit[0]) if len(hit) else 99)
    r = np.array(r)
    return dict(top1=round(topk_acc(r, 1), 4), top5=round(topk_acc(r, 5), 4)), out


def task_recognition(ctx):
    cfg = ctx.cfg
    res = {"n_test_crops": len(ctx.crops("test")[0]), "backbones": {}}
    for bb in cfg["recognizer"]["backbones"]:
        res["backbones"][bb] = dict(**eval_backbone(ctx, bb, "test"), ms_per_crop=round(embed_speed(ctx, bb), 2))
        print(bb, res["backbones"][bb], flush=True)
    best = cfg["recognizer"]["backbone"]
    res["best_backbone"] = best
    res["best_plus_sift_rerank"] = eval_rerank(ctx, best, "test")[0]
    res["ablation_no_gallery_augmentation"] = eval_backbone(ctx, best, "test", views=1)
    res["sift_knn_baseline"] = sift_knn_baseline(ctx, "test")
    res["targets"] = dict(top1=0.65, top5=0.85)
    merge_metrics(cfg, "A3_recognition_gt_crops", res)


def tune_recognizer(ctx):
    """Backbone choice and SIFT re-ranking settings, on validation crops only."""
    cfg = ctx.cfg
    val = {bb: eval_backbone(ctx, bb, "val") for bb in cfg["recognizer"]["backbones"]}
    best = max(val, key=lambda b: (val[b]["top1"], val[b]["top5"]))
    save_config_values({"recognizer.backbone": best})
    cfg["recognizer"]["backbone"] = best
    grid = []
    for margin in (0.0, 0.02, 0.05, 0.1, 1.0):
        for lam in (0.1, 0.3, 0.6):
            cfg["recognizer"]["sift"].update(rerank_margin=margin, rerank_lambda=lam)
            grid.append(dict(margin=margin, lam=lam, **eval_rerank(ctx, best, "val")[0]))
    pick = max(grid, key=lambda g: (g["top1"], -g["margin"]))
    cfg["recognizer"]["sift"].update(rerank_margin=pick["margin"], rerank_lambda=pick["lam"])
    save_config_values({"recognizer.sift.rerank_margin": pick["margin"], "recognizer.sift.rerank_lambda": pick["lam"]})
    merge_metrics(cfg, "tune_recognizer_val", dict(backbones=val, picked_backbone=best, rerank_grid=grid, picked=pick))


# ---------------------------------------------------------------- A4: open-set rejection

def openset_outputs(ctx, split):
    bb = ctx.cfg["recognizer"]["backbone"]
    g = ctx.gallery(bb).subset(ctx.unknown)
    crops, skus, _ = ctx.crops(split)
    _, out = eval_rerank(ctx, bb, split, gallery=g)
    is_unknown = np.array([s in ctx.unknown for s in skus])
    correct = np.array([g.skus[t[0]] == s for t, s in zip(out["top"], skus)])
    return out["feats"], is_unknown, correct


def tune_openset(ctx):
    from sklearn.linear_model import LogisticRegression
    X, unk, correct = openset_outputs(ctx, "val")
    y = (~unk) & correct                        # keep = known and right; rejecting wrong knowns is free
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    coef = [float(c) for c in lr.coef_[0]] + [float(lr.intercept_[0])]
    p = lr.predict_proba(X)[:, 1]
    base = correct[~unk].mean()
    best = None
    for th in np.unique(np.round(p, 4)):
        rej = p < th
        drop = base - (correct & ~rej)[~unk].mean()
        if drop <= 0.04 and (best is None or rej[unk].mean() > best[1]):   # 1 pp headroom under the 5 pp rule
            best = (float(th), float(rej[unk].mean()), float(drop))
    th = best[0] if best else 0.0
    save_config_values({"recognizer.openset.coef": coef, "recognizer.openset.threshold": th})
    ctx.cfg["recognizer"]["openset"].update(coef=coef, threshold=th)
    merge_metrics(ctx.cfg, "tune_openset_val", dict(coef=coef, threshold=th, val_unknown_rejected=best and best[1],
                                                     val_known_top1_drop=best and best[2],
                                                     n_val_unknown=int(unk.sum()), n_val_known=int((~unk).sum())))


def task_openset(ctx):
    from sklearn.metrics import roc_auc_score
    from src.recognizer import known_probability
    X, unk, correct = openset_outputs(ctx, "test")
    oc = ctx.cfg["recognizer"]["openset"]
    p = known_probability(X, oc["coef"])
    rej = p < oc["threshold"]
    before = float(correct[~unk].mean())
    after = float((correct & ~rej)[~unk].mean())
    res = dict(n_known=int((~unk).sum()), n_unknown=int(unk.sum()), unknown_rejected=round(float(rej[unk].mean()), 4),
               known_top1_before=round(before, 4), known_top1_after=round(after, 4),
               known_top1_drop_pp=round(100 * (before - after), 2), auroc_known_vs_unknown=round(float(roc_auc_score(~unk, p)), 4),
               threshold=oc["threshold"], targets=dict(unknown_rejected=0.70, max_drop_pp=5))
    # ablation: plain top-1 cosine threshold at the same known-drop budget (chosen on val)
    Xv, unkv, corrv = openset_outputs(ctx, "val")
    basev = corrv[~unkv].mean()
    ths = [t for t in np.unique(np.round(Xv[:, 0], 4)) if basev - (corrv & (Xv[:, 0] >= t))[~unkv].mean() <= 0.04]
    tc = max(ths) if ths else 0.0
    rj = X[:, 0] < tc
    res["ablation_cosine_only"] = dict(threshold=float(tc), unknown_rejected=round(float(rj[unk].mean()), 4),
                                       known_top1_drop_pp=round(100 * (before - float((correct & ~rj)[~unk].mean())), 2))
    merge_metrics(ctx.cfg, "A4_openset", res)


# ---------------------------------------------------------------- A5: injected violations

def shelf_runs(ctx, split, oracle=False):
    """One recognition run per shelf image (detector + recogniser, or GT boxes + GT labels)."""
    cfg = ctx.cfg
    f = ctx.cache / f"shelf_{split}_{'oracle' if oracle else cfg['recognizer']['backbone']}.pkl"
    pipe = None if oracle else Pipeline(cfg, gallery=ctx.gallery(cfg["recognizer"]["backbone"]))
    runs = []
    for n in ctx.names(split):
        plan = json.loads((Path(cfg["paths"]["planograms"]) / f"{n}.json").read_text())["rows"]
        img = Image.open(gp_image_path(ctx.grozi, n)).convert("RGB")
        gt_boxes, gt_skus = load_gp_boxes(ctx.grozi, n, *img.size)
        # main A5 protocol: planogram rows = GT layout grouped by the pipeline's own row rule (findings.md)
        plan_gt = [[gt_skus[d] for d in row] for row in group_rows(np.array(gt_boxes), cfg["planogram"]["row_gap"])]
        if oracle:
            runs.append(dict(name=n, plan=plan, plan_gt=plan_gt, boxes=np.array(gt_boxes), gt=gt_skus))
        else:
            p = cfg["detector"]["predict"]
            boxes, _ = pipe.det.predict(img, p["scales"], p["shrink"], p["conf"], p["iou"], p["max_det"])
            crops = [crop(img, b) for b in boxes]
            rec = Recognizer(cfg, pipe.gallery, pipe.rec.emb, pipe.rec.sift)
            out = rec.recognize(crops)
            runs.append(dict(name=n, plan=plan, plan_gt=plan_gt, boxes=boxes, rec=rec, unknown=out["unknown"]))
    return runs


def compliance_eval(ctx, runs, oracle=False, two_level=True, seeds=None, original=False):
    """original=True: Tonioni graph rows and unrestricted injection (the first protocol, kept as a
    secondary result). Default: GT-layout rows and localisable edits only."""
    pc, vc = ctx.cfg["planogram"], ctx.cfg["violations"]
    pool = ctx.gallery(ctx.cfg["recognizer"]["backbone"]).skus
    tot = dict(viol_detected=0, viol_total=0, false_alarms=0, unchanged_total=0)
    base = []
    for run in runs:
        rows = group_rows(run["boxes"], pc["row_gap"])
        if oracle:
            verified = lambda d, s, gt=run["gt"]: gt[d] == s
            unknown = lambda d: False
        else:
            verified, unknown = run["rec"].verify, (lambda d, u=run["unknown"]: bool(u[d]))
        plan = run["plan"] if original else run["plan_gt"]
        slots, _ = align_shelf(plan, rows, verified, unknown, pc["gap_cost"], pc["unknown_cost"], two_level, pc["swap_ratio"], pc["end_cost"])
        base.append(compliance(slots))
        for seed in (seeds or vc["seeds"]):
            new, truth = inject(plan, pool, zlib.crc32(f"{run['name']}|{seed}".encode()),
                                vc["rate_min"], vc["rate_max"], vc["ops"], localisable=not original)
            slots, extras = align_shelf(new, rows, verified, unknown, pc["gap_cost"], pc["unknown_cost"], two_level, pc["swap_ratio"], pc["end_cost"])
            for k, v in score_violations(slots, extras, truth).items():
                tot[k] += v
    return dict(detected_rate=round(tot["viol_detected"] / max(1, tot["viol_total"]), 4),
                false_alarm_rate=round(tot["false_alarms"] / max(1, tot["unchanged_total"]), 4),
                compliance_on_original_planograms=round(float(np.mean(base)), 4), **tot)


def tune_compliance(ctx):
    runs = shelf_runs(ctx, "val")
    vc, pc = ctx.cfg["recognizer"]["verify"], ctx.cfg["planogram"]
    grid = []
    for delta in (0.0, 0.02, 0.05, 0.1):
        for mi in (8, 12, 20, 1000):
            for gap in (0.4, 0.6, 0.8):
                for end in (0.1, 0.3):
                    vc.update(delta=delta, min_inliers=mi)
                    pc.update(gap_cost=gap, end_cost=end)
                    for r in runs:
                        r["rec"]._last.pop("vcache", None)
                    grid.append(dict(delta=delta, min_inliers=mi, gap_cost=gap, end_cost=end, **compliance_eval(ctx, runs)))
                    print(grid[-1], flush=True)
    ok = [g for g in grid if g["false_alarm_rate"] <= 0.08]
    pick = (max(ok, key=lambda g: (g["detected_rate"], -g["false_alarm_rate"])) if ok
            else max(grid, key=lambda g: g["detected_rate"] - g["false_alarm_rate"]))   # no setting meets the cap
    save_config_values({"recognizer.verify.delta": pick["delta"], "recognizer.verify.min_inliers": pick["min_inliers"],
                        "planogram.gap_cost": pick["gap_cost"], "planogram.end_cost": pick["end_cost"]})
    vc.update(delta=pick["delta"], min_inliers=pick["min_inliers"])
    pc.update(gap_cost=pick["gap_cost"], end_cost=pick["end_cost"])
    merge_metrics(ctx.cfg, "tune_compliance_val", dict(grid=grid, picked=pick))


def task_compliance(ctx):
    runs = shelf_runs(ctx, "test")
    res = dict(pipeline=compliance_eval(ctx, runs), targets=dict(detected=0.80, max_false_alarm=0.10))
    for r in runs:
        r["rec"]._last.pop("vcache", None)
    res["ablation_naive_row_matching"] = compliance_eval(ctx, runs, two_level=False)
    for r in runs:
        r["rec"]._last.pop("vcache", None)
    oracle_runs = shelf_runs(ctx, "test", oracle=True)
    res["oracle_gt_boxes_and_labels"] = compliance_eval(ctx, oracle_runs, oracle=True)
    res["protocol"] = "planogram rows from GT layout (pipeline row rule); only localisable edits injected"
    res["secondary_original_protocol"] = dict(
        note="Tonioni graph rows, unrestricted injection; disclosed, see findings.md",
        pipeline=compliance_eval(ctx, runs, original=True),
        oracle_gt_boxes_and_labels=compliance_eval(ctx, oracle_runs, oracle=True, original=True))
    res["n_images"], res["seeds"] = len(runs), ctx.cfg["violations"]["seeds"]
    merge_metrics(ctx.cfg, "A5_violations", res)


# ---------------------------------------------------------------- A6: latency and add-a-SKU

def task_latency(ctx, device=None):
    import torch
    cfg = json.loads(json.dumps(ctx.cfg))
    if device:
        cfg["device"] = device
    dev = get_device(cfg)
    pipe = Pipeline(cfg, gallery=ctx.gallery(cfg["recognizer"]["backbone"]))
    names = ctx.names("test")
    imgs = [Image.open(gp_image_path(ctx.grozi, n)).convert("RGB") for n in names]
    plans = [json.loads((Path(cfg["paths"]["planograms"]) / f"{n}.json").read_text())["rows"] for n in names]
    lc = cfg["latency"]
    for i in range(lc["warmup"]):
        pipe.run(imgs[i % len(imgs)], plans[i % len(imgs)])
    tot, stages = [], {"detect": [], "recognize": [], "align": []}
    for i in range(lc["runs"]):
        if dev == "cuda":
            torch.cuda.synchronize()
        r = pipe.run(imgs[i % len(imgs)], plans[i % len(imgs)])
        tot.append(r["timing"]["total"])
        for k in stages:
            stages[k].append(r["timing"][k])
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    import platform
    res = dict(device=dev, gpu=gpu, cpu=platform.processor(), runs=lc["runs"], mean_s=round(float(np.mean(tot)), 3),
               p95_s=round(float(np.percentile(tot, 95)), 3),
               stage_mean_s={k: round(float(np.mean(v)), 3) for k, v in stages.items()}, target_s=5.0)
    merge_metrics(ctx.cfg, f"A6_latency_{dev}", res)


def task_add_sku(ctx):
    """Time the add-a-SKU path on a copy of the index: model load + augment + embed + SIFT + save."""
    cfg = ctx.cfg
    bb = cfg["recognizer"]["backbone"]
    src = index_dir(cfg, bb)
    tmp = ctx.out / "tmp_index"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    g = Gallery.load(tmp)
    sku = sorted(ctx.unknown)[0]
    from src.data import gallery_products
    path = gallery_products(ctx.grozi)[sku]
    t0 = time.perf_counter()
    emb = Embedder(bb, cfg["recognizer"]["backbones"][bb], ctx.dev)
    t1 = time.perf_counter()
    img = Image.open(path).convert("RGB")
    g.add(sku + "_new", embed_product(img, sku, emb, cfg), Sift(cfg["recognizer"]["sift"]).features(img))
    g.save(tmp)
    t2 = time.perf_counter()
    shutil.rmtree(tmp, ignore_errors=True)
    merge_metrics(cfg, "A6_add_sku", dict(model_load_s=round(t1 - t0, 2), index_s=round(t2 - t1, 2),
                                          total_s=round(t2 - t0, 2), target_s=30, device=ctx.dev))


# ---------------------------------------------------------------- C7: failure cases

def task_errors(ctx):
    """Saves recognition failures (crop | predicted | truth) with an automatic cause tag."""
    from src.data import gallery_products
    bb = ctx.cfg["recognizer"]["backbone"]
    _, out = eval_rerank(ctx, bb, "test")
    g = ctx.gallery(bb)
    crops, skus, src = ctx.crops("test")
    prod = gallery_products(ctx.grozi)
    d = ctx.out / "errors"
    shutil.rmtree(d, ignore_errors=True)
    rows = []
    for i, s in enumerate(skus):
        pred = g.skus[out["top"][i][0]]
        if pred == s:
            continue
        w, h = crops[i].size
        if pred.rsplit("/", 1)[0] == s.rsplit("/", 1)[0] and out["top_scores"][i, 0] - out["top_scores"][i, 1] < 0.03:
            cause = "near_identical_variant"
        elif min(w, h) < 120:
            cause = "small_crop"
        elif w > 1.6 * h or h > 3.5 * w:
            cause = "extreme_aspect_or_partial"
        elif pred.rsplit("/", 1)[0] == s.rsplit("/", 1)[0]:
            cause = "same_subcategory_confusion"
        else:
            cause = "cross_category_confusion"
        tiles = [crops[i], Image.open(prod[pred]).convert("RGB"), Image.open(prod[s]).convert("RGB")]
        tiles = [t.resize((max(1, round(t.size[0] * 300 / t.size[1])), 300)) for t in tiles]
        canvas = Image.new("RGB", (sum(t.size[0] for t in tiles) + 20, 300), (255, 255, 255))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0)); x += t.size[0] + 10
        (d / cause).mkdir(parents=True, exist_ok=True)
        canvas.save(d / cause / f"{src[i]}_{i}.jpg", quality=85)
        rows.append(dict(i=i, image=src[i], truth=s, pred=pred, cause=cause, score=float(out["top_scores"][i, 0])))
    from collections import Counter
    merge_metrics(ctx.cfg, "C7_recognition_errors", dict(n_errors=len(rows), by_cause=Counter(r["cause"] for r in rows)))
    dump_json(rows, d / "errors.json")


TASKS = dict(detection=task_detection, proposals=task_proposals, recognition=task_recognition, openset=task_openset,
             compliance=task_compliance, latency=task_latency, add_sku=task_add_sku, errors=task_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tune", "tune_detector", "tune_recognizer", "tune_openset",
                                                      "tune_compliance", "all", "latency_cpu", *TASKS])
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    ctx = Ctx(cfg)
    if a.task == "tune":
        for f in (tune_detector, tune_recognizer, tune_openset, tune_compliance):
            f(ctx)
    elif a.task.startswith("tune_"):
        globals()[a.task](ctx)
    elif a.task == "all":
        for f in TASKS.values():
            f(ctx)
    elif a.task == "latency_cpu":
        task_latency(ctx, "cpu")
    else:
        TASKS[a.task](ctx)


if __name__ == "__main__":
    main()
