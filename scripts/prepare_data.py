"""Prepare data and freeze splits (C4, C10). Run once, commit splits/, then never change them.

  python scripts/prepare_data.py             # splits + hashes + row planograms (+ SKU-110K if present)
  python scripts/prepare_data.py --download  # also download SKU110K_fixed.tar.gz (13.6 GB) first

Outputs:
  splits/gp_val.txt, splits/gp_test.txt      GP-180: 20 val / 50 test (seeded, store-stratified)
  splits/openset_unknown.txt                 20% of gallery products held out as unknown (seeded)
  splits/sku110k_test_subset.txt             500 seeded SKU-110K test images for A1
  splits/SHA256SUMS
  outputs/planograms/<name>.json             editable row planograms {"image", "rows"}
  data/sku110k_yolo/{images,labels}/{train,val,test}, configs/sku110k.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from src.data import (gallery_products, gp_names, load_gp_boxes, load_tonioni_planogram, make_gp_split,
                      make_openset_split, planogram_path, write_hashes, _write_list)
from src.utils import ROOT, dump_json, load_config

SKU_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"


def _resize_one(args):
    src, dst, long_side = args
    if dst.exists():
        return True
    try:
        im = Image.open(src)
        im.load()
        im = im.convert("RGB")
        s = long_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
        im.save(dst, quality=92)
        return True
    except Exception as e:                       # SKU-110K has a few corrupt files
        print(f"skip {src.name}: {e}")
        return False


def prepare_sku110k(cfg):
    raw = Path(cfg["paths"]["sku110k_raw"])
    out = Path(cfg["paths"]["sku110k_yolo"])
    if not (raw / "images").exists():
        print(f"SKU-110K images not found at {raw}; skipping (use --download).")
        return
    ann_dir = raw / "annotations" if (raw / "annotations").exists() else ROOT / "data" / "sku110k" / "annotations"
    for split in ("train", "val", "test"):
        boxes = defaultdict(list)
        with open(ann_dir / f"annotations_{split}.csv", newline="") as f:
            for name, x1, y1, x2, y2, _, w, h in csv.reader(f):
                w, h = float(w), float(h)
                x1, x2 = max(0.0, float(x1)), min(w, float(x2))
                y1, y2 = max(0.0, float(y1)), min(h, float(y2))
                if x2 > x1 and y2 > y1:
                    boxes[name].append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs = [(raw / "images" / n, out / "images" / split / n, cfg["data"]["sku110k_resize"]) for n in sorted(boxes)]
        with ProcessPoolExecutor() as ex:
            ok = list(ex.map(_resize_one, jobs, chunksize=32))
        for (src, dst, _), good in zip(jobs, ok):
            lab = out / "labels" / split / (src.stem + ".txt")
            if good:
                lab.write_text("".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in boxes[src.name]))
            elif lab.exists():
                lab.unlink()
        print(f"SKU-110K {split}: {sum(ok)}/{len(jobs)} images")
    yaml_text = (f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 f"nc: 1\nnames:\n  0: object\n")
    (ROOT / "configs" / "sku110k.yaml").write_text(yaml_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    split_dir = Path(cfg["paths"]["splits"])
    grozi = Path(cfg["paths"]["grozi"])

    if a.download:
        raw = Path(cfg["paths"]["sku110k_raw"])
        tgz = raw.parent / "SKU110K_fixed.tar.gz"
        if not (raw / "images").exists():
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", str(tgz), SKU_URL], check=True)
            with tarfile.open(tgz) as t:
                t.extractall(raw.parent, filter="data")

    names = gp_names(grozi)
    if (split_dir / "SHA256SUMS").exists():
        print("splits already frozen; not rewriting them")
    else:
        val, test = make_gp_split(names, cfg)
        _write_list(split_dir / "gp_val.txt", val)
        _write_list(split_dir / "gp_test.txt", test)
        _write_list(split_dir / "openset_unknown.txt", make_openset_split(list(gallery_products(grozi)), cfg))
        raw = Path(cfg["paths"]["sku110k_raw"])
        ann = raw / "annotations" if (raw / "annotations").exists() else ROOT / "data/sku110k/annotations"
        test_imgs = sorted({r.split(",")[0] for r in (ann / "annotations_test.csv").read_text().splitlines()})
        rng = random.Random(cfg["seed"])
        _write_list(split_dir / "sku110k_test_subset.txt",
                    sorted(rng.sample(test_imgs, cfg["data"]["sku110k_test_subset"])))
        write_hashes(split_dir)
        print(f"GP-180: {len(val)} val / {len(test)} test frozen in {split_dir}")

    # row planograms for every paired image (editable by hand for the viva change)
    pdir = Path(cfg["paths"]["planograms"])
    n_ok = 0
    for n in names:
        try:
            rows = load_tonioni_planogram(planogram_path(cfg, n))
        except Exception as e:
            print(f"planogram {n}: {e}")
            continue
        n_boxes = len(load_gp_boxes(grozi, n)[0])
        dump_json({"image": n, "n_annotated_boxes": n_boxes, "rows": rows}, pdir / f"{n}.json")
        n_ok += 1
    print(f"row planograms: {n_ok}/{len(names)} -> {pdir}")
    prepare_sku110k(cfg)


if __name__ == "__main__":
    main()
