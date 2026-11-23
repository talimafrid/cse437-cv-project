"""Dataset access: Grocery Products gallery, GP-180 shelf annotations and planograms, splits.

GP-180 files (Tonioni & Di Stefano 2017) as released:
  bb/sX_N.csv      one row per product facing: img_path,[x,y],[x,y],[x,y],[x,y]  (full-res pixels)
  planos/sX_N.json {"oggetti": [{width,height,img_path}], "grafo": [{n,ne,e,se,s,sw,w,nw,ogg}]}
A graph node is one facing; "ogg" is a 0-based index into "oggetti"; "n" points up in the image.
Shelf image for sX_N is Testing/storeX/images/N.jpg (3264 x 2448).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_ORIGINALS = re.compile(r"/[Oo]riginals?/")


def sku_from_path(p: str) -> str:
    """'X:/.../Training//Food/Rice/17.jpg' or '.../TrainingFood/Rice/17.jpg' -> 'Food/Rice/17'."""
    p = p.replace("\\", "/").replace("TrainingFood", "Training/Food")
    m = re.search(r"(?:Training/+)?(Food/.+?)\.\w+$", p)
    if not m:
        raise ValueError(f"cannot parse product path: {p}")
    return m.group(1)


def gallery_products(grozi: Path) -> dict[str, Path]:
    """One studio image per Food product. Skips the duplicate original(s)/ folders."""
    out = {}
    for p in sorted((grozi / "Training" / "Food").rglob("*")):
        s = p.as_posix()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not _ORIGINALS.search(s):
            out[s.split("/Training/", 1)[1].rsplit(".", 1)[0]] = p
    return out


def gp_names(grozi: Path) -> list[str]:
    """Images that have both a box CSV and a planogram."""
    csvs = {p.stem for p in (grozi / "bb").glob("*.csv")}
    planos = {p.stem for p in (grozi / "planos").glob("*.json")}
    return sorted(csvs & planos, key=_name_key)


def _name_key(name: str):
    s, n = name.split("_")
    return int(s[1:]), int(n)


def gp_image_path(grozi: Path, name: str) -> Path:
    s, n = name.split("_")
    return grozi / "Testing" / f"store{s[1:]}" / "images" / f"{n}.jpg"


def load_gp_boxes(grozi: Path, name: str, width: int | None = None, height: int | None = None):
    """Returns [(x1, y1, x2, y2), ...] clipped to the image, and the SKU of each box.
    Some shelf photos are portrait, so the size is read from the image when not given."""
    if width is None or height is None:
        from PIL import Image
        with Image.open(gp_image_path(grozi, name)) as im:   # header only, no decode
            width, height = im.size
    boxes, skus = [], []
    for line in (grozi / "bb" / f"{name}.csv").read_text().splitlines():
        if not line.strip():
            continue
        path, rest = line.split(",", 1)
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", rest)]
        xs, ys = nums[0::2], nums[1::2]
        x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
        x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((x1, y1, x2, y2))
        skus.append(sku_from_path(path))
    return boxes, skus


def load_tonioni_planogram(path: Path) -> list[list[str]]:
    """Graph planogram -> rows of SKUs, top row first, each row left to right.

    Rows are chains that start at a node with no west neighbour and follow "e".
    Row order comes from the "n"/"s" links (a topological sort, so one broken link does not
    reorder everything). Nodes left unvisited by inconsistent links start their own chain.
    """
    d = json.loads(Path(path).read_text())
    objs = d.get("objects") or d["oggetti"]
    graph = d.get("graph") or d["grafo"]
    sku = [sku_from_path(objs[g["ogg"]]["img_path"]) for g in graph]
    n = len(graph)

    row_of = [-1] * n
    rows: list[list[int]] = []

    def chain(start):
        r, cur = [], start
        while 0 <= cur < n and row_of[cur] == -1:
            row_of[cur] = len(rows)
            r.append(cur)
            cur = graph[cur]["e"]
        rows.append(r)

    for i in range(n):
        if graph[i]["w"] == -1 and row_of[i] == -1:
            chain(i)
    for i in range(n):                       # nodes reachable only through broken links
        if row_of[i] == -1:
            j, seen = i, set()
            while graph[j]["w"] >= 0 and row_of[graph[j]["w"]] == -1 and j not in seen:
                seen.add(j)
                j = graph[j]["w"]
            chain(j)

    # row A above row B if a node of B has a north-ish neighbour in A (or A has south-ish in B)
    below = defaultdict(set)
    for i, g in enumerate(graph):
        for k in ("n", "ne", "nw"):
            if g.get(k, -1) >= 0 and row_of[g[k]] != row_of[i]:
                below[row_of[g[k]]].add(row_of[i])
        for k in ("s", "se", "sw"):
            if g.get(k, -1) >= 0 and row_of[g[k]] != row_of[i]:
                below[row_of[i]].add(row_of[g[k]])
    # depth = longest path from a top row; chains at the same depth are one broken shelf row
    depth = [0] * len(rows)
    for _ in range(len(rows)):               # Bellman-Ford style; bounded, so cycles cannot hang
        changed = False
        for a in below:
            for b in below[a]:
                if depth[b] < depth[a] + 1 <= len(rows):
                    depth[b] = depth[a] + 1
                    changed = True
        if not changed:
            break

    def left_of(a, b):                       # any horizontal link from chain a to chain b
        return any(graph[i].get(k, -1) in rows[b] for i in rows[a] for k in ("e", "ne", "se")) or \
            any(graph[j].get(k, -1) in rows[a] for j in rows[b] for k in ("w", "nw", "sw"))

    out = []
    for d in sorted(set(depth)):
        group = [r for r in range(len(rows)) if depth[r] == d]
        ordered = []
        for r in group:                      # insertion by horizontal links, else keep index order
            pos = next((k for k, o in enumerate(ordered) if left_of(r, o)), len(ordered))
            ordered.insert(pos, r)
        out.append([sku[i] for r in ordered for i in rows[r]])
    return out


def planogram_path(cfg: dict, name: str) -> Path:
    fixed = Path(cfg["paths"]["planogram_fixes"]) / f"{name}.json"
    return fixed if fixed.exists() else Path(cfg["paths"]["grozi"]) / "planos" / f"{name}.json"


def load_rows_planogram(path: Path) -> list[list[str]]:
    """Editable row planogram written by scripts/prepare_data.py: {"image":..., "rows": [[sku,...],...]}."""
    return json.loads(Path(path).read_text())["rows"]


# ---------------------------------------------------------------- splits (C4)

def _write_list(path: Path, items: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8", newline="\n")


def read_list(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def make_gp_split(names: list[str], cfg: dict) -> tuple[list[str], list[str]]:
    """Seeded, store-stratified choice of 70 images, 20 val (incl. cvpce's 8) / 50 test."""
    dc = cfg["data"]
    rng = random.Random(cfg["seed"])
    cand = [n for n in names if n not in set(dc["gp_exclude"])]
    forced = [n for n in dc["gp_val_forced"] if n in cand]
    by_store = defaultdict(list)
    for n in cand:
        if n not in forced:
            by_store[n.split("_")[0]].append(n)
    need = dc["gp_n_images"] - len(forced)
    total = sum(len(v) for v in by_store.values())
    # largest-remainder allocation proportional to store size
    quota = {s: need * len(v) / total for s, v in by_store.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True)[: need - sum(alloc.values())]:
        alloc[s] += 1
    chosen = []
    for s in sorted(by_store):
        pool = sorted(by_store[s], key=_name_key)
        rng.shuffle(pool)
        chosen += pool[: alloc[s]]
    rng.shuffle(chosen)
    n_val_extra = dc["gp_n_val"] - len(forced)
    val = sorted(forced + chosen[:n_val_extra], key=_name_key)
    test = sorted(chosen[n_val_extra:], key=_name_key)
    assert len(val) == dc["gp_n_val"] and len(val) + len(test) == dc["gp_n_images"]
    assert not set(val) & set(test)
    return val, test


def make_openset_split(products: list[str], cfg: dict) -> list[str]:
    rng = random.Random(cfg["seed"])
    k = round(len(products) * cfg["data"]["openset_fraction"])
    return sorted(rng.sample(sorted(products), k))


def write_hashes(split_dir: Path):
    lines = []
    for p in sorted(split_dir.glob("*.txt")):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (split_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_hashes(split_dir: Path) -> bool:
    for line in (split_dir / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ")
        if hashlib.sha256((split_dir / name).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"split file changed after freezing: {name}")
    return True
