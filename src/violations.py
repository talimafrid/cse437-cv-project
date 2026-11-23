"""F5: seeded violation injection on row planograms, with per-slot ground truth.

The shelf photo stays unchanged (GP-180 photos match their planograms); the planogram is edited,
so every edited slot is a known violation:
  swap     two different SKUs in a row exchange places      -> both slots changed
  replace  a slot gets another product (half same subcategory, the hard case)
  insert   a new slot that has no product on the shelf       -> should come out "missing"
  remove   an interior slot is deleted, so its product on the shelf becomes an "extra"
"""
from __future__ import annotations

import random


def subcategory(sku: str) -> str:
    return sku.rsplit("/", 1)[0]


def inject(rows: list[list[str]], pool: list[str], seed: int, rate_min=0.10, rate_max=0.20,
           ops=("swap", "remove", "insert", "replace")):
    """Returns (new_rows, truth). truth = {"changed": [[bool]] aligned with new_rows,
    "removed": [(row, gap_pos)], "log": [...]}; gap_pos = number of slots left of the removed one."""
    rng = random.Random(seed)
    by_sub = {}
    for s in pool:
        by_sub.setdefault(subcategory(s), []).append(s)
    R = [[{"sku": s, "changed": False} for s in row] for row in rows]
    n_slots = sum(len(r) for r in rows)
    n_edits = max(1, round(n_slots * rng.uniform(rate_min, rate_max)))
    log = []

    def live(r):
        return [k for k, e in enumerate(R[r]) if not e.get("removed")]

    def fresh(r):
        return [k for k in live(r) if not R[r][k]["changed"]]

    def other_product(orig, hard):
        cands = [s for s in by_sub.get(subcategory(orig), []) if s != orig] if hard else []
        if not cands:
            cands = [s for s in pool if s != orig]
        return rng.choice(cands)

    attempts = 0
    while len(log) < n_edits and attempts < 50 * n_edits:
        attempts += 1
        op = rng.choice(list(ops))
        r = rng.randrange(len(R))
        f = fresh(r)
        if op == "swap":
            pairs = [(a, b) for a in f for b in f if a < b and R[r][a]["sku"] != R[r][b]["sku"]]
            if not pairs:
                continue
            a, b = rng.choice(pairs)
            R[r][a]["sku"], R[r][b]["sku"] = R[r][b]["sku"], R[r][a]["sku"]
            R[r][a]["changed"] = R[r][b]["changed"] = True
            log.append(dict(op="swap", row=r, slots=[a, b]))
        elif op == "replace":
            if not f:
                continue
            k = rng.choice(f)
            hard = rng.random() < 0.5
            old = R[r][k]["sku"]
            R[r][k].update(sku=other_product(old, hard), changed=True)
            log.append(dict(op="replace", row=r, slot=k, old=old, new=R[r][k]["sku"], hard=hard))
        elif op == "insert":
            pos = rng.randint(0, len(R[r]))
            new = other_product(rng.choice(rows[r]), rng.random() < 0.5)
            R[r].insert(pos, {"sku": new, "changed": True})
            log.append(dict(op="insert", row=r, sku=new))
        elif op == "remove":
            lv = live(r)
            interior = [k for k in f if lv and lv[0] < k < lv[-1]
                        and not R[r][k - 1].get("removed") and not R[r][k + 1].get("removed")]
            if not interior:
                continue
            k = rng.choice(interior)
            R[r][k]["removed"] = True
            log.append(dict(op="remove", row=r, sku=R[r][k]["sku"]))

    new_rows, changed, removed = [], [], []
    for r, row in enumerate(R):
        kept = []
        for e in row:
            if e.get("removed"):
                removed.append((r, len(kept)))
            else:
                kept.append(e)
        new_rows.append([e["sku"] for e in kept])
        changed.append([e["changed"] for e in kept])
    return new_rows, {"changed": changed, "removed": removed, "log": log}


def score(slots: list[dict], extras: list[dict], truth: dict) -> dict:
    """Counts for A5. A changed slot is caught when its status is not "correct"; a removal is
    caught when an extra detection is reported at the same gap position of the same row."""
    tp = fn = fa = ok = 0
    for s in slots:
        flagged = s["status"] != "correct"
        if truth["changed"][s["row"]][s["col"]]:
            tp += flagged; fn += not flagged
        else:
            fa += flagged; ok += not flagged
    extra_pos = {(e["row"], e["gap_pos"]) for e in extras}
    rem_tp = sum(1 for rp in truth["removed"] if tuple(rp) in extra_pos)
    rem_fn = len(truth["removed"]) - rem_tp
    return dict(viol_detected=tp + rem_tp, viol_total=tp + fn + rem_tp + rem_fn,
                false_alarms=fa, unchanged_total=fa + ok)
