"""F4: group detections into shelf rows and align them with the planogram.

Two-level Needleman-Wunsch:
  inner: one planogram row vs one detected row. Substitution cost 0 when the crop verifies as the
         expected SKU, `unknown_cost` when the crop was rejected as unknown, 1 otherwise.
         Leading/trailing detections are free (planograms often cover only part of the photo).
  outer: planogram rows vs detected rows, pair cost = inner cost. Skipping a detected row is free
         (shelves outside the planogram); skipping a planogram row costs every slot as missing.
         One spurious or missed row therefore costs one gap and does not shift the rows below it.
Slot status: correct | misplaced | missing | unknown. Interior unmatched detections are "extra".
"""
from __future__ import annotations

import numpy as np


def group_rows(boxes: np.ndarray, row_gap: float = 0.5) -> list[list[int]]:
    """Indices of boxes per shelf row, top row first, left to right.

    Products stand on shelves, so rows are clustered by box bottom edge. The photo slant is
    estimated as the median slope of bottoms within first-pass rows and removed before re-clustering.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    xc = (boxes[:, 0] + boxes[:, 2]) / 2
    bottom = boxes[:, 3]
    gap = row_gap * np.median(boxes[:, 3] - boxes[:, 1])

    def cluster(y):
        order = np.argsort(y)
        rows, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if y[b] - y[a] > gap:
                rows.append(cur)
                cur = []
            cur.append(b)
        rows.append(cur)
        return rows

    slopes = []
    for r in cluster(bottom):
        if len(r) >= 3 and np.ptp(xc[r]) > 0:
            slopes.append(np.polyfit(xc[r], bottom[r], 1)[0])
    slope = float(np.median(slopes)) if slopes else 0.0
    rows = cluster(bottom - slope * xc)
    return [sorted(r, key=lambda i: xc[i]) for r in rows]


def align_row(n_plan: int, n_det: int, sub_cost, gap: float, swap: float | None = None, end: float = 0.0):
    """Semi-global NW. sub_cost(i, j) for planogram slot i vs detection j.
    swap: cost of an adjacent transposition (Damerau step) when both crossed pairs verify; without
    it an adjacent swap is always explained more cheaply as one product moved (extra + missing).
    end: cost per detection skipped before/after the planogram row (products outside the planogram).
    Zero makes "skip a correct leading product and call its slot missing" as cheap as the right answer.
    Returns (cost, ops) with ops as ('match', i, j) | ('missing', i, None) | ('extra', gap_pos, j)."""
    INF = float("inf")
    D = np.full((n_plan + 1, n_det + 1), INF)
    D[0, :] = np.arange(n_det + 1) * end                      # cheap leading detections
    for i in range(1, n_plan + 1):
        D[i, 0] = i * gap
    C = np.array([[sub_cost(i, j) for j in range(n_det)] for i in range(n_plan)]).reshape(n_plan, n_det)
    def crossed(i, j):
        return swap is not None and i >= 2 and j >= 2 and C[i - 2, j - 1] == 0 and C[i - 1, j - 2] == 0

    for i in range(1, n_plan + 1):
        for j in range(1, n_det + 1):
            extra = end if i == n_plan else gap               # cheap trailing detections
            D[i, j] = min(D[i - 1, j - 1] + C[i - 1, j - 1], D[i - 1, j] + gap, D[i, j - 1] + extra)
            if crossed(i, j):
                D[i, j] = min(D[i, j], D[i - 2, j - 2] + swap)
    ops, i, j = [], n_plan, n_det
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(D[i, j], D[i - 1, j - 1] + C[i - 1, j - 1]):
            ops.append(("match", i - 1, j - 1)); i -= 1; j -= 1
        elif crossed(i, j) and np.isclose(D[i, j], D[i - 2, j - 2] + swap):
            ops += [("match", i - 1, j - 1), ("match", i - 2, j - 2)]; i -= 2; j -= 2   # straight pairing -> both misplaced
        elif i > 0 and np.isclose(D[i, j], D[i - 1, j] + gap):
            ops.append(("missing", i - 1, None)); i -= 1
        else:
            if 0 < i < n_plan:                                 # end extras are outside the planogram
                ops.append(("extra", i, j - 1))
            j -= 1
            if j < 0:
                break
    return float(D[n_plan, n_det]), ops[::-1]


def align_shelf(plan_rows: list[list[str]], det_rows: list[list[int]], verified, unknown,
                gap: float = 0.6, unknown_cost: float = 0.5, two_level: bool = True, swap_ratio: float | None = 0.75,
                end_cost: float = 0.0):
    """verified(det_idx, sku) -> bool; unknown(det_idx) -> bool.
    swap_ratio: adjacent-swap cost as a fraction of (extra + missing) = 2 * gap; None disables it.
    Returns (slots, extras): slots in planogram order with keys row, col, expected, det, status."""
    def sub(prow, drow):
        def c(i, j):
            d = drow[j]
            if verified(d, prow[i]):
                return 0.0
            return unknown_cost if unknown(d) else 1.0
        return c

    P, R = len(plan_rows), len(det_rows)
    pair = {}
    for a in range(P):
        for b in range(R):
            pair[a, b] = align_row(len(plan_rows[a]), len(det_rows[b]), sub(plan_rows[a], det_rows[b]), gap,
                                   None if swap_ratio is None else swap_ratio * 2 * gap, end_cost)

    if two_level:
        miss = [gap * len(r) for r in plan_rows]
        D = np.full((P + 1, R + 1), np.inf)
        D[0, :] = 0.0
        for a in range(1, P + 1):
            D[a, 0] = D[a - 1, 0] + miss[a - 1]
        for a in range(1, P + 1):
            for b in range(1, R + 1):
                D[a, b] = min(D[a - 1, b - 1] + pair[a - 1, b - 1][0], D[a - 1, b] + miss[a - 1], D[a, b - 1])
        match, a, b = {}, P, R
        while a > 0:
            if b > 0 and np.isclose(D[a, b], D[a - 1, b - 1] + pair[a - 1, b - 1][0]):
                match[a - 1] = b - 1; a -= 1; b -= 1
            elif np.isclose(D[a, b], D[a - 1, b] + miss[a - 1]):
                a -= 1
            else:
                b -= 1
    else:                                                     # naive: planogram row k <-> detected row k
        match = {a: a for a in range(min(P, R))}

    slots, extras = [], []
    for a, prow in enumerate(plan_rows):
        if a not in match:
            slots += [dict(row=a, col=i, expected=s, det=None, status="missing") for i, s in enumerate(prow)]
            continue
        drow = det_rows[match[a]]
        for op, i, j in pair[a, match[a]][1]:
            if op == "extra":
                extras.append(dict(row=a, gap_pos=i, det=drow[j], status="extra"))
                continue
            if op == "missing":
                slots.append(dict(row=a, col=i, expected=prow[i], det=None, status="missing"))
                continue
            d = drow[j]
            status = "correct" if verified(d, prow[i]) else ("unknown" if unknown(d) else "misplaced")
            slots.append(dict(row=a, col=i, expected=prow[i], det=d, status=status))
    slots.sort(key=lambda s: (s["row"], s["col"]))
    return slots, extras


def compliance(slots: list[dict]) -> float:
    return sum(s["status"] == "correct" for s in slots) / max(1, len(slots))
