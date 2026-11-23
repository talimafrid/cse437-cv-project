"""Unit tests for the parts that need no model weights: rows, alignment, injection, scoring, data, NMS."""
import numpy as np
import pytest

from src.data import gp_names, load_gp_boxes, load_tonioni_planogram, planogram_path, sku_from_path
from src.planogram import align_row, align_shelf, compliance, group_rows
from src.utils import iou_matrix, load_config, nms
from src.violations import inject, score


def boxes_grid(rows, cols, w=100, h=150, slant=0.0):
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = c * w, r * 200 + c * w * slant
            out.append([x, y, x + w - 5, y + h])
    return np.array(out, float)


def oracle(labels):
    """verified/unknown callbacks for detections whose true SKU is labels[d] (None = unknown)."""
    return (lambda d, s: labels[d] == s), (lambda d: labels[d] is None)


# ---- rows -------------------------------------------------------------------------------------
@pytest.mark.parametrize("slant", [0.0, 0.15])
def test_group_rows_grid_and_slant(slant):
    rows = group_rows(boxes_grid(3, 6, slant=slant))
    assert [len(r) for r in rows] == [6, 6, 6]
    assert rows[0] == list(range(6))                          # top row first, left to right


def test_group_rows_empty():
    assert group_rows(np.zeros((0, 4))) == []


# ---- alignment --------------------------------------------------------------------------------
def test_align_row_perfect():
    cost, ops = align_row(3, 3, lambda i, j: 0.0 if i == j else 1.0, 0.6)
    assert cost == 0 and [o[0] for o in ops] == ["match"] * 3


def test_align_row_missing_slot():
    plan, det = ["A", "B", "C"], ["A", "C"]
    cost, ops = align_row(3, 2, lambda i, j: 0.0 if plan[i] == det[j] else 1.0, 0.6)
    assert ("missing", 1, None) in ops and cost == pytest.approx(0.6)


def test_align_shelf_statuses():
    plan = [["A", "B", "C", "D"], ["E", "F", "G"]]
    labels = ["A", "X", None, "D", "E", "G"]                 # B wrong product, C unknown, F missing
    ver, unk = oracle(labels)
    slots, extras = align_shelf(plan, [[0, 1, 2, 3], [4, 5]], ver, unk, end_cost=0.2)
    st = {s["expected"]: s["status"] for s in slots}
    assert st == {"A": "correct", "B": "misplaced", "C": "unknown", "D": "correct",
                  "E": "correct", "F": "missing", "G": "correct"}
    assert compliance(slots) == pytest.approx(4 / 7) and extras == []


def test_adjacent_swap_flags_both_slots():
    plan = [["A", "C", "B", "D", "E"]]                        # planogram edited; shelf is A B C D E
    ver, unk = oracle(["A", "B", "C", "D", "E"])
    slots, _ = align_shelf(plan, [[0, 1, 2, 3, 4]], ver, unk, end_cost=0.2)
    assert [s["status"] for s in slots] == ["correct", "misplaced", "misplaced", "correct", "correct"]
    slots, _ = align_shelf(plan, [[0, 1, 2, 3, 4]], ver, unk, end_cost=0.2, swap_ratio=None)
    assert sum(s["status"] == "correct" for s in slots) == 4   # without the swap step one slot is lost


def test_two_level_survives_spurious_row():
    plan = [["A", "B", "C"], ["D", "E", "F"]]
    labels = ["Z", "Y", "W", "A", "B", "C", "D", "E", "F"]    # an unplanned shelf on top
    ver, unk = oracle(labels)
    det_rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    slots, _ = align_shelf(plan, det_rows, ver, unk, two_level=True)
    assert compliance(slots) == 1.0
    naive, _ = align_shelf(plan, det_rows, ver, unk, two_level=False)
    assert compliance(naive) < 1.0


# ---- violation injection + A5 scoring ---------------------------------------------------------
PLAN = [[f"Food/Snacks/{i}" for i in range(r * 8, r * 8 + 8)] for r in range(3)]
POOL = [f"Food/Snacks/{i}" for i in range(100)] + [f"Food/Tea/{i}" for i in range(50)]


@pytest.mark.parametrize("seed", range(5))
def test_inject_rate_and_determinism(seed):
    a = inject(PLAN, POOL, seed)
    b = inject(PLAN, POOL, seed)
    assert a == b
    rows, truth = a
    assert [len(r) for r in rows] == [len(c) for c in truth["changed"]]
    n_viol = sum(map(sum, truth["changed"])) + len(truth["removed"])
    assert 1 <= len(truth["log"]) <= round(24 * 0.2) and n_viol >= len(truth["log"])


def test_oracle_pipeline_a5_bound():
    """Shelf matches the original planogram; aligning against edited planograms with oracle
    recognition is the A5 ceiling of the aligner + scorer. Measured: ~0.95 detected, ~0.005 FA."""
    labels = [s for r in PLAN for s in r]
    det_rows = [list(range(8 * r, 8 * r + 8)) for r in range(3)]
    ver, unk = oracle(labels)
    tot = dict(viol_detected=0, viol_total=0, false_alarms=0, unchanged_total=0)
    for seed in range(100):
        rows, truth = inject(PLAN, POOL, seed)
        slots, extras = align_shelf(rows, det_rows, ver, unk, end_cost=0.3)
        for k, v in score(slots, extras, truth).items():
            tot[k] += v
    assert tot["viol_detected"] / tot["viol_total"] >= 0.90
    assert tot["false_alarms"] / tot["unchanged_total"] <= 0.02


# ---- utils ------------------------------------------------------------------------------------
def test_iou_and_nms():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    iou = iou_matrix(b, b)
    assert iou[0, 0] == pytest.approx(1) and iou[0, 2] == 0 and iou[0, 1] == pytest.approx(0.81)
    keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(np.asarray(keep).tolist()) == [0, 2]


def test_sku_from_path_variants():
    assert sku_from_path("X:/Grocery stores/x/Training//Food/Rice/17.jpg") == "Food/Rice/17"
    assert sku_from_path("X:/a/TrainingFood/Snacks/25.jpg") == "Food/Snacks/25"


# ---- data (skipped if GroZi is not on disk) ---------------------------------------------------
cfg = load_config()
from pathlib import Path                                          # noqa: E402

GROZI = Path(cfg["paths"]["grozi"])
needs_data = pytest.mark.skipif(not (GROZI / "bb").exists(), reason="GroZi not downloaded")


@needs_data
def test_planogram_node_count_matches_boxes():
    names = [n for n in gp_names(GROZI)][:10]
    for n in names:
        rows = load_tonioni_planogram(planogram_path(cfg, n))
        boxes = load_gp_boxes(GROZI, n)[0]
        assert sum(map(len, rows)) == len(boxes), n
        assert all(s.startswith("Food/") for r in rows for s in r)


def test_augment_views_deterministic():
    from PIL import Image
    from src.recognizer import augment_views
    img = Image.fromarray((np.random.default_rng(0).random((120, 80, 3)) * 255).astype(np.uint8))
    a, b = augment_views(img, 8, "Food/Tea/1"), augment_views(img, 8, "Food/Tea/1")
    assert len(a) == 8 and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
