"""Gradio demo (C11): audit a shelf, add/remove a SKU, inject violations. Thresholds come from config.yaml.

  python -m src.app
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from src.data import gp_image_path, load_gp_boxes, read_list
from src.gallery import embed_product, index_dir
from src.pipeline import Pipeline, write_report
from src.planogram import group_rows
from src.recognizer import Sift
from src.utils import load_config
from src.violations import inject

CFG = load_config()
PLANOS = Path(CFG["paths"]["planograms"])
OUT = Path(CFG["paths"]["outputs"]) / "demo"
_pipe: Pipeline | None = None


def pipe() -> Pipeline:
    global _pipe
    if _pipe is None:                                         # load models on first use
        _pipe = Pipeline(CFG)
    return _pipe


def test_images() -> list[str]:
    f = Path(CFG["paths"]["splits"]) / "gp_test.txt"
    return read_list(f) if f.exists() else sorted(p.stem for p in PLANOS.glob("*.json"))


def pick_image(name: str):
    """Planogram = GT layout grouped by the pipeline's row rule (same as the main A5 protocol)."""
    path = gp_image_path(Path(CFG["paths"]["grozi"]), name)
    boxes, skus = load_gp_boxes(Path(CFG["paths"]["grozi"]), name)
    rows = [[skus[d] for d in row] for row in group_rows(np.array(boxes), CFG["planogram"]["row_gap"])]
    return str(path), json.dumps(rows, indent=1)


def run_audit(image, plan_text: str, name: str):
    if image is None:
        raise gr.Error("Pick or upload a shelf image.")
    try:
        rows = json.loads(plan_text)
        assert isinstance(rows, list) and all(isinstance(r, list) for r in rows)
    except Exception:
        raise gr.Error('Planogram must be JSON rows, e.g. [["Food/Rice/17", "Food/Rice/11"], [...]]')
    res = pipe().run(image, rows)
    stem = name or "upload"
    write_report(res, OUT, stem)
    table = pd.DataFrame([{k: s.get(k) for k in ("row", "col", "expected", "detected", "status", "score")}
                          for s in res["slots"]])
    counts = table["status"].value_counts().to_dict() if len(table) else {}
    summary = (f"**Compliance {res['compliance']:.1%}**  ·  {counts}  ·  extras {len(res['extras'])}  ·  "
               f"{res['timing']['total']:.2f} s")
    return str(OUT / f"{stem}_audit.jpg"), summary, table, str(OUT / f"{stem}_report.json")


def edit_gallery(action: str, image, sku: str):
    p = pipe()
    sku = (sku or "").strip()
    if not sku:
        raise gr.Error("Enter a SKU id such as Food/Tea/999.")
    t0 = time.perf_counter()
    if action == "add":
        if image is None:
            raise gr.Error("Upload the product's studio image.")
        p.gallery.add(sku, embed_product(image.convert("RGB"), sku, p.rec.emb, CFG),
                      Sift(CFG["recognizer"]["sift"]).features(image.convert("RGB")))
    elif p.gallery.index_of(sku) is None:
        raise gr.Error(f"{sku} is not in the gallery.")
    else:
        p.gallery.remove(sku)
    p.gallery.save(index_dir(CFG, CFG["recognizer"]["backbone"]))
    return f"{action} {sku}: gallery has {len(p.gallery)} products ({time.perf_counter() - t0:.1f} s)"


def make_violations(plan_text: str, seed: float):
    rows = json.loads(plan_text)
    pool = sorted({s for r in rows for s in r}) if _pipe is None else pipe().gallery.skus
    vc = CFG["violations"]
    new, truth = inject(rows, pool, int(seed), vc["rate_min"], vc["rate_max"], vc["ops"])
    return json.dumps(new, indent=1), json.dumps(truth["log"], indent=1)


def ui() -> gr.Blocks:
    names = test_images()
    with gr.Blocks(title="Planogram compliance") as demo:
        gr.Markdown("## Shelf planogram compliance with open-set SKU recognition")
        with gr.Tab("Audit"):
            with gr.Row():
                with gr.Column():
                    name = gr.Dropdown(names, label="GP-180 test image", value=None)
                    image = gr.Image(type="filepath", label="Shelf image")
                    plan = gr.Code(language="json", label="Planogram rows (editable)")
                    go = gr.Button("Run audit", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Audit")
                    summary = gr.Markdown()
                    report = gr.File(label="report.json")
            table = gr.Dataframe(label="Slots")
            name.change(pick_image, name, [image, plan])
            go.click(run_audit, [image, plan, name], [out_img, summary, table, report])
        with gr.Tab("Gallery (add / remove SKU)"):
            g_img = gr.Image(type="pil", label="Studio image (add only)")
            g_sku = gr.Textbox(label="SKU id", placeholder="Food/Tea/999")
            with gr.Row():
                add = gr.Button("Add and index")
                rem = gr.Button("Remove")
            g_out = gr.Markdown()
            add.click(lambda i, s: edit_gallery("add", i, s), [g_img, g_sku], g_out)
            rem.click(lambda i, s: edit_gallery("remove", i, s), [g_img, g_sku], g_out)
        with gr.Tab("Inject violations"):
            v_in = gr.Code(language="json", label="Planogram rows")
            v_seed = gr.Number(value=CFG["seed"], label="Seed", precision=0)
            v_go = gr.Button("Inject 10-20% edits")
            v_out = gr.Code(language="json", label="Edited planogram (paste into the Audit tab)")
            v_log = gr.Code(language="json", label="Edit log")
            v_go.click(make_violations, [v_in, v_seed], [v_out, v_log])
    return demo


if __name__ == "__main__":
    ui().launch()
