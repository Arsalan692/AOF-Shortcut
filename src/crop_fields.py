"""
Phase C2 - cut one image per field out of the registered scan.

Everything a handwriting reader needs is decided here, not in the model:

  * WHAT to read - the rect comes from the blank template, so the crop is guaranteed
    to be the right field. The model is never asked to work out layout, which is what
    makes whole-page OCR attach values to the wrong labels.
  * HOW MUCH to include - handwriting on these forms routinely overshoots its box, so
    the crop is padded. Padding is asymmetric and deliberately modest vertically:
    rows are ~16pt apart, so a generous pad would pull in the line above or below.

Crops are taken from the registered greyscale, not the ink mask - a reader does much
better on natural strokes than on a hard binary silhouette.

Outputs
  build/crops/<doc>/p<N>/<field_id>.png   one crop per field
  build/crops/<doc>/index.json            crop path + label + type per field
  build/qa/crops_<doc>_p<N>.png           contact sheet for eyeballing
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import cv2
import fitz
import numpy as np

from register import binarize, render_gray

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "build", "template_schema.json")
BLANK = os.path.join(ROOT, "Forms", "Account_Opening_Form_blank template.pdf")
DPI = 300
S = DPI / 72.0

PAD_X = 2.0        # pt
PAD_Y = 3.0        # pt - kept tight; form rows are only ~16pt apart
BLOCK_PAD_Y = 2.0
UPSCALE = 2        # small boxes are ~50px tall at 300 dpi; VLMs read 2x much better

READABLE = ("text", "grid", "block")

# Blankness is measured on the field INTERIOR, inset past the printed border, for the
# same reason the checkbox reader insets: a border sliver left by 1px of registration
# error otherwise reads as ink. A gate is worth having because a vision model shown an
# empty box tends to invent a plausible value instead of reporting nothing.
BLANK_INSET = 2.0
# Ink is counted after one erosion. Pen strokes are 3-6px thick at 300 dpi and keep a
# core; the 1-2px residue left behind by printed placeholder glyphs (date grids carry
# a faint "D D M M Y Y Y Y") disappears. Measured on this scan, that takes an empty
# date grid from 325 raw pixels down to 125 while a filled one stays at 259-3453.
ERODE = 1
BLANK_FLOOR = 60
BASELINE_SHARE = 0.06

# Only an exactly-zero field is treated as certainly empty and skipped. Everything
# else goes to the model, which is allowed to answer "empty".
#
# This is deliberate rather than lazy. Because handwriting overflows its box, a box can
# contain real ink belonging to the NEIGHBOURING field - on this form the ID Number
# digits spill into the CIF Number cells - so "is there ink here" and "is this field
# filled" are genuinely different questions. Measured against known ground truth, no
# pixel statistic separates them: stroke thickness overlaps (filled min 2.87 vs empty
# max 3.82) and so does eroded area (41 vs 251). Only a reader that can recognise
# characters can arbitrate, so the gate stays conservative and records a hint instead.
EMPTY, FAINT, STRONG = "empty", "faint", "strong"


def interior_ink(ink: np.ndarray, rect, inset=BLANK_INSET) -> int:
    h, w = ink.shape[:2]
    x0 = max(0, int(round((rect[0] + inset) * S)))
    y0 = max(0, int(round((rect[1] + inset) * S)))
    x1 = min(w, int(round((rect[2] - inset) * S)))
    y1 = min(h, int(round((rect[3] - inset) * S)))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(ink[y0:y1, x0:x1]))


# How far a crop may grow beyond the printed box to chase its handwriting. Measured
# on a real form, overflow runs to 26pt right and 34pt vertically, so these are set
# past the p90 rather than the median. Growth is still capped: without a cap, one
# runaway stroke could drag a crop across half the page.
MAX_GROW_X = 44.0
MAX_GROW_Y = 30.0
OWN_INSET = 1.0    # boxes are shrunk this much when seeding ownership so that
                   # table cells sharing a border stay separate regions
# A nearest-box partition assigns every pixel on the page to some field, so residue
# 40pt away would be credited to an empty field. Ink must start within this distance
# of the box to be claimed at all; strokes that then run further are recovered by
# completing their connected component.
OWN_MAX_DIST = 8.0


def ownership_labels(shape, fields) -> tuple[np.ndarray, np.ndarray, dict]:
    """Partition the page: every pixel is labelled with the field whose box is
    nearest to it.

    This is the answer to handwriting that leaves its box. A fixed pad cannot work -
    rows are ~16pt apart but writing strays up to 34pt - and a plain
    connected-component rule fails when one person's "Karachi East" touches the next
    field's "Karachi", merging two answers into one blob. Nearest-box ownership both
    reaches out to strokes that never touched the box and cuts a merged blob along the
    boundary between the two fields that share it.
    """
    seeds = np.full(shape, 255, np.uint8)
    for f in fields:
        x0, y0, x1, y1 = (v * S for v in f["rect"])
        a = int(round(x0 + OWN_INSET * S)); b = int(round(y0 + OWN_INSET * S))
        c = int(round(x1 - OWN_INSET * S)); d = int(round(y1 - OWN_INSET * S))
        a, b = max(0, a), max(0, b)
        c, d = min(shape[1], max(c, a + 1)), min(shape[0], max(d, b + 1))
        seeds[b:d, a:c] = 0

    dist, labels = cv2.distanceTransformWithLabels(
        seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)

    fmap = {}
    for i, f in enumerate(fields):
        x0, y0, x1, y1 = (v * S for v in f["rect"])
        cx = min(max(int((x0 + x1) / 2), 0), shape[1] - 1)
        cy = min(max(int((y0 + y1) / 2), 0), shape[0] - 1)
        fmap[i] = int(labels[cy, cx])
    return dist, labels, fmap


def window(shape, rect):
    h, w = shape[:2]
    return (max(0, int((rect[0] - MAX_GROW_X) * S)), max(0, int((rect[1] - MAX_GROW_Y) * S)),
            min(w, int((rect[2] + MAX_GROW_X) * S)), min(h, int((rect[3] + MAX_GROW_Y) * S)))


def owned_mask(ink, dist, labels, cc, lab, rect):
    """(mask, window) - the ink this field owns.

    Three rules, in order: the pixel's nearest box must be ours (so a blob straddling
    two fields is cut along the boundary rather than duplicated into both crops); some
    of the stroke must lie within OWN_MAX_DIST of the box (so distant residue is not
    credited to an empty field); and once a stroke qualifies, the rest of its connected
    component comes too, which is what recovers writing that runs far outside the box.
    """
    x0, y0, x1, y1 = window(ink.shape, rect)
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0, 0, 0)
    win = (slice(y0, y1), slice(x0, x1))
    mine = (labels[win] == lab) & (ink[win] > 0)
    near = mine & (dist[win] <= OWN_MAX_DIST * S)
    if not near.any():
        return np.zeros_like(mine), (x0, y0, x1, y1)
    ids = np.unique(cc[win][near])
    ids = ids[ids != 0]
    return mine & np.isin(cc[win], ids), (x0, y0, x1, y1)


TRIM_PCT = 0.4   # ignore this share of extreme owned pixels when sizing the crop


def owned_extent(ink, dist, labels, cc, lab, rect) -> list:
    """The rect grown to contain the strokes this field owns, in PDF points.

    Extremes are trimmed by a fraction of a percent: a single stray speck that
    survived cleaning would otherwise stretch the crop tens of points and drag a
    neighbouring row into view.
    """
    m, (x0, y0, _, _) = owned_mask(ink, dist, labels, cc, lab, rect)
    if m is None or not m.any():
        return list(rect)
    ys, xs = np.nonzero(m)
    lo_x, hi_x = np.percentile(xs, [TRIM_PCT, 100 - TRIM_PCT])
    lo_y, hi_y = np.percentile(ys, [TRIM_PCT, 100 - TRIM_PCT])
    return [min(rect[0], (x0 + float(lo_x)) / S),
            min(rect[1], (y0 + float(lo_y)) / S),
            max(rect[2], (x0 + float(hi_x) + 1) / S),
            max(rect[3], (y0 + float(hi_y) + 1) / S)]


def render_crop(reg, ink, owned, win, ext, pad_x, pad_y) -> np.ndarray:
    """Crop `ext` out of the registered page with every OTHER field's handwriting
    erased.

    Growing a crop to chase overflowing handwriting inevitably makes its rectangle
    overlap neighbouring rows - the Next-of-Kin Name box reaches down over the
    Telephone Number answer. Ownership already says which strokes belong here, so the
    strokes that do not are painted out to paper. The model then sees exactly one
    answer, and the printed labels that remain are obvious typography it can be told
    to ignore.
    """
    wx0, wy0, wx1, wy1 = win
    patch = reg[wy0:wy1, wx0:wx1].copy()
    if owned is not None and ink is not None:
        foreign = (ink[wy0:wy1, wx0:wx1] > 0) & ~owned
        if foreign.any():
            paper = int(np.percentile(patch, 92))     # local unmarked paper level
            patch[foreign] = paper
            # feather so the erasure does not leave hard-edged ghosts
            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            grown = cv2.dilate(foreign.astype(np.uint8), np.ones((3, 3), np.uint8))
            patch = np.where(grown > 0, blur, patch).astype(np.uint8)

    # ext is in page points; convert to offsets inside the window
    ax = max(0, int(round((ext[0] - pad_x) * S)) - wx0)
    ay = max(0, int(round((ext[1] - pad_y) * S)) - wy0)
    bx = min(patch.shape[1], int(round((ext[2] + pad_x) * S)) - wx0)
    by = min(patch.shape[0], int(round((ext[3] + pad_y) * S)) - wy0)
    if bx <= ax or by <= ay:
        return np.full((10, 10), 255, np.uint8)
    return patch[ay:by, ax:bx]


def crop_rect(img: np.ndarray, rect, pad_x=PAD_X, pad_y=PAD_Y) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, int(round((rect[0] - pad_x) * S)))
    y0 = max(0, int(round((rect[1] - pad_y) * S)))
    x1 = min(w, int(round((rect[2] + pad_x) * S)))
    y1 = min(h, int(round((rect[3] + pad_y) * S)))
    if x1 <= x0 or y1 <= y0:
        return np.full((10, 10), 255, np.uint8)
    return img[y0:y1, x0:x1]


def run(doc: str, upscale: int = UPSCALE) -> list[dict]:
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    regdir = os.path.join(ROOT, "build", "registered", doc)
    outdir = os.path.join(ROOT, "build", "crops", doc)
    blank_doc = fitz.open(BLANK)

    by_page = defaultdict(list)
    all_page = defaultdict(list)
    for f in schema["fields"]:
        if f["mirror"]:
            continue
        # Checkboxes take part in the ownership partition even though they are never
        # cropped: a tick is ink, and if no box claims it the nearest text field will,
        # which both inflates that field's ink and drags the tick into its crop.
        all_page[f["page"]].append(f)
        if f["kind"] in READABLE:
            by_page[f["page"]].append(f)

    index = []
    for pno in sorted(by_page):
        reg_path = os.path.join(regdir, f"reg_p{pno}.png")
        if not os.path.exists(reg_path):
            continue
        reg = cv2.imread(reg_path, cv2.IMREAD_GRAYSCALE)
        ink = cv2.imread(os.path.join(regdir, f"ink_p{pno}.png"), cv2.IMREAD_GRAYSCALE)
        tpl_ink = binarize(render_gray(blank_doc[pno - 1]), min_area=1)
        k = np.ones((3, 3), np.uint8)
        ink_e = cv2.erode(ink, k, iterations=ERODE) if ink is not None else None
        pdir = os.path.join(outdir, f"p{pno}")
        os.makedirs(pdir, exist_ok=True)

        seed_fields = sorted(all_page[pno], key=lambda z: (z["rect"][1], z["rect"][0]))
        dist, labels, fmap = ownership_labels(reg.shape[:2], seed_fields)
        lab_of = {f["id"]: fmap[i] for i, f in enumerate(seed_fields)}
        # connected components of the handwriting, used to complete part-claimed strokes
        _, cc = cv2.connectedComponents(ink, 8) if ink is not None else (0, None)
        _, cc_e = cv2.connectedComponents(ink_e, 8) if ink_e is not None else (0, None)

        for f in sorted(by_page[pno], key=lambda z: (z["rect"][1], z["rect"][0])):
            lab = lab_of[f["id"]]
            pad_y = BLOCK_PAD_Y if f["kind"] == "block" else PAD_Y
            # crop the box grown to its own handwriting, not the bare printed box
            if ink is not None:
                ext = owned_extent(ink, dist, labels, cc, lab, f["rect"])
                om, win = owned_mask(ink, dist, labels, cc, lab, f["rect"])
                c = render_crop(reg, ink, om, win, ext, PAD_X, pad_y)
            else:
                ext, om = list(f["rect"]), None
                c = crop_rect(reg, ext, PAD_X, pad_y)
            if upscale != 1:
                c = cv2.resize(c, None, fx=upscale, fy=upscale,
                               interpolation=cv2.INTER_CUBIC)
            path = os.path.join(pdir, f"{f['id']}.png")
            cv2.imwrite(path, c)
            # Blankness is judged on OWNED ink, not on the box interior. Nearly a
            # quarter of all handwriting on this form sits wholly outside its box, so
            # an interior-only test would call those fields empty and silently drop
            # the answer. raw asks "is there any stroke at all", core asks "is it a
            # pen stroke or just border residue".
            oe = owned_mask(ink_e, dist, labels, cc_e, lab, f["rect"])[0] if ink_e is not None else None
            raw = int(om.sum()) if om is not None else -1
            core = int(oe.sum()) if oe is not None else -1
            base = interior_ink(tpl_ink, f["rect"])
            gate = BLANK_FLOOR + BASELINE_SHARE * base
            hint = EMPTY if raw == 0 else (FAINT if core < gate else STRONG)
            index.append({
                "field": f["id"], "page": pno, "kind": f["kind"],
                "label": f["label"], "section": f["section"],
                "table": f["table"], "grid_hint": f["grid_hint"],
                # carried through so the reader can keep a wrapped line out of the same
                # montage as the line it continues - see split_batches() in read_text.py
                "continuation_of": f.get("continuation_of", ""),
                "cells": len(f["cells"]), "crop": os.path.relpath(path, ROOT),
                "rect": [round(v, 2) for v in f["rect"]],
                "crop_rect": [round(v, 2) for v in ext],
                "ink_px": raw, "core_px": core, "baseline_px": base,
                "gate": round(gate),
                "ink_hint": hint, "needs_model": raw != 0,
            })

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)
    return index


# ------------------------------------------------------------------ contact sheet
LABEL_W = 430
GAP = 6


def contact_sheet(doc: str, pno: int, index: list[dict], out_path: str,
                  max_crop_w: int = 2150) -> str:
    rows = [r for r in index if r["page"] == pno]
    imgs = []
    for r in rows:
        img = cv2.imread(os.path.join(ROOT, r["crop"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[1] > max_crop_w:
            f = max_crop_w / img.shape[1]
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        imgs.append((r, img))

    W = LABEL_W + max_crop_w + GAP * 3
    H = sum(im.shape[0] + GAP for _, im in imgs) + GAP
    sheet = np.full((H, W), 245, np.uint8)

    y = GAP
    for r, im in imgs:
        sheet[y:y + im.shape[0], LABEL_W + GAP * 2:LABEL_W + GAP * 2 + im.shape[1]] = im
        tag = {"text": "T", "grid": f"G{r['cells']}", "block": "B"}[r["kind"]]
        txt = f"[{tag}] {r['label']}"
        cv2.putText(sheet, txt[:52], (GAP, y + im.shape[0] // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, 0, 1, cv2.LINE_AA)
        cv2.line(sheet, (0, y + im.shape[0] + GAP // 2), (W, y + im.shape[0] + GAP // 2),
                 205, 1)
        y += im.shape[0] + GAP

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


if __name__ == "__main__":
    import sys
    doc = sys.argv[1] if len(sys.argv) > 1 else "Cif-form"
    pages = [int(x) for x in sys.argv[2:]] or [1]
    idx = run(doc)
    print(f"cropped {len(idx)} readable fields -> build/crops/{doc}/")
    by = defaultdict(lambda: defaultdict(int))
    for r in idx:
        by[r["page"]][r["ink_hint"]] += 1
    order = [STRONG, FAINT, EMPTY]
    print("  page " + "".join(f"{s:>11}" for s in order))
    tot = defaultdict(int)
    for p in sorted(by):
        print(f"{p:>6} " + "".join(f"{by[p][s]:>11}" for s in order))
        for s in order:
            tot[s] += by[p][s]
    print(f"{'ALL':>6} " + "".join(f"{tot[s]:>11}" for s in order))
    skip = tot[EMPTY]
    print(f"  -> {skip} fields have zero ink and skip the model "
          f"({100 * skip / max(len(idx), 1):.0f}%); {len(idx) - skip} are read")
    for pno in pages:
        p = contact_sheet(doc, pno, idx,
                          os.path.join(ROOT, "build", "qa", f"crops_{doc}_p{pno}.png"))
        print("  sheet:", p)
