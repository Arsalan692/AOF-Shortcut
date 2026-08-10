"""
Phase C1 - locate every field on the registered scan and measure the ink in it.

This used to cut and write one PNG per field, because the reader was given one field at a
time. It no longer is: a whole page goes to the model in a single call, so nothing here
writes an image during a normal run. What it still does is the part the model cannot do
for itself - decide, from the template geometry and the ink layer, which strokes belong to
which field:

  * ink evidence per field, so a box with no ink at all never costs a model call and can
    never come back with an invented value;
  * the grown rectangle each answer really occupies, which is what the page image's
    numbered markers are drawn against and what the reviewer's crop is cut from.

Ownership is what makes both possible. Handwriting on this form overflows its box by up to
34pt against 16pt row spacing, and 23.5% of all ink sits wholly outside every box, so
neither "the printed rectangle" nor "the connected blob" is the answer on its own.

Output: build/fields/<doc>.json - one record per readable field.

Crops are still produced on demand, for two callers only: the second-pass re-read of
fields that failed their format check, and the reviewer clicking a value in the web UI.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

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
GHOST_K = 5        # erasing a foreign stroke must also cover its sub-threshold rim, or the
                   # rim survives as a glowing outline (see render_crop)

READABLE = ("text", "grid", "block")

# Blankness is measured on the field INTERIOR, inset past the printed border: a border
# sliver left by 1px of registration error otherwise reads as ink.
BLANK_INSET = 2.0
# Ink is counted after one erosion. Pen strokes are 3-6px thick at 300 dpi and keep a core;
# the 1-2px residue left by printed placeholder glyphs (date grids carry a faint
# "D D M M Y Y Y Y") disappears.
ERODE = 1
BLANK_FLOOR = 60
BASELINE_SHARE = 0.06

# Only an exactly-zero field is treated as certainly empty and skipped. Because handwriting
# overflows, a box can hold real ink belonging to its NEIGHBOUR - on this form the page 3
# ID Number digits spill into the CIF Number cells - so "is there ink here" and "is this
# field filled" are different questions, and no pixel statistic separates them (measured
# against ground truth: filled stroke thickness min 2.87 vs empty max 3.82; eroded area
# 41 vs 251). The gate stays conservative and records a hint instead.
EMPTY, FAINT, STRONG = "empty", "faint", "strong"

# How far a crop may grow beyond the printed box to chase its handwriting. Measured on a
# real form, overflow runs to 26pt right and 34pt vertically, so these sit past the p90.
MAX_GROW_X = 44.0
MAX_GROW_Y = 30.0
OWN_INSET = 1.0    # boxes are shrunk this much when seeding ownership so that table cells
                   # sharing a border stay separate regions
# A nearest-box partition assigns every pixel to some field, so residue 40pt away would be
# credited to an empty field. Ink must start within this distance of the box to be claimed.
OWN_MAX_DIST = 8.0
# Share of a stroke's connected component this field must already hold before the whole
# component is claimed. Above it, the stroke is one answer leaning past the boundary; near
# half, it is a blob two answers wrote into each other and must stay cut.
COMP_MAJORITY = 0.6
TRIM_PCT = 0.4     # ignore this share of extreme owned pixels when sizing the crop


def interior_ink(ink: np.ndarray, rect, inset=BLANK_INSET) -> int:
    h, w = ink.shape[:2]
    x0 = max(0, int(round((rect[0] + inset) * S)))
    y0 = max(0, int(round((rect[1] + inset) * S)))
    x1 = min(w, int(round((rect[2] - inset) * S)))
    y1 = min(h, int(round((rect[3] - inset) * S)))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(ink[y0:y1, x0:x1]))


def ownership_labels(shape, fields) -> tuple[np.ndarray, np.ndarray, dict]:
    """Partition the page: every pixel is labelled with the field whose box is nearest.

    This is the answer to handwriting that leaves its box. A fixed pad cannot work - rows
    are ~16pt apart but writing strays up to 34pt - and a plain connected-component rule
    fails when one person's "Karachi East" touches the next field's "Karachi", merging two
    answers into one blob. Nearest-box ownership both reaches out to strokes that never
    touched the box and cuts a merged blob along the boundary between the two fields.
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


def owned_mask(ink, dist, labels, cc, lab, rect, reclaim: bool = False):
    """(mask, window) - the ink this field owns.

    Three rules, in order: the pixel's nearest box must be ours (so a blob straddling two
    fields is cut along the boundary rather than duplicated into both); some of the stroke
    must lie within OWN_MAX_DIST of the box (so distant residue is not credited to an empty
    field); and once a stroke qualifies, the rest of its connected component comes too,
    which is what recovers writing that runs far outside the box.
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
    comp = cc[win]
    owned = mine & np.isin(comp, ids)

    # Reclaim a stroke that merely leans into the next row. Nearest-box ownership cuts along
    # the midline between two boxes, and grid digits are written far taller than their 12pt
    # cells: the bottom of "02136102837" fell on the row below, so it was both excluded from
    # the crop and painted out as foreign ink - which turned the 2 into a 9 and the 3 into a
    # 2, because a digit loses its identity when its base is missing.
    #
    # The test is how much of the component we already hold. One answer leaning past the
    # boundary is still overwhelmingly ours; a blob genuinely shared by two answers - the
    # "Karachi East" that touches the next field's "Karachi" - is split near half and stays
    # cut, which is the case nearest-box ownership exists to handle.
    if reclaim:
        for cid in ids:
            m = comp == cid
            total = int(m.sum())
            if total and int((mine & m).sum()) / total >= COMP_MAJORITY:
                owned |= m
    return owned, (x0, y0, x1, y1)


def owned_extent(ink, dist, labels, cc, lab, rect, reclaim: bool = False) -> list:
    """The rect grown to contain the strokes this field owns, in PDF points.

    Extremes are trimmed by a fraction of a percent: a single stray speck that survived
    cleaning would otherwise stretch the rectangle tens of points and drag a neighbouring
    row into view.
    """
    m, (x0, y0, _, _) = owned_mask(ink, dist, labels, cc, lab, rect, reclaim)
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
    """Crop `ext` out of the registered page with every OTHER field's handwriting erased.

    A rectangle grown to chase overflowing handwriting inevitably overlaps neighbouring
    rows - the Next-of-Kin Name box reaches down over the Telephone Number answer.
    Ownership already says which strokes belong here, so the strokes that do not are
    painted out to paper and the model sees exactly one answer.
    """
    wx0, wy0, wx1, wy1 = win
    patch = reg[wy0:wy1, wx0:wx1].copy()
    if owned is not None and ink is not None:
        foreign = (ink[wy0:wy1, wx0:wx1] > 0) & ~owned
        if foreign.any():
            # Paint with the paper *immediately around* each stroke, not one level for the
            # whole window: a single bright value taken from anywhere in the crop made
            # erased strokes come out brighter than the grey CamScanner band they sat on,
            # so the removed letters glowed white instead of disappearing. Over a 31px
            # window a pen stroke is the minority, so the median is the paper under it.
            paper = cv2.medianBlur(patch, 31)
            # The ink mask is a threshold, so each foreign stroke keeps an anti-aliased rim
            # a pixel or two wide that falls below it; filling only the mask leaves that rim
            # behind as a hollow outline of the letter. Widen the fill to swallow it - but
            # never into ink this field owns, because an overflowing neighbour physically
            # crosses the answer and a blind dilation would erase part of the value.
            wide = cv2.dilate(foreign.astype(np.uint8), np.ones((GHOST_K, GHOST_K), np.uint8))
            guard = cv2.dilate(owned.astype(np.uint8), np.ones((3, 3), np.uint8))
            fill = (wide > 0) & (guard == 0)
            patch[fill] = paper[fill]
            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            grown = cv2.dilate(fill.astype(np.uint8), np.ones((3, 3), np.uint8))
            patch = np.where(grown > 0, blur, patch).astype(np.uint8)

    ax = max(0, int(round((ext[0] - pad_x) * S)) - wx0)
    ay = max(0, int(round((ext[1] - pad_y) * S)) - wy0)
    bx = min(patch.shape[1], int(round((ext[2] + pad_x) * S)) - wx0)
    by = min(patch.shape[0], int(round((ext[3] + pad_y) * S)) - wy0)
    if bx <= ax or by <= ay:
        return np.full((10, 10), 255, np.uint8)
    return patch[ay:by, ax:bx]


def crop_rect(img: np.ndarray, rect, pad_x=PAD_X, pad_y=PAD_Y) -> np.ndarray:
    """A plain cut of `rect` out of a 300 dpi page image, with no ink erased."""
    h, w = img.shape[:2]
    x0 = max(0, int(round((rect[0] - pad_x) * S)))
    y0 = max(0, int(round((rect[1] - pad_y) * S)))
    x1 = min(w, int(round((rect[2] + pad_x) * S)))
    y1 = min(h, int(round((rect[3] + pad_y) * S)))
    if x1 <= x0 or y1 <= y0:
        return np.full((10, 10), 255, np.uint8)
    return img[y0:y1, x0:x1]


# --------------------------------------------------------------- page context
@dataclass
class Page:
    """One registered page with everything needed to reason about its ink.

    Built once per page and passed around: the distance transform and the two connected
    component maps cost about a second each, and both index building and any crop the
    model or a reviewer asks for want the same ones.
    """
    page: int
    reg: np.ndarray                 # registered greyscale, 300 dpi, template space
    ink: np.ndarray | None          # handwriting with the printed form subtracted
    ink_e: np.ndarray | None        # the same, eroded once, for the blankness gate
    dist: np.ndarray
    labels: np.ndarray
    cc: np.ndarray | None
    cc_e: np.ndarray | None
    tpl_ink: np.ndarray
    lab_of: dict                    # field id -> ownership label

    def crop(self, field: dict) -> np.ndarray:
        """The image for one field: its box grown to its own handwriting, with every other
        field's strokes painted out."""
        rect = field["rect"]
        lab = self.lab_of[field.get("field") or field["id"]]
        pad_y = BLOCK_PAD_Y if field["kind"] == "block" else PAD_Y
        if self.ink is None:
            return crop_rect(self.reg, rect, PAD_X, pad_y)
        # only a character grid may reclaim a leaning stroke: its printed cells make
        # "this ink is in my box" a fact, which free text cannot claim
        reclaim = field["kind"] == "grid"
        ext = owned_extent(self.ink, self.dist, self.labels, self.cc, lab, rect, reclaim)
        om, win = owned_mask(self.ink, self.dist, self.labels, self.cc, lab, rect, reclaim)
        return render_crop(self.reg, self.ink, om, win, ext, PAD_X, pad_y)


def load_page(doc: str, pno: int, schema: dict | None = None) -> Page | None:
    """Build the Page context for one registered page, or None if it was never scanned."""
    schema = schema or json.load(open(SCHEMA, encoding="utf-8"))
    regdir = os.path.join(ROOT, "build", "registered", doc)
    reg_path = os.path.join(regdir, f"reg_p{pno}.png")
    if not os.path.exists(reg_path):
        return None
    reg = cv2.imread(reg_path, cv2.IMREAD_GRAYSCALE)
    if reg is None:
        return None
    ink = cv2.imread(os.path.join(regdir, f"ink_p{pno}.png"), cv2.IMREAD_GRAYSCALE)
    with fitz.open(BLANK) as blank:
        tpl_ink = binarize(render_gray(blank[pno - 1]), min_area=1)

    # Checkboxes take part in the ownership partition even though they are never cropped:
    # a tick is ink, and if no box claims it the nearest text field will, which both
    # inflates that field's ink and drags the tick into its crop.
    seeds = sorted((f for f in schema["fields"] if f["page"] == pno and not f["mirror"]),
                   key=lambda z: (z["rect"][1], z["rect"][0]))
    dist, labels, fmap = ownership_labels(reg.shape[:2], seeds)
    lab_of = {f["id"]: fmap[i] for i, f in enumerate(seeds)}

    k = np.ones((3, 3), np.uint8)
    ink_e = cv2.erode(ink, k, iterations=ERODE) if ink is not None else None
    cc = cv2.connectedComponents(ink, 8)[1] if ink is not None else None
    cc_e = cv2.connectedComponents(ink_e, 8)[1] if ink_e is not None else None
    return Page(pno, reg, ink, ink_e, dist, labels, cc, cc_e, tpl_ink, lab_of)


def index_path(doc: str) -> str:
    return os.path.join(ROOT, "build", "fields", f"{doc}.json")


# Rows on this form sit ~16pt apart, and boxes sharing one are not aligned to the point.
ROW_TOL = 4.0


def readable_fields(schema: dict, pno: int) -> list[dict]:
    """The page's readable fields in reading order: top to bottom, then left to right.

    The left-to-right part has to be enforced, not assumed. Sorting on (row_y, x) looks
    right and is not: page 4's second reference has Tel. No. (Off.) at row_y 155.8 against
    Mobile's 155.7, so a tenth of a point put the marker for the right-hand box before the
    one in the middle of the row. The numbers then ran 12, 14, 13 across the page, the
    reader took them in sequence anyway, and two phone numbers were returned under each
    other's labels - both perfectly well-formed, so nothing downstream could notice.

    Reference 1's identical row shares one row_y exactly and came out correct, which is what
    makes this worth guarding: the bug appears or not depending on sub-point jitter in the
    template.
    """
    fields = sorted((f for f in schema["fields"]
                     if f["page"] == pno and not f["mirror"] and f["kind"] in READABLE),
                    key=lambda z: (z["row_y"], z["rect"][0]))
    rows: list[list[dict]] = []
    for f in fields:
        if rows and abs(f["row_y"] - rows[-1][0]["row_y"]) <= ROW_TOL:
            rows[-1].append(f)          # anchored on the row's first box, so a run of
        else:                           # near-misses cannot chain into one long row
            rows.append([f])
    return [f for row in rows for f in sorted(row, key=lambda z: z["rect"][0])]


def build_index(doc: str, pages: list[int] | None = None) -> list[dict]:
    """One record per readable field on every registered page, with its ink evidence."""
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    want = sorted({f["page"] for f in schema["fields"]}
                  if pages is None else set(pages))

    index: list[dict] = []
    for pno in want:
        pg = load_page(doc, pno, schema)
        if pg is None:
            continue
        for f in readable_fields(schema, pno):
            lab = pg.lab_of[f["id"]]
            reclaim = f["kind"] == "grid"
            if pg.ink is not None:
                ext = owned_extent(pg.ink, pg.dist, pg.labels, pg.cc, lab, f["rect"], reclaim)
                om = owned_mask(pg.ink, pg.dist, pg.labels, pg.cc, lab, f["rect"], reclaim)[0]
            else:
                ext, om = list(f["rect"]), None
            # Blankness is judged on OWNED ink, not on the box interior: nearly a quarter of
            # all handwriting on this form sits wholly outside its box, so an interior-only
            # test would call those fields empty and silently drop the answer. raw asks "is
            # there any stroke at all", core asks "is it a pen stroke or border residue".
            oe = (owned_mask(pg.ink_e, pg.dist, pg.labels, pg.cc_e, lab, f["rect"], reclaim)[0]
                  if pg.ink_e is not None else None)
            raw = int(om.sum()) if om is not None else -1
            core = int(oe.sum()) if oe is not None else -1
            base = interior_ink(pg.tpl_ink, f["rect"])
            gate = BLANK_FLOOR + BASELINE_SHARE * base
            index.append({
                "field": f["id"], "page": pno, "kind": f["kind"],
                "label": f["label"], "section": f["section"], "table": f["table"],
                "grid_hint": f["grid_hint"], "cells": len(f["cells"]),
                "continuation_of": f.get("continuation_of", ""),
                "row_y": f["row_y"],
                "rect": [round(v, 2) for v in f["rect"]],
                "crop_rect": [round(v, 2) for v in ext],
                "ink_px": raw, "core_px": core, "baseline_px": base, "gate": round(gate),
                "ink_hint": EMPTY if raw == 0 else (FAINT if core < gate else STRONG),
                "needs_model": raw != 0,
            })

    os.makedirs(os.path.dirname(index_path(doc)), exist_ok=True)
    with open(index_path(doc), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)
    return index


def load_index(doc: str) -> list[dict]:
    with open(index_path(doc), encoding="utf-8") as fh:
        return json.load(fh)


def review_crop(doc: str, rec: dict, scale: float = 1.0) -> np.ndarray | None:
    """The patch a value was read from, for a human to look at.

    Cut straight from the registered page using the extent already stored in the index, so
    a reviewer's click costs a file read rather than a rebuild of the page's ownership map.
    Foreign strokes are left in place on purpose: a person looking at the scan should see
    what is actually on the paper.
    """
    path = os.path.join(ROOT, "build", "registered", doc, f"reg_p{rec['page']}.png")
    reg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if reg is None:
        return None
    pad_y = BLOCK_PAD_Y if rec["kind"] == "block" else PAD_Y
    img = crop_rect(reg, rec.get("crop_rect") or rec["rect"], PAD_X, pad_y)
    if scale and scale != 1.0 and 0.2 <= scale <= 4.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "Cif-form"
    idx = build_index(d)
    print(f"{len(idx)} readable fields -> {os.path.relpath(index_path(d), ROOT)}")
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
    print(f"  -> {tot[EMPTY]} fields have zero ink and are never sent to the model; "
          f"{len(idx) - tot[EMPTY]} are read")
