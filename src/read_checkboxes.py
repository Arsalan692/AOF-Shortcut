"""
Phase C1 - read tick boxes with computer vision, not OCR.

Checkboxes are the one part of this form that can be read deterministically, and
they are roughly 45% of all fields, so it is worth doing properly.

Method: measure pen ink inside each box interior on the registered ink layer, then
resolve each *group* competitively - the option with the most ink wins. Competitive
resolution is what makes this robust to the two things that actually go wrong on
these scans:

  * ticks are drawn far larger than the box and spill over neighbouring options, so
    an absolute per-box threshold produces multiple "checked" boxes per question;
  * a light tick can fall below any threshold that is high enough to reject
    speckle, so a fixed cut loses real answers.

Groups the form marks "(Select all that apply)" are scored independently instead.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "build", "template_schema.json")
DPI = 300
S = DPI / 72.0

INSET = 1.3        # pt trimmed off each edge so the printed border is excluded
HALO = 4.0         # pt of margin added around the box for the second measurement
ON = 0.030         # ink fraction that counts as a mark (calibrated: empty boxes
                   # measure 0.000 once the ink layer is denoised, marked ones 0.024+)
DOMINANCE = 0.55   # winner must hold this share of its group's total ink


def ink_fraction(ink: np.ndarray, rect, inset=INSET) -> float:
    x0 = int(round((rect[0] + inset) * S))
    y0 = int(round((rect[1] + inset) * S))
    x1 = int(round((rect[2] - inset) * S))
    y1 = int(round((rect[3] - inset) * S))
    h, w = ink.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(ink[y0:y1, x0:x1])) / ((x1 - x0) * (y1 - y0))


def score_box(ink: np.ndarray, rect) -> tuple[float, float]:
    """(interior, halo) ink fractions.

    Two measurements are needed because ticks on these forms are drawn much bigger
    than the 11pt box: some pass straight through the interior, others hook in from
    outside and barely clip it. The interior is the trustworthy signal, the halo the
    sensitive one.
    """
    inner = ink_fraction(ink, rect, INSET)
    halo = ink_fraction(ink, [rect[0] - HALO, rect[1] - HALO,
                              rect[2] + HALO, rect[3] + HALO], 0.0)
    return inner, halo


def read_page(fields, ink: np.ndarray) -> dict:
    boxes = [f for f in fields if f["kind"] == "checkbox" and not f["mirror"]]
    for f in boxes:
        f["_in"], f["_halo"] = score_box(ink, f["rect"])

    groups: dict[str, list] = defaultdict(list)
    for f in boxes:
        groups[f["group"] or f"__lone__{f['id']}"].append(f)

    out = {}
    for members in groups.values():
        multi = any(m["select_mode"] == "many" for m in members)

        # Choose one measurement for the whole group. If any option was struck
        # through its interior, interiors settle it and halos are ignored - halos of
        # the losing options pick up bleed from adjacent handwriting and can
        # otherwise out-vote a clear winner.
        use_halo = max(m["_in"] for m in members) < ON
        for m in members:
            m["_s"] = m["_halo"] if use_halo else m["_in"]

        total = sum(m["_s"] for m in members)
        best = max(m["_s"] for m in members)

        for m in members:
            s = m["_s"]
            if multi or len(members) == 1:
                checked = s >= ON
            else:
                # single-answer question: the strongest option wins, provided it
                # clearly dominates and carries real ink
                checked = bool(s >= ON and s == best and s / max(total, 1e-9) >= DOMINANCE)

            if checked:
                # a mark seen only in the halo never touched the box, so it may
                # belong to a neighbour - surface that for review
                conf = "high" if not use_halo and s / max(total, 1e-9) > 0.75 else "low"
            else:
                conf = "low" if s >= ON * 0.5 else "high"

            out[m["id"]] = {
                "field": m["id"], "page": m["page"], "section": m["section"],
                "group": m["group"], "option": m["label"], "kind": "checkbox",
                "value": checked, "score": round(s, 4),
                "interior": round(m["_in"], 4), "halo": round(m["_halo"], 4),
                "confidence": conf,
                "select_mode": m["select_mode"] or ("many" if multi else "one"),
            }
    return out


def run(regdir: str) -> dict:
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    by_page = defaultdict(list)
    for f in schema["fields"]:
        by_page[f["page"]].append(f)

    results = {}
    for pno in sorted(by_page):
        path = os.path.join(regdir, f"ink_p{pno}.png")
        if not os.path.exists(path):
            continue
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        results.update(read_page(by_page[pno], ink))
    return results


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Cif-form"
    regdir = os.path.join(ROOT, "build", "registered", name)
    res = run(regdir)

    ticked = {k: v for k, v in res.items() if v["value"]}
    print(f"{len(res)} checkboxes read, {len(ticked)} ticked\n")
    seen = set()
    for v in sorted(res.values(), key=lambda z: (z["page"], z["group"])):
        if not v["value"]:
            continue
        head = f"p{v['page']} {v['group']}"
        if head not in seen:
            seen.add(head)
        print(f"  p{v['page']:<2} {v['group'][:48]:<50} = {v['option'][:34]:<36}"
              f" ink={v['score']:.3f} {v['confidence']}")
