"""
QA overlay: draw the harvested schema on top of the blank template so a human can
verify field geometry, type classification and label association at a glance.

  text     -> blue box,  label drawn above
  grid     -> green box, each cell outlined
  checkbox -> red box

Output: build/qa/schema_p<N>.png
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import cv2
import fitz
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLANK = os.path.join(ROOT, "Forms", "Account_Opening_Form_blank template.pdf")
SCHEMA = os.path.join(ROOT, "build", "template_schema.json")
OUTDIR = os.path.join(ROOT, "build", "qa")
DPI = 150
S = DPI / 72.0

COLORS = {"text": (255, 120, 0), "grid": (0, 170, 0), "checkbox": (0, 0, 255),
          "block": (200, 0, 200)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    by_page = defaultdict(list)
    for f in schema["fields"]:
        by_page[f["page"]].append(f)

    doc = fitz.open(BLANK)
    for pno in sorted(by_page):
        pix = doc[pno - 1].get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for f in by_page[pno]:
            x0, y0, x1, y1 = (int(v * S) for v in f["rect"])
            col = COLORS[f["kind"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
            for c in f.get("cells", []):
                cx0, cy0, cx1, cy1 = (int(v * S) for v in c)
                cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (0, 220, 220), 1)
            lab = f["label"][:34] if f["label"] else "!! NO LABEL"
            cv2.putText(img, lab, (x0, max(9, y0 - 2)),
                        cv2.FONT_HERSHEY_PLAIN, 0.62,
                        col if f["label"] else (255, 0, 255), 1, cv2.LINE_AA)

        out = os.path.join(OUTDIR, f"schema_p{pno}.png")
        cv2.imwrite(out, img)
        print(out, len(by_page[pno]), "fields")


if __name__ == "__main__":
    main()
