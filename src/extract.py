"""
Assemble the final answer set.

Pulls together the four sources of truth produced upstream and turns them into one
reviewable record per field:

  template_schema.json  what fields exist, and what each one is
  alignment.json        whether the page it came from registered properly
  crops/<doc>/index.json    the ink evidence and crop provenance
  reads/<doc>.json      what the local vision model transcribed
  read_checkboxes       tick boxes, measured rather than read

Every value carries where it came from and whether it can be trusted. Nothing is
silently dropped or silently corrected: a field that fails its format check, disagrees
with the ink evidence, or sits on a badly registered page is emitted with
needs_review set, so a human looks at a short list instead of the whole form.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import read_checkboxes
from validate import assess, cross_check

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "build", "template_schema.json")

SOURCE_NAMES = {"batch": "vlm-batch", "retry": "vlm-recheck",
                "single": "vlm-single",      # kept for caches written before batched re-reads
                "ink-gate": "ink-gate"}


def load(doc: str):
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    crops = json.load(open(os.path.join(ROOT, "build", "crops", doc, "index.json"),
                           encoding="utf-8"))
    reads_path = os.path.join(ROOT, "build", "reads", f"{doc}.json")
    reads = json.load(open(reads_path, encoding="utf-8")) if os.path.exists(reads_path) else {}
    align_path = os.path.join(ROOT, "build", "registered", doc, "alignment.json")
    align = json.load(open(align_path, encoding="utf-8")) if os.path.exists(align_path) else {}
    page_ok = {p["template_page"]: p for p in align.get("pages", [])}
    return schema, {c["field"]: c for c in crops}, reads, page_ok


def build(doc: str) -> dict:
    schema, crops, reads, page_ok = load(doc)
    boxes = read_checkboxes.run(os.path.join(ROOT, "build", "registered", doc))

    records = []
    for f in schema["fields"]:
        page = f["page"]
        pinfo = page_ok.get(page)
        page_problems = [] if (pinfo is None or pinfo["ok"]) else list(pinfo["problems"])

        if f["kind"] == "checkbox":
            cb = boxes.get(f["id"])
            if cb is None:                     # Urdu mirror, or a page never scanned
                continue
            rec = {
                "field": f["id"], "page": page, "section": f["section"],
                "label": f["label"], "group": f["group"], "kind": "checkbox",
                "type": "boolean", "value": cb["value"], "raw": cb["score"],
                "source": "checkbox-cv", "confidence": cb["confidence"],
                "valid": True, "note": "", "select_mode": cb["select_mode"],
            }
        else:
            c = crops.get(f["id"])
            if c is None:
                continue
            r = reads.get(f["id"])
            raw = (r or {}).get("value", "")
            src = SOURCE_NAMES.get((r or {}).get("source", ""), "not-read")
            a = assess(raw, c)

            note = a["note"]
            conf = "high"
            if src == "not-read":
                conf, note = "low", note or "no value produced"
            elif not a["valid"]:
                conf = "low"
            elif src == "ink-gate":
                conf = "high"                  # zero ink: empty is a measurement
            elif c["ink_hint"] == "faint" and not a["empty"]:
                # the pixels say barely-anything, the model says there is a value
                conf = "low"
                note = note or "faint ink but a value was read - check for spill-over"
            elif c["ink_hint"] != "empty" and a["empty"] and not a.get("explicit_none"):
                # ink was measured but nothing was transcribed - either a missed value
                # or spill-over from a neighbour. An explicit "N/A" is not this case.
                conf = "low"
                note = note or "ink present but read as empty"

            rec = {
                "field": f["id"], "page": page, "section": f["section"],
                "label": f["label"], "group": f["table"], "kind": f["kind"],
                "type": a["type"], "value": a["value"], "raw": raw,
                "source": src, "confidence": conf,
                "valid": a["valid"], "note": note,
                "ink_hint": c["ink_hint"], "ink_px": c["ink_px"],
            }

        rec["page_problems"] = page_problems
        rec["needs_review"] = bool(rec["confidence"] == "low" or not rec["valid"]
                                   or page_problems)
        records.append(rec)

    issues = cross_check({r["field"]: {"value": r["value"], "empty": r["value"] in (None, "")}
                          for r in records if r["kind"] != "checkbox"})
    return {"document": doc, "fields": records, "cross_check": issues,
            "alignment": [p for p in page_ok.values()]}


# ------------------------------------------------------------------ rendering
def answers_view(records) -> dict:
    """A human-shaped summary: one entry per question, grouped by section."""
    out: dict = defaultdict(dict)
    groups: dict = defaultdict(list)
    for r in records:
        if r["kind"] == "checkbox":
            groups[(r["section"], r["group"] or r["label"])].append(r)
        elif r["value"] not in (None, ""):
            out[r["section"] or "(unsectioned)"][r["label"]] = r["value"]
    for (section, q), members in groups.items():
        sel = [m["label"] for m in members if m["value"]]
        if sel:
            out[section or "(unsectioned)"][q] = sel if len(sel) > 1 else sel[0]
    return {k: out[k] for k in out}


def write_outputs(doc: str, data: dict) -> tuple[str, str]:
    outdir = os.path.join(ROOT, "build", "output")
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, f"{doc}.json")
    payload = dict(data)
    payload["answers"] = answers_view(data["fields"])
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    cpath = os.path.join(outdir, f"{doc}.csv")
    cols = ["page", "section", "group", "label", "kind", "type", "value", "raw",
            "source", "confidence", "valid", "needs_review", "note", "field"]
    with open(cpath, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(data["fields"], key=lambda z: (z["page"], z["field"])):
            w.writerow(r)
    return jpath, cpath


def report(data: dict) -> None:
    recs = data["fields"]
    read = [r for r in recs if r["kind"] != "checkbox"]
    cb = [r for r in recs if r["kind"] == "checkbox"]
    filled = [r for r in read if r["value"] not in (None, "")]
    review = [r for r in recs if r["needs_review"]]

    print(f"\n{'':-<74}")
    print(f"{len(recs)} fields | {len(cb)} checkboxes ({sum(1 for r in cb if r['value'])} ticked)"
          f" | {len(read)} readable ({len(filled)} with a value)")
    print(f"invalid format: {sum(1 for r in read if not r['valid'])}"
          f" | needs review: {len(review)}")

    bad_pages = [p for p in data["alignment"] if not p["ok"]]
    if bad_pages:
        print("\nALIGNMENT PROBLEMS:")
        for p in bad_pages:
            print(f"  template p{p['template_page']}: {', '.join(p['problems'])}")

    if data["cross_check"]:
        print("\nCROSS-FIELD ISSUES:")
        for m in data["cross_check"]:
            print(f"  - {m}")

    if review:
        print(f"\nFIELDS TO REVIEW ({len(review)}):")
        for r in sorted(review, key=lambda z: (z["page"], z["label"]))[:40]:
            v = r["value"] if r["kind"] != "checkbox" else r["value"]
            print(f"  p{r['page']} {r['label'][:38]:<40} = {str(v)[:24]:<26} {r['note'][:34]}")
        if len(review) > 40:
            print(f"  ... and {len(review) - 40} more")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", nargs="?", default="Cif-form")
    a = ap.parse_args()
    data = build(a.doc)
    j, c = write_outputs(a.doc, data)
    report(data)
    print(f"\nwrote {os.path.relpath(j, ROOT)} and {os.path.relpath(c, ROOT)}")
