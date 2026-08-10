"""
Assemble the final answer set.

Pulls together the four sources of truth produced upstream and turns them into one
reviewable record per field:

  template_schema.json  what fields exist, and what each one is
  alignment.json        whether the page it came from registered properly
  fields/<doc>.json     the ink evidence and the extent each answer occupies
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

import fields as fieldmod
import read_checkboxes
from prompts import ink_verdict
from validate import assess, cross_check

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "build", "template_schema.json")

SOURCE_NAMES = {"page": "vlm-page",          # read with the rest of its page, in one call
                "ocr-sweep": "ocr-verified",  # a structured field, second-opinioned by OCR
                "retry": "vlm-recheck",      # re-read on its own after failing its format
                "ink-gate": "ink-gate",      # no ink in the box, so never sent to a model
                # caches written by the earlier per-field batch reader
                "batch": "vlm-batch", "single": "vlm-single",
                "with-parent": "read-with-parent"}

# "Is this box filled in?" is answered in exactly one place - prompts.ink_verdict() - so
# that what the model is told, what gets re-read, and what a person is asked to look at all
# rest on the same measurement and the same thresholds. Flagging a field is never a reason
# to discard its value: everything here is reported either way.


def load(doc: str):
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    crops = fieldmod.load_index(doc)
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
            elif (r or {}).get("disputed"):
                # Two readers read this box and got different, individually valid values -
                # the OCR reader's was taken. No rule can settle it (that is what "both
                # valid" means), so it goes to a human. These are the errors that otherwise
                # ship silently: a misread digit in a CNIC still yields a valid CNIC.
                conf = "low"
                note = note or (f"readers disagreed - page read {(r or {}).get('was')!r}, "
                                f"the OCR reader read {raw!r} and was taken")
            elif src == "ink-gate":
                conf = "high"                  # zero ink: empty is a measurement
            elif src == "read-with-parent":
                # this box holds the second line of the box above it, and both were read
                # from one strip. Its text is in the parent's value, so an empty value
                # here is correct and the ink in it is already accounted for.
                conf = "high"
                note = note or "read together with the line above"
            elif c["ink_hint"] == "faint" and not a["empty"]:
                # the pixels say barely-anything, the model says there is a value
                conf = "low"
                note = note or "faint ink but a value was read - check for spill-over"
            elif (ink_verdict(c) == "filled" and a["empty"]
                    and not a.get("explicit_none")):
                # Ink was measured but nothing was transcribed - either a missed value or
                # spill-over from a neighbour. An explicit "N/A" is not this case.
                #
                # The test is the same conclusive one the prompt and the re-read use, and it
                # has to be: a weaker floor made this the noisiest flag on the form. Page 3's
                # blank supplementary-card grids carry 500-2200px of residue where the
                # printed cell borders survive registration, so a bare "any ink over 500px"
                # rule reported 30 empty boxes as unread answers - on a form with 118 filled
                # fields, that is a review list a person stops reading.
                conf = "low"
                note = note or "ink present but read as empty"

            rec = {
                "field": f["id"], "page": page, "section": f["section"],
                "label": f["label"], "group": f["table"], "kind": f["kind"],
                "type": a["type"], "value": a["value"], "raw": raw,
                "source": src, "confidence": conf,
                "valid": a["valid"], "note": note,
                "ink_hint": c["ink_hint"], "ink_px": c["ink_px"],
                "continuation_of": f.get("continuation_of", ""),
                # which reader produced this: numeric fields go to an OCR model, words to
                # a general VLM, so the value's provenance is part of the record
                "model": (r or {}).get("model", ""),
            }

        rec["page_problems"] = page_problems
        rec["needs_review"] = bool(rec["confidence"] == "low" or not rec["valid"]
                                   or page_problems)
        records.append(rec)

    # cross-field rules also need the ticked options: the CNIC's last digit encodes gender,
    # which only means something alongside the Gender question
    ticks = {r["group"] or r["label"]: r["label"]
             for r in records if r["kind"] == "checkbox" and r["value"]}
    issues = cross_check({r["field"]: {"value": r["value"], "empty": r["value"] in (None, "")}
                          for r in records if r["kind"] != "checkbox"}, ticks)
    return {"document": doc, "fields": records, "cross_check": issues,
            "alignment": [p for p in page_ok.values()]}


# ------------------------------------------------------------------ rendering
def answers_view(records) -> dict:
    """A human-shaped summary: one entry per question, grouped by section.

    A wrapped second line is one answer split across two boxes on the page, so it is
    rejoined here rather than reported as its own field. The per-field records stay
    one-per-box, because each line still has its own crop to review.

    Labels are not unique, so an entry must never overwrite another one: page 1 has two
    "P.A. No." boxes, page 2 repeats the whole address block for the office address, and
    page 4 carries "Name of the Bank/DFI (row 1)" inside six different tables. Keying on
    the label alone silently dropped 68 values across the form. A table name disambiguates
    where there is one, and a counter covers the rest - the summary is allowed to be ugly,
    it is not allowed to lose an answer.
    """
    out: dict = defaultdict(dict)
    groups: dict = defaultdict(list)
    conts: dict = defaultdict(list)
    for r in records:
        if r.get("continuation_of"):
            conts[r["continuation_of"]].append(r)

    def put(section: str, label: str, value) -> None:
        sec = out[section or "(unsectioned)"]
        key, n = label, 1
        while key in sec:
            n += 1
            key = f"{label} ({n})"
        sec[key] = value

    for r in records:
        if r["kind"] == "checkbox":
            groups[(r["section"], r["group"] or r["label"])].append(r)
            continue
        if r.get("continuation_of"):
            continue                                  # folded into its parent below
        parts = [r["value"]] + [c["value"] for c in conts.get(r["field"], [])]
        # a line often ends with the comma that leads into the next one
        parts = [str(p).strip().rstrip(",").strip() for p in parts if p not in (None, "")]
        if parts:
            label = f"{r['group']} - {r['label']}" if r.get("group") else r["label"]
            put(r["section"], label, ", ".join(parts))
    for (section, q), members in groups.items():
        sel = [m["label"] for m in members if m["value"]]
        if sel:
            put(section, q, sel if len(sel) > 1 else sel[0])
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
            "source", "model", "confidence", "valid", "needs_review", "note", "field"]
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
