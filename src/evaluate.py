"""
Score the pipeline against hand-transcribed ground truth.

Reports two rates, because they answer different questions:

  exact       character-for-character after trimming - what a downstream system that
              consumes the value verbatim would see
  normalised  ignoring case, spaces and punctuation - whether the reader actually
              recognised the writing, as opposed to formatting it differently

Also scores the empty/non-empty decision separately: a value invented in a blank field
and a value missed in a filled one are the two failures that matter most, and both are
invisible in a plain accuracy number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from validate import NONE_MARKERS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def equivalent(want: str, have: str) -> bool:
    """Is `have` an acceptable rendering of `want`?

    Beyond normalisation, a none-marker counts as satisfied by an empty value: if the
    customer wrote "N/A" then both the literal "N/A" and a resolved null are correct
    extractions, and scoring one of them as a miss would just measure a convention.
    """
    if norm(have) == norm(want):
        return True
    return want.strip().lower() in NONE_MARKERS and have.strip() == ""


def main(doc: str, page: int):
    gt_path = os.path.join(ROOT, "tests", f"ground_truth_{doc}_p{page}.json")
    gt = json.load(open(gt_path, encoding="utf-8"))
    out = json.load(open(os.path.join(ROOT, "build", "output", f"{doc}.json"),
                         encoding="utf-8"))
    recs = [r for r in out["fields"] if r["page"] == page]

    # ---------------------------------------------------------------- values
    # keyed by field id: labels are not unique (two page-1 fields are both "Address")
    got = {r["field"]: r for r in recs if r["kind"] != "checkbox"}
    truth = dict(gt["values"])

    # A wrapped answer is read as one strip, so its whole text lands in the parent box
    # while the continuation box is legitimately empty. Ground truth was transcribed
    # box-by-box, so compare the joined value against the joined truth - otherwise both
    # halves score as misses purely because of where the line break was recorded.
    folded = []
    for r in recs:
        parent = r.get("continuation_of")
        if not parent or parent not in truth or r["field"] not in truth:
            continue
        if str((got.get(parent) or {}).get("value") or "") == "":
            continue                      # parent empty: nothing was folded, score as-is
        truth[parent] = ", ".join(p.strip().rstrip(",").strip()
                                  for p in (truth[parent], truth[r["field"]]) if p.strip())
        truth.pop(r["field"])
        folded.append(r["label"])
    if folded:
        print(f"  wrapped answers scored as one value: {', '.join(folded)}")
    exact = nrm = 0
    empty_right = empty_total = 0
    rows = []
    for fid, want in truth.items():
        r = got.get(fid)
        have = "" if r is None or r["value"] in (None, "") else str(r["value"])
        blank_expected = want.strip() == "" or want.strip().lower() in NONE_MARKERS
        e = have.strip() == want.strip() or (blank_expected and have == "")
        n = equivalent(want, have)
        exact += e
        nrm += n
        empty_total += 1
        empty_right += blank_expected == (have == "")
        rows.append((n, (r or {}).get("label", fid), want, have,
                     "" if r is None else r["confidence"],
                     "" if r is None else r["note"]))
    missing = set(truth) - set(got)
    if missing:
        print(f"  ground-truth ids not present in output: {sorted(missing)}")

    n_tot = len(truth)
    print(f"page {page} readable fields: {n_tot}")
    print(f"  exact           {exact}/{n_tot}  ({100*exact/n_tot:.0f}%)")
    print(f"  normalised      {nrm}/{n_tot}  ({100*nrm/n_tot:.0f}%)")
    print(f"  empty/filled    {empty_right}/{empty_total}  ({100*empty_right/empty_total:.0f}%)")

    misses = [r for r in rows if not r[0]]
    if misses:
        print(f"\n  MISREADS ({len(misses)}):")
        for _, lab, want, have, conf, note in misses:
            flag = "flagged" if conf == "low" else "NOT flagged"
            print(f"    {lab[:34]:<36} want {want[:22]!r:<24} got {have[:22]!r:<24} {flag}")
            if note:
                print(f"      note: {note[:80]}")

    # ------------------------------------------------------------ checkboxes
    cbs = [r for r in recs if r["kind"] == "checkbox"]
    sel: dict[str, list[str]] = {}
    for r in cbs:
        if r["value"]:
            sel.setdefault(r["group"] or r["label"], []).append(r["label"])
    cb_truth = gt.get("checkboxes", {})
    cb_ok = 0
    cb_bad = []
    for q, want in cb_truth.items():
        have = sel.get(q, [])
        if len(have) == 1 and norm(have[0]) == norm(want):
            cb_ok += 1
        else:
            cb_bad.append((q, want, have))
    if cb_truth:
        print(f"\ncheckbox questions: {cb_ok}/{len(cb_truth)} "
              f"({100*cb_ok/len(cb_truth):.0f}%)")
        for q, want, have in cb_bad:
            print(f"    {q[:44]:<46} want {want!r} got {have}")

    spurious = {q: v for q, v in sel.items() if q not in cb_truth}
    if spurious:
        print(f"  ticks in questions not in ground truth: {len(spurious)}")

    flagged = sum(1 for r in recs if r["needs_review"])
    caught = sum(1 for r in misses if r[4] == "low")
    print(f"\nreview list: {flagged} of {len(recs)} fields; "
          f"{caught}/{len(misses)} misreads were flagged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", nargs="?", default="Cif-form")
    ap.add_argument("--page", type=int, default=1)
    a = ap.parse_args()
    main(a.doc, a.page)
