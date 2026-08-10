"""
Guard: no field's instruction may contain a value that could be an answer to that field.

A spec describes the shape of a value. The moment it also carries a literal example, the
example becomes something a reader can hand back instead of reading - measured, not
theorised: a postcode spec that offered "54000 Lahore" as illustration got 54000 back for a
box that plainly read 12345.

The trap this catches is specific and easy to fall into: writing the specs while looking at
a filled sample form, so that form's own answers end up quoted as "examples" and are then
fed back on every real form. Ground truth is exactly the list of that form's answers, so it
is exactly the right thing to test against.

Run: python tests/test_prompts_leak.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import fields as fieldmod          # noqa: E402
import prompts                     # noqa: E402

# Words that are legitimately part of a format rule rather than a sample answer: the country
# and nationality this form is issued in, and the bank whose form it is, cannot be described
# without naming them.
ALLOWED = {"pakistan", "pakistani", "hbl", "habb", "none", "n/a"}
# Kinship terms are the exception the rule allows: for a Relationship box the vocabulary IS
# the answer space, not a sample drawn from it, and naming it is what lets a reader resolve
# a scrawled "Cousin" or "Spouse" at all. Listing them cannot teach a guess the way an
# invented house number or postcode does, because there is nothing else the box can hold.
ALLOWED |= {"wife", "husband", "son", "daughter", "father", "mother", "brother", "sister",
            "uncle", "aunt", "cousin", "friend", "colleague", "spouse", "parent",
            "clean", "secured", "cnic", "passport"}
MIN_LEN = 4                        # shorter strings collide with ordinary prose


def leaks(doc: str = "Cif-form") -> list[tuple[str, str]]:
    index = {r["field"]: r for r in fieldmod.load_index(doc)}
    found: list[tuple[str, str]] = []
    for path in sorted(os.listdir(os.path.join(ROOT, "tests"))):
        if not path.startswith(f"ground_truth_{doc}_"):
            continue
        with open(os.path.join(ROOT, "tests", path), encoding="utf-8") as fh:
            truth = json.load(fh)["values"]
        for fid, want in truth.items():
            rec = index.get(fid)
            value = str(want).strip()
            if rec is None or len(value) < MIN_LEN or value.lower() in ALLOWED:
                continue
            spec = prompts.spec(rec).lower()
            if value.lower() in spec:
                found.append((rec["label"], value))
    return found


if __name__ == "__main__":
    bad = leaks()
    if bad:
        print(f"FAIL: {len(bad)} field instruction(s) quote that field's own answer:")
        for label, value in bad:
            print(f"  {label[:44]:<46} contains {value!r}")
        print("\nDescribe the shape of the value instead of giving an example of it.")
        sys.exit(1)
    print("OK: no field instruction quotes a ground-truth answer")
