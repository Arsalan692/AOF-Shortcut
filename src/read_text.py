"""
Phase C3 - read the handwritten values with a LOCAL vision model (Ollama).

Nothing leaves the machine: the model runs at http://localhost:11434.

Fields are read in numbered batches stacked into one image. That is not a nicety - on
CPU this hardware costs ~120s per distinct image almost regardless of its size, so the
only way to make a 183-field form practical is to put many fields in one image.
Measured: 10 per batch = ~20s/field, a 6x speed-up, with values still correctly
attached to their labels because each strip is numbered.

Anything that comes back malformed, or fails its format check, is re-read - but in small
montages, not one image per field. Re-reading individually cost 978s of a 1764s page-one
run (55% of the total) for just 7 fields, because a single crop costs the same as a full
batch. Re-read batches carry fewer strips at roughly double the height, so each field
still gets far more pixels than in a first-pass batch.

Results are cached per field, so an interrupted run resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from validate import assess, infer_type

# field labels carry non-ASCII typography; never let printing them kill a long run
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("HBL_VLM", "qwen2.5vl:7b")

BATCH = 10            # fields per first-pass image. 10 keeps the montage near 1 Mpx,
                      # which is inside the model's pixel budget - past that the input
                      # gets downscaled and the handwriting stops being legible.
ROW_H = 86            # target strip height for a first-pass batch
RETRY_BATCH = 5       # fewer strips per re-read image...
RETRY_ROW_H = 168     # ...so each field gets more height...
RETRY_MAX_W = 2000    # ...and more width, which is what the long fields actually need:
                      # an address crop is ~4400x224px, so width binds first and extra
                      # row height alone buys it nothing.
MAX_W = 1000
NUM_W = 74            # gutter holding the strip number
# Hard ceiling on the assembled image. Qwen2.5-VL caps input pixels and downscales
# anything larger, which would shrink the handwriting invisibly - better to do the
# shrinking here, where it is predictable and logged, than to let the server do it.
# A 10-field first-pass montage is ~0.95 Mpx, so this never touches the tuned path.
MAX_MONTAGE_PX = 1_200_000
TIMEOUT = 3600


# ------------------------------------------------------------------ prompting
def describe(field: dict) -> str:
    """A short expectation for the prompt. Telling the model the shape of the answer
    measurably reduces digit slips on the structured fields."""
    t = infer_type(field)
    cells = field.get("cells") or 0
    hint = (field.get("grid_hint") or "").upper()
    if t == "date":
        order = hint if hint else "DDMMYYYY"
        return f"a date written as {len(order)} digits in {order} order"
    if t == "cnic":
        return "13 digits, shown as XXXXX-XXXXXXX-X"
    if t == "iban":
        return "24 characters beginning PK, then digits/letters"
    if t == "email":
        return "an e-mail address"
    if t == "phone":
        return "a phone number (digits, may contain a dash)"
    if t == "amount":
        return "a money amount (digits)"
    if t in ("integer", "postcode", "digits", "account"):
        return f"digits only{f', {cells} boxes' if cells else ''}"
    if t == "name":
        return "a person or company name"
    return f"{cells} character boxes" if cells else "free text"


def build_prompt(items: list[dict]) -> str:
    lines = "\n".join(
        f'{i}. "{it["label"]}" - expect {describe(it)}'
        for i, it in enumerate(items, 1))
    return f"""This image contains {len(items)} separate fields cropped from one scanned bank form.
They are stacked vertically, separated by black lines, and numbered on the left.

The numbered fields are:
{lines}

For each number, read ONLY the handwritten value in that strip.
Ignore printed form text, printed labels and box borders - the handwriting is what a
customer wrote by hand. Transcribe exactly what is written; do not correct spelling or
invent anything. If a strip contains no handwriting, use "EMPTY".

Reply with JSON only: one entry for every number from 1 to {len(items)}, mapping the
number to the value you read."""


SINGLE_PROMPT = """This image is one field cropped from a scanned bank form.
The field is labelled: "{label}".
Expect {desc}.
Read ONLY the handwritten value. Ignore printed form text, labels and box borders.
Reply with the value alone and nothing else. If there is no handwriting, reply EMPTY."""


# -------------------------------------------------------------------- imaging
def strip(path: str, row_h: int = ROW_H, max_w: int = MAX_W) -> np.ndarray | None:
    """One field crop scaled to fit a row, aspect ratio preserved.

    Preserving aspect matters: address lines are ~4400px wide against ~250px tall, so
    scaling to the row height and then clamping the width - as this did originally -
    squashed them horizontally by about 1.5x. Fitting by whichever axis binds first
    keeps the letters the shape the model was trained on.
    """
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape[:2]
    f = min(row_h / h, max_w / w)
    if abs(f - 1.0) > 0.01:
        im = cv2.resize(im, None, fx=f, fy=f,
                        interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return im


def montage(items: list[dict], row_h: int = ROW_H, max_w: int = MAX_W) -> np.ndarray:
    """Numbered strips stacked into one image. Rows may differ in height - the number
    in the gutter is what ties a value to its field, not the row geometry."""
    tiles = []
    for n, it in enumerate(items, 1):
        im = strip(os.path.join(ROOT, it["crop"]), row_h, max_w)
        if im is None:
            im = np.full((24, 40), 255, np.uint8)
        h = im.shape[0]
        tile = np.full((h, NUM_W + max_w), 255, np.uint8)
        tile[:, NUM_W:NUM_W + im.shape[1]] = im
        cv2.putText(tile, str(n), (8, h // 2 + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, 0, 2, cv2.LINE_AA)
        tiles.append(tile)
        tiles.append(np.zeros((3, NUM_W + max_w), np.uint8))
    sheet = np.vstack(tiles)

    px = sheet.shape[0] * sheet.shape[1]
    if px > MAX_MONTAGE_PX:
        f = (MAX_MONTAGE_PX / px) ** 0.5
        sheet = cv2.resize(sheet, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return sheet


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


# --------------------------------------------------------------------- ollama
def keys_schema(n: int) -> dict:
    """A JSON schema requiring exactly keys "1".."n", all strings.

    Necessary, not decorative. With a bare format:"json" the model is free to emit any
    valid object, and it reliably stopped after two keys - mirroring the shape of the
    example in the prompt. Requiring the keys makes short output impossible.
    """
    keys = [str(i) for i in range(1, n + 1)]
    return {"type": "object",
            "properties": {k: {"type": "string"} for k in keys},
            "required": keys}


def generate(prompt: str, img_b64: str, fmt, num_predict: int) -> str:
    body = {"model": MODEL, "prompt": prompt, "images": [img_b64], "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read()).get("response", "")


def read_batch(items: list[dict], row_h: int = ROW_H,
               max_w: int = MAX_W) -> dict[str, str]:
    raw = generate(build_prompt(items), png_b64(montage(items, row_h, max_w)),
                   keys_schema(len(items)), 80 + 40 * len(items))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for i, it in enumerate(items, 1):
        v = data.get(str(i), data.get(i))
        if isinstance(v, (str, int, float)):
            out[it["field"]] = str(v)
    return out


def read_single(item: dict) -> str:
    """Read one field from its own full-size image.

    Not used by the pipeline - re-reads go through read_batch, because one image per
    field costs the same as a batch of ten. Kept for inspecting a single awkward field
    by hand, which is where it earns its place.
    """
    img = cv2.imread(os.path.join(ROOT, item["crop"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    prompt = SINGLE_PROMPT.format(label=item["label"], desc=describe(item))
    return generate(prompt, png_b64(img), None, 64).strip().strip('"')


# ----------------------------------------------------------------------- main
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def run(doc: str, pages=None, batch=BATCH, retry=True, force=False, limit=None,
        progress=None):
    """progress: optional callable(done_fields, total_fields, message)."""
    index = json.load(open(os.path.join(ROOT, "build", "crops", doc, "index.json"),
                           encoding="utf-8"))
    by_id = {r["field"]: r for r in index}
    cache_path = os.path.join(ROOT, "build", "reads", f"{doc}.json")
    cache = {} if force else load_cache(cache_path)

    todo = [r for r in index
            if r["needs_model"] and (pages is None or r["page"] in pages)
            and r["field"] not in cache]
    # zero-ink fields are settled without spending a call on them
    for r in index:
        if not r["needs_model"] and r["field"] not in cache:
            cache[r["field"]] = {"value": "", "source": "ink-gate", "seconds": 0.0}

    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    if limit:
        batches = batches[:limit]
    print(f"{len(todo)} fields to read in {len(batches)} batches of <= {batch} "
          f"({MODEL} at {OLLAMA})")

    t0 = time.time()
    for bi, items in enumerate(batches, 1):
        t = time.time()
        try:
            got = read_batch(items)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"  batch {bi}/{len(batches)} FAILED: {e}")
            got = {}
        dt = time.time() - t
        for it in items:
            if it["field"] in got:
                cache[it["field"]] = {"value": got[it["field"]], "source": "batch",
                                      "seconds": round(dt / len(items), 2)}
        miss = [it for it in items if it["field"] not in got]
        print(f"  batch {bi}/{len(batches)}: {len(items) - len(miss)}/{len(items)} "
              f"in {dt:.0f}s ({dt/max(len(items),1):.1f}s/field)"
              f"{f' | {len(miss)} unparsed' if miss else ''}")
        sys.stdout.flush()          # long runs are usually watched from a log file
        if progress:
            done = min(bi * batch, len(todo))
            progress(done, len(todo), f"Read {done} of {len(todo)} handwritten fields")
        save_cache(cache_path, cache)

    if retry:
        # re-read individually: anything the batch did not return, plus anything that
        # fails its format check
        bad = []
        for r in index:
            if pages is not None and r["page"] not in pages:
                continue
            c = cache.get(r["field"])
            if not r["needs_model"]:
                continue
            if c is None:
                bad.append((r, "no value"))
            elif c["source"] == "batch":
                a = assess(c["value"], r)
                if not a["valid"]:
                    bad.append((r, a["note"]))
        groups = [bad[i:i + RETRY_BATCH] for i in range(0, len(bad), RETRY_BATCH)]
        if bad:
            print(f"\nre-reading {len(bad)} field(s) in {len(groups)} batch(es) "
                  f"at {RETRY_ROW_H}px per field:")
        for gi, grp in enumerate(groups, 1):
            if progress:
                progress(len(todo), len(todo),
                         f"Re-checking uncertain fields ({gi} of {len(groups)})")
            t = time.time()
            try:
                got = read_batch([r for r, _ in grp], RETRY_ROW_H, RETRY_MAX_W)
            except (urllib.error.URLError, OSError, RuntimeError) as e:
                print(f"  re-read batch {gi}/{len(groups)} FAILED: {e}")
                continue
            dt = time.time() - t
            per = round(dt / max(len(grp), 1), 2)
            for r, why in grp:
                v = got.get(r["field"])
                if v is None:
                    print(f"  {r['label'][:34]:<36} ({why}) -> nothing returned, "
                          f"keeping first-pass value")
                    continue
                # A second reading only wins if it is well-formed; never trade a valid
                # value for an invalid one just because it came later.
                keep = assess(v, r)["valid"] or cache.get(r["field"]) is None
                print(f"  {r['label'][:34]:<36} ({why}) -> {v!r} "
                      f"{'accepted' if keep else 'rejected, keeping first-pass value'}")
                if keep:
                    cache[r["field"]] = {"value": v, "source": "retry", "seconds": per}
            print(f"  re-read batch {gi}/{len(groups)}: {dt:.0f}s ({per:.1f}s/field)")
            sys.stdout.flush()          # long runs are usually watched from a log file
            save_cache(cache_path, cache)

    print(f"\ntotal {time.time() - t0:.0f}s -> {cache_path}")
    return cache, by_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", nargs="?", default="Cif-form")
    ap.add_argument("--pages", type=int, nargs="*")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--limit", type=int, help="only this many batches (for testing)")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached reads")
    a = ap.parse_args()
    run(a.doc, a.pages, a.batch, not a.no_retry, a.force, a.limit)
