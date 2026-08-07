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
import re
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from validate import OCR_TYPES, assess, infer_type

# field labels carry non-ASCII typography; never let printing them kill a long run
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Two readers, each given the fields its architecture suits. Measured on page 1:
#
#                    structured (16)   words (28)
#   qwen2.5vl:7b       14 (87%)         27 (96%)
#   glm-ocr (1.1B)     16 (100%)        22 (79%)
#
# A general VLM reads cursive well because it knows the language - shown a blurry scrawl
# it recovers "Fountain". That same prior is what makes it invent a digit: a CNIC has no
# vocabulary to fall back on, so "35810" came back as "35820" - well-formed, and therefore
# undetectable by any format check. An OCR model has no such prior and simply copies the
# characters, which is why the 1.1B model beats the 8.3B one on every structured field.
# Prompting cannot close this gap: the shape of each value is already in the prompt (see
# describe()), and pushing harder on wording plus double row height scored 12/16 and began
# attaching values to the wrong strip.
MODEL = os.environ.get("HBL_VLM", "qwen2.5vl:7b")            # names, addresses, free text
MODEL_NUM = os.environ.get("HBL_VLM_NUM", "glm-ocr:latest")  # dates, CNIC, phone, amounts

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
ADDRESSY = re.compile(r"house|appt|apartment|flat|plot|street|road|address|building|"
                      r"floor|block|sector|village|mohallah?|landmark|town|office no")
# Place names are where a general VLM does its worst damage, because its prior pulls an
# unfamiliar Pakistani name toward a familiar English word - "Jhang" came back as "Thang".
PLACEY = re.compile(r"\bcity\b|district|\bcountry\b|province|\bplace\b|nationalit|"
                    r"landmark|\barea\b|\btown\b|birth")


def describe(field: dict) -> str:
    """A short expectation for the prompt, using the real Pakistani format for the field.

    These are the formats as issued in Pakistan, not patterns copied off a sample form.
    Stating the exact digit count is what stops a reader inventing or dropping one: told
    only "a phone number", a reader will happily return ten digits or twelve.
    """
    t = infer_type(field)
    label = (field.get("label") or "").lower()
    cells = field.get("cells") or 0
    hint = (field.get("grid_hint") or "").upper()
    if t == "date":
        order = hint if hint else "DDMMYYYY"
        return (f"a date as {len(order)} digits in {order} order "
                "(day 01-31, month 01-12, 4-digit year)")
    # The template's own grid encodes the real format including its separators: a CNIC box
    # is 15 cells for 13 digits and 2 dashes, a mobile 12 for 11 digits and 1 dash. Giving
    # the reader the cell count as well as the rule lets it count boxes against the answer.
    boxes = f", across {cells} character boxes" if cells else ""
    if t == "cnic":
        return ("a Pakistani CNIC: exactly 13 digits as XXXXX-XXXXXXX-X"
                f"{boxes} (each dash takes a box). "
                "The first digit is the issuing region, 1-8, never 0")
    if t == "iban":
        return ("a Pakistani IBAN: exactly 24 characters - PK, 2 digits, a 4-LETTER bank "
                f"code (HBL is HABB), then 16 digits/letters{boxes}")
    if t == "passport":
        return "a Pakistani passport number: 2 capital letters then 7 digits"
    if t == "email":
        return "an e-mail address"
    if t == "mobile":
        return ("a Pakistani mobile: exactly 11 digits starting 03, as 03XX-XXXXXXX"
                f"{boxes} (the dash takes one). Real prefixes are 0300-0349 and 0355")
    if t == "phone":
        return ("a Pakistani landline: the city code then the subscriber number, one dash "
                f"between them{boxes}. Karachi 021 and Lahore 042 take 8-digit subscriber "
                "numbers (11 digits in total); every other city takes 7 (10 in total) - "
                "e.g. 051 Islamabad, 061 Multan, 091 Peshawar, 042 Lahore")
    if t == "amount":
        return ("an amount in Pakistani Rupees - digits, possibly with thousands commas "
                "and a trailing /- ; transcribe the digits as written")
    if t == "postcode":
        return "a Pakistani postal code: exactly 5 digits (e.g. 74000 Karachi, 54000 Lahore)"
    if t == "ntn":
        return ("an FBR National Tax Number: 7 digits. An individual may instead write "
                "their 13-digit CNIC, which is also valid")
    if t in ("integer", "digits", "account"):
        return f"digits only{f', {cells} boxes' if cells else ''}"
    if t == "name":
        return ("a Pakistani person or company name: letters and spaces only, no digits. "
                "It may contain an initial with a full stop")
    if ADDRESSY.search(label):
        return ("an address line. It mixes letters and digits and may contain / or - "
                "(e.g. House No. 43/2, Flat 5-B, Street 12). After Block, Sector or Phase "
                "either a DIGIT or a LETTER is possible - copy exactly which one is written")
    if PLACEY.search(label):
        return ("a Pakistani place name - letters, spaces and hyphens. Unfamiliar spellings "
                "are normal here (Jhang, Toba Tek Singh, Gulshan-e-Iqbal, Nazimabad, "
                "Bahawalpur). Copy the letters exactly; never substitute a better-known "
                "name that merely looks similar")
    return f"{cells} character boxes" if cells else "free text"


def build_prompt(items: list[dict], conts: dict | None = None,
                 model: str = "") -> str:
    conts = conts or {}

    def line(i, it):
        n = len(conts.get(it["field"], ()))
        wrapped = (f", written across {n + 1} lines - read them as one value" if n else "")
        return f'{i}. "{it["label"]}" - expect {describe(it)}{wrapped}'

    lines = "\n".join(line(i, it) for i, it in enumerate(items, 1))

    # Each reader is warned about the mistake it actually makes. The OCR model has no
    # language prior, so it needs the digit counts held in front of it and must be told to
    # stop talking. The general VLM has too strong a prior: it pulled "Convention" to
    # "Conversion" and "35810" to "35820" because the commoner form fitted better.
    if model == MODEL_NUM:
        care = """These are Pakistani bank form fields. Count the characters before you answer:
if the field says 11 digits, return exactly 11 digits - do not pad and do not drop one.
Read each digit from its own pen strokes, not from what would look like a plausible number.
Give the value only. Do not repeat it, do not explain it, do not add any other text."""
    else:
        care = """These are Pakistani bank form fields, filled in by hand in Pakistan.
Transcribe exactly what the pen shows, letter by letter. Do NOT change an unusual word into
a more common one - Pakistani names, companies and places are often unfamiliar, and the
unfamiliar reading is usually the correct one. Never "tidy" a value: keep the writer's
spelling, spacing and capitalisation."""

    return f"""This image contains {len(items)} separate fields cropped from one scanned bank form.
They are stacked vertically, separated by black lines, and numbered on the left.

The numbered fields are:
{lines}

{care}

For each number, read ONLY the handwritten value in that strip.
Ignore printed form text, printed labels, Urdu text and box borders - some of it bleeds into
the crop, and a printed unit such as "years" beside a handwritten "20" is NOT part of the
answer. If a strip contains no handwriting, use "EMPTY".

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


def field_strip(it: dict, row_h: int, max_w: int, extra=()) -> np.ndarray | None:
    """The image for one field: its own crop, with any continuation lines stacked beneath.

    A wrapped answer is deliberately shown as one strip. Read separately, the first line of
    the Next-of-Kin address came back as "Block B"; read with its second line visible, the
    same pixels came back "Block 6" - with "Gulshan-e-Iqbal" following it, a block *number*
    is the reading that fits. Presenting them joined is how the model gets that context on
    purpose, instead of us hoping it merges two adjacent strips by itself, which it
    sometimes does and which silently empties the second box when it happens.

    The lines are separated by white, not by the montage's black rule: within a field they
    are one wrapped answer, and the black rule means "different field".
    """
    ims = []
    for part in (it, *extra):
        im = strip(os.path.join(ROOT, part["crop"]), row_h, max_w)
        if im is not None:
            ims.append(im)
    if not ims:
        return None
    if len(ims) == 1:
        return ims[0]
    w = max(i.shape[1] for i in ims)
    rows: list[np.ndarray] = []
    for i in ims:
        if i.shape[1] < w:
            i = np.hstack([i, np.full((i.shape[0], w - i.shape[1]), 255, np.uint8)])
        rows.append(i)
        rows.append(np.full((2, w), 255, np.uint8))      # a line gap, not a field divider
    return np.vstack(rows[:-1])


def montage(items: list[dict], row_h: int = ROW_H, max_w: int = MAX_W,
            conts: dict | None = None) -> np.ndarray:
    """Numbered strips stacked into one image. Rows may differ in height - the number
    in the gutter is what ties a value to its field, not the row geometry."""
    conts = conts or {}
    tiles = []
    for n, it in enumerate(items, 1):
        im = field_strip(it, row_h, max_w, conts.get(it["field"], ()))
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


def reader_for(field: dict) -> str:
    """Which model reads this field. The template already told us every field's type at
    schema time, so the routing costs nothing to work out."""
    return MODEL_NUM if infer_type(field) in OCR_TYPES else MODEL


def first_line(v: str) -> str:
    """glm-ocr answers correctly and then keeps talking - '1023\\n```markdown\\n1023'.
    The first non-empty line is the value. Scoped to the OCR reader: qwen returns clean
    strings, and a blanket first-line rule would silently truncate anything multi-line.
    """
    for ln in str(v).splitlines():
        ln = ln.strip().strip("`").strip()
        if ln:
            return ln
    return ""


def generate(prompt: str, img_b64: str, fmt, num_predict: int, model: str = "") -> str:
    body = {"model": model or MODEL, "prompt": prompt, "images": [img_b64],
            "stream": False, "keep_alive": "1h",     # never reload 6GB between batches
            "options": {"temperature": 0, "num_predict": num_predict}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read()).get("response", "")


def read_batch(items: list[dict], row_h: int = ROW_H, max_w: int = MAX_W,
               model: str = "", conts: dict | None = None) -> dict[str, str]:
    model = model or MODEL
    raw = generate(build_prompt(items, conts, model),
                   png_b64(montage(items, row_h, max_w, conts)),
                   keys_schema(len(items)), 80 + 40 * len(items), model)
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
            out[it["field"]] = first_line(v) if model == MODEL_NUM else str(v)
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
def _clashes(it: dict, batch: list[dict]) -> bool:
    """True if `it` and something already in `batch` are two lines of one answer."""
    cont = it.get("continuation_of", "")
    if cont and any(b["field"] == cont for b in batch):
        return True
    return any(b.get("continuation_of", "") == it["field"] for b in batch)


def split_batches(items: list[dict], size: int) -> list[list[dict]]:
    """Batches of <= size that never hold a field and its own continuation.

    Normally moot: run() folds a continuation into its parent's strip, so it is not a
    batch item at all. This stays as the guard for a continuation that reaches the queue
    anyway - a crop index built before `continuation_of` existed, say. Both in one montage
    is the case to avoid: two adjacent strips of visually continuous handwriting get read
    as a single value, which returns the whole answer in the first box and empties the
    second. Continuations go to the front so the pair separates without stranding an
    under-filled image, which costs as much to run as a full one.
    """
    items = ([it for it in items if it.get("continuation_of")]
             + [it for it in items if not it.get("continuation_of")])
    out: list[list[dict]] = []
    cur: list[dict] = []
    for it in items:
        if cur and (len(cur) >= size or _clashes(it, cur)):
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def available_models() -> set[str]:
    """What Ollama actually has pulled. Returns an empty set if it cannot be asked, which
    the caller reads as "assume it is there" - a health probe must not abort a real run."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=15) as r:
            return {m["name"] for m in json.loads(r.read()).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return set()


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

    # A wrapped answer is read as one strip, so a continuation never gets a call of its
    # own: it is shown beneath its parent and its text arrives inside the parent's value.
    conts: dict[str, list] = {}
    for r in index:
        parent = r.get("continuation_of")
        if parent and (pages is None or r["page"] in pages):
            conts.setdefault(parent, []).append(r)

    todo = [r for r in index
            if r["needs_model"] and (pages is None or r["page"] in pages)
            and r["field"] not in cache and not r.get("continuation_of")]
    # zero-ink fields are settled without spending a call on them
    for r in index:
        if not r["needs_model"] and r["field"] not in cache:
            cache[r["field"]] = {"value": "", "source": "ink-gate", "seconds": 0.0}

    # A montage may only hold one reader's fields, so the two groups are batched apart.
    # This is also why routing shifts the text values slightly: taking the numeric fields
    # out changes which text fields share an image, and the montage is a shared canvas.
    avail = available_models()
    numeric_reader = MODEL_NUM if (not avail or MODEL_NUM in avail) else MODEL
    if numeric_reader != MODEL_NUM:
        print(f"WARNING: {MODEL_NUM} is not installed - reading numeric fields with "
              f"{MODEL} instead (expect digit slips it cannot detect)")

    groups: dict[str, list] = {}
    for r in todo:
        m = numeric_reader if reader_for(r) == MODEL_NUM else MODEL
        groups.setdefault(m, []).append(r)

    batches = [(m, chunk) for m, g in groups.items()
               for chunk in split_batches(g, batch)]
    if limit:
        batches = batches[:limit]
    print(f"{len(todo)} fields to read in {len(batches)} batches of <= {batch} at {OLLAMA}")
    for m, g in groups.items():
        print(f"  {m:<18} {len(g)} fields "
              f"({'digits' if m == numeric_reader and m != MODEL else 'words'})")

    t0 = time.time()
    for bi, (model, items) in enumerate(batches, 1):
        t = time.time()
        try:
            got = read_batch(items, model=model, conts=conts)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"  batch {bi}/{len(batches)} FAILED: {e}")
            got = {}
        dt = time.time() - t
        for it in items:
            if it["field"] in got:
                cache[it["field"]] = {"value": got[it["field"]], "source": "batch",
                                      "model": model,
                                      "seconds": round(dt / len(items), 2)}
                # its continuation lines were part of that same strip, so they are
                # settled too - recorded explicitly so nothing downstream reads them
                # as a filled box that came back empty
                for c in conts.get(it["field"], ()):
                    cache[c["field"]] = {"value": "", "source": "with-parent",
                                         "parent": it["field"], "model": model,
                                         "seconds": 0.0}
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
            if not r["needs_model"] or r.get("continuation_of"):
                continue        # a continuation has no reading of its own to re-check
            if c is None:
                bad.append((r, "no value"))
            elif c["source"] == "batch":
                a = assess(c["value"], r)
                if not a["valid"]:
                    bad.append((r, a["note"]))
        # grouped by reader as well as by size: a re-read goes back to the model that
        # owns the field, so a field never changes hands between passes
        per_model: dict[str, list] = {}
        for r, why in bad:
            m = numeric_reader if reader_for(r) == MODEL_NUM else MODEL
            per_model.setdefault(m, []).append((r, why))
        groups = []
        for m, g in per_model.items():
            why_of = {r["field"]: w for r, w in g}
            for chunk in split_batches([r for r, _ in g], RETRY_BATCH):
                groups.append((m, [(r, why_of[r["field"]]) for r in chunk]))
        if bad:
            print(f"\nre-reading {len(bad)} field(s) in {len(groups)} batch(es) "
                  f"at {RETRY_ROW_H}px per field:")
        for gi, (model, grp) in enumerate(groups, 1):
            if progress:
                progress(len(todo), len(todo),
                         f"Re-checking uncertain fields ({gi} of {len(groups)})")
            t = time.time()
            try:
                got = read_batch([r for r, _ in grp], RETRY_ROW_H, RETRY_MAX_W,
                                 model, conts)
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
                    cache[r["field"]] = {"value": v, "source": "retry",
                                         "model": model, "seconds": per}
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
